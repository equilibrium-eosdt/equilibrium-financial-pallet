// Copyright (C) 2020 equilibrium.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//#![warn(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]

pub mod capvec;

use core::time::Duration;
use frame_support::dispatch::DispatchError;
use sp_std::convert::{From, TryFrom};
use sp_std::vec::Vec;

pub trait OnPriceSet {
    type Asset;
    type Price;

    fn on_price_set(asset: Self::Asset, value: Self::Price) -> Result<(), DispatchError>;
}

pub trait BalanceAware {
    type AccountId;
    type Asset;
    type Balance;

    fn balances(
        account_id: &Self::AccountId,
        assets: &[Self::Asset],
    ) -> Result<Vec<Self::Balance>, DispatchError>;
}

#[derive(Debug, Eq, PartialEq)]
pub enum PricePeriodError {
    DivisionByZero,
    Overflow,
}

pub type PricePeriodResult<T> = Result<T, PricePeriodError>;

pub struct PricePeriod(pub u32);

impl PricePeriod {
    pub fn get_period_id(&self, now: Duration) -> Result<u64, PricePeriodError> {
        let seconds = now.as_secs();
        let period = self.0 as u64;

        seconds
            .checked_div(60)
            .and_then(|x| x.checked_div(period))
            .ok_or(PricePeriodError::DivisionByZero)
    }

    pub fn get_period_id_start(&self, period_id: u64) -> Result<Duration, PricePeriodError> {
        let period = self.0 as u64;
        let seconds: Result<_, PricePeriodError> = period_id
            .checked_mul(60)
            .and_then(|x| x.checked_mul(period))
            .ok_or(PricePeriodError::Overflow);

        Ok(Duration::from_secs(seconds?))
    }

    pub fn get_period_start(&self, now: Duration) -> Result<Duration, PricePeriodError> {
        self.get_period_id_start(self.get_period_id(now)?)
    }

    pub fn is_valid_period_start(&self, period_start: Duration) -> PricePeriodResult<bool> {
        Ok(period_start == self.get_period_start(period_start)?)
    }
}

/// Indicates what returns will be used in calculations of volatilities, correlations, and value at
/// risk: `Regular` or `Log` returns.
///
/// The choice of return type also governs the method for Value at Risk (VAR) calculation:
/// * `Regular` type should be used when arithmetic returns are used and are assumed to be normally
/// distributed;
/// * `Log` normal type should be used when geometric returns (log returns) are used
/// and are assumed to be normally distributed.
///
/// We suggest using the latter approach, as it doesn't
/// lead to losses greater than a portfolio value unlike the normal VaR.
#[derive(Copy, Clone, Debug)]
pub enum CalcReturnType {
    /// Regular returns.
    Regular,

    /// Log returns.
    Log,
}

impl CalcReturnType {
    pub const fn into_u32(&self) -> u32 {
        match self {
            CalcReturnType::Regular => 0,
            CalcReturnType::Log => 1,
        }
    }
}

impl TryFrom<u32> for CalcReturnType {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(CalcReturnType::Regular),
            1 => Ok(CalcReturnType::Log),
            _ => Err(()),
        }
    }
}

impl From<CalcReturnType> for u32 {
    fn from(value: CalcReturnType) -> Self {
        value.into_u32()
    }
}

/// Indicates the method for calculating volatility: `Regular` or `Exponential`.
#[derive(Copy, Clone, Debug)]
pub enum CalcVolatilityType {
    /// Regular type is a standard statistical approach of calculating standard deviation of
    /// returns using simple average.
    Regular,

    /// Exponentially weighted type gives more weight to most recent data given the decay value or
    /// period of exponentially weighted moving average.
    Exponential(u32),
}

impl CalcVolatilityType {
    pub const fn into_i64(&self) -> i64 {
        match self {
            CalcVolatilityType::Regular => -1,
            CalcVolatilityType::Exponential(n) => *n as i64,
        }
    }
}

impl TryFrom<i64> for CalcVolatilityType {
    type Error = ();

    fn try_from(value: i64) -> Result<Self, Self::Error> {
        match value {
            -1 => Ok(CalcVolatilityType::Regular),
            n if n >= 0 => Ok(CalcVolatilityType::Exponential(n as u32)),
            _ => Err(()),
        }
    }
}

impl From<CalcVolatilityType> for i64 {
    fn from(value: CalcVolatilityType) -> i64 {
        value.into_i64()
    }
}

#[cfg(test)]
mod tests {
    #[cfg(test)]
    mod price_period {
        use super::super::*;
        use chrono::prelude::*;

        fn create_duration(
            year: i32,
            month: u32,
            day: u32,
            hour: u32,
            minute: u32,
            second: u32,
        ) -> Duration {
            let timestamp = Utc
                .ymd(year, month, day)
                .and_hms(hour, minute, second)
                .timestamp();
            Duration::from_secs(timestamp as u64)
        }

        #[test]
        fn period_id_are_same_within_period() {
            let time1 = create_duration(2020, 9, 14, 12, 31, 0);
            let time2 = create_duration(2020, 9, 14, 12, 54, 0);

            // period is one hour
            let period = PricePeriod(60);

            let period_id_1 = period.get_period_id(time1);
            let period_id_2 = period.get_period_id(time2);

            assert!(period_id_1.is_ok());
            assert!(period_id_2.is_ok());
            assert_eq!(period_id_1, period_id_2);
        }

        #[test]
        fn period_id_are_different_for_different_periods() {
            let time1 = create_duration(2020, 9, 14, 12, 31, 0);
            let time2 = create_duration(2020, 9, 14, 13, 2, 0);

            // period is one hour
            let period = PricePeriod(60);

            let period_id_1 = period.get_period_id(time1);
            let period_id_2 = period.get_period_id(time2);

            assert!(period_id_1.is_ok());
            assert!(period_id_2.is_ok());
            assert!(period_id_1 != period_id_2);
        }
    }
}
