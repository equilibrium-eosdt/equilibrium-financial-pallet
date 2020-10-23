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
use sp_std::iter::Iterator;

pub struct Asset;

pub trait OnPriceSet {
    type Asset;
    type Price;

    fn on_price_set(asset: Self::Asset, value: Self::Price) -> Result<(), DispatchError>;
}

pub trait IntoTypeIterator: Sized {
    type Iterator: Iterator<Item = Self>;

    fn into_type_iter() -> Self::Iterator;
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
