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

//! # Financial Pallet Module
//! ## Overview
//!
//! Equilibrium's Financial Pallet is an open-source substrate module that subscribes to external
//! price feed/oracle, gathers asset prices and calculates financial metrics based on the
//! information collected.
//!
//! ## Genesis Config
//!
//! You can provide initial price logs for the assets using [`GenesisConfig`](./struct.GenesisConfig.html).

#![warn(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::ops::{AddAssign, BitOrAssign, ShlAssign};
use core::time::Duration;
use financial_primitives::capvec::CapVec;
use financial_primitives::{IntoTypeIterator, OnPriceSet, PricePeriod, PricePeriodError};
use frame_support::codec::{Codec, Decode, Encode};
use frame_support::dispatch::{DispatchError, Parameter};
use frame_support::traits::{Get, UnixTime};
use frame_support::{decl_error, decl_event, decl_module, decl_storage, dispatch, ensure};
use frame_system::ensure_signed;
use math::{
    calc_return_func, calc_return_func_exp_vola, calc_return_iter, decay, demeaned, exp_corr,
    exp_vola, last_recurrent_ewma, mean, mul, regular_corr, regular_vola, squared, sum, ConstType,
    MathError, MathResult,
};
use sp_std::cmp::{max, min};
use sp_std::convert::TryInto;
use sp_std::iter::Iterator;
use sp_std::ops::Range;
use sp_std::prelude::Vec;
use sp_std::vec;
use substrate_fixed::traits::{FixedSigned, ToFixed};
use substrate_fixed::transcendental::sqrt;

pub use math::{CalcCorrelationType, CalcReturnType, CalcVolatilityType};

mod math;

#[cfg(test)]
mod mock;

#[cfg(test)]
mod tests;

/// The module configuration trait.
pub trait Trait: frame_system::Trait {
    /// The overarching event type.
    type Event: From<Event<Self>> + Into<<Self as frame_system::Trait>::Event>;
    /// Implementation for the current unix timestamp provider. The
    /// [`pallet_timestamp`](https://crates.parity.io/pallet_timestamp/index.html) is
    /// right choice in most cases.
    type UnixTime: UnixTime;
    /// Number of price data points stored and used for calculations.
    type PriceCount: Get<u32>;
    /// The period of the collected prices in minutes.
    type PricePeriod: Get<u32>;
    /// System wide type for representing various assets such as BTC, ETH, EOS, etc.
    type Asset: Parameter + Copy + IntoTypeIterator;
    /// Primitive integer type that [`FixedNumber`](#associatedtype.FixedNumber) based on.
    type FixedNumberBits: Copy + ToFixed + AddAssign + BitOrAssign + ShlAssign;
    /// Fixed point data type with a required precision that used for all financial calculations.
    type FixedNumber: Clone
        + Copy
        + Codec
        + FixedSigned<Bits = Self::FixedNumberBits>
        + PartialOrd<ConstType>
        + From<ConstType>;
    /// System wide type for representing price values. It must be convertable to and
    /// from [`FixedNumber`](#associatedtype.FixedNumber).
    type Price: Parameter + Clone + From<Self::FixedNumber> + Into<Self::FixedNumber>;
}

/// Information about latest price update.
#[derive(Encode, Decode, Clone, Default, PartialEq, Eq, Debug)]
pub struct PriceUpdate<P> {
    /// Timestamp of the price period start for the latest price received.
    pub period_start: Duration,
    /// Latest price timestamp.
    pub time: Duration,
    /// Latest price value.
    pub price: P,
}

impl<P> PriceUpdate<P> {
    #[cfg(test)]
    fn new(period_start: Duration, time: Duration, price: P) -> PriceUpdate<P> {
        PriceUpdate {
            period_start,
            time,
            price,
        }
    }
}

/// History of price changes for asset
#[derive(Encode, Decode, Clone, Default, PartialEq, Eq, Debug)]
pub struct PriceLog<F> {
    /// Timestamp of the latest point in the log.
    pub latest_timestamp: Duration,
    /// History of prices changes for last [`PriceCount`](./trait.Trait.html#associatedtype.PriceCount) periods in succesion.
    pub prices: CapVec<F>,
}

/// Financial metrics for asset
#[derive(Encode, Decode, Clone, Default, PartialEq, Eq, Debug)]
pub struct FinancialMetrics<A, P> {
    /// Start of the period inclusive for which metcis were calculated.
    pub period_start: Duration,
    /// End of the period exclusive for which metcis were calculated.
    pub period_end: Duration,
    /// Log returns
    pub log_returns: Vec<P>,
    /// Volatility
    pub volatility: P,
    /// Correlations for all assets
    pub correlations: Vec<(A, P)>,
}

decl_storage! {
    trait Store for Module<T: Trait> as FinancialModule {
        /// Latest price updates on per asset basis.
        Updates get(fn updates): map hasher(blake2_128_concat) T::Asset => Option<PriceUpdate<T::FixedNumber>>;
        /// Price log on per asset basis.
        PriceLogs get(fn price_logs): map hasher(blake2_128_concat) T::Asset => PriceLog<T::FixedNumber>;
        /// Financial metrics on per asset basis.
        Metrics get(fn metrics): map hasher(blake2_128_concat) T::Asset => Option<FinancialMetrics<T::Asset, T::Price>>;
    }

    add_extra_genesis {
        /// Initial price logs on per asset basis.
        config(prices): Vec<(T::Asset, Vec<T::Price>, Duration)>;

        build(|config| {
            let max_price_count = 180;
            // Limit max value of `PricePeriod` to 7 days
            let max_price_period = 7 * 24 * 60;

            let price_count = T::PriceCount::get();
            assert!(price_count <= max_price_count, "PriceCount can not be greater than {}", max_price_count);

            let price_period = T::PricePeriod::get();
            assert!(price_period <= max_price_period, "PricePeriod can not be greater than {}", max_price_period);

            // We assume that each config item for a given asset contains prices of the past perionds going in
            // succession. Timestamp of the last period is specified in `latest_timestamp`.
            for (asset, values, latest_timestamp) in config.prices.iter() {
                let mut prices = CapVec::<T::FixedNumber>::new(price_count);

                for v in values.iter() {
                    prices.push(v.clone().into());
                }

                PriceLogs::<T>::insert(asset, PriceLog {
                    latest_timestamp: *latest_timestamp,
                    prices
                });
            }
        });
    }
}

decl_event!(
    pub enum Event<T>
    where
        Asset = <T as Trait>::Asset,
    {
        /// Financial parameters for the specified asset have been recalculeted
        Recalculated(Asset),
    }
);

decl_error! {
    /// Error for the Financial Pallet module.
    pub enum Error for Module<T: Trait> {
        /// Timestamp of the received price is in the past.
        PeriodIsInThePast,
        /// Overflow occured during financial calculation process.
        Overflow,
        /// Division by zero occured during financial calculation process.
        DivisionByZero,
        /// Price log is not long enough to calculate required value.
        NotEnoughPoints,
        /// Required functionality is not implemented.
        NotImplemented,
        /// Storage of the pallet is in an unexpected state.
        InvalidStorage,
        /// Specified period start timestamp is invalid for current
        /// [`PricePeriod`](./trait.Trait.html#associatedtype.PricePeriod) value.
        InvalidPeriodStart,
        /// An invalid argument passed to the transcendental function (i.e. log, sqrt, etc.)
        /// during financial calculation process.
        Transcendental,
        /// An invalid argument passed to the function.
        InvalidArgument,
    }
}

decl_module! {
    /// Financial Pallet module declaration.
    pub struct Module<T: Trait> for enum Call where origin: T::Origin {
        type Error = Error<T>;

        fn deposit_event() = default;

        const PriceCount: u32 = T::PriceCount::get();
        const PricePeriod: u32 = T::PricePeriod::get();

        /// Recalculates financial metrics for a given asset
        #[weight = 10_000 + T::DbWeight::get().writes(1)]
        pub fn recalc(origin, asset: T::Asset) -> dispatch::DispatchResult {
            ensure_signed(origin)?;

            let return_type = CalcReturnType::Log;
            let log_returns = <Module<T> as Financial>::calc_return(return_type, asset)?;
            let volatility = <Module<T> as Financial>::calc_vol(return_type, CalcVolatilityType::Regular, asset)?;

            let mut correlations = vec![];

            let mut period: Option<Range<Duration>> = None;

            for asset2 in T::Asset::into_type_iter() {
                let (correlation, curr_period) = <Module<T> as Financial>::calc_corr(return_type, CalcCorrelationType::Regular, asset, asset2)?;

                if let Some(ref prev_period) = period {
                    // Ensure that all correlations calculated for the same period
                    if prev_period != &curr_period {
                        return Err(Error::<T>::NotEnoughPoints.into());
                    }
                } else {
                    period = Some(curr_period);
                }

                correlations.push((asset2, correlation));
            }

            if let Some(period) = period {
                Metrics::<T>::mutate(&asset, |metrics| {
                    *metrics = Some(FinancialMetrics {
                        period_start: period.start,
                        period_end: period.end,
                        log_returns,
                        volatility,
                        correlations,
                    });
                });

                Ok(())
            } else {
                Err(Error::<T>::NotEnoughPoints.into())
            }
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
enum GetNewPricesError {
    Overflow,
}

fn get_new_prices<P: Clone>(
    last_price: Option<P>,
    new_price: P,
    empty_periods: u32,
    max_periods: u32,
) -> Result<Vec<P>, GetNewPricesError> {
    match last_price {
        None => Ok(vec![new_price]),
        Some(last_price) => {
            // Calculate how many values to pre-populate the array with
            // We will pre-fill up to `max_periods` elements (leaving out one for the new price)
            let prices_size = min(
                empty_periods,
                max_periods
                    .checked_sub(1)
                    .ok_or(GetNewPricesError::Overflow)?,
            ) as usize;

            // Init the vector filled with last_price
            let mut new_prices = vec![last_price.clone(); prices_size];

            new_prices.push(new_price);

            Ok(new_prices)
        }
    }
}

impl<T: Trait> From<GetNewPricesError> for Error<T> {
    fn from(error: GetNewPricesError) -> Self {
        match error {
            GetNewPricesError::Overflow => Error::<T>::Overflow,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
enum PricePeriodAction {
    RemainsUnchanged,
    StartedNew(u32),
}

#[derive(Debug, Eq, PartialEq)]
struct PricePeriodChange {
    pub period_start: Duration,
    pub action: PricePeriodAction,
}

#[derive(Debug, Eq, PartialEq)]
enum PricePeriodChangeError {
    DivisionByZero,
    Overflow,
    PeriodIsInThePast,
    InvalidPeriodStart,
}

impl From<PricePeriodError> for PricePeriodChangeError {
    fn from(error: PricePeriodError) -> Self {
        match error {
            PricePeriodError::DivisionByZero => PricePeriodChangeError::DivisionByZero,
            PricePeriodError::Overflow => PricePeriodChangeError::Overflow,
        }
    }
}

impl<T: Trait> From<PricePeriodChangeError> for Error<T> {
    fn from(error: PricePeriodChangeError) -> Self {
        match error {
            PricePeriodChangeError::DivisionByZero => Error::<T>::DivisionByZero,
            PricePeriodChangeError::Overflow => Error::<T>::Overflow,
            PricePeriodChangeError::PeriodIsInThePast => Error::<T>::PeriodIsInThePast,
            PricePeriodChangeError::InvalidPeriodStart => Error::<T>::InvalidPeriodStart,
        }
    }
}

/// Calculates start timestamp of the period which contains `now` timestamp.
/// Also  calculates number of periods elapsed since `prev_start` timestamp up until `now` timestamp.
fn get_curr_period_info(
    price_period: &PricePeriod,
    prev_start: Duration,
    now: Duration,
) -> Result<(Duration, i32), PricePeriodError> {
    let prev_period_id = price_period.get_period_id(prev_start)?;
    let curr_period_id = price_period.get_period_id(now)?;

    let prev: i32 = prev_period_id
        .try_into()
        .map_err(|_| PricePeriodError::Overflow)?;
    let curr: i32 = curr_period_id
        .try_into()
        .map_err(|_| PricePeriodError::Overflow)?;
    let delta: Result<_, PricePeriodError> =
        curr.checked_sub(prev).ok_or(PricePeriodError::Overflow);
    Ok((price_period.get_period_id_start(curr_period_id)?, delta?))
}

/// Dicides if the period change took place.
fn get_period_change(
    price_period: &PricePeriod,
    period_start: Option<Duration>,
    now: Duration,
) -> Result<PricePeriodChange, PricePeriodChangeError> {
    if let Some(period_start) = period_start {
        ensure!(
            price_period.is_valid_period_start(period_start)?,
            PricePeriodChangeError::InvalidPeriodStart
        );
    }

    match period_start {
        // No `period_start` exists. It means that we received price update for the first time.
        None => {
            let period_start = price_period.get_period_start(now)?;

            Ok(PricePeriodChange {
                period_start,
                action: PricePeriodAction::StartedNew(0),
            })
        }
        Some(last_start) => {
            let (current_start, periods_elapsed) =
                get_curr_period_info(price_period, last_start, now)?;

            if periods_elapsed < 0 {
                // Current period is in the past

                Err(PricePeriodChangeError::PeriodIsInThePast)
            } else if periods_elapsed == 0 {
                // Period is not changed

                Ok(PricePeriodChange {
                    period_start: last_start,
                    action: PricePeriodAction::RemainsUnchanged,
                })
            } else {
                // Period is changed

                let empty_periods = (periods_elapsed - 1) as u32;

                Ok(PricePeriodChange {
                    period_start: current_start,
                    action: PricePeriodAction::StartedNew(empty_periods),
                })
            }
        }
    }
}

impl<T: Trait> OnPriceSet for Module<T> {
    type Asset = T::Asset;
    type Price = T::Price;

    fn on_price_set(asset: T::Asset, value: T::Price) -> Result<(), DispatchError> {
        let value: T::FixedNumber = value.into();
        let now = T::UnixTime::now();
        let price_count = T::PriceCount::get();

        let update = Self::updates(asset);
        let log = Self::price_logs(asset);

        // Maximum price log size can be either 0 or `PriceCount`.
        // If size is 0 it means then no information exists for a given `asset` yet.
        // If size is `PriceCount` then we already have some price points received by previous
        // `on_price_set` call or from genesis block.
        ensure!(
            log.prices.len_cap() == 0 || log.prices.len_cap() == price_count,
            Error::<T>::InvalidStorage
        );
        let init_prices = log.prices.len_cap() == 0;

        let period_start = update.map(|x| x.period_start);
        let price_period = PricePeriod(T::PricePeriod::get());
        let period_change =
            get_period_change(&price_period, period_start, now).map_err(Into::<Error<T>>::into)?;

        let period_start = period_change.period_start;

        // Every point received is stores to the `Update`.
        // Meanwhile only first point received in the current period is stored in the price log.
        match period_change.action {
            PricePeriodAction::RemainsUnchanged => {
                Updates::<T>::mutate(&asset, |update| {
                    *update = Some(PriceUpdate {
                        period_start: period_start,
                        time: now,
                        price: value,
                    });
                });
            }
            PricePeriodAction::StartedNew(empty_periods) => {
                let new_prices = get_new_prices(
                    log.prices.last().cloned(),
                    value,
                    empty_periods,
                    price_count,
                )
                .map_err(Into::<Error<T>>::into)?;

                Updates::<T>::mutate(&asset, |update| {
                    *update = Some(PriceUpdate {
                        period_start: period_start,
                        time: now,
                        price: value,
                    })
                });
                PriceLogs::<T>::mutate(&asset, |log| {
                    log.latest_timestamp = now;

                    if init_prices {
                        log.prices = CapVec::<T::FixedNumber>::new(price_count);
                    }

                    for p in new_prices {
                        log.prices.push(p);
                    }
                });
            }
        }

        Ok(())
    }
}

/// Important financial functions that uses [`prices`](./struct.Module.html#method.prices) as a data source.
pub trait Financial {
    /// System wide type for representing various assets such as BTC, ETH, EOS, etc.
    type Asset;
    /// System wide type for representing price values.
    type Price;
    /// System wide type for representing account id.
    type AccountId;

    /// Calculates return.
    fn calc_return(
        return_type: CalcReturnType,
        asset: Self::Asset,
    ) -> Result<Vec<Self::Price>, DispatchError>;
    /// Calculates volatility.
    fn calc_vol(
        return_type: CalcReturnType,
        volatility_type: CalcVolatilityType,
        asset: Self::Asset,
    ) -> Result<Self::Price, DispatchError>;
    /// Calculates pairwise correlation between two specified assets.
    fn calc_corr(
        return_type: CalcReturnType,
        correlation_type: CalcCorrelationType,
        asset1: Self::Asset,
        asset2: Self::Asset,
    ) -> Result<(Self::Price, Range<Duration>), DispatchError>;
    /// Calculates portfolio volatility.
    fn calc_portf_vol(account_id: Self::AccountId) -> Result<Self::Price, DispatchError>;
    /// Calculates portfolio value at risk.
    fn calc_portf_var(
        account_id: Self::AccountId,
        return_type: CalcReturnType,
        z_score: u32,
    ) -> Result<Self::Price, DispatchError>;
}

fn calc_volatility<F>(
    return_type: CalcReturnType,
    volatility_type: CalcVolatilityType,
    prices: &[F],
) -> MathResult<F>
where
    F: FixedSigned + PartialOrd<ConstType> + From<ConstType>,
    F::Bits: Copy + ToFixed + AddAssign + BitOrAssign + ShlAssign,
{
    match volatility_type {
        CalcVolatilityType::Regular => {
            let return_func = calc_return_func(return_type);
            let returns: MathResult<Vec<F>> = calc_return_iter(&prices, return_func).collect();
            let returns = returns?;

            let mean_return: F = mean(&returns)?;

            let demeaned_return = demeaned(returns.iter(), mean_return);
            let squared_demeaned_return = squared(demeaned_return);

            let volatility: F = sqrt(regular_vola(returns.len(), sum(squared_demeaned_return)?)?)
                .map_err(|_| MathError::Transcendental)?;

            Ok(volatility)
        }
        CalcVolatilityType::Exponential(ewma_length) => {
            let return_func = calc_return_func_exp_vola(return_type);
            let returns = calc_return_iter(&prices, return_func);
            let squared_returns = squared(returns);
            let decay = decay(ewma_length)?;
            let var = last_recurrent_ewma(squared_returns, decay)?;
            let last_price = prices.last().copied().ok_or(MathError::NotEnoughPoints)?;
            let vola = exp_vola(return_type, var, last_price)?;

            Ok(vola)
        }
    }
}

impl From<PricePeriodError> for MathError {
    fn from(error: PricePeriodError) -> Self {
        match error {
            PricePeriodError::DivisionByZero => MathError::DivisionByZero,
            PricePeriodError::Overflow => MathError::Overflow,
        }
    }
}

fn calc_correlation<F>(
    return_type: CalcReturnType,
    correlation_type: CalcCorrelationType,
    prices1: &[F],
    prices2: &[F],
) -> MathResult<F>
where
    F: FixedSigned + PartialOrd<ConstType> + From<ConstType>,
    F::Bits: Copy + ToFixed + AddAssign + BitOrAssign + ShlAssign,
{
    let return_func = calc_return_func(return_type);

    let returns1: MathResult<Vec<F>> = calc_return_iter(&prices1, return_func).collect();
    let returns1 = returns1?;

    let returns2: MathResult<Vec<F>> = calc_return_iter(&prices2, return_func).collect();
    let returns2 = returns2?;

    let mean_return1: F = mean(&returns1)?;
    let mean_return2: F = mean(&returns2)?;

    let demeaned_returns1: MathResult<Vec<F>> = demeaned(returns1.iter(), mean_return1).collect();
    let demeaned_returns1 = demeaned_returns1?;
    let demeaned_returns2: MathResult<Vec<F>> = demeaned(returns2.iter(), mean_return2).collect();
    let demeaned_returns2 = demeaned_returns2?;

    let demeaned_returns_product = mul(
        demeaned_returns1.iter().copied(),
        demeaned_returns2.iter().copied(),
    );

    let squared_demeaned_returns1 = squared(demeaned_returns1.iter().map(|&x| Ok(x)));
    let squared_demeaned_returns2 = squared(demeaned_returns2.iter().map(|&x| Ok(x)));

    let volatility1: F = sqrt(regular_vola(
        returns1.len(),
        sum(squared_demeaned_returns1)?,
    )?)
    .map_err(|_| MathError::Transcendental)?;
    let volatility2: F = sqrt(regular_vola(
        returns2.len(),
        sum(squared_demeaned_returns2)?,
    )?)
    .map_err(|_| MathError::Transcendental)?;

    match correlation_type {
        CalcCorrelationType::Regular => {
            let products_sum = sum(demeaned_returns_product)?;
            let products_len = min(returns1.len(), returns2.len());
            let result = regular_corr(products_len, products_sum, volatility1, volatility2)?;
            Ok(result)
        }
        CalcCorrelationType::Exponential(ewma_length) => {
            let decay = decay(ewma_length)?;
            let last_covar = last_recurrent_ewma(demeaned_returns_product, decay)?;
            let result = exp_corr(last_covar, volatility1, volatility2)?;
            Ok(result)
        }
    }
}

impl<T: Trait> From<MathError> for Error<T> {
    fn from(error: MathError) -> Self {
        match error {
            MathError::NotEnoughPoints => Error::<T>::NotEnoughPoints,
            MathError::Overflow => Error::<T>::Overflow,
            MathError::DivisionByZero => Error::<T>::DivisionByZero,
            MathError::Transcendental => Error::<T>::Transcendental,
            MathError::InvalidArgument => Error::<T>::InvalidArgument,
        }
    }
}

/// Calculates price period range for a given `items_length` periods elapsed number
/// up to `latest_timestamp`.
fn get_period_id_range(
    price_period: &PricePeriod,
    items_length: usize,
    latest_timestamp: Duration,
) -> MathResult<Range<u64>> {
    let last_period_id = price_period.get_period_id(latest_timestamp)?;
    let next_after_last_period_id = last_period_id.checked_add(1).ok_or(MathError::Overflow)?;
    let items_length: u64 = items_length.try_into().map_err(|_| MathError::Overflow)?;
    let first_period_id = next_after_last_period_id
        .checked_sub(items_length)
        .ok_or(MathError::Overflow)?;

    Ok(Range {
        start: first_period_id,
        end: next_after_last_period_id,
    })
}

fn get_range_intersection<T>(range1: &Range<T>, range2: &Range<T>) -> Range<T>
where
    T: Ord + Copy,
{
    let start = max(range1.start, range2.start);
    let end = min(range1.end, range2.end);

    Range { start, end }
}

/// Calculates price log indices for the subrange `intersection` of the price period range `range`.
/// Price log contains only prices within range `range`.
fn get_index_range(range: &Range<u64>, intersection: &Range<u64>) -> MathResult<Range<usize>> {
    if intersection.is_empty() {
        Ok(Range { start: 0, end: 0 })
    } else {
        ensure!(
            range.start <= intersection.start && range.end >= intersection.end,
            MathError::InvalidArgument
        );
        let offset = range.start;
        let start = intersection.start - offset;
        let end = intersection.end - offset;

        let start: usize = start.try_into().map_err(|_| MathError::Overflow)?;
        let end: usize = end.try_into().map_err(|_| MathError::Overflow)?;
        Ok(Range { start, end })
    }
}

/// Gets prices for two given assets `asset1` and `asset2` only for those price periods that
/// are present in both price logs.
/// Returns tuple which contains respectively prices from the log for `asset1`, prices from the log
/// to `asset2`, a time range which is common for both `asset1` and `asset2` price logs.
fn get_prices_for_common_periods<T: Trait>(
    asset1: T::Asset,
    asset2: T::Asset,
) -> MathResult<(Vec<T::FixedNumber>, Vec<T::FixedNumber>, Range<Duration>)> {
    let price_period = PricePeriod(T::PricePeriod::get());

    let log1 = PriceLogs::<T>::get(asset1);
    let period_id_range1 =
        get_period_id_range(&price_period, log1.prices.len(), log1.latest_timestamp)?;

    let log2 = PriceLogs::<T>::get(asset2);
    let period_id_range2 =
        get_period_id_range(&price_period, log2.prices.len(), log2.latest_timestamp)?;

    let intersection = get_range_intersection(&period_id_range1, &period_id_range2);

    let index_range1 = get_index_range(&period_id_range1, &intersection)?;
    let prices1: Vec<_> = log1.prices.iter_range(&index_range1).copied().collect();

    let index_range2 = get_index_range(&period_id_range2, &intersection)?;
    let prices2: Vec<_> = log2.prices.iter_range(&index_range2).copied().collect();
    let temporal_range = Range {
        start: price_period.get_period_id_start(intersection.start)?,
        end: price_period.get_period_id_start(intersection.end)?,
    };

    Ok((prices1, prices2, temporal_range))
}

impl<T: Trait> Financial for Module<T> {
    type Asset = T::Asset;
    type Price = T::Price;
    type AccountId = <T as frame_system::Trait>::AccountId;

    fn calc_return(
        return_type: CalcReturnType,
        asset: T::Asset,
    ) -> Result<Vec<T::Price>, DispatchError> {
        let prices: Vec<_> = PriceLogs::<T>::get(asset).prices.iter().cloned().collect();

        let result: MathResult<Vec<T::Price>> =
            calc_return_iter(&prices, calc_return_func(return_type))
                .map(|x| x.map(|y| y.into()))
                .collect();
        let result = result.map_err(Into::<Error<T>>::into)?;

        if result.len() == 0 {
            Err(Error::<T>::NotEnoughPoints.into())
        } else {
            Ok(result)
        }
    }

    fn calc_vol(
        return_type: CalcReturnType,
        volatility_type: CalcVolatilityType,
        asset: T::Asset,
    ) -> Result<Self::Price, DispatchError> {
        let prices: Vec<_> = PriceLogs::<T>::get(asset).prices.iter().cloned().collect();

        let result = calc_volatility::<T::FixedNumber>(return_type, volatility_type, &prices)
            .map_err(Into::<Error<T>>::into)?;

        Ok(result.into())
    }

    fn calc_corr(
        return_type: CalcReturnType,
        correlation_type: CalcCorrelationType,
        asset1: T::Asset,
        asset2: T::Asset,
    ) -> Result<(Self::Price, Range<Duration>), DispatchError> {
        // We should only use those points for which price periods are present in both price logs
        let (prices1, prices2, temporal_range) =
            get_prices_for_common_periods::<T>(asset1, asset2).map_err(Into::<Error<T>>::into)?;

        let result =
            calc_correlation::<T::FixedNumber>(return_type, correlation_type, &prices1, &prices2)
                .map_err(Into::<Error<T>>::into)?;

        Ok((result.into(), temporal_range))
    }

    fn calc_portf_vol(_account_id: Self::AccountId) -> Result<Self::Price, DispatchError> {
        Err(Error::<T>::NotImplemented.into())
    }

    fn calc_portf_var(
        _account_id: Self::AccountId,
        _return_type: CalcReturnType,
        _z_score: u32,
    ) -> Result<Self::Price, DispatchError> {
        Err(Error::<T>::NotImplemented.into())
    }
}
