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
use financial_primitives::{
    BalanceAware, CalcReturnType, CalcVolatilityType, OnPriceSet, PricePeriod, PricePeriodError,
};
use frame_support::codec::{Decode, Encode, FullCodec};
use frame_support::dispatch::{DispatchError, Parameter};
use frame_support::storage::{IterableStorageMap, StorageMap};
use frame_support::traits::{Get, UnixTime};
use frame_support::{decl_error, decl_event, decl_module, decl_storage, dispatch, ensure};
use frame_system::ensure_signed;
use math::{
    calc_return_func, calc_return_iter, covariance, decay, demeaned, exp_corr, from_num,
    last_recurrent_ewma, log_value_at_risk, mean, mul, regular_corr, regular_value_at_risk,
    regular_vola, squared, sum, ConstType, MathError, MathResult,
};
use sp_std::cmp::{max, min};
use sp_std::convert::{TryFrom, TryInto};
use sp_std::iter::Iterator;
use sp_std::ops::Range;
use sp_std::prelude::Vec;
use sp_std::vec;
use substrate_fixed::traits::{Fixed, FixedSigned, ToFixed};
use substrate_fixed::transcendental::sqrt;

pub use math::CalcCorrelationType;

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
    /// Default type of calculation for return: Regular or Log.
    type ReturnType: Get<u32>;
    /// Default type of calculation for volatility and correlation: Regular or Exponential.
    type VolCorType: Get<i64>;
    /// System wide type for representing various assets such as BTC, ETH, EOS, etc.
    type Asset: Parameter + Copy + Ord + Eq;
    /// Primitive integer type that [`FixedNumber`](#associatedtype.FixedNumber) based on.
    type FixedNumberBits: Copy + ToFixed + AddAssign + BitOrAssign + ShlAssign;
    /// Fixed point data type with a required precision that used for all financial calculations.
    type FixedNumber: Clone
        + Copy
        + FullCodec
        + FixedSigned<Bits = Self::FixedNumberBits>
        + PartialOrd<ConstType>
        + From<ConstType>;
    /// System wide type for representing price values. It must be convertable to and
    /// from [`FixedNumber`](#associatedtype.FixedNumber).
    type Price: Parameter + Clone + From<Self::FixedNumber> + Into<Self::FixedNumber>;
    /// Type that gets user balances for a given AccountId
    type Balances: BalanceAware<
        AccountId = Self::AccountId,
        Asset = Self::Asset,
        Balance = Self::Price,
    >;
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
pub struct AssetMetrics<A, P> {
    /// Start of the period inclusive for which metrics were calculated.
    pub period_start: Duration,
    /// End of the period exclusive for which metrics were calculated.
    pub period_end: Duration,
    /// Log returns
    pub returns: Vec<P>,
    /// Volatility
    pub volatility: P,
    /// Correlations for all assets
    pub correlations: Vec<(A, P)>,
}

/// Financial metrics for all assets
#[derive(Encode, Decode, Clone, Default, PartialEq, Eq, Debug)]
pub struct FinancialMetrics<A, P> {
    /// Start of the period inclusive for which metrics were calculated.
    pub period_start: Duration,
    /// End of the period exclusive for which metrics were calculated.
    pub period_end: Duration,

    /// Assets for which metrics were calculated.
    pub assets: Vec<A>,

    /// Mean returns for all assets. Mean returns are in the same order as the assets in the `assets` field.
    pub mean_returns: Vec<P>,

    /// Volatilities for all assets. Volatilities are in the same order as the assets in the `assets` field.
    pub volatilities: Vec<P>,

    /// Correlation matrix for all assets.
    /// Rows and columns are in the same order as the assets in the `assets` field.
    /// Matrix is stored by rows. For example, let matrix A =
    ///
    /// ```pseudocode
    /// a11 a12 a13
    /// a21 a22 a21
    /// a31 a32 a33
    /// ```
    ///
    /// Then vector for this matrix will be:
    ///
    /// ```pseudocode
    /// vec![a11, a12, a13, a21, a22, a23, a31, a32, a33]
    /// ```
    pub correlations: Vec<P>,

    /// Covariance matrix for all assets.
    /// Rows and columns are in the same order as the assets in the `assets` field.
    /// Matrix is stored by rows. See example for `correlations` field.
    pub covariances: Vec<P>,
}

decl_storage! {
    trait Store for Module<T: Trait> as FinancialModule {
        /// Latest price updates on per asset basis.
        pub Updates get(fn updates): map hasher(blake2_128_concat) T::Asset => Option<PriceUpdate<T::FixedNumber>>;

        /// Price log on per asset basis.
        pub PriceLogs get(fn price_logs): map hasher(blake2_128_concat) T::Asset => Option<PriceLog<T::FixedNumber>>;

        /// Financial metrics on per asset basis.
        pub PerAssetMetrics get(fn per_asset_metrics): map hasher(blake2_128_concat) T::Asset => Option<AssetMetrics<T::Asset, T::Price>>;

        /// Financial metrics for all known assets.
        pub Metrics get(fn metrics): Option<FinancialMetrics<T::Asset, T::Price>>;
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

                assert!(values.len() > 0, "Initial price vector can not be empty. Asset: {:?}.");

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
        /// Financial metrics for the specified asset have been recalculeted
        Recalculated(Asset),
        /// Financial metrics for all assets have been recalculeted
        MetricsRecalculated(),
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
        /// Default return type or default correlation type is initialized with a value that can
        /// not be converted to type `CalcReturnType` or `CalcVolatilityType` respectively.
        InvalidConstant,
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
        pub fn recalc_asset(origin, asset: T::Asset) -> dispatch::DispatchResult {
            ensure_signed(origin)?;

            let return_type = CalcReturnType::try_from(T::ReturnType::get()).map_err(|_| Error::<T>::InvalidConstant)?;
            let vol_cor_type = CalcVolatilityType::try_from(T::VolCorType::get()).map_err(|_| Error::<T>::InvalidConstant)?;

            let mut correlations = vec![];

            let mut asset_logs: Vec<_> = PriceLogs::<T>::iter().collect();
            ensure!(asset_logs.len() > 0, Error::<T>::NotEnoughPoints);
            asset_logs.sort_by(|(a1, _), (a2, _)| a1.cmp(a2));

            let price_period = PricePeriod(T::PricePeriod::get());
            let ranges = asset_logs.iter().map(|(_, l)| get_period_id_range(&price_period, l.prices.len(), l.latest_timestamp)).collect::<MathResult<Vec<_>>>().map_err(Into::<Error<T>>::into)?;

            // Ensure that all correlations calculated for the same period
            let intersection = get_range_intersection(ranges.iter()).map_err(Into::<Error<T>>::into)?;

            let (log1, period_id_range1) = asset_logs.iter().zip(ranges.iter()).find_map(|((a, l), r)| if *a == asset {Some((l, r))} else {None}).ok_or(Error::<T>::NotEnoughPoints)?;
            let range1 = get_index_range(period_id_range1, &intersection).map_err(Into::<Error<T>>::into)?;
            let prices1 = log1.prices.iter_range(&range1).copied().collect::<Vec<_>>();

            let ret1 = Ret::<T::FixedNumber>::new(&prices1, return_type).map_err(Into::<Error<T>>::into)?;
            let vol1 = Vol::<T::FixedNumber>::new(&ret1, vol_cor_type).map_err(Into::<Error<T>>::into)?;

            for ((asset2, log2), period_id_range2) in asset_logs.iter().zip(ranges.iter()) {
                if *asset2 == asset {
                    // Correlation for any asset with itself sould be 1
                    correlations.push((*asset2, from_num::<T::FixedNumber>(1).into()));
                } else {
                    let range2 = get_index_range(period_id_range2, &intersection).map_err(Into::<Error<T>>::into)?;
                    let prices2 = log2.prices.iter_range(&range2).copied().collect::<Vec<_>>();

                    let ret2 = Ret::<T::FixedNumber>::new(&prices2, return_type).map_err(Into::<Error<T>>::into)?;
                    let vol2 = Vol::<T::FixedNumber>::new(&ret2, vol_cor_type).map_err(Into::<Error<T>>::into)?;

                    let corre = cor(&ret1, &vol1, &ret2, &vol2).map_err(Into::<Error<T>>::into)?;

                    correlations.push((*asset2, corre.into()));
                }
            }

            let temporal_range = Range {
                start: price_period.get_period_id_start(intersection.start).map_err(Into::<MathError>::into).map_err(Into::<Error<T>>::into)?,
                end: price_period.get_period_id_start(intersection.end).map_err(Into::<MathError>::into).map_err(Into::<Error<T>>::into)?,
            };

            let returns: Vec<T::Price> = ret1.ret.into_iter().map(|x| x.into()).collect();
            let volatility: T::Price = vol1.vol.into();

            PerAssetMetrics::<T>::insert(&asset, AssetMetrics {
                    period_start: temporal_range.start,
                    period_end: temporal_range.end,
                    returns,
                    volatility,
                    correlations,
                }
            );

            Self::deposit_event(RawEvent::Recalculated(asset));

            Ok(())
        }

        /// Recalculates financial metrics for all known assets.
        #[weight = 10_000 + T::DbWeight::get().writes(1)]
        pub fn recalc(origin) -> dispatch::DispatchResult {
            ensure_signed(origin)?;

            let return_type = CalcReturnType::try_from(T::ReturnType::get()).map_err(|_| Error::<T>::InvalidConstant)?;
            let vol_cor_type = CalcVolatilityType::try_from(T::VolCorType::get()).map_err(|_| Error::<T>::InvalidConstant)?;

            let price_period = PricePeriod(T::PricePeriod::get());

            let mut asset_logs: Vec<_> = PriceLogs::<T>::iter().collect();
            ensure!(asset_logs.len() > 0, Error::<T>::NotEnoughPoints);
            asset_logs.sort_by(|(a1, _), (a2, _)| a1.cmp(a2));

            let metrics = financial_metrics::<T::Asset, T::FixedNumber, T::Price>(return_type, vol_cor_type, &price_period, &asset_logs).map_err(Into::<Error<T>>::into)?;

            Metrics::<T>::put(metrics);

            Self::deposit_event(RawEvent::MetricsRecalculated());

            Ok(())
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
enum GetNewPricesError {
    Overflow,
}

fn get_new_prices<P: Clone>(
    last_price: P,
    new_price: P,
    empty_periods: u32,
    max_periods: u32,
) -> Result<Vec<P>, GetNewPricesError> {
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

        // If `PriceLog` for a given asset is not empty then it must contain fully initialized `CapVec`. It's `cap_len` should be equal to `PriceCount`.
        if let Some(ref log) = log {
            ensure!(
                log.prices.len_cap() == price_count,
                Error::<T>::InvalidStorage
            );
        }

        let period_start = update.map(|x| x.period_start);
        let price_period = PricePeriod(T::PricePeriod::get());
        let period_change =
            get_period_change(&price_period, period_start, now).map_err(Into::<Error<T>>::into)?;

        let period_start = period_change.period_start;

        // Every point received is stores to the `Update`.
        // Meanwhile only first point received in the current period is stored in the price log.
        match period_change.action {
            PricePeriodAction::RemainsUnchanged => {
                Updates::<T>::insert(
                    &asset,
                    PriceUpdate {
                        period_start: period_start,
                        time: now,
                        price: value,
                    },
                );
            }
            PricePeriodAction::StartedNew(empty_periods) => {
                let log_to_insert = if let Some(mut existing_log) = log {
                    existing_log.latest_timestamp = now;

                    // If `PriceLog` for a given asset is not empty, then it should contain at
                    // least one price value. It was set during prevoius `on_price_update` call or
                    // during genesis state generation routine.
                    let last_price = existing_log
                        .prices
                        .last()
                        .copied()
                        .ok_or(Error::<T>::InvalidStorage)?;

                    let new_prices = get_new_prices(last_price, value, empty_periods, price_count)
                        .map_err(Into::<Error<T>>::into)?;

                    for p in new_prices {
                        existing_log.prices.push(p);
                    }

                    existing_log
                } else {
                    let mut new_prices = CapVec::<T::FixedNumber>::new(price_count);
                    new_prices.push(value);

                    PriceLog {
                        latest_timestamp: now,
                        prices: new_prices,
                    }
                };

                Updates::<T>::insert(
                    &asset,
                    PriceUpdate {
                        period_start: period_start,
                        time: now,
                        price: value,
                    },
                );
                PriceLogs::<T>::insert(&asset, log_to_insert);
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
        correlation_type: CalcVolatilityType,
        asset1: Self::Asset,
        asset2: Self::Asset,
    ) -> Result<(Self::Price, Range<Duration>), DispatchError>;
    /// Calculates portfolio volatility.
    fn calc_portf_vol(
        return_type: CalcReturnType,
        vol_cor_type: CalcVolatilityType,
        account_id: Self::AccountId,
    ) -> Result<Self::Price, DispatchError>;
    /// Calculates portfolio value at risk.
    fn calc_portf_var(
        return_type: CalcReturnType,
        vol_cor_type: CalcVolatilityType,
        account_id: Self::AccountId,
        z_score: u32,
    ) -> Result<Self::Price, DispatchError>;
}

#[derive(Debug)]
struct Ret<'prices, F> {
    prices: &'prices [F],
    return_type: CalcReturnType,
    ret: Vec<F>,
}

impl<'prices, F> Ret<'prices, F>
where
    F: FixedSigned + PartialOrd<ConstType> + From<ConstType>,
    F::Bits: Copy + ToFixed + AddAssign + BitOrAssign + ShlAssign,
{
    fn new(prices: &'prices [F], return_type: CalcReturnType) -> MathResult<Ret<'prices, F>> {
        let ret = calc_return_iter(prices, calc_return_func(return_type))
            .collect::<MathResult<Vec<_>>>()?;
        Ok(Ret {
            prices,
            return_type,
            ret,
        })
    }
}

#[derive(Debug)]
struct Vol<F> {
    volatility_type: CalcVolatilityType,
    mean_return: F,
    demeaned_return: Vec<F>,
    decay: Option<F>,
    vol: F,
}

impl<F> Vol<F>
where
    F: FixedSigned + PartialOrd<ConstType> + From<ConstType>,
    F::Bits: Copy + ToFixed + AddAssign + BitOrAssign + ShlAssign,
{
    fn new<'prices>(
        ret: &Ret<'prices, F>,
        volatility_type: CalcVolatilityType,
    ) -> MathResult<Vol<F>> {
        let mean_return: F = mean(&ret.ret)?;

        let demeaned_return =
            demeaned(ret.ret.iter(), mean_return).collect::<MathResult<Vec<_>>>()?;
        let squared_demeaned_return = squared(demeaned_return.iter().copied().map(Ok));

        match volatility_type {
            CalcVolatilityType::Regular => {
                let vol: F = sqrt(regular_vola(ret.ret.len(), sum(squared_demeaned_return)?)?)
                    .map_err(|_| MathError::Transcendental)?;

                Ok(Vol {
                    volatility_type,
                    mean_return,
                    demeaned_return,
                    decay: None,
                    vol,
                })
            }
            CalcVolatilityType::Exponential(ewma_length) => {
                let decay = decay(ewma_length)?;
                let var = last_recurrent_ewma(squared_demeaned_return, decay)?;
                let vol = sqrt(var).map_err(|_| MathError::Transcendental)?;

                Ok(Vol {
                    volatility_type: CalcVolatilityType::Exponential(ewma_length),
                    mean_return,
                    demeaned_return,
                    decay: Some(decay),
                    vol,
                })
            }
        }
    }
}

fn cor<'prices, F>(
    ret1: &Ret<'prices, F>,
    volatility1: &Vol<F>,
    ret2: &Ret<'prices, F>,
    volatility2: &Vol<F>,
) -> MathResult<F>
where
    F: FixedSigned + PartialOrd<ConstType> + From<ConstType>,
    F::Bits: Copy + ToFixed + AddAssign + BitOrAssign + ShlAssign,
{
    let zero = from_num::<F>(0);

    match (&volatility1, &volatility2) {
        (Vol { vol: vol1, .. }, _) if *vol1 == zero => Ok(zero),
        (_, Vol { vol: vol2, .. }) if *vol2 == zero => Ok(zero),
        (
            Vol {
                volatility_type: CalcVolatilityType::Regular,
                demeaned_return: dr1,
                vol: vol1,
                ..
            },
            Vol {
                volatility_type: CalcVolatilityType::Regular,
                demeaned_return: dr2,
                vol: vol2,
                ..
            },
        ) => {
            let demeaned_returns_product = mul(dr1.iter().copied(), dr2.iter().copied());

            let products_sum = sum(demeaned_returns_product)?;
            let products_len = min(ret1.ret.len(), ret2.ret.len());
            let result = regular_corr(products_len, products_sum, *vol1, *vol2)?;
            Ok(result)
        }
        (
            Vol {
                volatility_type: CalcVolatilityType::Exponential(n1),
                demeaned_return: dr1,
                decay: Some(d1),
                vol: vol1,
                ..
            },
            Vol {
                volatility_type: CalcVolatilityType::Exponential(n2),
                demeaned_return: dr2,
                vol: vol2,
                ..
            },
        ) if n1 == n2 => {
            let demeaned_returns_product = mul(dr1.iter().copied(), dr2.iter().copied());

            let decay = *d1;
            let last_covar = last_recurrent_ewma(demeaned_returns_product, decay)?;
            let result = exp_corr(last_covar, *vol1, *vol2)?;
            Ok(result)
        }
        _ => Err(MathError::InvalidArgument),
    }
}

fn financial_metrics<A, F, P>(
    return_type: CalcReturnType,
    vol_cor_type: CalcVolatilityType,
    price_period: &PricePeriod,
    asset_logs: &[(A, PriceLog<F>)],
) -> MathResult<FinancialMetrics<A, P>>
where
    F: FixedSigned + PartialOrd<ConstType> + From<ConstType>,
    F::Bits: Copy + ToFixed + AddAssign + BitOrAssign + ShlAssign,
    F: Into<P>,
    A: Eq + Copy,
{
    ensure!(asset_logs.len() > 0, MathError::NotEnoughPoints);

    let mut mean_returns = Vec::with_capacity(asset_logs.len());
    let mut volatilities = Vec::with_capacity(asset_logs.len());
    let mut correlations = Vec::with_capacity(asset_logs.len() * asset_logs.len());

    let period_id_ranges = asset_logs
        .iter()
        .map(|(_, l)| get_period_id_range(&price_period, l.prices.len(), l.latest_timestamp))
        .collect::<MathResult<Vec<_>>>()?;

    // Ensure that all correlations calculated for the same period
    let intersection = get_range_intersection(period_id_ranges.iter())?;

    for ((asset1, log1), period_id_range1) in asset_logs.iter().zip(period_id_ranges.iter()) {
        let range1 = get_index_range(period_id_range1, &intersection)?;
        let prices1 = log1.prices.iter_range(&range1).copied().collect::<Vec<_>>();

        let ret1 = Ret::<F>::new(&prices1, return_type)?;
        let vol1 = Vol::<F>::new(&ret1, vol_cor_type)?;

        mean_returns.push(vol1.mean_return);
        volatilities.push(vol1.vol);

        for ((asset2, log2), period_id_range2) in asset_logs.iter().zip(period_id_ranges.iter()) {
            if *asset2 == *asset1 {
                // Correlation for any asset with itself sould be 1
                correlations.push(from_num::<F>(1));
            } else {
                let range2 = get_index_range(period_id_range2, &intersection)?;
                let prices2 = log2.prices.iter_range(&range2).copied().collect::<Vec<_>>();

                let ret2 = Ret::<F>::new(&prices2, return_type)?;
                let vol2 = Vol::<F>::new(&ret2, vol_cor_type)?;

                let corre = cor(&ret1, &vol1, &ret2, &vol2)?;

                correlations.push(corre);
            }
        }
    }

    let covariances = covariance(&correlations, &volatilities)
        .map(|x| x.map(|y| y.into()))
        .collect::<MathResult<Vec<P>>>()?;

    let temporal_range = Range {
        start: price_period.get_period_id_start(intersection.start)?,
        end: price_period.get_period_id_start(intersection.end)?,
    };

    Ok(FinancialMetrics {
        period_start: temporal_range.start,
        period_end: temporal_range.end,
        assets: asset_logs.iter().map(|(a, _)| a).copied().collect(),
        mean_returns: mean_returns.into_iter().map(|x| x.into()).collect(),
        volatilities: volatilities.into_iter().map(|x| x.into()).collect(),
        correlations: correlations.into_iter().map(|x| x.into()).collect(),
        covariances: covariances,
    })
}

fn latest_prices<'logs, A, F>(
    asset_logs: &'logs [(A, PriceLog<F>)],
) -> impl Iterator<Item = MathResult<F>> + 'logs
where
    F: Copy,
{
    asset_logs.iter().map(|(_, l)| {
        l.prices
            .last()
            .map(|&x| x)
            .ok_or(MathError::NotEnoughPoints)
    })
}

fn weights<F>(quantity: &[F], prices: &[F]) -> MathResult<Vec<F>>
where
    F: Fixed,
{
    let products =
        mul(quantity.iter().copied(), prices.iter().copied()).collect::<MathResult<Vec<_>>>()?;
    let sum_product = sum(products.iter().copied().map(Ok))?;

    products
        .iter()
        .copied()
        .map(|x| x.checked_div(sum_product).ok_or(MathError::DivisionByZero))
        .collect()
}

fn portfolio_vol<F>(weights: &[F], cov: &[F]) -> MathResult<F>
where
    F: FixedSigned + PartialOrd<ConstType> + From<ConstType>,
    F::Bits: Copy + ToFixed + AddAssign + BitOrAssign + ShlAssign,
{
    let n = weights.len();
    ensure!(cov.len() == n * n, MathError::InvalidArgument);

    let matrix_1xn = cov
        .chunks(n)
        .map(|cov_row| sum(mul(weights.iter().copied(), cov_row.iter().copied())))
        .collect::<MathResult<Vec<_>>>()?;

    let matrix_1x1 = sum(mul(matrix_1xn.into_iter(), weights.iter().copied()))?;

    sqrt(matrix_1x1).map_err(|_| MathError::Transcendental)
}

impl From<PricePeriodError> for MathError {
    fn from(error: PricePeriodError) -> Self {
        match error {
            PricePeriodError::DivisionByZero => MathError::DivisionByZero,
            PricePeriodError::Overflow => MathError::Overflow,
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

fn get_range_intersection2<T>(range1: &Range<T>, range2: &Range<T>) -> Range<T>
where
    T: Ord + Copy,
{
    let start = max(range1.start, range2.start);
    let end = min(range1.end, range2.end);

    Range { start, end }
}

fn get_range_intersection<'a, T, I>(mut ranges: I) -> MathResult<Range<T>>
where
    T: 'a + Ord + Copy,
    I: Iterator<Item = &'a Range<T>>,
{
    let intersection = ranges.try_fold::<Option<Range<T>>, _, Option<_>>(None, |acc, r| {
        if let Some(i) = acc {
            let new_intersection = get_range_intersection2(&i, r);
            if new_intersection.is_empty() {
                None
            } else {
                Some(Some(new_intersection))
            }
        } else {
            Some(Some(r.clone()))
        }
    });

    intersection
        .and_then(|i| i)
        .ok_or(MathError::NotEnoughPoints)
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

    let log1 = PriceLogs::<T>::get(asset1).ok_or(MathError::NotEnoughPoints)?;
    let period_id_range1 =
        get_period_id_range(&price_period, log1.prices.len(), log1.latest_timestamp)?;

    let log2 = PriceLogs::<T>::get(asset2).ok_or(MathError::NotEnoughPoints)?;
    let period_id_range2 =
        get_period_id_range(&price_period, log2.prices.len(), log2.latest_timestamp)?;

    let intersection = get_range_intersection2(&period_id_range1, &period_id_range2);

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
        let log = PriceLogs::<T>::get(asset).ok_or(Error::<T>::NotEnoughPoints)?;
        let prices: Vec<_> = log.prices.iter().cloned().collect();

        let ret =
            Ret::<T::FixedNumber>::new(&prices, return_type).map_err(Into::<Error<T>>::into)?;
        let result: Vec<T::Price> = ret.ret.into_iter().map(|x| x.into()).collect();

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
        let log = PriceLogs::<T>::get(asset).ok_or(Error::<T>::NotEnoughPoints)?;
        let prices: Vec<_> = log.prices.iter().cloned().collect();

        let returns =
            Ret::<T::FixedNumber>::new(&prices, return_type).map_err(Into::<Error<T>>::into)?;
        let result = Vol::<T::FixedNumber>::new(&returns, volatility_type)
            .map_err(Into::<Error<T>>::into)?;

        Ok(result.vol.into())
    }

    fn calc_corr(
        return_type: CalcReturnType,
        correlation_type: CalcVolatilityType,
        asset1: T::Asset,
        asset2: T::Asset,
    ) -> Result<(Self::Price, Range<Duration>), DispatchError> {
        // We should only use those points for which price periods are present in both price logs
        let (prices1, prices2, temporal_range) =
            get_prices_for_common_periods::<T>(asset1, asset2).map_err(Into::<Error<T>>::into)?;

        let ret1 =
            Ret::<T::FixedNumber>::new(&prices1, return_type).map_err(Into::<Error<T>>::into)?;
        let vol1 =
            Vol::<T::FixedNumber>::new(&ret1, correlation_type).map_err(Into::<Error<T>>::into)?;

        let ret2 =
            Ret::<T::FixedNumber>::new(&prices2, return_type).map_err(Into::<Error<T>>::into)?;
        let vol2 =
            Vol::<T::FixedNumber>::new(&ret2, correlation_type).map_err(Into::<Error<T>>::into)?;

        let corre = cor(&ret1, &vol1, &ret2, &vol2).map_err(Into::<Error<T>>::into)?;

        Ok((corre.into(), temporal_range))
    }

    fn calc_portf_vol(
        return_type: CalcReturnType,
        vol_cor_type: CalcVolatilityType,
        account_id: Self::AccountId,
    ) -> Result<Self::Price, DispatchError> {
        let price_period = PricePeriod(T::PricePeriod::get());

        let mut asset_logs: Vec<_> = PriceLogs::<T>::iter().collect();
        ensure!(asset_logs.len() > 0, Error::<T>::NotEnoughPoints);
        asset_logs.sort_by(|(a1, _), (a2, _)| a1.cmp(a2));

        let metrics = financial_metrics::<T::Asset, T::FixedNumber, T::FixedNumber>(
            return_type,
            vol_cor_type,
            &price_period,
            &asset_logs,
        )
        .map_err(Into::<Error<T>>::into)?;

        let prices = latest_prices::<T::Asset, T::FixedNumber>(&asset_logs)
            .collect::<MathResult<Vec<_>>>()
            .map_err(Into::<Error<T>>::into)?;

        let balances = T::Balances::balances(&account_id, &metrics.assets)?
            .into_iter()
            .map(|x| x.into())
            .collect::<Vec<_>>();

        let ws = weights(&balances, &prices).map_err(Into::<Error<T>>::into)?;

        let vol = portfolio_vol(&ws, &metrics.covariances).map_err(Into::<Error<T>>::into)?;

        Ok(vol.into())
    }

    fn calc_portf_var(
        return_type: CalcReturnType,
        vol_cor_type: CalcVolatilityType,
        account_id: Self::AccountId,
        z_score: u32,
    ) -> Result<Self::Price, DispatchError> {
        let price_period = PricePeriod(T::PricePeriod::get());

        let mut asset_logs: Vec<_> = PriceLogs::<T>::iter().collect();
        ensure!(asset_logs.len() > 0, Error::<T>::NotEnoughPoints);
        asset_logs.sort_by(|(a1, _), (a2, _)| a1.cmp(a2));

        let metrics = financial_metrics::<T::Asset, T::FixedNumber, T::FixedNumber>(
            return_type,
            vol_cor_type,
            &price_period,
            &asset_logs,
        )
        .map_err(Into::<Error<T>>::into)?;

        let prices = latest_prices::<T::Asset, T::FixedNumber>(&asset_logs)
            .collect::<MathResult<Vec<_>>>()
            .map_err(Into::<Error<T>>::into)?;

        let balances = T::Balances::balances(&account_id, &metrics.assets)?
            .into_iter()
            .map(|x| x.into())
            .collect::<Vec<_>>();

        let ws = weights(&balances, &prices).map_err(Into::<Error<T>>::into)?;

        let vol = portfolio_vol(&ws, &metrics.covariances).map_err(Into::<Error<T>>::into)?;
        let total_weighted_mean_return = sum(mul(ws.into_iter(), metrics.mean_returns.into_iter()))
            .map_err(Into::<Error<T>>::into)?;

        match return_type {
            CalcReturnType::Regular => {
                let portf_var = regular_value_at_risk(z_score, vol, total_weighted_mean_return)
                    .map_err(Into::<Error<T>>::into)?;
                Ok(portf_var.into())
            }
            CalcReturnType::Log => {
                let portf_var = log_value_at_risk(z_score, vol, total_weighted_mean_return)
                    .map_err(Into::<Error<T>>::into)?;
                Ok(portf_var.into())
            }
        }
    }
}
