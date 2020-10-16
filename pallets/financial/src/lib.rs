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

use frame_support::{decl_module, decl_storage, decl_event, decl_error, ensure, dispatch};
use frame_support::traits::{Get, UnixTime};
use frame_support::codec::{Codec, Decode, Encode};
use frame_support::dispatch::{DispatchError, Parameter};
use frame_system::ensure_signed;
use financial_primitives::capvec::CapVec;
use financial_primitives::{OnPriceSet, IntoTypeIterator};
use core::time::Duration;
use sp_std::cmp::min;
use sp_std::prelude::Vec;
use sp_std::convert::TryInto;
use sp_std::iter::Iterator;
use sp_std::vec;
use substrate_fixed::transcendental::sqrt;
use substrate_fixed::traits::{FixedSigned, ToFixed};
use core::ops::{AddAssign, BitOrAssign, ShlAssign};
use math::{ConstType, MathError, MathResult, diff, squared, decay, last_recurrent_ewma, exp_vola, calc_return_func, calc_return_iter, sum, mean, demeaned, regular_vola, mul, regular_corr, exp_corr};

pub use math::{CalcReturnType, CalcVolatilityType, CalcCorrelationType};

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
	type FixedNumber: Clone + Copy + Codec + FixedSigned<Bits = Self::FixedNumberBits> + PartialOrd<ConstType> + From<ConstType>;
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

/// Financial metrics for asset
#[derive(Encode, Decode, Clone, Default, PartialEq, Eq, Debug)]
pub struct FinancialMetrics<A, P> {
    /// Timestamp of when last data point was updated
    timestamp: Duration,
    /// Log returns
    log_returns: Vec<P>,
    /// Volatility
    volatility: P,
    /// Correlations for all assets
    correlations: Vec<(A, P)>,
}

decl_storage! {
	trait Store for Module<T: Trait> as FinancialModule {
        /// Latest price updates on per asset basis.
		Updates get(fn updates): map hasher(blake2_128_concat) T::Asset => Option<PriceUpdate<T::FixedNumber>>;
        /// Price log on per asset basis.
		Prices get(fn prices): map hasher(blake2_128_concat) T::Asset => CapVec<T::FixedNumber>;
        /// Financial metrics on per asset basis.
		Metrics get(fn metrics): map hasher(blake2_128_concat) T::Asset => Option<FinancialMetrics<T::Asset, T::Price>>;
	}

	add_extra_genesis {
        /// Initial price logs on per asset basis.
		config(prices): Vec<(T::Asset, Vec<T::Price>)>;

		build(|config| {
			let price_count = T::PriceCount::get();

			for (asset, values) in config.prices.iter() {
				let mut prices = CapVec::<T::FixedNumber>::new(price_count);

				for v in values.iter() {
					prices.push(v.clone().into());
				}

				Prices::<T>::insert(asset, prices);
			}
		});
	}
}

decl_event!(
	pub enum Event<T> where Asset = <T as Trait>::Asset {
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
        /// An invalid argument was passed to the transcendental function (i.e. log, sqrt, etc.)
        /// during financial calculation process.
		Transcendental,
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

            let update = Self::updates(asset);
			ensure!(update.is_some(), Error::<T>::NotEnoughPoints);
            let update = update.unwrap();

            let timestamp = update.time;

            let return_type = CalcReturnType::Log;
            let log_returns = <Module<T> as Financial>::calc_return(return_type, asset)?;
            let volatility = <Module<T> as Financial>::calc_vol(return_type, CalcVolatilityType::Regular, asset)?;

            let mut correlations = vec![];

            for asset2 in T::Asset::into_type_iter() {
                let correlation = <Module<T> as Financial>::calc_corr(return_type, CalcCorrelationType::Regular, asset, asset2)?;
                correlations.push((asset2, correlation));
            }

            Metrics::<T>::mutate(&asset, |metrics| {
                *metrics = Some(FinancialMetrics {
                    timestamp,
                    log_returns,
                    volatility,
                    correlations,
                });
            });

            Ok(())
        }
	}
}

#[derive(Debug, Eq, PartialEq)]
enum PeriodAction {
	RemainsUnchanged,
	StartedNew(u32),
}

#[derive(Debug, Eq, PartialEq)]
struct PeriodChange {
	period_start: Duration,
	action: PeriodAction,
}

#[derive(Debug, Eq, PartialEq)]
enum PricePeriodError {
	DivisionByZero,
	Overflow,
	PeriodIsInThePast,
	InvalidPeriodStart,
}

struct PricePeriod(u32);

impl PricePeriod {
	fn get_period_id(&self, now: Duration) -> Result<u64, PricePeriodError> {
		let seconds = now.as_secs();
		let period = self.0 as u64;

		seconds.checked_div(60).and_then(|x| x.checked_div(period)).ok_or(PricePeriodError::DivisionByZero)
	}

	fn get_period_id_start(&self, period_id: u64) -> Result<Duration, PricePeriodError> {
		let period = self.0 as u64;
		let seconds: Result<_, PricePeriodError> = period_id.checked_mul(60).and_then(|x| x.checked_mul(period)).ok_or(PricePeriodError::Overflow);

		Ok(Duration::from_secs(seconds?))
	}

	fn get_period_start(&self, now: Duration) -> Result<Duration, PricePeriodError> {
		self.get_period_id_start(self.get_period_id(now)?)
	}

	fn get_curr_period_info(&self, prev_start: Duration, now: Duration) -> Result<(Duration, i32), PricePeriodError> {
		let prev_period_id = self.get_period_id(prev_start)?;
		let curr_period_id = self.get_period_id(now)?;

		let prev: i32 = prev_period_id.try_into().map_err(|_| PricePeriodError::Overflow)?;
		let curr: i32 = curr_period_id.try_into().map_err(|_| PricePeriodError::Overflow)?;
		let delta: Result<_, PricePeriodError> = curr.checked_sub(prev).ok_or(PricePeriodError::Overflow);
		Ok((self.get_period_id_start(curr_period_id)?, delta?))
	}

	fn get_period_change(&self, period_start: Option<Duration>, now: Duration) -> Result<PeriodChange, PricePeriodError> {
		if let Some(period_start) = period_start {
			ensure!(period_start == self.get_period_start(period_start)?, PricePeriodError::InvalidPeriodStart);
		}

		match period_start {
			None => {
				let period_start = self.get_period_start(now)?;

				Ok(PeriodChange {
					period_start,
					action: PeriodAction::StartedNew(0),
				})
			},
			Some(last_start) => {
				let (current_start, periods_elapsed) = self.get_curr_period_info(last_start, now)?;

				if periods_elapsed < 0 {
					// Current period is in the past

					Err(PricePeriodError::PeriodIsInThePast)
				} else if periods_elapsed == 0 {
					// Period is not changed

					Ok(PeriodChange {
						period_start: last_start,
						action: PeriodAction::RemainsUnchanged,
					})
				} else {
					// Period is changed
					
					let empty_periods = (periods_elapsed - 1) as u32;

					Ok(PeriodChange {
						period_start: current_start,
						action: PeriodAction::StartedNew(empty_periods),
					})
				}
			}
		}
	}
}

impl<T: Trait> From<PricePeriodError> for Error<T> {
	fn from(error: PricePeriodError) -> Self {
		match error {
			PricePeriodError::DivisionByZero => Error::<T>::DivisionByZero,
			PricePeriodError::Overflow => Error::<T>::Overflow,
			PricePeriodError::PeriodIsInThePast => Error::<T>::PeriodIsInThePast,
			PricePeriodError::InvalidPeriodStart => Error::<T>::InvalidPeriodStart,
		}
	}
}

#[derive(Debug, Eq, PartialEq)]
enum GetNewPricesError {
	Overflow,
}

fn get_new_prices<P: Clone>(last_price: Option<P>, new_price: P, empty_periods: u32, max_periods: u32) -> Result<Vec<P>, GetNewPricesError> {
	match last_price {
		None => {
			Ok(vec![new_price])
		},
		Some(last_price) => {
			// Calculate how many values to pre-populate the array with
			// We will pre-fill up to `max_periods` elements (leaving out one for the new price)
			let prices_size = min(empty_periods, max_periods.checked_sub(1).ok_or(GetNewPricesError::Overflow)?) as usize;

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

impl<T: Trait> OnPriceSet for Module<T> {
	type Asset = T::Asset;
	type Price = T::Price;

	fn on_price_set(asset: T::Asset, value: T::Price) -> Result<(), DispatchError> {
		let value: T::FixedNumber = value.into();
		let now = T::UnixTime::now();
		let price_count = T::PriceCount::get();

		let update = Self::updates(asset);
		let prices = Self::prices(asset);

		ensure!(prices.len_cap() == 0 || prices.len_cap() == price_count, Error::<T>::InvalidStorage);
		let init_prices = prices.len_cap() == 0;

		let period_start = update.map(|x| x.period_start);
		let price_period = PricePeriod(T::PricePeriod::get());
		let period_change = price_period.get_period_change(period_start, now).map_err(Into::<Error<T>>::into)?;

		let period_start = period_change.period_start;

		match period_change.action {
			PeriodAction::RemainsUnchanged => {
				Updates::<T>::mutate(&asset, |update| {
					*update = Some(PriceUpdate {
						period_start: period_start,
						time: now,
						price: value,
					});
				});
			},
			PeriodAction::StartedNew(empty_periods) => {
				let new_prices = get_new_prices(prices.last().cloned(), value, empty_periods, price_count).map_err(Into::<Error<T>>::into)?;

				Updates::<T>::mutate(&asset, |update| {
					*update = Some(PriceUpdate {
						period_start: period_start,
						time: now,
						price: value,
					})
				});
				Prices::<T>::mutate(&asset, |prices| {
					if init_prices {
						*prices = CapVec::<T::FixedNumber>::new(price_count);
					}

					for p in new_prices {
						prices.push(p);
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
	fn calc_return(return_type: CalcReturnType, asset: Self::Asset) -> Result<Vec<Self::Price>, DispatchError>;
    /// Calculates volatility.
	fn calc_vol(return_type: CalcReturnType, volatility_type: CalcVolatilityType, asset: Self::Asset) -> Result<Self::Price, DispatchError>;
    /// Calculates pairwise correlation between two specified assets.
	fn calc_corr(return_type: CalcReturnType, correlation_type: CalcCorrelationType, asset1: Self::Asset, asset2: Self::Asset) -> Result<Self::Price, DispatchError>;
    /// Calculates portfolio volatility.
	fn calc_portf_vol(account_id: Self::AccountId) -> Result<Self::Price, DispatchError>;
    /// Calculates portfolio value at risk.
	fn calc_portf_var(account_id: Self::AccountId, return_type: CalcReturnType, z_score: u32) -> Result<Self::Price, DispatchError>;
}

fn calc_volatility<F>(return_type: CalcReturnType, volatility_type: CalcVolatilityType, prices: &[F]) -> MathResult<F>
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

            let volatility: F = sqrt(regular_vola(returns.len(), sum(squared_demeaned_return)?)?).map_err(|_| MathError::Transcendental)?;

            Ok(volatility)
        },
        CalcVolatilityType::Exponential(ewma_length) => {
            let diffs = diff(prices);
            let squared_diffs = squared(diffs);
            let decay = decay(ewma_length)?;
            let var = last_recurrent_ewma(squared_diffs, decay)?;
            let last_price = prices.last().copied().ok_or(MathError::NotEnoughPoints)?;
            let vola = exp_vola(var, last_price)?;

            Ok(vola)
        }
    }
}

fn calc_correlation<F>(return_type: CalcReturnType, correlation_type: CalcCorrelationType, prices1: &[F], prices2: &[F]) -> MathResult<F>
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

	let demeaned_returns_product = mul(demeaned_returns1.iter().copied(), demeaned_returns2.iter().copied());

    let squared_demeaned_returns1 = squared(demeaned_returns1.iter().map(|&x| Ok(x)));
    let squared_demeaned_returns2 = squared(demeaned_returns2.iter().map(|&x| Ok(x)));

    let volatility1: F = sqrt(regular_vola(returns1.len(), sum(squared_demeaned_returns1)?)?).map_err(|_| MathError::Transcendental)?;
    let volatility2: F = sqrt(regular_vola(returns2.len(), sum(squared_demeaned_returns2)?)?).map_err(|_| MathError::Transcendental)?;

    match correlation_type {
        CalcCorrelationType::Regular => {
            let products_sum = sum(demeaned_returns_product)?;
            let products_len = min(returns1.len(), returns2.len());
            let result = regular_corr(products_len, products_sum, volatility1, volatility2)?;
            Ok(result)
        },
        CalcCorrelationType::Exponential(ewma_length) => {
            let decay = decay(ewma_length)?;
            let last_covar = last_recurrent_ewma(demeaned_returns_product, decay)?;
            let result = exp_corr(last_covar, volatility1, volatility2)?;
            Ok(result)
        },
    }
}

impl<T: Trait> From<MathError> for Error<T> {
	fn from(error: MathError) -> Self {
		match error {
			MathError::NotEnoughPoints => Error::<T>::NotEnoughPoints,
			MathError::Overflow => Error::<T>::Overflow,
			MathError::DivisionByZero => Error::<T>::DivisionByZero,
			MathError::Transcendental => Error::<T>::Transcendental,
		}
	}
}

impl<T: Trait> Financial for Module<T> {
	type Asset = T::Asset;
	type Price = T::Price;
	type AccountId = <T as frame_system::Trait>::AccountId;

	fn calc_return(return_type: CalcReturnType, asset: T::Asset) -> Result<Vec<T::Price>, DispatchError> {
		let prices: Vec<_> = Prices::<T>::get(asset).iter().cloned().collect();

		let result: MathResult<Vec<T::Price>> = calc_return_iter(&prices, calc_return_func(return_type)).map(|x| x.map(|y| y.into())).collect();
        let result = result.map_err(Into::<Error<T>>::into)?;

        if result.len() == 0 {
            Err(Error::<T>::NotEnoughPoints.into())
        } else {
            Ok(result)
        }
	}

	fn calc_vol(return_type: CalcReturnType, volatility_type: CalcVolatilityType, asset: T::Asset) -> Result<Self::Price, DispatchError> {
		let prices: Vec<_> = Prices::<T>::get(asset).iter().cloned().collect();

		let result = calc_volatility::<T::FixedNumber>(return_type, volatility_type, &prices).map_err(Into::<Error<T>>::into)?;

		Ok(result.into())
	}

	fn calc_corr(return_type: CalcReturnType, correlation_type: CalcCorrelationType, asset1: T::Asset, asset2: T::Asset) -> Result<Self::Price, DispatchError> {
		let prices1: Vec<_> = Prices::<T>::get(asset1).iter().cloned().collect();
		let prices2: Vec<_> = Prices::<T>::get(asset2).iter().cloned().collect();

		let result = calc_correlation::<T::FixedNumber>(return_type, correlation_type, &prices1, &prices2).map_err(Into::<Error<T>>::into)?;

		Ok(result.into())
	}

	fn calc_portf_vol(_account_id: Self::AccountId) -> Result<Self::Price, DispatchError> {
		Err(Error::<T>::NotImplemented.into())
	}

	fn calc_portf_var(_account_id: Self::AccountId, _return_type: CalcReturnType, _z_score: u32) -> Result<Self::Price, DispatchError> {
		Err(Error::<T>::NotImplemented.into())
	}
}
