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
//!
//! Equilibrium's Financial Pallet is an open-source substrate module that subscribes to external
//! price feed/oracle, gathers asset prices and calculates financial metrics based on the
//! information collected.
//!
//! - [`financial::Config`](./trait.Config.html)
//! - [`Call`](./enum.Call.html)
//! - [`Module`](./struct.Module.html)
//! - [`Financial`](./trait.Financial.html)
//!
//! ## Overview
//!
//! Financial pallet listens to the external Oracle pallet for price updates. It stores several recent updates so you
//! can calculate various financial metrics upon them:
//!
//! - Return (regular and logarithmic) for a given asset
//! - Volatility (regular and EWMA) for a given asset
//! - EWMA Rv for a given asset
//! - Correlation (regular and EWMA) between two given assets
//! - Portfolio volatility (regular and EWMA)
//! - Portfolio Value at Risk (regular and EWMA)
//!
//! ### Implementations
//!
//! Financial Pallet defined it's own trait [`Financial`](./trait.Financial.html) and implements it.
//!
//! ## Interface
//!
//! There are two main ways of interaction with Financial Pallet. Metrics can be recalculated
//! with dispatchable functions calls or called in other pallets logic using [`FinancialSystemTrait`](./trait.FinancialSystemTrait.html)
//!
//! ### System Functions
//!
//! - `recalc_inner` - Recalculates financial metrics for all known assets and saves them to the storage.
//! - `recalc_asset_inner` - Recalculates financial metrics for a given asset and saves them to the
//! storage.
//! - `recalc_portfolio_inner` - Recalculates financial metrics for a given user account and saves them to the
//! storage.
//!
//! ### Dispatchable Functions
//!
//! - `recalc_portfolio` - dispatchable function calling `recalc_portfolio_inner`. Must be signed.
//! - `recalc_asset` - dispatchable function calling `recalc_asset_inner`. Must be signed.
//! - `recalc` - dispatchable function calling `recalc_inner`. Must be signed.
//!
//! ## Usage
//!
//! ### Trait Financial
//!
//! Use `Financial` trait functions to calculate financial metrics without saving them to storage or
//! when you need to calculate some of metrics without calculating all metrics of a struct
//!
//! Suppose we defined `Config` for some another module that has associated type `type FinancialFuncs: financial_pallet::Financial`. Then we can calculate regular return for Btc like so:
//! regular return for Btc
//!
//! ```
//! # use financial_primitives::{CalcReturnType, CalcVolatilityType};
//! # use frame_support::codec::{Decode, Encode};
//! # use financial_pallet::Financial;
//! # use frame_support::dispatch::DispatchError;
//! # use core::ops::Range;
//! # use core::time::Duration;
//! # use serde::{Deserialize, Serialize};
//! #
//! # #[derive(Clone, Copy, Debug, Eq, PartialEq, Encode, Decode, Hash, Ord, PartialOrd)]
//! # #[cfg_attr(feature = "std", derive(Serialize, Deserialize))]
//! # pub enum Asset {
//! #     Btc,
//! # }
//! # struct FinancialFuncs;
//! # impl Financial for FinancialFuncs {
//! #     type Asset = Asset;
//! #     type Price = i32;
//! #     type AccountId = i32;
//! #     
//! #     
//! #    fn calc_return(
//! #        return_type: CalcReturnType,
//! #        asset: Self::Asset,
//! #    ) -> Result<Vec<Self::Price>, DispatchError> {Ok(vec![])}
//! #    fn calc_vol(
//! #        return_type: CalcReturnType,
//! #        volatility_type: CalcVolatilityType,
//! #        asset: Self::Asset,
//! #    ) -> Result<Self::Price, DispatchError> {Ok(0)}
//! #    fn calc_corr(
//! #        return_type: CalcReturnType,
//! #        correlation_type: CalcVolatilityType,
//! #        asset1: Self::Asset,
//! #        asset2: Self::Asset,
//! #    ) -> Result<(Self::Price, Range<Duration>), DispatchError> {Ok((0, Duration::new(5, 0)..Duration::new(5, 0)))}
//! #    fn calc_portf_vol(
//! #        return_type: CalcReturnType,
//! #        vol_cor_type: CalcVolatilityType,
//! #        account_id: Self::AccountId,
//! #    ) -> Result<Self::Price, DispatchError> {Ok(0)}
//! #    fn calc_portf_var(
//! #        return_type: CalcReturnType,
//! #        vol_cor_type: CalcVolatilityType,
//! #        account_id: Self::AccountId,
//! #        z_score: u32,
//! #    ) -> Result<Self::Price, DispatchError> {Ok(0)}
//! #    fn calc_rv(
//! #        return_type: CalcReturnType,
//! #        ewma_length: u32,
//! #        asset: Self::Asset,
//! #    ) -> Result<Self::Price, DispatchError> {Ok(0)}
//! # }
//! #
//! let ret = FinancialFuncs::calc_return(CalcReturnType::Regular, Asset::Btc);
//! ```
//! ### Trait FinancialSystemTrait
//!
//! Use `FinancialSystemTrait` trait to invoke recalculation of all metrics on an asset, user
//! portfolio or all system assets. Recalculated metrics would be saved in storage.
//!
//! This trait is useful for integration between pallets. You can use it as a type in other  
//! pallet's configuration trait and recalculate metrics automatically when your system
//! needs it.
//!
//! ## Setup
//!
//! Before using Financial Pallet in your code you need to setup it.
//!
//! ### Runtime crate
//!
//! First of all you need to define some parameters:
//!
//! ```
//! # use frame_support::parameter_types;
//! # use financial_primitives::{CalcReturnType, CalcVolatilityType};
//! parameter_types! {
//!     // Maximum number of points for each asset that Financial Pallet can store
//!     pub const PriceCount: u32 = 30;
//!     // Duration of the price period in minutes
//!     pub const PricePeriod: u32 = 1440;
//!     // CalcReturnType used by FinancialPallet's calc* extrinsics
//!     pub const ReturnType: u32 = CalcReturnType::Log.into_u32();
//!     // CalcVolatilityType used by FinancialPallet's calc* extrinsics
//!     pub const VolCorType: i64 = CalcVolatilityType::Regular.into_i64();
//! }
//! ```
//!
//! Implement Financial Pallet for your Runtime:
//!
//! ```
//! # use financial_primitives::{CalcReturnType, CalcVolatilityType};
//! # use frame_support::codec::{Decode, Encode};
//! # use frame_support::{impl_outer_origin, parameter_types, weights::Weight};
//! # use sp_runtime::{ testing::Header, traits::{BlakeTwo256, IdentityLookup}, Perbill};
//! # use sp_core::H256;
//! #
//! # fn main() {}
//! #
//! # parameter_types! {
//! #    // Maximum number of points for each asset that Financial Pallet can store
//! #    pub const PriceCount: u32 = 30;
//! #    // Duration of the price period in minutes
//! #    pub const PricePeriod: u32 = 1440;
//! #    // CalcReturnType used by FinancialPallet's calc* extrinsics
//! #    pub const ReturnType: u32 = CalcReturnType::Log.into_u32();
//! #    // CalcVolatilityType used by FinancialPallet's calc* extrinsics
//! #    pub const VolCorType: i64 = CalcVolatilityType::Regular.into_i64();
//! # }
//! #
//! # pub type FixedNumber = substrate_fixed::types::I64F64;
//! # pub type Balance = FixedNumber;
//! # pub type AccountId = u32;
//! # type UncheckedExtrinsic = frame_system::mocking::MockUncheckedExtrinsic<Runtime>;
//! # type Block = frame_system::mocking::MockBlock<Runtime>;
//! #
//! # #[derive(Clone, Copy, Debug, Eq, PartialEq, Encode, Decode, Hash, Ord, PartialOrd)]
//! # pub enum Asset {
//! #     Btc,
//! #     Eth,
//! # }
//! #
//! # mod portfolio {
//! #     use frame_support::dispatch::DispatchError;
//! #     use financial_primitives::BalanceAware;
//! #     use crate::{Asset, Balance, AccountId};
//! #
//! #     pub struct Module<T> {
//! #         _marker: sp_std::marker::PhantomData<T>,
//! #     }
//! #
//! #     impl <T> BalanceAware for Module<T> {
//! #         type AccountId = AccountId;
//! #         type Asset = Asset;
//! #         type Balance = Balance;
//! #
//! #         fn balances(
//! #             account_id: &Self::AccountId,
//! #             assets: &[Self::Asset],
//! #         ) -> Result<Vec<Self::Balance>, DispatchError> {
//! #             Ok(vec![])
//! #         }
//! #     }
//! # }
//! #
//! #
//! # use frame_support::weights::{DispatchClass, constants::{WEIGHT_PER_SECOND, BlockExecutionWeight, ExtrinsicBaseWeight}};
//! # use frame_system::limits::{BlockLength, BlockWeights};
//! #
//! # const AVERAGE_ON_INITIALIZE_RATIO: Perbill = Perbill::from_percent(10);
//! # const NORMAL_DISPATCH_RATIO: Perbill = Perbill::from_percent(75);
//! # const MAXIMUM_BLOCK_WEIGHT: Weight = 2 * WEIGHT_PER_SECOND;
//! #
//! # parameter_types! {
//! #     pub const BlockHashCount: u64 = 250;
//! #     pub RuntimeBlockLength: BlockLength =
//! #         BlockLength::max_with_normal_ratio(5 * 1024 * 1024, NORMAL_DISPATCH_RATIO);
//! #     pub RuntimeBlockWeights: BlockWeights = BlockWeights::builder()
//! #         .base_block(BlockExecutionWeight::get())
//! #         .for_class(DispatchClass::all(), |weights| {
//! #             weights.base_extrinsic = ExtrinsicBaseWeight::get();
//! #         })
//! #         .for_class(DispatchClass::Normal, |weights| {
//! #             weights.max_total = Some(NORMAL_DISPATCH_RATIO * MAXIMUM_BLOCK_WEIGHT);
//! #         })
//! #         .for_class(DispatchClass::Operational, |weights| {
//! #             weights.max_total = Some(MAXIMUM_BLOCK_WEIGHT);
//! #             weights.reserved = Some(
//! #                 MAXIMUM_BLOCK_WEIGHT - NORMAL_DISPATCH_RATIO * MAXIMUM_BLOCK_WEIGHT
//! #             );
//! #         })
//! #         .avg_block_initialization(AVERAGE_ON_INITIALIZE_RATIO)
//! #         .build_or_panic();
//! #     pub const SS58Prefix: u8 = 42;
//! # }
//! #
//! #  impl frame_system::Config for Runtime {
//! #      type BlockWeights = RuntimeBlockWeights;
//! #      type BlockLength = RuntimeBlockLength;
//! #      type SS58Prefix = SS58Prefix;
//! #      type BaseCallFilter = ();
//! #      type Origin = Origin;
//! #      type Call = Call;
//! #      type Index = u64;
//! #      type BlockNumber = u64;
//! #      type Hash = H256;
//! #      type Hashing = BlakeTwo256;
//! #      type AccountId = AccountId;
//! #      type Lookup = IdentityLookup<Self::AccountId>;
//! #      type Header = Header;
//! #      type Event = Event;
//! #      type BlockHashCount = BlockHashCount;
//! #      type DbWeight = ();
//! #      type Version = ();
//! #      type PalletInfo = PalletInfo;
//! #      type AccountData = ();
//! #      type OnNewAccount = ();
//! #      type OnKilledAccount = ();
//! #      type SystemWeightInfo = ();
//! #  }
//!
//! #  frame_support::construct_runtime!(
//! #     pub enum Runtime where
//! #         Block = Block,
//! #         NodeBlock = Block,
//! #         UncheckedExtrinsic = UncheckedExtrinsic,
//! #     {
//! #         System: frame_system::{Module, Call, Config, Storage, Event<T>},
//! #         Financial: financial_pallet::{Module, Call, Storage, Event<T> },
//! #     }
//! # );
//!
//! #
//! # mod pallet_timestamp {
//! #     use core::time::Duration;
//! #     use frame_support::traits::UnixTime;
//! #
//! #     pub struct Module<T> {
//! #         _marker: sp_std::marker::PhantomData<T>,
//! #     }
//! #     impl <T> UnixTime for Module<T> {
//! #         fn now() -> Duration {
//! #             Duration::new(5, 0)
//! #         }
//! #     }
//! # }
//! #
//! impl financial_pallet::Config for Runtime {
//!     type Event = Event;
//!
//!     // In most cases you should use pallet_timestamp as a UnixTime trait implementation
//!     type UnixTime = pallet_timestamp::Module<Runtime>;
//!
//!     // Specify parameters here you defined before
//!     type PriceCount = PriceCount;
//!     type PricePeriod = PricePeriod;
//!     type ReturnType = ReturnType;
//!     type VolCorType = VolCorType;
//!
//!     // Construct fixed number type that substrate-fixed crate provides.
//!     // This type defines valid range of values and precision
//!     // for all calculations Financial Pallet performs.
//!     type FixedNumber = substrate_fixed::types::I64F64;
//!     // FixedNumber underlying type should be defined explicitly because
//!     // rust compiler could not determine it on its own.
//!     type FixedNumberBits = i128;
//!
//!     // Specify here system wide type used for balance values.
//!     // You should also provide conversions to and from FixedNumber
//!     // which is used for all calculations under the hood.
//!     type Price = Balance;
//!
//!     // Asset type specific to your system. It can be as simple as
//!     // enum. See example below.
//!     type Asset = Asset;
//!
//!     // Provide BalanceAware trait implementation.
//!     // Financial Pallet uses it to check user balances.
//!     type Balances = portfolio::Module<Runtime>;
//! }
//! ```
//!
//! ### Asset Type
//!
//! Financial Pallet requires you to define Asset type. This type should be system wide. Oracle
//! pallet uses it to notify that new price received through `on_price_update` call. The
//! `BalanceAware` trait which gets information about user balances also uses it.
//!
//! The `Asset` type declaration can be as simple as enum with some derived trait implementations:
//!
//! ```
//! # use frame_support::codec::{Decode, Encode};
//! #[derive(Clone, Copy, Debug, Eq, PartialEq, Encode, Decode, Hash, Ord, PartialOrd)]
//! pub enum Asset {
//!     Btc,
//!     Eth,
//! }
//! ```
//!
//! But you can define more sophisticated type which values are not predefined and are lied in the
//! storage.
//!
//! Note that `Asset` type must be instance of `Copy` trait, so it's values are supposed to be lightweight.
//!
//! ### Genesis Config
//!
//! You can provide initial price logs for the assets using [`GenesisConfig`](./struct.GenesisConfig.html).
//!
//! ## Assumptions
//!
//! - Value of the `PriceCount` parameter can not be greater than 180.
//! - Value of the `PricePeriod` parameter can not be greater than 10080 (one week).

#![warn(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::ops::{AddAssign, BitOrAssign, ShlAssign};
use financial_primitives::capvec::CapVec;
use financial_primitives::{
    BalanceAware, CalcReturnType, CalcVolatilityType, OnPriceSet, PricePeriod, PricePeriodError,
};
use frame_support::codec::{Decode, Encode, FullCodec};
use frame_support::dispatch::{DispatchError, Parameter};
use frame_support::storage::{IterableStorageMap, StorageMap};
use frame_support::traits::{Get, UnixTime};
use frame_support::weights::Weight;
use frame_support::{decl_error, decl_event, decl_module, decl_storage, dispatch, ensure};
use frame_system::ensure_signed;
use math::{
    calc_log_return, calc_return_exp_vola, calc_return_func, calc_return_iter, covariance, decay,
    demeaned, exp_corr, exp_vola, from_num, last_price, last_recurrent_ewma, log_value_at_risk,
    mean, mul, regular_corr, regular_value_at_risk, regular_vola, squared, sum, ConstType,
    MathError, MathResult,
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
pub trait Config: frame_system::Config {
    /// The overarching event type.
    type RuntimeEvent: From<Event<Self>> + Into<<Self as frame_system::Config>::Event>;
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
        + From<ConstType>
        + scale_info::TypeInfo;
    /// System wide type for representing price values. It must be convertible to and
    /// from [`FixedNumber`](#associatedtype.FixedNumber).
    type Price: Parameter + Clone + From<Self::FixedNumber> + Into<Self::FixedNumber>;
    /// Type that gets user balances for a given AccountId
    type Balances: BalanceAware<
        AccountId = Self::AccountId,
        Asset = Self::Asset,
        Balance = Self::Price,
    >;
}

#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Default,
    Encode,
    Decode,
    Debug,
    scale_info::TypeInfo,
)]
pub struct Duration {
    secs: u64,
    nanos: u32,
}

impl Duration {
    const NANOS_PER_SEC: u32 = 1_000_000_000;

    ///
    pub const fn new(secs: u64, nanos: u32) -> Duration {
        let secs = match secs.checked_add((nanos / Self::NANOS_PER_SEC) as u64) {
            Some(secs) => secs,
            None => panic!("overflow in Duration::new"),
        };
        let nanos = nanos % Self::NANOS_PER_SEC;
        Duration { secs, nanos }
    }

    ///
    pub const fn from_secs(secs: u64) -> Duration {
        Duration { secs, nanos: 0 }
    }

    ///
    pub const fn checked_add(self, rhs: Duration) -> Option<Duration> {
        if let Some(mut secs) = self.secs.checked_add(rhs.secs) {
            let mut nanos = self.nanos + rhs.nanos;
            if nanos >= Self::NANOS_PER_SEC {
                nanos -= Self::NANOS_PER_SEC;
                if let Some(new_secs) = secs.checked_add(1) {
                    secs = new_secs;
                } else {
                    return None;
                }
            }
            debug_assert!(nanos < Self::NANOS_PER_SEC);
            Some(Duration { secs, nanos })
        } else {
            None
        }
    }
}

impl core::ops::Add for Duration {
    type Output = Duration;

    fn add(self, rhs: Self) -> Self::Output {
        self.checked_add(rhs)
            .expect("overflow when adding durations")
    }
}

impl From<core::time::Duration> for Duration {
    fn from(duration: core::time::Duration) -> Self {
        Self {
            secs: duration.as_secs(),
            nanos: duration.subsec_nanos(),
        }
    }
}

impl From<Duration> for core::time::Duration {
    fn from(this: Duration) -> Self {
        Self::new(this.secs, this.nanos)
    }
}

/// Information about latest price update.
#[derive(Encode, Decode, Clone, Default, PartialEq, Eq, Debug, scale_info::TypeInfo)]
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
#[derive(Encode, Decode, Clone, Default, PartialEq, Eq, Debug, scale_info::TypeInfo)]
pub struct PriceLog<F> {
    /// Timestamp of the latest point in the log.
    pub latest_timestamp: Duration,
    /// History of prices changes for last [`PriceCount`](./trait.Config.html#associatedtype.PriceCount) periods in succession.
    pub prices: CapVec<F>,
}

/// Financial metrics for asset
#[derive(Encode, Decode, Clone, Default, PartialEq, Eq, Debug, scale_info::TypeInfo)]
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

/// Financial metrics for portfolio
#[derive(Encode, Decode, Clone, Default, PartialEq, Eq, Debug, scale_info::TypeInfo)]
pub struct PortfolioMetrics<P> {
    /// Start of the period inclusive for which metrics were calculated.
    pub period_start: Duration,
    /// End of the period exclusive for which metrics were calculated.
    pub period_end: Duration,

    ///  Number of standard deviations to consider.
    pub z_score: u32,

    /// Portfolio volatility
    pub volatility: P,
    /// Value at Risk for portfolio
    pub value_at_risk: P,
}

/// Financial metrics for all assets
#[derive(Encode, Decode, Clone, Default, PartialEq, Eq, Debug, scale_info::TypeInfo)]
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
    trait Store for Module<T: Config> as FinancialModule {
        /// Latest price updates on per asset basis.
        pub Updates get(fn updates): map hasher(blake2_128_concat) T::Asset => Option<PriceUpdate<T::FixedNumber>>;
        /// Price log on per asset basis.
        pub PriceLogs get(fn price_logs): map hasher(blake2_128_concat) T::Asset => Option<PriceLog<T::FixedNumber>>;
        /// Financial metrics on per asset basis.
        pub PerAssetMetrics get(fn per_asset_metrics): map hasher(blake2_128_concat) T::Asset => Option<AssetMetrics<T::Asset, T::Price>>;

        /// Financial metrics on per portfolio basis.
        pub PerPortfolioMetrics get(fn per_portfolio_metrics): map hasher(blake2_128_concat) T::AccountId => Option<PortfolioMetrics<T::Price>>;

        /// Financial metrics for all known assets.
        pub Metrics get(fn metrics): Option<FinancialMetrics<T::Asset, T::Price>>;
    }

    add_extra_genesis {
        /// Initial price logs on per asset basis.
        config(prices): Vec<(T::Asset, Vec<T::Price>, core::time::Duration)>;
        build(|config| {
            let max_price_count = 180;
            // Limit max value of `PricePeriod` to 7 days
            let max_price_period = 7 * 24 * 60;

            let price_count = T::PriceCount::get();
            assert!(price_count <= max_price_count, "PriceCount can not be greater than {}", max_price_count);

            let price_period = T::PricePeriod::get();
            assert!(price_period <= max_price_period, "PricePeriod can not be greater than {}", max_price_period);

            // We assume that each config item for a given asset contains prices of the past periods going in
            // succession. Timestamp of the last period is specified in `latest_timestamp`.
            for (asset, values, latest_timestamp) in config.prices.iter() {
                let mut prices = CapVec::<T::FixedNumber>::new(price_count);

                assert!(values.len() > 0, "Initial price vector can not be empty. Asset: {:?}.", asset);

                for v in values.iter() {
                    prices.push(v.clone().into());
                }

                PriceLogs::<T>::insert(asset, PriceLog {
                    latest_timestamp: Into::<Duration>::into(*latest_timestamp),
                    prices
                });
            }
        });
    }
}

decl_event!(
    pub enum Event<T>
    where
        Asset = <T as Config>::Asset,
        AccountId = <T as frame_system::Config>::AccountId,
    {
        /// Financial metrics for the specified asset have been recalculated
        /// \[asset\]
        AssetMetricsRecalculated(Asset),
        /// Financial metrics for all assets have been recalculated
        MetricsRecalculated(),
        /// Financial metrics for the specified portfolio have been recalculated
        /// \[portfolio\]
        PortfolioMetricsRecalculated(AccountId),
    }
);

decl_error! {
    /// Error for the Financial Pallet module.
    pub enum Error for Module<T: Config> {
        /// Timestamp of the received price is in the past.
        PeriodIsInThePast,
        /// Overflow occurred during financial calculation process.
        Overflow,
        /// Division by zero occurred during financial calculation process.
        DivisionByZero,
        /// Price log is not long enough to calculate required value.
        NotEnoughPoints,
        /// Required functionality is not implemented.
        NotImplemented,
        /// Storage of the pallet is in an unexpected state.
        InvalidStorage,
        /// Specified period start timestamp is invalid for current
        /// [`PricePeriod`](./trait.Config.html#associatedtype.PricePeriod) value.
        InvalidPeriodStart,
        /// An invalid argument passed to the transcendental function (i.e. log, sqrt, etc.)
        /// during financial calculation process.
        Transcendental,
        /// An invalid argument passed to the function.
        InvalidArgument,
        /// Default return type or default correlation type is initialized with a value that can
        /// not be converted to type `CalcReturnType` or `CalcVolatilityType` respectively.
        InvalidConstant,
        /// This method is not allowed in production. Method is used in testing only
        MethodNotAllowed,
    }
}

decl_module! {
    /// Financial Pallet module declaration.
    pub struct Module<T: Config> for enum Call where origin: T::Origin {
        type Error = Error<T>;

        fn deposit_event() = default;

        const PriceCount: u32 = T::PriceCount::get();
        const PricePeriod: u32 = T::PricePeriod::get();

        /// Recalculates financial metrics for a given asset
        #[weight = Weight::from_ref_time(10_000).saturating_add(T::DbWeight::get().writes(1))]
        pub fn recalc_asset(origin, asset: T::Asset) -> dispatch::DispatchResult {
            ensure_signed(origin)?;
            Self::recalc_asset_inner(asset)?;
            Ok(())
        }

        /// Recalculates financial metrics for a given portfolio
        #[weight = Weight::from_ref_time(10_000).saturating_add(T::DbWeight::get().writes(1))]
        pub fn recalc_portfolio(origin, account_id: T::AccountId, z_score: u32) -> dispatch::DispatchResult {
            ensure_signed(origin)?;
            Self::recalc_portfolio_inner(account_id, z_score)?;
            Ok(())
        }

        /// Recalculates financial metrics for all known assets.
        #[weight = Weight::from_ref_time(10_000).saturating_add(T::DbWeight::get().writes(1))]
        pub fn recalc(origin) -> dispatch::DispatchResult {
            ensure_signed(origin)?;
            Self::recalc_inner()?;
            Ok(())
        }

        /// Test utility function for setting metrics, not allowed in production
        #[weight = Weight::from_ref_time(10_000).saturating_add(T::DbWeight::get().writes(1))]
        pub fn set_metrics(origin, metrics: FinancialMetrics<T::Asset, T::Price>) -> dispatch::DispatchResult {
            if cfg!(feature = "production") {
                log::error!(
                    "{}:{}. Setting metrics is not allowed in production",
                    file!(),
                    line!(),
                );
                frame_support::fail!(Error::<T>::MethodNotAllowed);
            }

            Metrics::<T>::put(metrics);
            Ok(())
        }

        /// Test utility function for setting asset metrics, not allowed in production
        #[weight = Weight::from_ref_time(10_000).saturating_add(T::DbWeight::get().writes(1))]
        pub fn set_per_asset_metrics(origin, asset: T::Asset, metrics: AssetMetrics<T::Asset, T::Price>) -> dispatch::DispatchResult {
            if  cfg!(feature = "production") {
                log::error!(
                    "{}:{}. Setting metrics is not allowed in production",
                    file!(),
                    line!(),
                );
                frame_support::fail!(Error::<T>::MethodNotAllowed);
            }

            PerAssetMetrics::<T>::insert(&asset, metrics);
            Ok(())
        }

    }
}

/// Trait with main system calculations implemented for Financial Pallet `Module`. Recalculates
/// metrics for assets and user portfolios and saves them to storage.
pub trait FinancialSystemTrait {
    /// System wide type for representing various assets such as BTC, ETH, EOS, etc.
    type Asset;
    /// Substrate `AccountId` type
    type AccountId;

    /// Recalculates `FinancialMetrics` (metrics for all system assets) and updates this
    /// information in storage
    fn recalc_inner() -> dispatch::DispatchResult;
    /// Recalculates `PerAssetMetrics` for given asset and updates this information in storage
    fn recalc_asset_inner(asset: Self::Asset) -> dispatch::DispatchResult;
    /// Recalculates `PerPortfolioMetrics` for `account_id`'s portfolio and updates this
    /// information in storage
    fn recalc_portfolio_inner(
        account_id: Self::AccountId,
        z_score: u32,
    ) -> dispatch::DispatchResult;
}

impl<T: Config> FinancialSystemTrait for Module<T> {
    type Asset = T::Asset;
    type AccountId = T::AccountId;

    fn recalc_inner() -> dispatch::DispatchResult {
        let return_type = CalcReturnType::try_from(T::ReturnType::get())
            .map_err(|_| Error::<T>::InvalidConstant)?;
        let vol_cor_type = CalcVolatilityType::try_from(T::VolCorType::get())
            .map_err(|_| Error::<T>::InvalidConstant)?;

        let price_period = PricePeriod(T::PricePeriod::get());

        let mut asset_logs: Vec<_> = PriceLogs::<T>::iter().collect();
        ensure!(asset_logs.len() > 0, Error::<T>::NotEnoughPoints);
        asset_logs.sort_by(|(a1, _), (a2, _)| a1.cmp(a2));

        let metrics = financial_metrics::<T::Asset, T::FixedNumber, T::Price>(
            return_type,
            vol_cor_type,
            &price_period,
            &asset_logs,
        )
        .map_err(Into::<Error<T>>::into)?;

        Metrics::<T>::put(metrics);

        Self::deposit_event(RawEvent::MetricsRecalculated());

        Ok(())
    }

    fn recalc_asset_inner(asset: Self::Asset) -> dispatch::DispatchResult {
        let return_type = CalcReturnType::try_from(T::ReturnType::get())
            .map_err(|_| Error::<T>::InvalidConstant)?;
        let vol_cor_type = CalcVolatilityType::try_from(T::VolCorType::get())
            .map_err(|_| Error::<T>::InvalidConstant)?;

        let mut correlations = vec![];

        let mut asset_logs: Vec<_> = PriceLogs::<T>::iter().collect();
        ensure!(asset_logs.len() > 0, Error::<T>::NotEnoughPoints);
        asset_logs.sort_by(|(a1, _), (a2, _)| a1.cmp(a2));

        let price_period = PricePeriod(T::PricePeriod::get());
        let ranges = asset_logs
            .iter()
            .map(|(_, l)| get_period_id_range(&price_period, l.prices.len(), l.latest_timestamp))
            .collect::<MathResult<Vec<_>>>()
            .map_err(Into::<Error<T>>::into)?;

        // Ensure that all correlations calculated for the same period
        let intersection = get_range_intersection(ranges.iter()).map_err(Into::<Error<T>>::into)?;

        let (log1, period_id_range1) = asset_logs
            .iter()
            .zip(ranges.iter())
            .find_map(|((a, l), r)| if *a == asset { Some((l, r)) } else { None })
            .ok_or(Error::<T>::NotEnoughPoints)?;
        let range1 =
            get_index_range(period_id_range1, &intersection).map_err(Into::<Error<T>>::into)?;
        let prices1 = log1.prices.iter_range(&range1).copied().collect::<Vec<_>>();

        let ret1 =
            Ret::<T::FixedNumber>::new(&prices1, return_type).map_err(Into::<Error<T>>::into)?;
        let vol1 =
            Vol::<T::FixedNumber>::new(&ret1, vol_cor_type).map_err(Into::<Error<T>>::into)?;

        for ((asset2, log2), period_id_range2) in asset_logs.iter().zip(ranges.iter()) {
            if *asset2 == asset {
                // Correlation for any asset with itself should be 1
                correlations.push((*asset2, from_num::<T::FixedNumber>(1).into()));
            } else {
                let range2 = get_index_range(period_id_range2, &intersection)
                    .map_err(Into::<Error<T>>::into)?;
                let prices2 = log2.prices.iter_range(&range2).copied().collect::<Vec<_>>();

                let ret2 = Ret::<T::FixedNumber>::new(&prices2, return_type)
                    .map_err(Into::<Error<T>>::into)?;
                let vol2 = Vol::<T::FixedNumber>::new(&ret2, vol_cor_type)
                    .map_err(Into::<Error<T>>::into)?;

                let corre = cor(&ret1, &vol1, &ret2, &vol2).map_err(Into::<Error<T>>::into)?;

                correlations.push((*asset2, corre.into()));
            }
        }

        let temporal_range = Range {
            start: price_period
                .get_period_id_start(intersection.start)
                .map_err(Into::<MathError>::into)
                .map_err(Into::<Error<T>>::into)?,
            end: price_period
                .get_period_id_start(intersection.end)
                .map_err(Into::<MathError>::into)
                .map_err(Into::<Error<T>>::into)?,
        };

        let returns: Vec<T::Price> = ret1.ret.into_iter().map(|x| x.into()).collect();
        let volatility: T::Price = vol1.vol.into();

        PerAssetMetrics::<T>::insert(
            &asset,
            AssetMetrics {
                period_start: temporal_range.start.into(),
                period_end: temporal_range.end.into(),
                returns,
                volatility,
                correlations,
            },
        );

        Self::deposit_event(RawEvent::AssetMetricsRecalculated(asset));

        Ok(())
    }

    fn recalc_portfolio_inner(
        account_id: Self::AccountId,
        z_score: u32,
    ) -> dispatch::DispatchResult {
        let return_type = CalcReturnType::try_from(T::ReturnType::get())
            .map_err(|_| Error::<T>::InvalidConstant)?;
        let vol_cor_type = CalcVolatilityType::try_from(T::VolCorType::get())
            .map_err(|_| Error::<T>::InvalidConstant)?;

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

        let volatility =
            portfolio_vol(&ws, &metrics.covariances).map_err(Into::<Error<T>>::into)?;
        let total_weighted_mean_return = sum(mul(ws.into_iter(), metrics.mean_returns.into_iter()))
            .map_err(Into::<Error<T>>::into)?;

        let value_at_risk = match return_type {
            CalcReturnType::Regular => {
                regular_value_at_risk(z_score, volatility, total_weighted_mean_return)
                    .map_err(Into::<Error<T>>::into)?
            }
            CalcReturnType::Log => {
                log_value_at_risk(z_score, volatility, total_weighted_mean_return)
                    .map_err(Into::<Error<T>>::into)?
            }
        };

        PerPortfolioMetrics::<T>::insert(
            &account_id,
            PortfolioMetrics {
                period_start: metrics.period_start,
                period_end: metrics.period_end,
                z_score,
                volatility: volatility.into(),
                value_at_risk: value_at_risk.into(),
            },
        );

        Self::deposit_event(RawEvent::PortfolioMetricsRecalculated(account_id));

        Ok(())
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

impl<T: Config> From<GetNewPricesError> for Error<T> {
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

impl<T: Config> From<PricePeriodChangeError> for Error<T> {
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
    let prev_period_id = price_period.get_period_id(prev_start.into())?;
    let curr_period_id = price_period.get_period_id(now.into())?;

    let prev: i32 = prev_period_id
        .try_into()
        .map_err(|_| PricePeriodError::Overflow)?;
    let curr: i32 = curr_period_id
        .try_into()
        .map_err(|_| PricePeriodError::Overflow)?;
    let delta: Result<_, PricePeriodError> =
        curr.checked_sub(prev).ok_or(PricePeriodError::Overflow);
    Ok((
        price_period.get_period_id_start(curr_period_id)?.into(),
        delta?,
    ))
}

/// Decides whether the period change took place.
fn get_period_change(
    price_period: &PricePeriod,
    period_start: Option<Duration>,
    now: Duration,
) -> Result<PricePeriodChange, PricePeriodChangeError> {
    if let Some(period_start) = period_start {
        ensure!(
            price_period.is_valid_period_start(period_start.into())?,
            PricePeriodChangeError::InvalidPeriodStart
        );
    }

    match period_start {
        // No `period_start` exists. It means that we received price update for the first time.
        None => {
            let period_start = price_period.get_period_start(now.into())?;

            Ok(PricePeriodChange {
                period_start: period_start.into(),
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

impl<T: Config> OnPriceSet for Module<T> {
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
        let period_change = get_period_change(&price_period, period_start, now.into())
            .map_err(Into::<Error<T>>::into)?;

        let period_start = period_change.period_start;

        // Every point received is stores to the `Update`.
        // Meanwhile only first point received in the current period is stored in the price log.
        match period_change.action {
            PricePeriodAction::RemainsUnchanged => {
                Updates::<T>::insert(
                    &asset,
                    PriceUpdate {
                        period_start: period_start,
                        time: now.into(),
                        price: value,
                    },
                );
            }
            PricePeriodAction::StartedNew(empty_periods) => {
                let log_to_insert = if let Some(mut existing_log) = log {
                    existing_log.latest_timestamp = now.into();

                    // If `PriceLog` for a given asset is not empty, then it should contain at
                    // least one price value. It was set during previous `on_price_update` call or
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
                        latest_timestamp: now.into(),
                        prices: new_prices,
                    }
                };

                Updates::<T>::insert(
                    &asset,
                    PriceUpdate {
                        period_start: period_start,
                        time: now.into(),
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
    /// Calculates rv.
    fn calc_rv(
        return_type: CalcReturnType,
        ewma_length: u32,
        asset: Self::Asset,
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

struct Rv;

impl Rv {
    fn exp_vol<F, I>(
        returns: I,
        decay: F,
        last_price: F,
        return_type: CalcReturnType,
    ) -> MathResult<F>
    where
        F: FixedSigned + PartialOrd<ConstType> + From<ConstType>,
        F::Bits: Copy + ToFixed + AddAssign + BitOrAssign + ShlAssign,
        I: Iterator<Item = MathResult<F>>,
    {
        let squared_returns = squared(returns);
        let var = last_recurrent_ewma(squared_returns, decay)?;
        let vol = exp_vola(return_type, var, last_price)?;

        Ok(vol)
    }

    pub fn regular<F>(prices: &[F], ewma_length: u32) -> MathResult<F>
    where
        F: FixedSigned + PartialOrd<ConstType> + From<ConstType>,
        F::Bits: Copy + ToFixed + AddAssign + BitOrAssign + ShlAssign,
    {
        let returns = calc_return_iter(prices, calc_return_exp_vola);
        let decay = decay(ewma_length)?;
        let last_price = last_price(prices)?;

        Self::exp_vol(returns, decay, last_price, CalcReturnType::Regular)
    }

    pub fn log<F, I>(last_price: F, log_returns: I, decay: F) -> MathResult<F>
    where
        F: FixedSigned + PartialOrd<ConstType> + From<ConstType>,
        F::Bits: Copy + ToFixed + AddAssign + BitOrAssign + ShlAssign,
        I: Iterator<Item = MathResult<F>>,
    {
        Self::exp_vol(log_returns, decay, last_price, CalcReturnType::Log)
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
                // Correlation for any asset with itself should be 1
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
        period_start: temporal_range.start.into(),
        period_end: temporal_range.end.into(),
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

impl<T: Config> From<MathError> for Error<T> {
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
pub fn get_period_id_range(
    price_period: &PricePeriod,
    items_length: usize,
    latest_timestamp: Duration,
) -> MathResult<Range<u64>> {
    let last_period_id = price_period.get_period_id(latest_timestamp.into())?;
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

/// Gets ranges intersection
pub fn get_range_intersection<'a, T, I>(mut ranges: I) -> MathResult<Range<T>>
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
pub fn get_index_range(range: &Range<u64>, intersection: &Range<u64>) -> MathResult<Range<usize>> {
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
fn get_prices_for_common_periods<T: Config>(
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
        start: price_period.get_period_id_start(intersection.start)?.into(),
        end: price_period.get_period_id_start(intersection.end)?.into(),
    };

    Ok((prices1, prices2, temporal_range))
}

impl<T: Config> Financial for Module<T> {
    type Asset = T::Asset;
    type Price = T::Price;
    type AccountId = <T as frame_system::Config>::AccountId;

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

    fn calc_rv(
        return_type: CalcReturnType,
        ewma_length: u32,
        asset: T::Asset,
    ) -> Result<Self::Price, DispatchError> {
        let log = PriceLogs::<T>::get(asset).ok_or(Error::<T>::NotEnoughPoints)?;
        let prices: Vec<_> = log.prices.iter().copied().collect();

        match return_type {
            CalcReturnType::Regular => Ok(Rv::regular(&prices, ewma_length)
                .map_err(Into::<Error<T>>::into)?
                .into()),
            CalcReturnType::Log => {
                let last_price = last_price(&prices).map_err(Into::<Error<T>>::into)?;
                let log_returns = calc_return_iter(&prices, calc_log_return);
                let decay = decay(ewma_length).map_err(Into::<Error<T>>::into)?;

                Ok(Rv::log(last_price, log_returns, decay)
                    .map_err(Into::<Error<T>>::into)?
                    .into())
            }
        }
    }
}
