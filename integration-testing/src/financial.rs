use codec::{Codec, Encode};
use common::Asset;
use core::marker::PhantomData;
use pallet_financial::{AssetMetrics, FinancialMetrics, PriceLog, PriceUpdate};
use sp_runtime::traits::Member;
use std::fmt::Debug;
use substrate_fixed::types::I64F64;
use substrate_subxt::system::{System, SystemEventsDecoder};
use substrate_subxt_proc_macro::{module, Call, Store};

pub type FixedNumber = I64F64;
pub type Price = FixedNumber;

#[module]
pub trait Financial: System {
    type Asset: Default + Codec + Member;
    type FixedNumber: Default + Codec + Member;
    type Price: Default + Codec + Member;
}

#[derive(Clone, Debug, Eq, PartialEq, Store, Encode)]
pub struct UpdatesStore<T: Financial> {
    #[store(returns = Option<PriceUpdate<T::FixedNumber>>)]
    pub asset: Asset,
    pub _runtime: PhantomData<T>,
}

#[derive(Clone, Debug, Eq, PartialEq, Store, Encode)]
pub struct PriceLogsStore<T: Financial> {
    #[store(returns = Option<PriceLog<T::FixedNumber>>)]
    pub asset: Asset,
    pub _runtime: PhantomData<T>,
}

#[derive(Clone, Debug, Eq, PartialEq, Store, Encode)]
pub struct MetricsStore<T: Financial> {
    #[store(returns = Option<FinancialMetrics<T::Asset, T::Price>>)]
    pub _runtime: PhantomData<T>,
}

#[derive(Clone, Debug, Eq, PartialEq, Store, Encode)]
pub struct PerAssetMetricsStore<T: Financial> {
    #[store(returns = Option<AssetMetrics<T::Asset, T::Price>>)]
    pub asset: Asset,
    pub _runtime: PhantomData<T>,
}

#[derive(Clone, Debug, PartialEq, Call, Encode)]
pub struct RecalcAssetCall<T: Financial> {
    pub asset: Asset,
    pub _runtime: PhantomData<T>,
}

#[derive(Clone, Debug, PartialEq, Call, Encode)]
pub struct RecalcCall<T: Financial> {
    pub _runtime: PhantomData<T>,
}
