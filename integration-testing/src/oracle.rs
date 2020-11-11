use codec::{Codec, Encode};
use common::Asset;
use core::marker::PhantomData;
use sp_runtime::traits::Member;
use std::fmt::Debug;
use substrate_fixed::types::I64F64;
use substrate_subxt::system::{System, SystemEventsDecoder};
use substrate_subxt_proc_macro::{module, Call, Store};

pub type Price = I64F64;

#[module]
pub trait Oracle: System {
    type Asset: Default + Codec + Member;
    type Price: Default + Codec + Member;
}

#[derive(Clone, Debug, Eq, PartialEq, Store, Encode)]
pub struct PricePointsStore<T: Oracle> {
    #[store(returns = Price)]
    pub asset: <T as Oracle>::Asset,
}

#[derive(Clone, Debug, PartialEq, Call, Encode)]
pub struct SetPriceCall<T: Oracle> {
    pub asset: Asset,
    pub value: Price,
    pub _runtime: PhantomData<T>,
}
