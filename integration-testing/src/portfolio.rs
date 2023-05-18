use codec::{Codec, Encode};
use common::Asset;
use sp_runtime::traits::Member;
use std::fmt::Debug;
use substrate_fixed::types::I64F64;
use substrate_subxt::system::{System, SystemEventsDecoder};
use substrate_subxt_proc_macro::{module, Call, Store};

pub type FixedNumber = I64F64;
pub type Balance = FixedNumber;

#[module]
pub trait Portfolio: System {
    type Asset: Default + Codec + Member;
    type Balance: Default + Codec + Member;
    type FixedNumber: Default + Codec + Member;
}

#[derive(Clone, Debug, Eq, PartialEq, Store, Encode)]
pub struct BalancesStore<T: Portfolio> {
    #[store(returns = <T as Portfolio>::Balance)]
    pub account_id: T::AccountId,
    pub asset: Asset,
}

#[derive(Clone, Debug, PartialEq, Call, Encode)]
pub struct SetBalanceCall<T: Portfolio> {
    pub account_id: T::AccountId,
    pub asset: Asset,
    pub balance: <T as Portfolio>::Balance,
}
