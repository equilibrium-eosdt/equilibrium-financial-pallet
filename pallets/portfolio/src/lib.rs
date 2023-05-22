#![cfg_attr(not(feature = "std"), no_std)]

use core::convert::TryInto;
use financial_primitives::BalanceAware;
use frame_support::codec::Codec;
use frame_support::dispatch::{DispatchError, DispatchResult, Parameter};
use frame_support::{
    decl_error, decl_event, decl_module, decl_storage, dispatch, traits::Get, weights::Weight,
};
use frame_system::ensure_signed;
use sp_std::vec::Vec;

#[cfg(test)]
mod mock;

#[cfg(test)]
mod tests;

pub trait Config: frame_system::Config {
    type Event: From<Event<Self>> + Into<<Self as frame_system::Config>::Event>;
    type Asset: Codec + Parameter + Default + Copy;
    type Balance: Codec + Parameter + Default + Copy;
}

decl_storage! {
    trait Store for Module<T: Config> as PortfolioModule {
        Balances get(fn balances): double_map hasher(blake2_128_concat) T::AccountId, hasher(identity) T::Asset => T::Balance;
    }
}

decl_event!(
    pub enum Event<T>
    where
        AccountId = <T as frame_system::Config>::AccountId,
        Asset = <T as Config>::Asset,
        Balance = <T as Config>::Balance,
    {
        /// \[who, currency, value, signer\]
        BalanceSet(AccountId, Asset, Balance, AccountId),
    }
);

decl_error! {
    pub enum Error for Module<T: Config> {}
}

decl_module! {
    pub struct Module<T: Config> for enum Call where origin: T::Origin {
        type Error = Error<T>;

        fn deposit_event() = default;

        #[weight = Weight::from_ref_time(10_000).saturating_add(T::DbWeight::get().writes(1))]
        pub fn set_balance(origin, account_id: T::AccountId,  asset: T::Asset, balance: T::Balance) -> dispatch::DispatchResult {
            let who = ensure_signed(origin)?;

            Balances::<T>::mutate(&account_id, asset, |b| -> DispatchResult {
                *b = balance;
                Ok(())
            })?;

            Self::deposit_event(RawEvent::BalanceSet(account_id, asset, balance, who));
            Ok(())
        }
    }
}

impl<T: Config> BalanceAware for Module<T> {
    type AccountId = T::AccountId;
    type Asset = T::Asset;
    type Balance = T::Balance;

    fn balances(
        account_id: &T::AccountId,
        assets: &[Self::Asset],
    ) -> Result<Vec<Self::Balance>, DispatchError> {
        Ok(assets
            .iter()
            .map(|a| <Balances<T>>::get(account_id, a))
            .collect())
    }
}
