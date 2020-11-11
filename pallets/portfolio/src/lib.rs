#![cfg_attr(not(feature = "std"), no_std)]

use financial_primitives::BalanceAware;
use frame_support::codec::Codec;
use frame_support::dispatch::{DispatchError, DispatchResult, Parameter};
use frame_support::{decl_error, decl_event, decl_module, decl_storage, dispatch, traits::Get};
use frame_system::ensure_signed;
use sp_std::vec::Vec;

#[cfg(test)]
mod mock;

#[cfg(test)]
mod tests;

pub trait Trait: frame_system::Trait {
    type Event: From<Event<Self>> + Into<<Self as frame_system::Trait>::Event>;
    type Asset: Codec + Parameter + Default + Copy;
    type Balance: Codec + Parameter + Default + Copy;
}

decl_storage! {
    trait Store for Module<T: Trait> as OracleModule {
        Balances get(fn balances): double_map hasher(blake2_128_concat) T::AccountId, hasher(identity) T::Asset => T::Balance;
    }
}

decl_event!(
    pub enum Event<T>
    where
        AccountId = <T as frame_system::Trait>::AccountId,
        Asset = <T as Trait>::Asset,
        Balance = <T as Trait>::Balance,
    {
        BalanceSet(AccountId, Asset, Balance, AccountId),
    }
);

decl_error! {
    pub enum Error for Module<T: Trait> {}
}

decl_module! {
    pub struct Module<T: Trait> for enum Call where origin: T::Origin {
        type Error = Error<T>;

        fn deposit_event() = default;

        #[weight = 10_000 + T::DbWeight::get().writes(1)]
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

impl<T: Trait> BalanceAware for Module<T> {
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
