#![cfg_attr(not(feature = "std"), no_std)]

use financial_primitives::OnPriceSet;
use frame_support::codec::Codec;
use frame_support::dispatch::Parameter;
use frame_support::{decl_error, decl_event, decl_module, decl_storage, dispatch, traits::Get, weights::Weight};
use frame_system::ensure_signed;
use core::convert::TryInto;

#[cfg(test)]
mod mock;

#[cfg(test)]
mod tests;

pub trait Config: frame_system::Config {
    type Event: From<Event<Self>> + Into<<Self as frame_system::Config>::Event>;
    type Asset: Codec + Parameter + Default + Copy;
    type Price: Codec + Parameter + Default + Copy;
    type OnPriceSet: OnPriceSet<Price = Self::Price, Asset = Self::Asset>;
}

decl_storage! {
    trait Store for Module<T: Config> as OracleModule {
        PricePoints get(fn price_points): map hasher(identity) T::Asset => T::Price;
    }
    add_extra_genesis {
        config(prices): Vec<(T::Asset, T::Price)>;

        build(|config| {
            for &(asset, value) in config.prices.iter() {
                PricePoints::<T>::insert(asset, value);
            }
        });
    }
}

decl_event!(
    pub enum Event<T>
    where
        AccountId = <T as frame_system::Config>::AccountId,
        Asset = <T as Config>::Asset,
        Price = <T as Config>::Price,
    {
        /// \[currency, value, who\]
        PriceSet(Asset, Price, AccountId),
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
        pub fn set_price(origin, asset: T::Asset, value: T::Price) -> dispatch::DispatchResult {
            let who = ensure_signed(origin)?;

            PricePoints::<T>::insert(asset, value);

            let _ = T::OnPriceSet::on_price_set(asset, value);
            Self::deposit_event(RawEvent::PriceSet(asset, value, who));
            Ok(())
        }
    }
}
