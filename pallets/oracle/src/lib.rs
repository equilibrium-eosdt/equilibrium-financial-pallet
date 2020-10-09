#![cfg_attr(not(feature = "std"), no_std)]

use financial_primitives::OnPriceSet;
use frame_support::{decl_error, decl_event, decl_module, decl_storage, dispatch, traits::Get};
use frame_system::ensure_signed;
use frame_support::codec::Codec;
use frame_support::dispatch::Parameter;

#[cfg(test)]
mod mock;

#[cfg(test)]
mod tests;

pub trait Trait: frame_system::Trait {
	type Event: From<Event<Self>> + Into<<Self as frame_system::Trait>::Event>;
	type Asset: Codec + Parameter + Default + Copy;
	type Price: Codec + Parameter + Default + Copy;
	type OnPriceSet: OnPriceSet<Price = Self::Price, Asset = Self::Asset>;
}

decl_storage! {
	trait Store for Module<T: Trait> as OracleModule {
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
		AccountId = <T as frame_system::Trait>::AccountId,
		Asset = <T as Trait>::Asset,
		Price = <T as Trait>::Price,
	{
		PriceSet(Asset, Price, AccountId),
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
		pub fn set_price(origin, asset: T::Asset, value: T::Price) -> dispatch::DispatchResult {
			let who = ensure_signed(origin)?;

			PricePoints::<T>::insert(asset, value);

			let _ = T::OnPriceSet::on_price_set(asset, value);
			Self::deposit_event(RawEvent::PriceSet(asset, value, who));
			Ok(())
		}
	}
}
