#![cfg_attr(not(feature = "std"), no_std)]

use frame_support::traits::UnixTime;
use frame_support::{decl_error, decl_event, decl_module, decl_storage, dispatch, traits::Get};
use frame_system::ensure_signed;

#[cfg(test)]
mod mock;

#[cfg(test)]
mod tests;

pub trait Trait: frame_system::Trait {
    type Event: From<Event<Self>> + Into<<Self as frame_system::Trait>::Event>;
}

decl_storage! {
    trait Store for Module<T: Trait> as OracleModule {
        pub Now get(fn now): u64;
    }
    add_extra_genesis {
        config(timestamp): u64;

        build(|config| {
            Now::put(config.timestamp);
        });
    }
}

decl_event!(
    pub enum Event<T>
    where
        AccountId = <T as frame_system::Trait>::AccountId,
    {
        NowAdvanced(u64, AccountId),
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
        pub fn advance_secs(origin, secs: u32) -> dispatch::DispatchResult {
            let who = ensure_signed(origin)?;
            let new_now = Self::now() + 1000 * (secs as u64);

            <Self as Store>::Now::put(new_now);

            Self::deposit_event(RawEvent::NowAdvanced(new_now, who));
            Ok(())
        }
    }
}

impl<T: Trait> UnixTime for Module<T> {
    fn now() -> core::time::Duration {
        core::time::Duration::from_millis(Self::now())
    }
}
