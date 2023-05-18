#![cfg_attr(not(feature = "std"), no_std)]

use frame_support::traits::UnixTime;
use frame_support::{decl_error, decl_event, decl_module, decl_storage, dispatch, traits::Get, weights::Weight};
use frame_system::ensure_signed;
use core::convert::TryInto;

#[cfg(test)]
mod mock;

#[cfg(test)]
mod tests;

pub trait Config: frame_system::Config {
    type Event: From<Event<Self>> + Into<<Self as frame_system::Config>::Event>;
}

decl_storage! {
    trait Store for Module<T: Config> as OracleModule {
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
        AccountId = <T as frame_system::Config>::AccountId,
    {
        /// `Now` timestamp set to a new value
        /// \[new_now, who\]
        NowAdvanced(u64, AccountId),
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
        pub fn advance_secs(origin, secs: u32) -> dispatch::DispatchResult {
            let who = ensure_signed(origin)?;
            let new_now = Self::now() + 1000 * (secs as u64);

            <Self as Store>::Now::put(new_now);

            Self::deposit_event(RawEvent::NowAdvanced(new_now, who));
            Ok(())
        }
    }
}

impl<T: Config> UnixTime for Module<T> {
    fn now() -> core::time::Duration {
        core::time::Duration::from_millis(Self::now())
    }
}
