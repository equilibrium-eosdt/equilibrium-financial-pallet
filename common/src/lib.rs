#![cfg_attr(not(feature = "std"), no_std)]

use core::slice::Iter;
use frame_support::codec::{Decode, Encode};
#[cfg(feature = "std")]
use serde::{Deserialize, Serialize};

#[derive(Encode, Decode, Debug, PartialEq, Eq, Clone, Copy, Hash)]
#[cfg_attr(feature = "std", derive(Serialize, Deserialize))]
pub enum Asset {
    Unknown,
    Usd,
    Eq,
    Eth,
    Btc,
    Eos,
}

impl Default for Asset {
    fn default() -> Asset {
        Asset::Unknown
    }
}

impl Asset {
    pub fn iterator() -> Iter<'static, Asset> {
        static ASSETS: [Asset; 5] = [Asset::Usd, Asset::Eq, Asset::Eth, Asset::Btc, Asset::Eos];
        ASSETS.iter()
    }
}
