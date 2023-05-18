#![cfg_attr(not(feature = "std"), no_std)]

use core::slice::Iter;
use frame_support::codec::{Decode, Encode};
#[cfg(feature = "std")]
use serde::{Deserialize, Serialize};

#[derive(Encode, Decode, Debug, PartialEq, Eq, Clone, Copy, Hash, Ord, PartialOrd, scale_info::TypeInfo)]
#[cfg_attr(feature = "std", derive(Serialize, Deserialize))]
pub enum Asset {
    Unknown,
    Usd,
    Eq,
    Eth,
    Btc,
    Eos,
    Dot,
}

impl Default for Asset {
    fn default() -> Asset {
        Asset::Unknown
    }
}

impl Asset {
    pub fn iterator() -> Iter<'static, Asset> {
        static ASSETS: [Asset; 6] = [
            Asset::Usd,
            Asset::Eq,
            Asset::Eth,
            Asset::Btc,
            Asset::Eos,
            Asset::Dot,
        ];
        ASSETS.iter()
    }
}

impl Asset {
    pub fn value(&self) -> u8 {
        match *self {
            Asset::Unknown => 0x0,
            Asset::Usd => 0x1,
            Asset::Eq => 0x2,
            Asset::Eth => 0x3,
            Asset::Btc => 0x4,
            Asset::Eos => 0x5,
            Asset::Dot => 0x6,
        }
    }
}
