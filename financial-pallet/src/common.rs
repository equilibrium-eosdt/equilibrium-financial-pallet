// Copyright (C) 2020 equilibrium.
// SPDX-License-Identifier: Apache-2.0
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
