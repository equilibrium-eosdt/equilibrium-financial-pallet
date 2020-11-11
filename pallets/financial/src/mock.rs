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

use crate::{Module, Trait};
use chrono::prelude::*;
use core::time::Duration;
use financial_primitives::{BalanceAware, CalcReturnType, CalcVolatilityType};
use frame_support::codec::{Decode, Encode};
use frame_support::dispatch::DispatchError;
use frame_support::traits::UnixTime;
use frame_support::{impl_outer_origin, parameter_types, weights::Weight};
use frame_system as system;
use sp_core::H256;
use sp_runtime::{
    testing::Header,
    traits::{BlakeTwo256, IdentityLookup},
    Perbill,
};
use std::cell::RefCell;
use std::collections::HashMap;
use substrate_fixed::types::I64F64;

impl_outer_origin! {
    pub enum Origin for Test {}
}

type AccountId = u64;

thread_local! {
    pub static NOW: RefCell<Duration> = RefCell::new(Duration::from_secs(0));
    pub static BALANCES: RefCell<HashMap<(AccountId, Asset), FixedNumber>> = RefCell::new(HashMap::new());
}

pub fn set_now(now: Duration) {
    NOW.with(|n| *n.borrow_mut() = now);
}

#[allow(dead_code)]
pub fn set_balance(account_id: AccountId, asset: Asset, balance: FixedNumber) {
    BALANCES.with(|b| b.borrow_mut().insert((account_id, asset), balance));
}

#[derive(Clone, Eq, PartialEq)]
pub struct Test;
parameter_types! {
    pub const BlockHashCount: u64 = 250;
    pub const MaximumBlockWeight: Weight = 1024;
    pub const MaximumBlockLength: u32 = 2 * 1024;
    pub const AvailableBlockRatio: Perbill = Perbill::from_percent(75);
}

impl system::Trait for Test {
    type BaseCallFilter = ();
    type Origin = Origin;
    type Call = ();
    type Index = u64;
    type BlockNumber = u64;
    type Hash = H256;
    type Hashing = BlakeTwo256;
    type AccountId = AccountId;
    type Lookup = IdentityLookup<Self::AccountId>;
    type Header = Header;
    type Event = ();
    type BlockHashCount = BlockHashCount;
    type MaximumBlockWeight = MaximumBlockWeight;
    type DbWeight = ();
    type BlockExecutionWeight = ();
    type ExtrinsicBaseWeight = ();
    type MaximumExtrinsicWeight = MaximumBlockWeight;
    type MaximumBlockLength = MaximumBlockLength;
    type AvailableBlockRatio = AvailableBlockRatio;
    type Version = ();
    type ModuleToIndex = ();
    type AccountData = ();
    type OnNewAccount = ();
    type OnKilledAccount = ();
    type SystemWeightInfo = ();
}

pub type FixedNumber = I64F64;

parameter_types! {
    pub const PriceCount: u32 = 10;
    pub const PricePeriod: u32 = 60;
    pub const ReturnType: u32 = CalcReturnType::Regular.into_u32();
    pub const VolCorType: i64 = CalcVolatilityType::Regular.into_i64();
}

pub struct TestUnixTime;

impl UnixTime for TestUnixTime {
    fn now() -> Duration {
        NOW.with(|n| *n.borrow())
    }
}

impl Trait for Test {
    type Event = ();
    type PriceCount = PriceCount;
    type PricePeriod = PricePeriod;
    type ReturnType = ReturnType;
    type VolCorType = VolCorType;
    type UnixTime = TestUnixTime;
    type Asset = Asset;
    type FixedNumberBits = i128;
    type FixedNumber = FixedNumber;
    type Price = FixedNumber;
    type Balances = Balances;
}

pub type FinancialModule = Module<Test>;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Encode, Decode, Hash, Ord, PartialOrd)]
pub enum Asset {
    Usd,
    Btc,
    Eos,
    Eq,
    Eth,
}

pub struct Balances;

impl BalanceAware for Balances {
    type AccountId = AccountId;
    type Asset = Asset;
    type Balance = FixedNumber;

    fn balances(
        account_id: &Self::AccountId,
        assets: &[Self::Asset],
    ) -> Result<Vec<Self::Balance>, DispatchError> {
        BALANCES.with(|b| {
            Ok(assets
                .iter()
                .map(|&a| {
                    b.borrow()
                        .get(&(*account_id, a))
                        .copied()
                        .unwrap_or(FixedNumber::default())
                })
                .collect())
        })
    }
}

pub fn initial_btc_prices() -> Vec<f64> {
    vec![
        7_117.21, 7_429.72, 7_550.90, 7_569.94, 7_679.87, 7_795.60, 7_807.06, 8_801.04, 8_658.55,
        8_864.77,
    ]
}

pub fn initial_eos_prices() -> Vec<f64> {
    vec![2.62, 2.67, 2.72, 2.72, 2.74, 2.75, 2.78, 3.02, 2.83, 2.89]
}

pub fn initial_eq_prices() -> Vec<FixedNumber> {
    vec![
        FixedNumber::max_value(),
        FixedNumber::max_value(),
        FixedNumber::max_value(),
        FixedNumber::max_value(),
        FixedNumber::max_value(),
        FixedNumber::max_value(),
        FixedNumber::max_value(),
        FixedNumber::max_value(),
        FixedNumber::max_value(),
        FixedNumber::max_value(),
    ]
}

pub fn initial_eth_prices() -> Vec<f64> {
    vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}

fn create_duration(
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
) -> Duration {
    let timestamp = Utc
        .ymd(year, month, day)
        .and_hms(hour, minute, second)
        .timestamp();
    Duration::from_secs(timestamp as u64)
}

pub fn new_test_ext() -> sp_io::TestExternalities {
    let mut t = system::GenesisConfig::default()
        .build_storage::<Test>()
        .unwrap();

    let prev_period_now = create_duration(2020, 9, 14, 11, 25, 0);
    crate::GenesisConfig::<Test> {
        prices: vec![
            (
                Asset::Btc,
                initial_btc_prices()
                    .into_iter()
                    .map(FixedNumber::from_num)
                    .collect(),
                prev_period_now,
            ),
            (
                Asset::Eos,
                initial_eos_prices()
                    .into_iter()
                    .map(FixedNumber::from_num)
                    .collect(),
                prev_period_now,
            ),
            (Asset::Eq, initial_eq_prices(), prev_period_now),
            (
                Asset::Eth,
                initial_eth_prices()
                    .into_iter()
                    .map(FixedNumber::from_num)
                    .collect(),
                prev_period_now,
            ),
        ],
    }
    .assimilate_storage(&mut t)
    .unwrap();

    t.into()
}

pub fn new_test_ext_empty_storage() -> sp_io::TestExternalities {
    system::GenesisConfig::default()
        .build_storage::<Test>()
        .unwrap()
        .into()
}

pub fn new_test_ext_btc_eos_only() -> sp_io::TestExternalities {
    let mut t = system::GenesisConfig::default()
        .build_storage::<Test>()
        .unwrap();

    let prev_period_now = create_duration(2020, 9, 14, 11, 25, 0);
    crate::GenesisConfig::<Test> {
        prices: vec![
            (
                Asset::Btc,
                initial_btc_prices()
                    .into_iter()
                    .map(FixedNumber::from_num)
                    .collect(),
                prev_period_now,
            ),
            (
                Asset::Eos,
                initial_eos_prices()
                    .into_iter()
                    .map(FixedNumber::from_num)
                    .collect(),
                prev_period_now,
            ),
        ],
    }
    .assimilate_storage(&mut t)
    .unwrap();

    t.into()
}
