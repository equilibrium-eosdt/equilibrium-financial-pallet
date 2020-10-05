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
use sp_core::H256;
use frame_support::{impl_outer_origin, parameter_types, weights::Weight};
use frame_support::traits::UnixTime;
use sp_runtime::{
	traits::{BlakeTwo256, IdentityLookup}, testing::Header, Perbill,
};
use frame_system as system;
use core::time::Duration;
use substrate_fixed::types::I64F64;
use std::cell::RefCell;
use crate::common::Asset;

impl_outer_origin! {
	pub enum Origin for Test {}
}

thread_local! {
	pub static NOW: RefCell<Duration> = RefCell::new(Duration::from_secs(0));
}

pub fn set_now(now: Duration) {
	NOW.with(|n| *n.borrow_mut() = now);
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
	type AccountId = u64;
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
	type UnixTime = TestUnixTime;
	type FixedNumberBits = i128;
	type FixedNumber = FixedNumber;
	type Price = FixedNumber;
}

pub type FinancialModule = Module<Test>;

pub fn new_test_ext() -> sp_io::TestExternalities {
	let mut t = system::GenesisConfig::default().build_storage::<Test>().unwrap();

	crate::GenesisConfig::<Test> {
		prices: vec![
			(Asset::Btc, vec![
				7_117.21,
				7_429.72,
				7_550.90,
				7_569.94,
				7_679.87,
				7_795.60,
				7_807.06,
				8_801.04,
				8_658.55,
				8_864.77,
			].into_iter().map(FixedNumber::from_num).collect()),
			(Asset::Eos, vec![
				2.62,
				2.67,
				2.72,
				2.72,
				2.74,
				2.75,
				2.78,
				3.02,
				2.83,
				2.89,
			].into_iter().map(FixedNumber::from_num).collect()),
		],
	}
		.assimilate_storage(&mut t)
		.unwrap();

	t.into()
}
