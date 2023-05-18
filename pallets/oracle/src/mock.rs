use crate::{Config, Module};
use common::Asset;
use financial_primitives::OnPriceSet;
use frame_support::dispatch::DispatchError;
use frame_support::traits::ConstU32;
use frame_support::{parameter_types, weights::Weight};
use frame_system as system;
use sp_core::H256;
use sp_runtime::{
    testing::Header,
    traits::{BlakeTwo256, IdentityLookup},
};
use core::convert::{TryFrom, TryInto};

use crate as oracle;

type UncheckedExtrinsic = frame_system::mocking::MockUncheckedExtrinsic<Test>;
type Block = frame_system::mocking::MockBlock<Test>;

frame_support::construct_runtime!(
    pub enum Test where
        Block = Block,
        NodeBlock = Block,
        UncheckedExtrinsic = UncheckedExtrinsic,
    {
        System: frame_system::{Pallet, Call, Config, Storage, Event<T>},
        Oracle: oracle::{Pallet, Call, Storage, Event<T>},
    }
);

type AccountId = u64;

parameter_types! {
    pub const BlockHashCount: u64 = 250;
    pub BlockWeights: frame_system::limits::BlockWeights =
        frame_system::limits::BlockWeights::simple_max(Weight::from_ref_time(1024)); 
}
impl frame_system::Config for Test {
    type BaseCallFilter = ();
    type BlockWeights = ();
    type BlockLength = ();
    type DbWeight = ();
    type Origin = Origin;
    type Call = Call;
    type Index = u64;
    type BlockNumber = u64;
    type Hash = H256;
    type Hashing = BlakeTwo256;
    type AccountId = AccountId;
    type Lookup = IdentityLookup<Self::AccountId>;
    type Header = Header;
    type Event = Event;
    type BlockHashCount = BlockHashCount;
    type Version = ();
    type PalletInfo = PalletInfo;
    type AccountData = ();
    type OnNewAccount = ();
    type OnKilledAccount = ();
    type SystemWeightInfo = ();
    type SS58Prefix = ();
    type OnSetCode = ();
    type MaxConsumers = ConstU32<1>;
}

impl Config for Test {
    type Event = Event;
    type Asset = Asset;
    type Price = u32;
    type OnPriceSet = TestOnPriceSet;
}

pub struct TestOnPriceSet;

impl OnPriceSet for TestOnPriceSet {
    type Asset = Asset;
    type Price = u32;
    fn on_price_set(_asset: Asset, _value: u32) -> Result<(), DispatchError> {
        Ok(())
    }
}

pub type OracleModule = Module<Test>;

// Build genesis storage according to the mock runtime.
pub fn new_test_ext() -> sp_io::TestExternalities {
    system::GenesisConfig::default()
        .build_storage::<Test>()
        .unwrap()
        .into()
}
