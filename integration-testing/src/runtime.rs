use super::{financial, manual_timestamp, oracle, portfolio, timestamp};
use crate::financial::FinancialEventsDecoder;
use crate::manual_timestamp::ManualTimestampEventsDecoder;
use crate::oracle::OracleEventsDecoder;
use crate::portfolio::PortfolioEventsDecoder;
use crate::timestamp::TimestampEventsDecoder;
use common::Asset;
use sp_runtime::traits::{BlakeTwo256, IdentifyAccount, Verify};
use sp_runtime::{generic, MultiSignature, OpaqueExtrinsic};
use substrate_fixed::types::I64F64;
use substrate_subxt::{
    balances::{AccountData, Balances, BalancesEventsDecoder}, // needed for sign extra
    extrinsic::DefaultExtra,
    sudo::Sudo,
    system::{System, SystemEventsDecoder},
    EventsDecoder,
    Metadata,
    Runtime,
};

#[derive(Clone, PartialEq, Debug)]
pub struct TestRuntime;

impl TestRuntime {
    pub fn create_decoder(metadata: Metadata) -> EventsDecoder<TestRuntime> {
        let mut decoder = EventsDecoder::<TestRuntime>::new(metadata);
        decoder.with_system();
        decoder.with_balances();
        decoder.with_oracle();
        decoder.with_timestamp();
        decoder.with_financial();
        decoder.with_manual_timestamp();
        decoder.with_portfolio();

        decoder.register_type_size::<u64>("u64");

        decoder
    }
}

impl Eq for TestRuntime {}

type Signature = MultiSignature;

impl Runtime for TestRuntime {
    type Signature = Signature;
    type Extra = DefaultExtra<Self>;
}

pub type AccountId = <<MultiSignature as Verify>::Signer as IdentifyAccount>::AccountId;

pub type Address = AccountId;

pub type FixedNumber = I64F64;
pub type Price = I64F64;

impl System for TestRuntime {
    type Index = u32;
    type BlockNumber = u32;
    type Hash = sp_core::H256;
    type Hashing = BlakeTwo256;
    type AccountId = AccountId;
    type Address = Self::AccountId;
    type Header = generic::Header<Self::BlockNumber, BlakeTwo256>;
    type Extrinsic = OpaqueExtrinsic;
    type AccountData = AccountData<<Self as Balances>::Balance>;
}

pub type Balance = u128;

impl Balances for TestRuntime {
    type Balance = Balance;
}

impl timestamp::Timestamp for TestRuntime {
    type Moment = u64;
}

impl Sudo for TestRuntime {}

impl oracle::Oracle for TestRuntime {
    type Asset = Asset;
    type Price = FixedNumber;
}

impl financial::Financial for TestRuntime {
    type Asset = Asset;
    type FixedNumber = FixedNumber;
    type Price = FixedNumber;
}

impl portfolio::Portfolio for TestRuntime {
    type Asset = Asset;
    type FixedNumber = FixedNumber;
    type Balance = FixedNumber;
}

impl manual_timestamp::ManualTimestamp for TestRuntime {}
