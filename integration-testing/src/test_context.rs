use super::runtime::AccountId;
use crate::financial::{PerAssetMetricsStoreExt, PriceLogsStoreExt, UpdatesStoreExt};
use crate::key::{AccountKey, DevPubKey, DevPubKeyId, PubKeyStore};
use crate::manual_timestamp::NowStoreExt;
use crate::oracle::PricePointsStoreExt;
use crate::runtime::{FixedNumber, Price, TestRuntime};
use common::Asset;
use pallet_financial::{AssetMetrics, PriceLog, PriceUpdate};
use serde::ser::SerializeTupleVariant;
use serde::{Serialize, Serializer};
use sp_keyring::sr25519::Keyring;
use sp_runtime::{AccountId32, ModuleId};
use std::collections::HashMap;
use std::fmt::Debug;
use std::{cell::RefCell, hash::Hash, str::FromStr};
use substrate_subxt::Client;

// Used for selfless traits implementations
thread_local! {
    static SAVED_TD: RefCell<Option<TestData>> = RefCell::new(Option::None)
}

#[derive(Debug, PartialEq, Eq, Ord, PartialOrd)]
pub enum AccountName {
    Id(DevPubKeyId),
    Unknown(AccountId),
}

impl Serialize for AccountName {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self.clone() {
            AccountName::Id(id) => {
                let mut state = serializer.serialize_tuple_variant("AccountName", 0, "Id", 1)?;
                state.serialize_field(&id)?;
                state.end()
            }
            AccountName::Unknown(u) => {
                let mut state =
                    serializer.serialize_tuple_variant("AccountName", 1, "Unknown", 1)?;
                state.serialize_field(&u)?;
                state.end()
            }
        }
    }
}

pub fn get_known_modules() -> Vec<DevPubKey> {
    vec![]
        .into_iter()
        .map(|&x| DevPubKey::from(ModuleId(x)))
        .collect()
}

pub fn register_common_pub_keys(pub_key_store: &mut PubKeyStore) {
    for k in Keyring::iter() {
        let acc_key = AccountKey::from(k);
        pub_key_store.register(acc_key.into());
    }

    let known_modules: Vec<_> = get_known_modules();

    for m in known_modules {
        pub_key_store.register(m);
    }

    let known_pub_keys: Vec<_> = vec![(
        "Alice//stash",
        "0xbe5ddb1579b72e84524fc29e78609e3caf42e85aa118ebfe0b0ad404b5bdd25f",
    )]
    .into_iter()
    .map(|(name, acc_id_str)| {
        DevPubKey::well_known(name, AccountId32::from_str(acc_id_str).unwrap())
    })
    .collect();

    for k in known_pub_keys {
        pub_key_store.register(k);
    }
}

#[derive(Debug, Clone)]
pub struct TestData {
    pub price_points: HashMap<Asset, FixedNumber>,
    pub updates: HashMap<Asset, Option<PriceUpdate<FixedNumber>>>,
    pub price_logs: HashMap<Asset, Option<PriceLog<FixedNumber>>>,
    pub per_asset_metrics: HashMap<Asset, Option<AssetMetrics<Asset, Price>>>,
    pub now: u64,
}

impl TestData {
    pub fn new() -> Self {
        TestData {
            price_points: HashMap::new(),
            updates: HashMap::new(),
            price_logs: HashMap::new(),
            per_asset_metrics: HashMap::new(),
            now: 0,
        }
    }

    pub fn clear(&mut self) {
        self.price_points.clear();
        self.updates.clear();
        self.price_logs.clear();
        self.per_asset_metrics.clear();
        self.now = 0;
    }

    pub fn to_static(&self) {
        SAVED_TD.with(|td_ref| {
            td_ref.replace(Some(self.clone()));
        });
    }

    pub fn get_static() -> Self {
        SAVED_TD.with(|td_ref| {
            let result = td_ref.borrow().clone();
            match result {
                Some(td) => td,
                None => panic!("Error during attempt to access LTS TestData - no data"),
            }
        })
    }
}

pub const EPS: i64 = 1_0;

pub trait EqWithEps {
    fn eq_with_eps(&self, other: &Self) -> bool;
}

impl EqWithEps for u64 {
    fn eq_with_eps(&self, other: &Self) -> bool {
        (*self as i64 - *other as i64).abs() < EPS
    }
}

impl<K, V> EqWithEps for HashMap<K, V>
where
    K: Eq + Hash,
    V: EqWithEps,
{
    fn eq_with_eps(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self
                .iter()
                .all(|(k, v)| (*other).get(k).map_or(false, |v2| v.eq_with_eps(v2)))
    }
}

pub struct TestContext {}

impl TestContext {
    pub async fn read_storage(
        client: &Client<TestRuntime>,
    ) -> Result<TestData, Box<dyn std::error::Error>> {
        let mut data = TestData::new();

        for asset in Asset::iterator() {
            let price_points = client.price_points(*asset, Option::None).await;
            data.price_points.insert(*asset, price_points.unwrap());

            let updates = client.updates(*asset, Option::None).await;
            data.updates.insert(*asset, updates.unwrap());

            let price_logs = client.price_logs(*asset, Option::None).await;
            data.price_logs.insert(*asset, price_logs.unwrap());

            let metrics = client.per_asset_metrics(*asset, Option::None).await;
            data.per_asset_metrics.insert(*asset, metrics.unwrap());
        }

        data.now = client.now(Option::None).await.unwrap();

        Ok(data)
    }
}
