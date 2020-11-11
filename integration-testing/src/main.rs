#![recursion_limit = "256"]

pub mod financial;
pub mod key;
pub mod keyring;
pub mod manual_timestamp;
pub mod oracle;
pub mod requester;
pub mod runtime;
pub mod test;
pub mod test_context;
pub mod test_scripts;
pub mod timestamp;

use crate::key::{AccountKey, DevNonces, PubKeyStore};
use crate::keyring::PubKey;
use crate::test::init_nonce;
use crate::test_context::register_common_pub_keys;

use futures::lock::Mutex;
use log::{Level, Metadata, Record};
use log::{LevelFilter, SetLoggerError};
use runtime::TestRuntime;
use serde::{Deserialize, Serialize};
use sp_keyring::AccountKeyring;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;
use substrate_subxt::ClientBuilder;
#[allow(unused_imports)]
use test_scripts::test_financial::test_financial;
use test_scripts::test_oracle::test_oracle;

pub const ONE: u64 = 1_000_000_000;

#[macro_use]
extern crate lazy_static;

struct SimpleLogger;
impl log::Log for SimpleLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Info
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            println!("{} - {}", record.level(), record.args());
        }
    }

    fn flush(&self) {}
}

static LOGGER: SimpleLogger = SimpleLogger;

pub fn init() -> Result<(), SetLoggerError> {
    log::set_logger(&LOGGER).map(|()| log::set_max_level(LevelFilter::Info))
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IntegrationTestConfig {
    local: bool,
    sudo: String,
    endpoint: String,
    tests_to_launch: Vec<String>,
    validators: Vec<String>,
}

pub fn read_integration_test_config() -> IntegrationTestConfig {
    let file = File::open("integration_test.json").unwrap();
    let reader = BufReader::new(file);
    let conf: IntegrationTestConfig =
        serde_json::from_reader(reader).expect("JSON was not well-formatted");
    println!("{:?}", conf);
    conf
}

#[async_std::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    //init();
    let config = read_integration_test_config();

    let client = ClientBuilder::<TestRuntime>::new()
        .set_page_size(1)
        .set_url("ws://localhost:9944/")
        .build()
        .await?;
    let nonces = Arc::new(Mutex::new(DevNonces::new()));
    let pub_key_store = Arc::new(Mutex::new(PubKeyStore::new()));
    {
        let mut store = pub_key_store.lock().await;
        register_common_pub_keys(&mut store);
    }

    let alice_acc = AccountKey::from(AccountKeyring::Alice);
    println!("alice {:?}", alice_acc.acc_id());
    init_nonce(&client, nonces.clone(), alice_acc).await?;

    let sudo_acc = AccountKey::from(&config.sudo);
    println!("sudo {:?}", sudo_acc.acc_id());
    init_nonce(&client, nonces.clone(), sudo_acc).await?;

    if config
        .tests_to_launch
        .contains(&"test_financial".to_string())
    {
        println!("\n\nRunning financial test \n\n");
        test_financial(&client, nonces.clone()).await?;
    }

    if config.tests_to_launch.contains(&"test_oracle".to_string()) {
        println!("\n\nRunning oracle test \n\n");
        test_oracle(&client, nonces.clone()).await?;
    }

    Ok(())
}
