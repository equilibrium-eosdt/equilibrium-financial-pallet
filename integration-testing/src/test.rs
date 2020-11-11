use crate::key::{AccountKey, DevNonces};
use crate::keyring::{NonceManager, PubKey};
use crate::runtime::TestRuntime;
use futures::lock::Mutex;
use sp_arithmetic::FixedI64;
use std::sync::Arc;
use substrate_subxt::Client;

pub fn i128_to_u64(n: i128) -> Option<u64> {
    if n < 0 || n > u64::MAX as i128 {
        return None;
    }
    Some(n as u64)
}

pub fn i128_to_fixedi64(n: i128) -> Option<FixedI64> {
    if n < i64::MIN as i128 || n > i64::MAX as i128 {
        return None;
    }

    Some(FixedI64::from_inner(n as i64))
}

pub async fn init_nonce(
    client: &Client<TestRuntime>,
    nonces: Arc<Mutex<DevNonces>>,
    account_key: AccountKey,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut nonces = nonces.lock().await;

    if nonces.is_initialized(account_key) {
        Ok(())
    } else {
        let id = account_key.acc_id();
        let initial_nonce = client
            .fetch_or_default(
                &substrate_subxt::system::AccountStore { account_id: &id },
                None,
            )
            .await
            .unwrap()
            .nonce;

        nonces.init_nonce(account_key, initial_nonce);

        println!("Initial nonce for {:?} is {}", account_key, initial_nonce);

        Ok(())
    }
}
