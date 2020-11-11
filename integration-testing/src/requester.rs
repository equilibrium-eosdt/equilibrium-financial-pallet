use super::TestRuntime;
use crate::key::{AccountKey, DevNonces};
use crate::keyring::{KeyPair, NonceManager};
use crate::runtime::{AccountId, Balance};
use core::marker::PhantomData;
use futures::lock::Mutex;
use pallet_transaction_payment_rpc_runtime_api::RuntimeDispatchInfo;
use sp_keyring::AccountKeyring;
use std::collections::HashMap;
use std::sync::Arc;
use substrate_subxt::ExtrinsicSuccessWithFee;
use substrate_subxt::{sudo::SudoCall, Call, Client, PairSigner};

lazy_static! {
    pub static ref FEES: HashMap<String, u64> = vec![
        (String::from("transfer_and_watch"), 125_000_146),
        (String::from("register_bailsman_and_watch"), 125_000_105),
        (String::from("unregister_bailsman_and_watch"), 125_000_105),
        (String::from("register_whitelist_and_watch"), 125_000_139),
        (String::from("unregister_whitelist_and_watch"), 125_000_139),
        (String::from("set_price_and_watch"), 125_000_114),
    ]
    .into_iter()
    .collect();
}

pub struct Requester {
    pub client: Client<super::TestRuntime>,
    nonces: HashMap<super::runtime::AccountId, u32>,
}

impl Requester {
    pub fn new(client: Client<TestRuntime>) -> Self {
        Requester {
            client,
            nonces: HashMap::new(),
        }
    }

    pub async fn with_sudo<C: Call<TestRuntime>>(
        &mut self,
        call: C,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let alice = AccountKeyring::Alice;
        let mut signer = PairSigner::new(alice.pair()); //hardcode Alice as sudo signer
        let sudo_nonce = self
            .increment_nonce(AccountKeyring::Alice.to_account_id())
            .await?;

        signer.set_nonce(sudo_nonce);
        let encoded = self.client.encode(call)?;
        let sudo = SudoCall {
            call: &encoded,
            _runtime: PhantomData,
        };
        let extrinsic = self.client.watch(sudo, &signer).await?;
        println!("with_sudo extrinsic {:?}", extrinsic);
        Ok(())
    }

    pub async fn increment_nonce(
        &mut self,
        who: AccountId,
    ) -> Result<u32, Box<dyn std::error::Error>> {
        let nonce = self
            .nonces
            .entry(who.clone())
            .and_modify(|a| *a += 1)
            .or_insert(
                self.client
                    .fetch_or_default(
                        &substrate_subxt::system::AccountStore { account_id: &who },
                        None,
                    )
                    .await
                    .unwrap()
                    .nonce,
            );
        println!("new nonce is {}", nonce);
        Ok(*nonce)
    }
}

pub type ChainCallSuccess = ExtrinsicSuccessWithFee<TestRuntime, (AccountKey, Balance)>;
pub type ChainCallResult = Result<Vec<ChainCallSuccess>, Box<dyn std::error::Error>>;

pub async fn call_chain<C: Call<TestRuntime> + Send + Sync>(
    client: &Client<TestRuntime>,
    nonces: Arc<Mutex<DevNonces>>,
    account_key: AccountKey,
    call: C,
) -> Result<Vec<ChainCallSuccess>, Box<dyn std::error::Error>> {
    let nonce = { nonces.lock().await.get_nonce_and_inc(account_key) };
    println!("Current nonce for {:?} is {}", account_key, nonce);

    let mut signer = PairSigner::new(account_key.key_pair());
    signer.set_nonce(nonce);
    let extrinsic = client.create_signed(call, &signer).await?;
    let decoder = TestRuntime::create_decoder(client.metadata().clone());
    let success = client
        .submit_and_watch_extrinsic_with_fee::<Balance>(extrinsic, decoder)
        .await?;
    let fee = success
        .dispatch_info
        .as_ref()
        .map(|x| x.partial_fee)
        .unwrap_or(0);

    println!("incoming fee for {:?}: {}", account_key, fee);

    Ok(vec![ExtrinsicSuccessWithFee {
        block: success.block,
        extrinsic: success.extrinsic,
        events: success.events,
        dispatch_info: success.dispatch_info.map(|x| RuntimeDispatchInfo {
            weight: x.weight,
            class: x.class,
            partial_fee: (account_key, x.partial_fee),
        }),
    }])
}

pub async fn sudo_call_chain<C: Call<TestRuntime> + Send + Sync>(
    client: &Client<TestRuntime>,
    nonces: Arc<Mutex<DevNonces>>,
    call: C,
) -> Result<Vec<ChainCallSuccess>, Box<dyn std::error::Error>> {
    let encoded = client.encode(call)?;
    let sudo_call = SudoCall {
        call: &encoded,
        _runtime: PhantomData,
    };

    call_chain(client, nonces, AccountKeyring::Alice.into(), sudo_call).await
}

pub async fn call_chain_unsigned<C: Call<TestRuntime> + Send + Sync>(
    client: &Client<TestRuntime>,
    call: C,
) -> Result<Vec<ChainCallSuccess>, Box<dyn std::error::Error>> {
    let extrinsic = client.create_unsigned(call)?;
    let decoder = TestRuntime::create_decoder(client.metadata().clone());
    let success = client
        .submit_and_watch_extrinsic(extrinsic, decoder)
        .await?;

    Ok(vec![ExtrinsicSuccessWithFee {
        block: success.block,
        extrinsic: success.extrinsic,
        events: success.events,
        dispatch_info: Option::None,
    }])
}

#[macro_export]
macro_rules! join_chain_calls {
    ( $($tokens:tt)* ) => {{
        let join_result = try_join!($($tokens)*)?;

        let vec_vec = tuple_to_vec!(join_result, $($tokens)*);

        let flatten: Vec<_> = vec_vec.into_iter().flatten().collect();

        flatten
    }};
}
