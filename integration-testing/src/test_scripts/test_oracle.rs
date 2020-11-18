use crate::key::{AccountKey, DevNonces};
use crate::oracle::SetPriceCall;
use crate::runtime::FixedNumber;
use crate::test::init_nonce;
use crate::test_context::TestContext;
use crate::{join_chain_calls, requester::call_chain, TestRuntime};
use common::Asset;
use futures::{lock::Mutex, try_join};
use sp_keyring::AccountKeyring;
use std::{marker::PhantomData, sync::Arc};
use substrate_subxt::Client;

pub async fn test_oracle(
    client: &Client<TestRuntime>,
    nonces: Arc<Mutex<DevNonces>>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Start oracle test");
    let alice_key = AccountKey::from(AccountKeyring::Alice);
    init_nonce(client, nonces.clone(), alice_key).await?;

    let btc_price = FixedNumber::from_num(20_000);
    let eth_price = FixedNumber::from_num(250);
    let eq_price = FixedNumber::from_num(25);
    let eos_price = FixedNumber::from_num(3);

    // ---------------------------------- Setting initial prices ----------------------------------

    println!("Initial price set");
    join_chain_calls!(
        call_chain(
            client,
            nonces.clone(),
            alice_key,
            SetPriceCall {
                asset: Asset::Btc,
                value: btc_price,
                _runtime: PhantomData,
            },
        ),
        call_chain(
            client,
            nonces.clone(),
            alice_key,
            SetPriceCall {
                asset: Asset::Eth,
                value: eth_price,
                _runtime: PhantomData,
            },
        ),
        call_chain(
            client,
            nonces.clone(),
            alice_key,
            SetPriceCall {
                asset: Asset::Eq,
                value: eq_price,
                _runtime: PhantomData,
            },
        ),
        call_chain(
            client,
            nonces.clone(),
            alice_key,
            SetPriceCall {
                asset: Asset::Eos,
                value: eos_price,
                _runtime: PhantomData,
            },
        ),
    );

    // ------------------------------------ Storing chain state ------------------------------------

    let initial_state = TestContext::read_storage(client).await?;
    let curr_btc = initial_state.price_points.get(&Asset::Btc).unwrap();
    let curr_eth = initial_state.price_points.get(&Asset::Eth).unwrap();
    let curr_eq = initial_state.price_points.get(&Asset::Eq).unwrap();
    let curr_eos = initial_state.price_points.get(&Asset::Eos).unwrap();

    assert!(curr_btc == &btc_price);
    assert!(curr_eth == &eth_price);
    assert!(curr_eq == &eq_price);
    assert!(curr_eos == &eos_price);

    // -------------------------------- Iteratively changing prices --------------------------------

    let iterations: i64 = 30;

    for i in 1..=iterations {
        // price_q slowly increases from ~1 to 1.5
        let price_q = FixedNumber::from_num(1)
            + (FixedNumber::from_num(i) / FixedNumber::from_num(iterations * 2));

        println!("Setting prices, iteration #{}", i);
        join_chain_calls!(
            call_chain(
                client,
                nonces.clone(),
                alice_key,
                SetPriceCall {
                    asset: Asset::Btc,
                    value: btc_price * price_q,
                    _runtime: PhantomData,
                },
            ),
            call_chain(
                client,
                nonces.clone(),
                alice_key,
                SetPriceCall {
                    asset: Asset::Eth,
                    value: eth_price / price_q,
                    _runtime: PhantomData,
                },
            ),
            call_chain(
                client,
                nonces.clone(),
                alice_key,
                SetPriceCall {
                    asset: Asset::Eq,
                    value: eq_price * price_q,
                    _runtime: PhantomData,
                },
            ),
            call_chain(
                client,
                nonces.clone(),
                alice_key,
                SetPriceCall {
                    asset: Asset::Eos,
                    value: eos_price / price_q,
                    _runtime: PhantomData,
                },
            ),
        );

        let iteration_state = TestContext::read_storage(client).await?;
        let curr_btc = iteration_state.price_points.get(&Asset::Btc).unwrap();
        let curr_eth = iteration_state.price_points.get(&Asset::Eth).unwrap();
        let curr_eq = iteration_state.price_points.get(&Asset::Eq).unwrap();
        let curr_eos = iteration_state.price_points.get(&Asset::Eos).unwrap();

        assert!(curr_btc == &(btc_price * price_q));
        assert!(curr_eth == &(eth_price / price_q));
        assert!(curr_eq == &(eq_price * price_q));
        assert!(curr_eos == &(eos_price / price_q));
    }

    // ------------------------------------ Test assertion ------------------------------------

    println!("Assertion successful, test passed");

    Ok(())
}
