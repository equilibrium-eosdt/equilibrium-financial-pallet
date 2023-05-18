use crate::financial::RecalcAssetCall;
use crate::financial::{PerAssetMetricsStoreExt, PriceLogsStoreExt, UpdatesStoreExt};
use crate::key::{AccountKey, DevNonces};
use crate::manual_timestamp::AdvanceSecsCall;
use crate::manual_timestamp::NowStoreExt;
use crate::oracle::PricePointsStoreExt;
use crate::oracle::SetPriceCall;
use crate::runtime::FixedNumber;
use crate::test::init_nonce;
use crate::{join_chain_calls, requester::call_chain, TestRuntime};
use approx::assert_abs_diff_eq;
use common::Asset;
use core::time::Duration;
use financial_primitives::capvec::CapVec;
use financial_primitives::PricePeriod;
use futures::{lock::Mutex, try_join};
use sp_keyring::AccountKeyring;
use std::{marker::PhantomData, sync::Arc};
use substrate_fixed::traits::LossyInto;
use substrate_subxt::Client;

fn fixed_to_f64(fixed_actual: FixedNumber) -> f64 {
    let actual: f64 = fixed_actual.lossy_into();
    actual
}

fn assert_cap_vecs_with_epsilon(
    step_name: &str,
    actual_vec: CapVec<FixedNumber>,
    expected_vec: CapVec<FixedNumber>,
) {
    println!("Assert CapVec elements. {}", step_name);
    let actual: Vec<f64> = actual_vec.iter().cloned().map(|x| x.lossy_into()).collect();
    let expected: Vec<f64> = expected_vec
        .iter()
        .cloned()
        .map(|x| x.lossy_into())
        .collect();

    for (a, e) in actual.into_iter().zip(expected.into_iter()) {
        assert_abs_diff_eq!(a, e, epsilon = 1e-8);
    }
}

fn assert_vectors_with_epsilon(
    step_name: &str,
    actual_vec: Vec<FixedNumber>,
    expected_vec: Vec<FixedNumber>,
) {
    println!("Assert vectors elements. {}", step_name);
    let actual: Vec<f64> = actual_vec.into_iter().map(|x| x.lossy_into()).collect();
    let expected: Vec<f64> = expected_vec.into_iter().map(|x| x.lossy_into()).collect();

    for (a, e) in actual.into_iter().zip(expected.into_iter()) {
        assert_abs_diff_eq!(a, e, epsilon = 1e-8);
    }
}

fn duration_to_period(timestamp: u64) -> Duration {
    let price_period: PricePeriod = PricePeriod(1440);
    let now = Duration::from_millis(timestamp);
    let period_start = price_period.get_period_start(now).unwrap();
    period_start
}

fn assert_correlations_with_epsilon(
    step_name: &str,
    actual_vec: Vec<(Asset, FixedNumber)>,
    expected_vec: Vec<(Asset, FixedNumber)>,
) {
    println!("Assert correlations. {}", step_name);
    let actual: Vec<(Asset, f64)> = actual_vec
        .into_iter()
        .map(|x| (x.0, x.1.lossy_into()))
        .collect();
    let expected: Vec<(Asset, f64)> = expected_vec
        .into_iter()
        .map(|x| (x.0, x.1.lossy_into()))
        .collect();

    for (a, e) in actual.into_iter().zip(expected.into_iter()) {
        assert!(a.0.value() == e.0.value());
        assert_abs_diff_eq!(a.1, e.1, epsilon = 1e-8);
    }
}

pub async fn test_recalc_asset(
    client: &Client<TestRuntime>,
    nonces: Arc<Mutex<DevNonces>>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Start recalc asset test");
    let alice_key = AccountKey::from(AccountKeyring::Alice);
    init_nonce(client, nonces.clone(), alice_key).await?;
    let secs_to_move = 86400;
    let initial_timestamp: u64 = client.now(Option::None).await?;

    let first_eos_price = FixedNumber::from_num(3.62);
    let first_btc_price = FixedNumber::from_num(10000.31);
    let first_eth_price = FixedNumber::from_num(250.53);
    let first_eq_price = FixedNumber::from_num(25.3);
    let first_usd_price = FixedNumber::from_num(1.2);

    // ---------------------------------- Setting initial prices ----------------------------------

    println!("Initial price set");
    join_chain_calls!(
        call_chain(
            client,
            nonces.clone(),
            alice_key,
            SetPriceCall {
                asset: Asset::Usd,
                value: first_usd_price,
                _runtime: PhantomData,
            },
        ),
        call_chain(
            client,
            nonces.clone(),
            alice_key,
            SetPriceCall {
                asset: Asset::Eq,
                value: first_eq_price,
                _runtime: PhantomData,
            },
        ),
        call_chain(
            client,
            nonces.clone(),
            alice_key,
            SetPriceCall {
                asset: Asset::Eth,
                value: first_eth_price,
                _runtime: PhantomData,
            },
        ),
        call_chain(
            client,
            nonces.clone(),
            alice_key,
            SetPriceCall {
                asset: Asset::Btc,
                value: first_btc_price,
                _runtime: PhantomData,
            },
        ),
        call_chain(
            client,
            nonces.clone(),
            alice_key,
            SetPriceCall {
                asset: Asset::Eos,
                value: first_eos_price,
                _runtime: PhantomData,
            },
        ),
    );

    // -------------------------------- Asserting chain state --------------------------------

    let mut curr_btc = client.price_points(Asset::Btc, Option::None).await?;
    let mut curr_eos = client.price_points(Asset::Eos, Option::None).await?;
    let mut curr_eth = client.price_points(Asset::Eth, Option::None).await?;
    let mut curr_eq = client.price_points(Asset::Eq, Option::None).await?;
    let mut curr_usd = client.price_points(Asset::Usd, Option::None).await?;

    assert!(curr_btc == first_btc_price);
    assert!(curr_eos == first_eos_price);
    assert!(curr_eth == first_eth_price);
    assert!(curr_eq == first_eq_price);
    assert!(curr_usd == first_usd_price);

    let btc_update = client.updates(Asset::Btc, Option::None).await?.unwrap();
    let eos_update = client.updates(Asset::Eos, Option::None).await?.unwrap();
    let eth_update = client.updates(Asset::Eth, Option::None).await?.unwrap();
    let eq_update = client.updates(Asset::Eq, Option::None).await?.unwrap();
    let usd_update = client.updates(Asset::Usd, Option::None).await?.unwrap();

    assert!(btc_update.price == first_btc_price);
    assert!(eos_update.price == first_eos_price);
    assert!(eth_update.price == first_eth_price);
    assert!(eq_update.price == first_eq_price);
    assert!(usd_update.price == first_usd_price);

    assert!(btc_update.time == Duration::from_millis(initial_timestamp));
    assert!(eos_update.time == Duration::from_millis(initial_timestamp));
    assert!(eth_update.time == Duration::from_millis(initial_timestamp));
    assert!(eq_update.time == Duration::from_millis(initial_timestamp));
    assert!(usd_update.time == Duration::from_millis(initial_timestamp));

    assert!(btc_update.period_start == duration_to_period(initial_timestamp));
    assert!(eos_update.period_start == duration_to_period(initial_timestamp));
    assert!(eth_update.period_start == duration_to_period(initial_timestamp));
    assert!(eq_update.period_start == duration_to_period(initial_timestamp));
    assert!(usd_update.period_start == duration_to_period(initial_timestamp));

    // -------------------------------------- Move time --------------------------------------

    println!("Moving time, initial timestamp: {:?}", initial_timestamp);
    call_chain(
        client,
        nonces.clone(),
        alice_key,
        AdvanceSecsCall {
            _runtime: PhantomData,
            secs: (secs_to_move as u32),
        },
    )
    .await?;

    let actual_timestamp = client.now(Option::None).await?;
    let expected_timestamp = initial_timestamp + 1000 * (secs_to_move as u64);

    assert!(actual_timestamp == expected_timestamp);

    // -------------------------------- Second prices changing --------------------------------

    let second_eos_price = FixedNumber::from_num(4.11);
    let second_btc_price = FixedNumber::from_num(10001.42);
    let second_eth_price = FixedNumber::from_num(251.61);
    let second_eq_price = FixedNumber::from_num(26.4);
    let second_usd_price = FixedNumber::from_num(1.25);

    println!("Setting prices on second iteration");
    join_chain_calls!(
        call_chain(
            client,
            nonces.clone(),
            alice_key,
            SetPriceCall {
                asset: Asset::Usd,
                value: second_usd_price,
                _runtime: PhantomData,
            },
        ),
        call_chain(
            client,
            nonces.clone(),
            alice_key,
            SetPriceCall {
                asset: Asset::Eq,
                value: second_eq_price,
                _runtime: PhantomData,
            },
        ),
        call_chain(
            client,
            nonces.clone(),
            alice_key,
            SetPriceCall {
                asset: Asset::Eth,
                value: second_eth_price,
                _runtime: PhantomData,
            },
        ),
        call_chain(
            client,
            nonces.clone(),
            alice_key,
            SetPriceCall {
                asset: Asset::Btc,
                value: second_btc_price,
                _runtime: PhantomData,
            },
        ),
        call_chain(
            client,
            nonces.clone(),
            alice_key,
            SetPriceCall {
                asset: Asset::Eos,
                value: second_eos_price,
                _runtime: PhantomData,
            },
        ),
    );

    curr_btc = client.price_points(Asset::Btc, Option::None).await?;
    curr_eos = client.price_points(Asset::Eos, Option::None).await?;
    curr_eth = client.price_points(Asset::Eth, Option::None).await?;
    curr_eq = client.price_points(Asset::Eq, Option::None).await?;
    curr_usd = client.price_points(Asset::Usd, Option::None).await?;

    assert!(curr_btc == second_btc_price);
    assert!(curr_eos == second_eos_price);
    assert!(curr_eth == second_eth_price);
    assert!(curr_eq == second_eq_price);
    assert!(curr_usd == second_usd_price);

    // ------------------------------- Recalc financial pallet -------------------------------

    println!("Recalc financial pallet for each asset");
    join_chain_calls!(
        call_chain(
            client,
            nonces.clone(),
            alice_key,
            RecalcAssetCall {
                asset: Asset::Btc,
                _runtime: PhantomData,
            },
        ),
        call_chain(
            client,
            nonces.clone(),
            alice_key,
            RecalcAssetCall {
                asset: Asset::Eos,
                _runtime: PhantomData,
            },
        ),
        call_chain(
            client,
            nonces.clone(),
            alice_key,
            RecalcAssetCall {
                asset: Asset::Eth,
                _runtime: PhantomData,
            },
        ),
        call_chain(
            client,
            nonces.clone(),
            alice_key,
            RecalcAssetCall {
                asset: Asset::Eq,
                _runtime: PhantomData,
            },
        ),
        call_chain(
            client,
            nonces.clone(),
            alice_key,
            RecalcAssetCall {
                asset: Asset::Usd,
                _runtime: PhantomData,
            },
        ),
    );

    let actual_btc_logs = client.price_logs(Asset::Btc, Option::None).await?.unwrap();
    let actual_eos_logs = client.price_logs(Asset::Eos, Option::None).await?.unwrap();
    let actual_eth_logs = client.price_logs(Asset::Eth, Option::None).await?.unwrap();
    let actual_eq_logs = client.price_logs(Asset::Eq, Option::None).await?.unwrap();
    let actual_usd_logs = client.price_logs(Asset::Usd, Option::None).await?.unwrap();

    let actual_btc_metrics = client
        .per_asset_metrics(Asset::Btc, Option::None)
        .await?
        .unwrap();
    let actual_eos_metrics = client
        .per_asset_metrics(Asset::Eos, Option::None)
        .await?
        .unwrap();
    let actual_eth_metrics = client
        .per_asset_metrics(Asset::Eth, Option::None)
        .await?
        .unwrap();
    let actual_eq_metrics = client
        .per_asset_metrics(Asset::Eq, Option::None)
        .await?
        .unwrap();
    let actual_usd_metrics = client
        .per_asset_metrics(Asset::Usd, Option::None)
        .await?
        .unwrap();

    let mut expected_btc_logs = CapVec::<FixedNumber>::new(30);
    expected_btc_logs.push(FixedNumber::from_num(10000)); // from genesis
    expected_btc_logs.push(FixedNumber::from_num(first_btc_price));
    expected_btc_logs.push(FixedNumber::from_num(second_btc_price));

    let mut expected_eos_logs = CapVec::<FixedNumber>::new(30);
    expected_eos_logs.push(FixedNumber::from_num(3)); // from genesis
    expected_eos_logs.push(FixedNumber::from_num(first_eos_price));
    expected_eos_logs.push(FixedNumber::from_num(second_eos_price));

    let mut expected_eth_logs = CapVec::<FixedNumber>::new(30);
    expected_eth_logs.push(FixedNumber::from_num(250)); // from genesis
    expected_eth_logs.push(FixedNumber::from_num(first_eth_price));
    expected_eth_logs.push(FixedNumber::from_num(second_eth_price));

    let mut expected_eq_logs = CapVec::<FixedNumber>::new(30);
    expected_eq_logs.push(FixedNumber::from_num(25)); // from genesis
    expected_eq_logs.push(FixedNumber::from_num(first_eq_price));
    expected_eq_logs.push(FixedNumber::from_num(second_eq_price));

    let mut expected_usd_logs = CapVec::<FixedNumber>::new(30);
    expected_usd_logs.push(FixedNumber::from_num(1)); // from genesis
    expected_usd_logs.push(FixedNumber::from_num(first_usd_price));
    expected_usd_logs.push(FixedNumber::from_num(second_usd_price));

    let expected_btc_log_return = vec![
        FixedNumber::from_num(0.00003099952),
        FixedNumber::from_num(0.000110990399444),
    ];

    let expected_eos_log_return = vec![
        FixedNumber::from_num(0.18786173716957),
        FixedNumber::from_num(0.126949002670464),
    ];

    let expected_eth_log_return = vec![
        FixedNumber::from_num(0.002117755971001),
        FixedNumber::from_num(0.004301595831182),
    ];

    let expected_usd_log_return = vec![
        FixedNumber::from_num(0.182321556793955),
        FixedNumber::from_num(0.040821994520255),
    ];

    let expected_eq_log_return = vec![
        FixedNumber::from_num(0.011928570865274),
        FixedNumber::from_num(0.042559614418796),
    ];

    assert_cap_vecs_with_epsilon("BTC price logs", actual_btc_logs.prices, expected_btc_logs);
    assert_cap_vecs_with_epsilon("EOS price logs", actual_eos_logs.prices, expected_eos_logs);
    assert_cap_vecs_with_epsilon("ETH price logs", actual_eth_logs.prices, expected_eth_logs);
    assert_cap_vecs_with_epsilon("USD price logs", actual_usd_logs.prices, expected_usd_logs);
    assert_cap_vecs_with_epsilon("EQ price logs", actual_eq_logs.prices, expected_eq_logs);

    assert_vectors_with_epsilon(
        "BTC log return",
        actual_btc_metrics.returns,
        expected_btc_log_return.clone(),
    );
    assert_vectors_with_epsilon(
        "EOS log return",
        actual_eos_metrics.returns,
        expected_eos_log_return,
    );
    assert_vectors_with_epsilon(
        "ETH log return",
        actual_eth_metrics.returns,
        expected_eth_log_return,
    );
    assert_vectors_with_epsilon(
        "USD log return",
        actual_usd_metrics.returns,
        expected_usd_log_return,
    );
    assert_vectors_with_epsilon(
        "EQ log return",
        actual_eq_metrics.returns,
        expected_eq_log_return,
    );

    assert_abs_diff_eq!(
        fixed_to_f64(actual_btc_metrics.volatility),
        0.000056562,
        epsilon = 1e-8
    );
    assert_abs_diff_eq!(
        fixed_to_f64(actual_eos_metrics.volatility),
        0.043071808,
        epsilon = 1e-8
    );
    assert_abs_diff_eq!(
        fixed_to_f64(actual_eq_metrics.volatility),
        0.021659419,
        epsilon = 1e-8
    );
    assert_abs_diff_eq!(
        fixed_to_f64(actual_eth_metrics.volatility),
        0.001544208,
        epsilon = 1e-8
    );
    assert_abs_diff_eq!(
        fixed_to_f64(actual_usd_metrics.volatility),
        0.1000553,
        epsilon = 1e-8
    );

    let expected_btc_correlations = vec![
        (Asset::Usd, FixedNumber::from_num(-1)),
        (Asset::Eq, FixedNumber::from_num(1)),
        (Asset::Eth, FixedNumber::from_num(1)),
        (Asset::Btc, FixedNumber::from_num(1)),
        (Asset::Eos, FixedNumber::from_num(-1)),
    ];

    assert_correlations_with_epsilon(
        "BTC correlations",
        actual_btc_metrics.correlations,
        expected_btc_correlations.clone(),
    );

    // ------------------------------------ Test assertion ------------------------------------

    println!("Assertion successful, test passed");

    Ok(())
}
