use crate::financial::{MetricsStoreExt, RecalcCall, UpdatesStoreExt};
use crate::key::{AccountKey, DevNonces};
use crate::manual_timestamp::{AdvanceSecsCall, NowStoreExt};
use crate::oracle::{PricePointsStoreExt, SetPriceCall};
use crate::runtime::FixedNumber;
use crate::test::init_nonce;
use crate::{join_chain_calls, requester::call_chain, TestRuntime};
use approx::assert_abs_diff_eq;
use common::Asset;
use core::time::Duration;
use futures::{lock::Mutex, try_join};
use integration_testing_macro::tuple_to_vec;
use itertools::izip;
use pallet_financial::FinancialMetrics;
use sp_keyring::AccountKeyring;
use std::{marker::PhantomData, sync::Arc};
use substrate_fixed::traits::LossyInto;
use substrate_subxt::Client;

pub async fn test_financial(
    client: &Client<TestRuntime>,
    nonces: Arc<Mutex<DevNonces>>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Start recalc asset test");
    let alice_key = AccountKey::from(AccountKeyring::Alice);
    init_nonce(client, nonces.clone(), alice_key).await?;
    let secs_to_move = 86400;

    // ------------------------------------ Setting prices ------------------------------------

    set_prices_and_move_time(client, alice_key, nonces.clone(), secs_to_move).await?;

    // ------------------------------- Recalc financial pallet -------------------------------

    println!("Recalc financial pallet for every asset");
    call_chain(
        client,
        nonces.clone(),
        alice_key,
        RecalcCall {
            _runtime: PhantomData,
        },
    )
    .await?;

    // ---------------------------------- Assert chain state ----------------------------------

    assert_first_recalc_chain_state(client).await?;

    // -------------------- Setting prices 1 day after month prices setted --------------------

    set_prices_after_month(client, alice_key, nonces.clone(), secs_to_move).await?;

    // ------------------------------- Recalc financial pallet -------------------------------

    println!("Recalc financial pallet for every asset");
    call_chain(
        client,
        nonces.clone(),
        alice_key,
        RecalcCall {
            _runtime: PhantomData,
        },
    )
    .await?;

    // ---------------------------------- Assert chain state ----------------------------------

    assert_second_recalc_chain_state(client).await?;

    // ------------------------------------ Test assertion ------------------------------------

    println!("Assertion successful, test passed");

    Ok(())
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

fn assert_metrics_assets(metrics: &FinancialMetrics<Asset, FixedNumber>) {
    for (actual, expect) in metrics.clone().assets.into_iter().zip(Asset::iterator()) {
        assert!(actual.value() == expect.value());
    }
}

async fn assert_second_recalc_chain_state(
    client: &Client<TestRuntime>,
) -> Result<(), Box<dyn std::error::Error>> {
    let metrics = client.metrics(Option::None).await?.unwrap();

    assert_metrics_assets(&metrics);

    let expected_volatilities: Vec<FixedNumber> = vec![
        FixedNumber::from_num(0.0),               // usd
        FixedNumber::from_num(0.041212132650566), // eq
        FixedNumber::from_num(0.026979767511806), // eth
        FixedNumber::from_num(0.019530767671752), // btc
        FixedNumber::from_num(0.022172261805909), // eos
    ];

    assert_vectors_with_epsilon(
        "Assert assets volatilities",
        metrics.volatilities,
        expected_volatilities,
    );

    let expected_correlations: Vec<FixedNumber> = vec![
        FixedNumber::from_num(1),
        FixedNumber::from_num(0),
        FixedNumber::from_num(0),
        FixedNumber::from_num(0),
        FixedNumber::from_num(0),
        FixedNumber::from_num(0),
        FixedNumber::from_num(1),
        FixedNumber::from_num(0.58314744717848),
        FixedNumber::from_num(0.5055289438838),
        FixedNumber::from_num(-0.14858602369776),
        FixedNumber::from_num(0),
        FixedNumber::from_num(0.58314744717848),
        FixedNumber::from_num(1),
        FixedNumber::from_num(0.72688913085808),
        FixedNumber::from_num(0.33944301098853),
        FixedNumber::from_num(0),
        FixedNumber::from_num(0.5055289438838),
        FixedNumber::from_num(0.72688913085808),
        FixedNumber::from_num(1),
        FixedNumber::from_num(0.18143650586153),
        FixedNumber::from_num(0),
        FixedNumber::from_num(-0.14858602369776),
        FixedNumber::from_num(0.33944301098853),
        FixedNumber::from_num(0.18143650586153),
        FixedNumber::from_num(1),
    ];

    let expected_covariances: Vec<FixedNumber> = vec![
        FixedNumber::from_num(0),
        FixedNumber::from_num(0),
        FixedNumber::from_num(0),
        FixedNumber::from_num(0),
        FixedNumber::from_num(0),
        FixedNumber::from_num(0),
        FixedNumber::from_num(0.001698439877608),
        FixedNumber::from_num(0.00064839800627),
        FixedNumber::from_num(0.00040690256633),
        FixedNumber::from_num(-0.00013577288546),
        FixedNumber::from_num(0),
        FixedNumber::from_num(0.00064839800627),
        FixedNumber::from_num(0.000727907854991),
        FixedNumber::from_num(0.0003830237393),
        FixedNumber::from_num(0.00020305564717),
        FixedNumber::from_num(0),
        FixedNumber::from_num(0.00040690256633),
        FixedNumber::from_num(0.0003830237393),
        FixedNumber::from_num(0.000381450885848),
        FixedNumber::from_num(0.00007856949929),
        FixedNumber::from_num(0),
        FixedNumber::from_num(-0.00013577288546),
        FixedNumber::from_num(0.00020305564717),
        FixedNumber::from_num(0.00007856949929),
        FixedNumber::from_num(0.00049160919359),
    ];

    let actual_correlations: Vec<FixedNumber> = metrics.correlations;
    let actual_covariances: Vec<FixedNumber> = metrics.covariances;

    assert_vectors_with_epsilon(
        "Assert correlations matrix",
        actual_correlations,
        expected_correlations,
    );

    assert_vectors_with_epsilon(
        "Assert covariances matrix",
        actual_covariances,
        expected_covariances,
    );

    // let actual_btc_metrics = client
    //     .per_asset_metrics(Asset::Btc, Option::None)
    //     .await?
    //     .unwrap();
    // let actual_eos_metrics = client
    //     .per_asset_metrics(Asset::Eos, Option::None)
    //     .await?
    //     .unwrap();
    // let actual_eth_metrics = client
    //     .per_asset_metrics(Asset::Eth, Option::None)
    //     .await?
    //     .unwrap();
    // let actual_eq_metrics = client
    //     .per_asset_metrics(Asset::Eq, Option::None)
    //     .await?
    //     .unwrap();
    // let actual_usd_metrics = client
    //     .per_asset_metrics(Asset::Usd, Option::None)
    //     .await?
    //     .unwrap();

    // let expected_btc_log_return = vec![
    //     FixedNumber::from_num(-0.017851775539992),
    //     FixedNumber::from_num(0.006485899505049),
    //     FixedNumber::from_num(0.023702618762709),
    //     FixedNumber::from_num(0.011824769398687),
    //     FixedNumber::from_num(0.021735125114429),
    //     FixedNumber::from_num(0.006526171853234),
    //     FixedNumber::from_num(0.014313388019069),
    //     FixedNumber::from_num(-0.009840284915377),
    //     FixedNumber::from_num(0.000175008750884),
    //     FixedNumber::from_num(0.006888460773841),
    //     FixedNumber::from_num(-0.015941477997688),
    //     FixedNumber::from_num(0.003701423103186),
    //     FixedNumber::from_num(0.012674440896728),
    //     FixedNumber::from_num(0.020971976194077),
    //     FixedNumber::from_num(0.013936624912131),
    //     FixedNumber::from_num(0.072074847729704),
    //     FixedNumber::from_num(0.013719550155069),
    //     FixedNumber::from_num(-0.004088407746743),
    //     FixedNumber::from_num(0.014503573115766),
    //     FixedNumber::from_num(-0.006496755376766),
    //     FixedNumber::from_num(0.002144772940134),
    //     FixedNumber::from_num(0.043203424778333),
    //     FixedNumber::from_num(-0.026735589150183),
    //     FixedNumber::from_num(0.013160052345472),
    //     FixedNumber::from_num(0.007695760935366),
    //     FixedNumber::from_num(0.017319297317554),
    //     FixedNumber::from_num(-0.002974789061153),
    //     FixedNumber::from_num(-0.013976007681055),
    //     FixedNumber::from_num(0.0328349845197),
    // ];

    // let expected_eos_log_return = vec![
    //     FixedNumber::from_num(0.063945225362233),
    //     FixedNumber::from_num(-0.018342269805096),
    //     FixedNumber::from_num(-0.014076701910362),
    //     FixedNumber::from_num(0.013698844358162),
    //     FixedNumber::from_num(-0.00151285959263),
    //     FixedNumber::from_num(0.006789916693474),
    //     FixedNumber::from_num(-0.004899199548643),
    //     FixedNumber::from_num(-0.002648006086739),
    //     FixedNumber::from_num(-0.011045627441233),
    //     FixedNumber::from_num(-0.004606534057638),
    //     FixedNumber::from_num(-0.023750386367157),
    //     FixedNumber::from_num(-0.004739345363897),
    //     FixedNumber::from_num(0.008671713781618),
    //     FixedNumber::from_num(0.014029848210439),
    //     FixedNumber::from_num(-0.021514618232527),
    //     FixedNumber::from_num(0.032292243726357),
    //     FixedNumber::from_num(0.022336722713639),
    //     FixedNumber::from_num(-0.01091672802769),
    //     FixedNumber::from_num(0.007917100731662),
    //     FixedNumber::from_num(0.017126256271163),
    //     FixedNumber::from_num(-0.028455682588786),
    //     FixedNumber::from_num(0.014703375160227),
    //     FixedNumber::from_num(-0.006382600980497),
    //     FixedNumber::from_num(-0.006423600398779),
    //     FixedNumber::from_num(-0.049738301151431),
    //     FixedNumber::from_num(0.006750075262365),
    //     FixedNumber::from_num(-0.003567892783902),
    //     FixedNumber::from_num(-0.033518179087602),
    //     FixedNumber::from_num(-0.027901039512794),
    // ];

    // let expected_eth_log_return = vec![
    //     FixedNumber::from_num(-0.037176219300342),
    //     FixedNumber::from_num(0.003105046726959),
    //     FixedNumber::from_num(0.026808317225208),
    //     FixedNumber::from_num(0.039664149077215),
    //     FixedNumber::from_num(0.014858549313211),
    //     FixedNumber::from_num(0.009366053036486),
    //     FixedNumber::from_num(0.032947494656537),
    //     FixedNumber::from_num(-0.014317195829359),
    //     FixedNumber::from_num(-0.006048511576352),
    //     FixedNumber::from_num(-0.003726170720288),
    //     FixedNumber::from_num(-0.032615096229953),
    //     FixedNumber::from_num(0.008362550947202),
    //     FixedNumber::from_num(0.026209855400584),
    //     FixedNumber::from_num(0.002559874723751),
    //     FixedNumber::from_num(-0.02852562591775),
    //     FixedNumber::from_num(0.059791550432133),
    //     FixedNumber::from_num(0.057189635500972),
    //     FixedNumber::from_num(-0.011965145719892),
    //     FixedNumber::from_num(0.007249581694894),
    //     FixedNumber::from_num(-0.015117944327008),
    //     FixedNumber::from_num(-0.033149995299452),
    //     FixedNumber::from_num(0.02668163550439),
    //     FixedNumber::from_num(-0.037783197312849),
    //     FixedNumber::from_num(-0.003168839842996),
    //     FixedNumber::from_num(-0.012410908361374),
    //     FixedNumber::from_num(0.010422040197932),
    //     FixedNumber::from_num(0.025476158297524),
    //     FixedNumber::from_num(-0.033679947915332),
    //     FixedNumber::from_num(0.011842738081395),
    // ];

    // let expected_usd_log_return = vec![
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    // ];

    // let expected_eq_log_return = vec![
    //     FixedNumber::from_num(-0.104400822685896),
    //     FixedNumber::from_num(0.027852562402598),
    //     FixedNumber::from_num(0.065930399912786),
    //     FixedNumber::from_num(0.033634655314806),
    //     FixedNumber::from_num(-0.003995774502009),
    //     FixedNumber::from_num(0.00070629785161),
    //     FixedNumber::from_num(0.014254242982982),
    //     FixedNumber::from_num(-0.012608152076313),
    //     FixedNumber::from_num(-0.01395976768789),
    //     FixedNumber::from_num(-0.02411498578672),
    //     FixedNumber::from_num(-0.04569512959085),
    //     FixedNumber::from_num(0.013954304827044),
    //     FixedNumber::from_num(0.035152134114813),
    //     FixedNumber::from_num(-0.030125734992472),
    //     FixedNumber::from_num(-0.018980703949385),
    //     FixedNumber::from_num(0.05686029412138),
    //     FixedNumber::from_num(0.030193664145268),
    //     FixedNumber::from_num(-0.000937207191374),
    //     FixedNumber::from_num(0.021795437311784),
    //     FixedNumber::from_num(-0.006442728222199),
    //     FixedNumber::from_num(0.083021433280261),
    //     FixedNumber::from_num(-0.00575265248945),
    //     FixedNumber::from_num(-0.069221835426132),
    //     FixedNumber::from_num(-0.073632490279114),
    //     FixedNumber::from_num(0.00712271217112),
    //     FixedNumber::from_num(0.023462196064631),
    //     FixedNumber::from_num(0.009280256966068),
    //     FixedNumber::from_num(-0.044807107555325),
    //     FixedNumber::from_num(-0.001983635661594),
    // ];

    // assert_vectors_with_epsilon(
    //     "BTC log return",
    //     actual_btc_metrics.returns,
    //     expected_btc_log_return,
    // );
    // assert_vectors_with_epsilon(
    //     "EOS log return",
    //     actual_eos_metrics.returns,
    //     expected_eos_log_return,
    // );
    // assert_vectors_with_epsilon(
    //     "ETH log return",
    //     actual_eth_metrics.returns,
    //     expected_eth_log_return,
    // );
    // assert_vectors_with_epsilon(
    //     "USD log return",
    //     actual_usd_metrics.returns,
    //     expected_usd_log_return,
    // );
    // assert_vectors_with_epsilon(
    //     "EQ log return",
    //     actual_eq_metrics.returns,
    //     expected_eq_log_return,
    // );

    Ok(())
}

async fn assert_first_recalc_chain_state(
    client: &Client<TestRuntime>,
) -> Result<(), Box<dyn std::error::Error>> {
    let metrics = client.metrics(Option::None).await?.unwrap();

    assert_metrics_assets(&metrics);

    let expected_volatilities: Vec<FixedNumber> = vec![
        FixedNumber::from_num(0.0),              // usd
        FixedNumber::from_num(0.04121390630829), // eq
        FixedNumber::from_num(0.02692861833717), // eth
        FixedNumber::from_num(0.01899396779000), // btc
        FixedNumber::from_num(0.02163976358360), // eos
    ];

    assert_vectors_with_epsilon(
        "Assert assets volatilities",
        metrics.volatilities,
        expected_volatilities,
    );

    let expected_correlations: Vec<FixedNumber> = vec![
        FixedNumber::from_num(1),
        FixedNumber::from_num(0),
        FixedNumber::from_num(0),
        FixedNumber::from_num(0),
        FixedNumber::from_num(0),
        FixedNumber::from_num(0),
        FixedNumber::from_num(1),
        FixedNumber::from_num(0.58444399870037),
        FixedNumber::from_num(0.52044196493031),
        FixedNumber::from_num(-0.15357736293063),
        FixedNumber::from_num(0),
        FixedNumber::from_num(0.58444399870037),
        FixedNumber::from_num(1),
        FixedNumber::from_num(0.73404325427508),
        FixedNumber::from_num(0.36263027397808),
        FixedNumber::from_num(0),
        FixedNumber::from_num(0.52044196493031),
        FixedNumber::from_num(0.73404325427508),
        FixedNumber::from_num(1),
        FixedNumber::from_num(0.24738115822628),
        FixedNumber::from_num(0),
        FixedNumber::from_num(-0.15357736293063),
        FixedNumber::from_num(0.36263027397808),
        FixedNumber::from_num(0.24738115822628),
        FixedNumber::from_num(1),
    ];

    let expected_covariances: Vec<FixedNumber> = vec![
        FixedNumber::from_num(0),
        FixedNumber::from_num(0),
        FixedNumber::from_num(0),
        FixedNumber::from_num(0),
        FixedNumber::from_num(0),
        FixedNumber::from_num(0),
        FixedNumber::from_num(0.001698586073),
        FixedNumber::from_num(0.0006486355597),
        FixedNumber::from_num(0.00040741009369),
        FixedNumber::from_num(-0.00013696938233),
        FixedNumber::from_num(0),
        FixedNumber::from_num(0.0006486355597),
        FixedNumber::from_num(0.0007251504855),
        FixedNumber::from_num(0.0003754494048),
        FixedNumber::from_num(0.00021131515315),
        FixedNumber::from_num(0),
        FixedNumber::from_num(0.00040741009369),
        FixedNumber::from_num(0.0003754494048),
        FixedNumber::from_num(0.00036077081241),
        FixedNumber::from_num(0.00010167983375),
        FixedNumber::from_num(0),
        FixedNumber::from_num(-0.00013696938233),
        FixedNumber::from_num(0.00021131515315),
        FixedNumber::from_num(0.00010167983375),
        FixedNumber::from_num(0.000468279368),
    ];

    let actual_correlations: Vec<FixedNumber> = metrics.correlations;
    let actual_covariances: Vec<FixedNumber> = metrics.covariances;

    assert_vectors_with_epsilon(
        "Assert correlations matrix",
        actual_correlations,
        expected_correlations,
    );

    assert_vectors_with_epsilon(
        "Assert covariances matrix",
        actual_covariances,
        expected_covariances,
    );

    // let actual_btc_metrics = client
    //     .per_asset_metrics(Asset::Btc, Option::None)
    //     .await?
    //     .unwrap();
    // let actual_eos_metrics = client
    //     .per_asset_metrics(Asset::Eos, Option::None)
    //     .await?
    //     .unwrap();
    // let actual_eth_metrics = client
    //     .per_asset_metrics(Asset::Eth, Option::None)
    //     .await?
    //     .unwrap();
    // let actual_eq_metrics = client
    //     .per_asset_metrics(Asset::Eq, Option::None)
    //     .await?
    //     .unwrap();
    // let actual_usd_metrics = client
    //     .per_asset_metrics(Asset::Usd, Option::None)
    //     .await?
    //     .unwrap();

    // let expected_btc_log_return = vec![
    //     FixedNumber::from_num(0.011084832424493),
    //     FixedNumber::from_num(-0.017851775539992),
    //     FixedNumber::from_num(0.006485899505049),
    //     FixedNumber::from_num(0.023702618762709),
    //     FixedNumber::from_num(0.011824769398687),
    //     FixedNumber::from_num(0.021735125114429),
    //     FixedNumber::from_num(0.006526171853234),
    //     FixedNumber::from_num(0.014313388019069),
    //     FixedNumber::from_num(-0.009840284915377),
    //     FixedNumber::from_num(0.000175008750884),
    //     FixedNumber::from_num(0.006888460773841),
    //     FixedNumber::from_num(-0.015941477997688),
    //     FixedNumber::from_num(0.003701423103186),
    //     FixedNumber::from_num(0.012674440896728),
    //     FixedNumber::from_num(0.020971976194077),
    //     FixedNumber::from_num(0.013936624912131),
    //     FixedNumber::from_num(0.072074847729704),
    //     FixedNumber::from_num(0.013719550155069),
    //     FixedNumber::from_num(-0.004088407746743),
    //     FixedNumber::from_num(0.014503573115766),
    //     FixedNumber::from_num(-0.006496755376766),
    //     FixedNumber::from_num(0.002144772940134),
    //     FixedNumber::from_num(0.043203424778333),
    //     FixedNumber::from_num(-0.026735589150183),
    //     FixedNumber::from_num(0.013160052345472),
    //     FixedNumber::from_num(0.007695760935366),
    //     FixedNumber::from_num(0.017319297317554),
    //     FixedNumber::from_num(-0.002974789061153),
    //     FixedNumber::from_num(-0.013976007681055),
    // ];

    // let expected_eos_log_return = vec![
    //     FixedNumber::from_num(0.003961970317355),
    //     FixedNumber::from_num(0.063945225362233),
    //     FixedNumber::from_num(-0.018342269805096),
    //     FixedNumber::from_num(-0.014076701910362),
    //     FixedNumber::from_num(0.013698844358162),
    //     FixedNumber::from_num(-0.00151285959263),
    //     FixedNumber::from_num(0.006789916693474),
    //     FixedNumber::from_num(-0.004899199548643),
    //     FixedNumber::from_num(-0.002648006086739),
    //     FixedNumber::from_num(-0.011045627441233),
    //     FixedNumber::from_num(-0.004606534057638),
    //     FixedNumber::from_num(-0.023750386367157),
    //     FixedNumber::from_num(-0.004739345363897),
    //     FixedNumber::from_num(0.008671713781618),
    //     FixedNumber::from_num(0.014029848210439),
    //     FixedNumber::from_num(-0.021514618232527),
    //     FixedNumber::from_num(0.032292243726357),
    //     FixedNumber::from_num(0.022336722713639),
    //     FixedNumber::from_num(-0.01091672802769),
    //     FixedNumber::from_num(0.007917100731662),
    //     FixedNumber::from_num(0.017126256271163),
    //     FixedNumber::from_num(-0.028455682588786),
    //     FixedNumber::from_num(0.014703375160227),
    //     FixedNumber::from_num(-0.006382600980497),
    //     FixedNumber::from_num(-0.006423600398779),
    //     FixedNumber::from_num(-0.049738301151431),
    //     FixedNumber::from_num(0.006750075262365),
    //     FixedNumber::from_num(-0.003567892783902),
    //     FixedNumber::from_num(-0.033518179087602),
    // ];

    // let expected_eth_log_return = vec![
    //     FixedNumber::from_num(0.003256086798801),
    //     FixedNumber::from_num(-0.037176219300342),
    //     FixedNumber::from_num(0.003105046726959),
    //     FixedNumber::from_num(0.026808317225208),
    //     FixedNumber::from_num(0.039664149077215),
    //     FixedNumber::from_num(0.014858549313211),
    //     FixedNumber::from_num(0.009366053036486),
    //     FixedNumber::from_num(0.032947494656537),
    //     FixedNumber::from_num(-0.014317195829359),
    //     FixedNumber::from_num(-0.006048511576352),
    //     FixedNumber::from_num(-0.003726170720288),
    //     FixedNumber::from_num(-0.032615096229953),
    //     FixedNumber::from_num(0.008362550947202),
    //     FixedNumber::from_num(0.026209855400584),
    //     FixedNumber::from_num(0.002559874723751),
    //     FixedNumber::from_num(-0.02852562591775),
    //     FixedNumber::from_num(0.059791550432133),
    //     FixedNumber::from_num(0.057189635500972),
    //     FixedNumber::from_num(-0.011965145719892),
    //     FixedNumber::from_num(0.007249581694894),
    //     FixedNumber::from_num(-0.015117944327008),
    //     FixedNumber::from_num(-0.033149995299452),
    //     FixedNumber::from_num(0.02668163550439),
    //     FixedNumber::from_num(-0.037783197312849),
    //     FixedNumber::from_num(-0.003168839842996),
    //     FixedNumber::from_num(-0.012410908361374),
    //     FixedNumber::from_num(0.010422040197932),
    //     FixedNumber::from_num(0.025476158297524),
    //     FixedNumber::from_num(-0.033679947915332),
    // ];

    // let expected_usd_log_return = vec![
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    //     FixedNumber::from_num(0),
    // ];

    // let expected_eq_log_return = vec![
    //     FixedNumber::from_num(-0.003354903697885),
    //     FixedNumber::from_num(-0.104400822685896),
    //     FixedNumber::from_num(0.027852562402598),
    //     FixedNumber::from_num(0.065930399912786),
    //     FixedNumber::from_num(0.033634655314806),
    //     FixedNumber::from_num(-0.003995774502009),
    //     FixedNumber::from_num(0.00070629785161),
    //     FixedNumber::from_num(0.014254242982982),
    //     FixedNumber::from_num(-0.012608152076313),
    //     FixedNumber::from_num(-0.01395976768789),
    //     FixedNumber::from_num(-0.02411498578672),
    //     FixedNumber::from_num(-0.04569512959085),
    //     FixedNumber::from_num(0.013954304827044),
    //     FixedNumber::from_num(0.035152134114813),
    //     FixedNumber::from_num(-0.030125734992472),
    //     FixedNumber::from_num(-0.018980703949385),
    //     FixedNumber::from_num(0.05686029412138),
    //     FixedNumber::from_num(0.030193664145268),
    //     FixedNumber::from_num(-0.000937207191374),
    //     FixedNumber::from_num(0.021795437311784),
    //     FixedNumber::from_num(-0.006442728222199),
    //     FixedNumber::from_num(0.083021433280261),
    //     FixedNumber::from_num(-0.00575265248945),
    //     FixedNumber::from_num(-0.069221835426132),
    //     FixedNumber::from_num(-0.073632490279114),
    //     FixedNumber::from_num(0.00712271217112),
    //     FixedNumber::from_num(0.023462196064631),
    //     FixedNumber::from_num(0.009280256966068),
    //     FixedNumber::from_num(-0.044807107555325),
    // ];

    // assert_vectors_with_epsilon(
    //     "BTC log return",
    //     actual_btc_metrics.returns,
    //     expected_btc_log_return,
    // );
    // assert_vectors_with_epsilon(
    //     "EOS log return",
    //     actual_eos_metrics.returns,
    //     expected_eos_log_return,
    // );
    // assert_vectors_with_epsilon(
    //     "ETH log return",
    //     actual_eth_metrics.returns,
    //     expected_eth_log_return,
    // );
    // assert_vectors_with_epsilon(
    //     "USD log return",
    //     actual_usd_metrics.returns,
    //     expected_usd_log_return,
    // );
    // assert_vectors_with_epsilon(
    //     "EQ log return",
    //     actual_eq_metrics.returns,
    //     expected_eq_log_return,
    // );

    Ok(())
}

async fn set_prices_after_month(
    client: &Client<TestRuntime>,
    alice_key: AccountKey,
    nonces: Arc<Mutex<DevNonces>>,
    secs_to_move: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Setting prices 1 day after month");
    let usd_price: FixedNumber = FixedNumber::from_num(1);
    let eq_price: FixedNumber = FixedNumber::from_num(0.68493);
    let eth_price: FixedNumber = FixedNumber::from_num(388.18);
    let btc_price: FixedNumber = FixedNumber::from_num(14024);
    let eos_price: FixedNumber = FixedNumber::from_num(2.368);

    let initial_timestamp: u64 = client.now(Option::None).await?;

    join_chain_calls!(
        call_chain(
            client,
            nonces.clone(),
            alice_key,
            SetPriceCall {
                asset: Asset::Usd,
                value: usd_price,
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
                asset: Asset::Eos,
                value: eos_price,
                _runtime: PhantomData,
            },
        ),
    );

    let curr_btc = client.price_points(Asset::Btc, Option::None).await?;
    let curr_eos = client.price_points(Asset::Eos, Option::None).await?;
    let curr_eth = client.price_points(Asset::Eth, Option::None).await?;
    let curr_eq = client.price_points(Asset::Eq, Option::None).await?;
    let curr_usd = client.price_points(Asset::Usd, Option::None).await?;

    assert!(curr_btc == btc_price);
    assert!(curr_eos == eos_price);
    assert!(curr_eth == eth_price);
    assert!(curr_eq == eq_price);
    assert!(curr_usd == usd_price);

    let btc_update = client.updates(Asset::Btc, Option::None).await?.unwrap();
    let eos_update = client.updates(Asset::Eos, Option::None).await?.unwrap();
    let eth_update = client.updates(Asset::Eth, Option::None).await?.unwrap();
    let eq_update = client.updates(Asset::Eq, Option::None).await?.unwrap();
    let usd_update = client.updates(Asset::Usd, Option::None).await?.unwrap();

    assert!(btc_update.price == btc_price);
    assert!(eos_update.price == eos_price);
    assert!(eth_update.price == eth_price);
    assert!(eq_update.price == eq_price);
    assert!(usd_update.price == usd_price);

    assert!(btc_update.time == Duration::from_millis(initial_timestamp));
    assert!(eos_update.time == Duration::from_millis(initial_timestamp));
    assert!(eth_update.time == Duration::from_millis(initial_timestamp));
    assert!(eq_update.time == Duration::from_millis(initial_timestamp));
    assert!(usd_update.time == Duration::from_millis(initial_timestamp));

    println!("Moving time");
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

    Ok(())
}

async fn set_prices_and_move_time(
    client: &Client<TestRuntime>,
    alice_key: AccountKey,
    nonces: Arc<Mutex<DevNonces>>,
    secs_to_move: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    let btc_prices: Vec<FixedNumber> = vec![
        FixedNumber::from_num(10676),
        FixedNumber::from_num(10795),
        FixedNumber::from_num(10604),
        FixedNumber::from_num(10673),
        FixedNumber::from_num(10929),
        FixedNumber::from_num(11059),
        FixedNumber::from_num(11302),
        FixedNumber::from_num(11376),
        FixedNumber::from_num(11540),
        FixedNumber::from_num(11427),
        FixedNumber::from_num(11429),
        FixedNumber::from_num(11508),
        FixedNumber::from_num(11326),
        FixedNumber::from_num(11368),
        FixedNumber::from_num(11513),
        FixedNumber::from_num(11757),
        FixedNumber::from_num(11922),
        FixedNumber::from_num(12813),
        FixedNumber::from_num(12990),
        FixedNumber::from_num(12937),
        FixedNumber::from_num(13126),
        FixedNumber::from_num(13041),
        FixedNumber::from_num(13069),
        FixedNumber::from_num(13646),
        FixedNumber::from_num(13286),
        FixedNumber::from_num(13462),
        FixedNumber::from_num(13566),
        FixedNumber::from_num(13803),
        FixedNumber::from_num(13762),
        FixedNumber::from_num(13571),
    ];

    let eos_prices: Vec<FixedNumber> = vec![
        FixedNumber::from_num(2.519),
        FixedNumber::from_num(2.529),
        FixedNumber::from_num(2.696),
        FixedNumber::from_num(2.647),
        FixedNumber::from_num(2.61),
        FixedNumber::from_num(2.646),
        FixedNumber::from_num(2.642),
        FixedNumber::from_num(2.66),
        FixedNumber::from_num(2.647),
        FixedNumber::from_num(2.64),
        FixedNumber::from_num(2.611),
        FixedNumber::from_num(2.599),
        FixedNumber::from_num(2.538),
        FixedNumber::from_num(2.526),
        FixedNumber::from_num(2.548),
        FixedNumber::from_num(2.584),
        FixedNumber::from_num(2.529),
        FixedNumber::from_num(2.612),
        FixedNumber::from_num(2.671),
        FixedNumber::from_num(2.642),
        FixedNumber::from_num(2.663),
        FixedNumber::from_num(2.709),
        FixedNumber::from_num(2.633),
        FixedNumber::from_num(2.672),
        FixedNumber::from_num(2.655),
        FixedNumber::from_num(2.638),
        FixedNumber::from_num(2.51),
        FixedNumber::from_num(2.527),
        FixedNumber::from_num(2.518),
        FixedNumber::from_num(2.435),
    ];

    let usd_prices: Vec<FixedNumber> = vec![
        FixedNumber::from_num(1),
        FixedNumber::from_num(1),
        FixedNumber::from_num(1),
        FixedNumber::from_num(1),
        FixedNumber::from_num(1),
        FixedNumber::from_num(1),
        FixedNumber::from_num(1),
        FixedNumber::from_num(1),
        FixedNumber::from_num(1),
        FixedNumber::from_num(1),
        FixedNumber::from_num(1),
        FixedNumber::from_num(1),
        FixedNumber::from_num(1),
        FixedNumber::from_num(1),
        FixedNumber::from_num(1),
        FixedNumber::from_num(1),
        FixedNumber::from_num(1),
        FixedNumber::from_num(1),
        FixedNumber::from_num(1),
        FixedNumber::from_num(1),
        FixedNumber::from_num(1),
        FixedNumber::from_num(1),
        FixedNumber::from_num(1),
        FixedNumber::from_num(1),
        FixedNumber::from_num(1),
        FixedNumber::from_num(1),
        FixedNumber::from_num(1),
        FixedNumber::from_num(1),
        FixedNumber::from_num(1),
        FixedNumber::from_num(1),
    ];

    let eth_prices: Vec<FixedNumber> = vec![
        FixedNumber::from_num(352.61),
        FixedNumber::from_num(353.76),
        FixedNumber::from_num(340.85),
        FixedNumber::from_num(341.91),
        FixedNumber::from_num(351.2),
        FixedNumber::from_num(365.41),
        FixedNumber::from_num(370.88),
        FixedNumber::from_num(374.37),
        FixedNumber::from_num(386.91),
        FixedNumber::from_num(381.41),
        FixedNumber::from_num(379.11),
        FixedNumber::from_num(377.7),
        FixedNumber::from_num(365.58),
        FixedNumber::from_num(368.65),
        FixedNumber::from_num(378.44),
        FixedNumber::from_num(379.41),
        FixedNumber::from_num(368.74),
        FixedNumber::from_num(391.46),
        FixedNumber::from_num(414.5),
        FixedNumber::from_num(409.57),
        FixedNumber::from_num(412.55),
        FixedNumber::from_num(406.36),
        FixedNumber::from_num(393.11),
        FixedNumber::from_num(403.74),
        FixedNumber::from_num(388.77),
        FixedNumber::from_num(387.54),
        FixedNumber::from_num(382.76),
        FixedNumber::from_num(386.77),
        FixedNumber::from_num(396.75),
        FixedNumber::from_num(383.61),
    ];

    let eq_prices: Vec<FixedNumber> = vec![
        FixedNumber::from_num(0.7106),
        FixedNumber::from_num(0.70822),
        FixedNumber::from_num(0.63801),
        FixedNumber::from_num(0.65603),
        FixedNumber::from_num(0.70074),
        FixedNumber::from_num(0.72471),
        FixedNumber::from_num(0.72182),
        FixedNumber::from_num(0.72233),
        FixedNumber::from_num(0.7327),
        FixedNumber::from_num(0.72352),
        FixedNumber::from_num(0.71349),
        FixedNumber::from_num(0.69649),
        FixedNumber::from_num(0.66538),
        FixedNumber::from_num(0.67473),
        FixedNumber::from_num(0.69887),
        FixedNumber::from_num(0.67813),
        FixedNumber::from_num(0.66538),
        FixedNumber::from_num(0.70431),
        FixedNumber::from_num(0.7259),
        FixedNumber::from_num(0.72522),
        FixedNumber::from_num(0.7412),
        FixedNumber::from_num(0.73644),
        FixedNumber::from_num(0.80019),
        FixedNumber::from_num(0.7956),
        FixedNumber::from_num(0.74239),
        FixedNumber::from_num(0.68969),
        FixedNumber::from_num(0.69462),
        FixedNumber::from_num(0.71111),
        FixedNumber::from_num(0.71774),
        FixedNumber::from_num(0.68629),
    ];

    assert!(
        btc_prices.len() == eos_prices.len()
            && eos_prices.len() == usd_prices.len()
            && usd_prices.len() == eth_prices.len()
            && eth_prices.len() == eq_prices.len()
            && btc_prices.len() == 30
    );

    let prices_tuple = izip!(usd_prices, eq_prices, eth_prices, btc_prices, eos_prices);

    println!("Setting initial monthly prices");

    let mut counter = 1;
    for (usd, eq, eth, btc, eos) in prices_tuple {
        println!("Setting prices on iteration #{}", counter);
        let initial_timestamp: u64 = client.now(Option::None).await?;

        join_chain_calls!(
            call_chain(
                client,
                nonces.clone(),
                alice_key,
                SetPriceCall {
                    asset: Asset::Usd,
                    value: usd,
                    _runtime: PhantomData,
                },
            ),
            call_chain(
                client,
                nonces.clone(),
                alice_key,
                SetPriceCall {
                    asset: Asset::Eq,
                    value: eq,
                    _runtime: PhantomData,
                },
            ),
            call_chain(
                client,
                nonces.clone(),
                alice_key,
                SetPriceCall {
                    asset: Asset::Eth,
                    value: eth,
                    _runtime: PhantomData,
                },
            ),
            call_chain(
                client,
                nonces.clone(),
                alice_key,
                SetPriceCall {
                    asset: Asset::Btc,
                    value: btc,
                    _runtime: PhantomData,
                },
            ),
            call_chain(
                client,
                nonces.clone(),
                alice_key,
                SetPriceCall {
                    asset: Asset::Eos,
                    value: eos,
                    _runtime: PhantomData,
                },
            ),
        );

        let curr_btc = client.price_points(Asset::Btc, Option::None).await?;
        let curr_eos = client.price_points(Asset::Eos, Option::None).await?;
        let curr_eth = client.price_points(Asset::Eth, Option::None).await?;
        let curr_eq = client.price_points(Asset::Eq, Option::None).await?;
        let curr_usd = client.price_points(Asset::Usd, Option::None).await?;

        assert!(curr_btc == btc);
        assert!(curr_eos == eos);
        assert!(curr_eth == eth);
        assert!(curr_eq == eq);
        assert!(curr_usd == usd);

        let btc_update = client.updates(Asset::Btc, Option::None).await?.unwrap();
        let eos_update = client.updates(Asset::Eos, Option::None).await?.unwrap();
        let eth_update = client.updates(Asset::Eth, Option::None).await?.unwrap();
        let eq_update = client.updates(Asset::Eq, Option::None).await?.unwrap();
        let usd_update = client.updates(Asset::Usd, Option::None).await?.unwrap();

        assert!(btc_update.price == btc);
        assert!(eos_update.price == eos);
        assert!(eth_update.price == eth);
        assert!(eq_update.price == eq);
        assert!(usd_update.price == usd);

        assert!(btc_update.time == Duration::from_millis(initial_timestamp));
        assert!(eos_update.time == Duration::from_millis(initial_timestamp));
        assert!(eth_update.time == Duration::from_millis(initial_timestamp));
        assert!(eq_update.time == Duration::from_millis(initial_timestamp));
        assert!(usd_update.time == Duration::from_millis(initial_timestamp));

        println!("Moving time");
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

        counter = counter + 1;
    }

    Ok(())
}
