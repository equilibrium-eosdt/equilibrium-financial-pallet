use crate::mock::*;
use common::Asset;
use financial_primitives::BalanceAware;
use frame_support::assert_ok;

#[test]
fn default_balance_is_zero() {
    new_test_ext().execute_with(|| {
        //assert_ok!(OracleModule::set_price(Origin::signed(1), Asset::Btc, 42));
        assert_eq!(PortfolioModule::balances(123, Asset::Btc), 0);
    });
}

#[test]
fn set_balance() {
    new_test_ext().execute_with(|| {
        assert_ok!(PortfolioModule::set_balance(
            Origin::signed(1),
            456,
            Asset::Btc,
            42
        ));
        assert_eq!(PortfolioModule::balances(456, Asset::Btc), 42);
    });
}

#[test]
fn balance_aware_balances() {
    new_test_ext().execute_with(|| {
        assert_ok!(PortfolioModule::set_balance(
            Origin::signed(1),
            456,
            Asset::Btc,
            42
        ));
        assert_ok!(PortfolioModule::set_balance(
            Origin::signed(1),
            456,
            Asset::Eos,
            11
        ));
        let actual = <PortfolioModule as BalanceAware>::balances(
            &456,
            &vec![Asset::Eos, Asset::Btc, Asset::Usd],
        );

        assert_eq!(actual, Ok(vec![11, 42, 0]));
    });
}
