use crate::mock::*;
use frame_support::{assert_ok};
use common::Asset;

#[test]
fn set_price_is_visible_through_storage() {
	new_test_ext().execute_with(|| {
		assert_ok!(OracleModule::set_price(Origin::signed(1), Asset::Btc, 42));
		assert_eq!(OracleModule::price_points(Asset::Btc), 42);
	});
}
