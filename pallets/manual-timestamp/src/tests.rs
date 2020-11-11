use crate::mock::*;
use core::time::Duration;
use frame_support::assert_ok;
use frame_support::traits::UnixTime;

#[test]
fn genesis_timestamp() {
    new_test_ext().execute_with(|| {
        assert_eq!(
            <TimestampModule as UnixTime>::now(),
            Duration::from_millis(INITIAL_TIMESTAMP)
        );
    });
}

#[test]
fn advance_secs() {
    new_test_ext().execute_with(|| {
        let delta = 543;
        assert_ok!(TimestampModule::advance_secs(Origin::signed(1), delta));
        assert_eq!(
            <TimestampModule as UnixTime>::now(),
            Duration::from_millis(INITIAL_TIMESTAMP + (1000 * delta as u64))
        );
    });
}
