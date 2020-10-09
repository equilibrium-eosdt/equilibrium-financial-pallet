// Copyright (C) 2020 equilibrium.
// SPDX-License-Identifier: Apache-2.0
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::mock::*;
use frame_support::assert_ok;
use frame_support::dispatch::DispatchError;
use crate::{Financial, CalcReturnType, PriceUpdate, Error};
use financial_primitives::OnPriceSet;
use approx::assert_abs_diff_eq;
use substrate_fixed::traits::LossyInto;
use chrono::prelude::*;
use core::time::Duration;

fn create_duration(year: i32, month: u32, day: u32, hour: u32, minute: u32, second: u32) -> Duration {
	let timestamp = Utc.ymd(year, month, day).and_hms(hour, minute, second).timestamp();
	Duration::from_secs(timestamp as u64)
}

#[cfg(test)]
mod price_period {
	use crate::*;
	use crate::tests::create_duration;

	#[test]
	fn period_id_are_same_within_period() {
		let time1 = create_duration(2020, 9, 14, 12, 31, 0);
		let time2 = create_duration(2020, 9, 14, 12, 54, 0);

		// period is one hour
		let period = PricePeriod(60);

		let period_id_1 = period.get_period_id(time1);
		let period_id_2 = period.get_period_id(time2);

		assert!(period_id_1.is_ok());
		assert!(period_id_2.is_ok());
		assert_eq!(period_id_1, period_id_2);
	}

	#[test]
	fn period_id_are_different_for_different_periods() {
		let time1 = create_duration(2020, 9, 14, 12, 31, 0);
		let time2 = create_duration(2020, 9, 14, 13, 2, 0);

		// period is one hour
		let period = PricePeriod(60);

		let period_id_1 = period.get_period_id(time1);
		let period_id_2 = period.get_period_id(time2);

		assert!(period_id_1.is_ok());
		assert!(period_id_2.is_ok());
		assert!(period_id_1 != period_id_2);
	}

	#[test]
	fn info_for_neighbour_periods() {
		let period_start = create_duration(2020, 9, 14, 12, 31, 0);
		let now = create_duration(2020, 9, 14, 13, 2, 0);

		// period is one hour
		let period = PricePeriod(60);

		let actual = period.get_curr_period_info(period_start, now);
		let expected = Ok((create_duration(2020, 9, 14, 13, 0, 0), 1));

		assert_eq!(actual, expected);
	}

	#[test]
	fn info_for_distant_periods() {
		let period_start = create_duration(2020, 9, 14, 12, 31, 0);
		let now = create_duration(2020, 9, 15, 7, 2, 0);

		// period is one hour
		let period = PricePeriod(60);

		let actual = period.get_curr_period_info(period_start, now);
		let expected = Ok((create_duration(2020, 9, 15, 7, 0, 0), 19));

		assert_eq!(actual, expected);
	}

	#[test]
	fn invalid_period_start() {
		let period_start = create_duration(2020, 9, 14, 12, 31, 0);
		let now = create_duration(2020, 9, 14, 12, 42, 0);

		// period is one hour
		let period = PricePeriod(60);

		let actual = period.get_period_change(Some(period_start), now);
		let expected = Err(PricePeriodError::InvalidPeriodStart);

		assert_eq!(actual, expected);
	}

	#[test]
	fn period_remains_unchanged() {
		let period_start = create_duration(2020, 9, 14, 12, 0, 0);
		let now = create_duration(2020, 9, 14, 12, 42, 0);

		// period is one hour
		let period = PricePeriod(60);

		let actual = period.get_period_change(Some(period_start), now);
		let expected = Ok(PeriodChange {
			period_start,
			action: PeriodAction::RemainsUnchanged,
		});

		assert_eq!(actual, expected);
	}

	#[test]
	fn period_changed_slightly() {
		let period_start = create_duration(2020, 9, 14, 12, 0, 0);
		let now = create_duration(2020, 9, 14, 13, 2, 0);

		// period is one hour
		let period = PricePeriod(60);

		let actual = period.get_period_change(Some(period_start), now);
		let expected = Ok(PeriodChange {
			period_start: create_duration(2020, 9, 14, 13, 0, 0),
			action: PeriodAction::StartedNew(0),
		});

		assert_eq!(actual, expected);
	}

	#[test]
	fn period_changed_significantly() {
		let period_start = create_duration(2020, 9, 14, 12, 0, 0);
		let now = create_duration(2020, 9, 15, 7, 2, 0);

		// period is one hour
		let period = PricePeriod(60);

		let actual = period.get_period_change(Some(period_start), now);
		let expected = Ok(PeriodChange {
			period_start: create_duration(2020, 9, 15, 7, 0, 0),
			action: PeriodAction::StartedNew(18),
		});

		assert_eq!(actual, expected);
	}

	#[test]
	fn period_is_in_the_past() {
		let period_start = create_duration(2020, 9, 14, 12, 0, 0);
		let now = create_duration(2020, 9, 12, 16, 27, 10);

		// period is one hour
		let period = PricePeriod(60);

		let actual = period.get_period_change(Some(period_start), now);
		let expected = Err(PricePeriodError::PeriodIsInThePast);

		assert_eq!(actual, expected);
	}

	#[test]
	fn zero_period() {
		let period_start = create_duration(2020, 9, 14, 12, 0, 0);
		let now = create_duration(2020, 9, 15, 7, 2, 0);

		// period is one hour
		let period = PricePeriod(0);

		let actual = period.get_period_change(Some(period_start), now);
		let expected = Err(PricePeriodError::DivisionByZero);

		assert_eq!(actual, expected);
	}

	#[test]
	fn elapsed_period_count_is_too_large() {
		let period_start = create_duration(2020, 9, 14, 12, 0, 0);
		let now = Duration::from_secs(u64::MAX);

		// period is one hour
		let period = PricePeriod(60);

		let actual = period.get_period_change(Some(period_start), now);
		let expected = Err(PricePeriodError::Overflow);

		assert_eq!(actual, expected);
	}
}

#[cfg(test)]
mod new_prices {
	use crate::*;

	#[test]
	fn no_last_price_no_empty_periods() {
		let actual = get_new_prices::<u32>(None, 123, 0, 5);
		let expected = Ok(vec![123]);

		assert_eq!(actual, expected);
	}

	#[test]
	fn no_last_price_some_empty_periods() {
		let actual = get_new_prices::<u32>(None, 123, 3, 5);
		let expected = Ok(vec![123]);

		assert_eq!(actual, expected);
	}

	#[test]
	fn some_last_price_no_empty_periods() {
		let actual = get_new_prices::<u32>(Some(555), 123, 0, 5);
		let expected = Ok(vec![123]);

		assert_eq!(actual, expected);
	}

	#[test]
	fn some_last_price_some_empty_periods_less_than_max() {
		let actual = get_new_prices::<u32>(Some(555), 123, 3, 5);
		let expected = Ok(vec![555, 555, 555, 123]);

		assert_eq!(actual, expected);
	}

	#[test]
	fn some_last_price_some_empty_periods_equal_to_max() {
		let actual = get_new_prices::<u32>(Some(555), 123, 4, 5);
		let expected = Ok(vec![555, 555, 555, 555, 123]);

		assert_eq!(actual, expected);
	}

	#[test]
	fn some_last_price_some_empty_periods_slightly_more_than_max() {
		let actual = get_new_prices::<u32>(Some(555), 123, 5, 5);
		let expected = Ok(vec![555, 555, 555, 555, 123]);

		assert_eq!(actual, expected);
	}

	#[test]
	fn some_last_price_some_empty_periods_significantly_more_than_max() {
		let actual = get_new_prices::<u32>(Some(555), 123, 27, 5);
		let expected = Ok(vec![555, 555, 555, 555, 123]);

		assert_eq!(actual, expected);
	}

	#[test]
	fn max_periods_is_zero() {
		let actual = get_new_prices::<u32>(Some(555), 123, 27, 0);
		let expected = Err(GetNewPricesError::Overflow);

		assert_eq!(actual, expected);
	}

	#[test]
	fn max_periods_is_one() {
		let actual = get_new_prices::<u32>(Some(555), 123, 3, 1);
		let expected = Ok(vec![123]);

		assert_eq!(actual, expected);
	}
}

#[cfg(test)]
mod calc_return {
	use crate::*;
	use crate::mock::FixedNumber;
	use substrate_fixed::traits::LossyInto;
	use approx::assert_abs_diff_eq;

	#[test]
	fn calc_return_valid() {
		let x1 = FixedNumber::from_num(8);
		let x2 = FixedNumber::from_num(6);
		let actual = calc_return(x1, x2);
		let expected = Ok(FixedNumber::from_num(-0.25));

		assert_eq!(actual, expected);
	}

	#[test]
	fn calc_return_valid_positive() {
		let x1 = FixedNumber::from_num(8);
		let x2 = FixedNumber::from_num(10);
		let actual = calc_return(x1, x2);
		let expected = Ok(FixedNumber::from_num(0.25));

		assert_eq!(actual, expected);
	}

	#[test]
	fn calc_return_x1_is_zero() {
		let x1 = FixedNumber::from_num(0);
		let x2 = FixedNumber::from_num(6);
		let actual = calc_return(x1, x2);
		let expected = Err(MathError::DivisionByZero);

		assert_eq!(actual, expected);
	}

	#[test]
	fn calc_log_return_valid() {
		let x1 = FixedNumber::from_num(8);
		let x2 = FixedNumber::from_num(6);
		let actual: f64 = calc_log_return(x1, x2).unwrap().lossy_into();
		let expected = -0.287682072452;

		assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
	}

	#[test]
	fn calc_log_return_x1_is_zero() {
		let x1 = FixedNumber::from_num(0);
		let x2 = FixedNumber::from_num(6);
		let actual = calc_log_return(x1, x2);
		let expected = Err(MathError::DivisionByZero);

		assert_eq!(actual, expected);
	}

	#[test]
	fn calc_log_return_x1_is_negative() {
		let x1 = FixedNumber::from_num(-3);
		let x2 = FixedNumber::from_num(6);
		let actual = calc_log_return(x1, x2);
		let expected = Err(MathError::Transcendental);

		assert_eq!(actual, expected);
	}

	#[test]
	fn calc_return_vec_empty() {
		let prices: Vec<FixedNumber> = Vec::new();

		let actual = calc_return_vec::<FixedNumber, FixedNumber>(CalcReturnType::Linear, prices);
		let expected = Err(CalcReturnError::NotEnoughPoints);

		assert_eq!(actual, expected);
	}

	#[test]
	fn calc_return_vec_one_item() {
		let prices: Vec<FixedNumber> = vec![1.5].into_iter().map(FixedNumber::from_num).collect();

		let actual = calc_return_vec::<FixedNumber, FixedNumber>(CalcReturnType::Linear, prices);
		let expected = Err(CalcReturnError::NotEnoughPoints);

		assert_eq!(actual, expected);
	}

	#[test]
	fn calc_return_vec_linear_valid() {
		let prices: Vec<FixedNumber> = vec![
			7_117.21,
			7_429.72,
			7_550.90,
			7_569.94,
			7_679.87,
			7_795.60,
			7_807.06,
			8_801.04,
			8_658.55,
			8_864.77,
		].into_iter().map(FixedNumber::from_num).collect();

		let actual: Vec<f64> = calc_return_vec::<FixedNumber, FixedNumber>(CalcReturnType::Linear, prices)
						.unwrap().into_iter().map(|x| x.lossy_into()).collect();
		let expected: Vec<f64> = vec![
			0.04390906,
			0.01631017,
			0.00252155,
			0.01452191,
			0.01506927,
			0.00147006,
			0.12731809,
			-0.01619013,
			0.02381692,
		];

		assert_eq!(actual.len(), expected.len());
		for (a, e) in actual.into_iter().zip(expected.into_iter()) {
			assert_abs_diff_eq!(a, e, epsilon = 1e-8);
		}
	}

	#[test]
	fn calc_return_vec_log_valid() {
		let prices: Vec<FixedNumber> = vec![
			7_117.21,
			7_429.72,
			7_550.90,
			7_569.94,
			7_679.87,
			7_795.60,
			7_807.06,
			8_801.04,
			8_658.55,
			8_864.77,
		].into_iter().map(FixedNumber::from_num).collect();

		let actual: Vec<f64> = calc_return_vec::<FixedNumber, FixedNumber>(CalcReturnType::Log, prices)
						.unwrap().into_iter().map(|x| x.lossy_into()).collect();
		let expected: Vec<f64> = vec![
			0.042972378,
			0.016178588,
			0.002518380,
			0.014417479,
			0.014956852,
			0.001468981,
			0.119841444,
			-0.016322624,
			0.023537722,
		];

		assert_eq!(actual.len(), expected.len());
		for (a, e) in actual.into_iter().zip(expected.into_iter()) {
			assert_abs_diff_eq!(a, e, epsilon = 1e-8);
		}
	}
}

#[test]
fn calc_linear_return_for_btc_using_only_genesis() {
	new_test_ext().execute_with(|| {
		let actual: Vec<f64> = <FinancialModule as Financial>::calc_return(CalcReturnType::Linear, Asset::Btc)
			.unwrap().into_iter().map(|x| x.lossy_into()).collect();
		let expected = vec![
			0.04390906,
			0.01631017,
			0.00252155,
			0.01452191,
			0.01506927,
			0.00147006,
			0.12731809,
			-0.01619013,
			0.02381692,
		];

		assert_eq!(actual.len(), expected.len());
		for (a, e) in actual.into_iter().zip(expected.into_iter()) {
			assert_abs_diff_eq!(a, e, epsilon = 1e-8);
		}
	});
}

#[test]
fn calc_log_return_for_btc_using_only_genesis() {
	new_test_ext().execute_with(|| {
		let actual: Vec<f64> = <FinancialModule as Financial>::calc_return(CalcReturnType::Log, Asset::Btc)
			.unwrap().into_iter().map(|x| x.lossy_into()).collect();
		let expected = vec![
			0.042972378,
			0.016178588,
			0.002518380,
			0.014417479,
			0.014956852,
			0.001468981,
			0.119841444,
			-0.016322624,
			0.023537722,
		];

		assert_eq!(actual.len(), expected.len());
		for (a, e) in actual.into_iter().zip(expected.into_iter()) {
			assert_abs_diff_eq!(a, e, epsilon = 1e-8);
		}
	});
}

#[test]
fn calc_linear_return_for_eos_using_only_genesis() {
	new_test_ext().execute_with(|| {
		let actual: Vec<f64> = <FinancialModule as Financial>::calc_return(CalcReturnType::Linear, Asset::Eos)
			.unwrap().into_iter().map(|x| x.lossy_into()).collect();
		let expected = vec![
			0.01908397,
			0.01872659,
			0.00000000,
			0.00735294,
			0.00364964,
			0.01090909,
			0.08633094,
			-0.06291391,
			0.02120141,
		];

		assert_eq!(actual.len(), expected.len());
		for (a, e) in actual.into_iter().zip(expected.into_iter()) {
			assert_abs_diff_eq!(a, e, epsilon = 1e-8);
		}
	});
}

#[test]
fn calc_log_return_for_eos_using_only_genesis() {
	new_test_ext().execute_with(|| {
		let actual: Vec<f64> = <FinancialModule as Financial>::calc_return(CalcReturnType::Log, Asset::Eos)
			.unwrap().into_iter().map(|x| x.lossy_into()).collect();
		let expected = vec![
			0.018904155,
			0.018553408,
			0.000000000,
			0.007326040,
			0.003642991,
			0.010850016,
			0.082805904,
			-0.064980120,
			0.020979790,
		];

		assert_eq!(actual.len(), expected.len());
		for (a, e) in actual.into_iter().zip(expected.into_iter()) {
			assert_abs_diff_eq!(a, e, epsilon = 1e-8);
		}
	});
}

#[test]
fn calc_linear_return_for_btc_using_some_oracle_prices() {
	new_test_ext().execute_with(|| {
		let prices: Vec<_> = vec![
			8_988.60,
			8_897.47,
			8_912.65,
			9_003.07,
		].into_iter().map(FixedNumber::from_num).collect();
 
		let mut now = create_duration(2020, 9, 14, 12, 31, 0);
		set_now(now);

		for p in prices.into_iter() {
			let result = <FinancialModule as OnPriceSet>::on_price_set(Asset::Btc, p);

			assert_ok!(result);

			now = now + Duration::from_secs(60 * 60);
			set_now(now);
		}

		let actual: Vec<f64> = <FinancialModule as Financial>::calc_return(CalcReturnType::Linear, Asset::Btc)
			.unwrap().into_iter().map(|x| x.lossy_into()).collect();
		let expected = vec![
			0.01506927,
			0.00147006,
			0.12731809,
			-0.01619013,
			0.02381692,
			0.01396878,
			-0.01013840,
			0.00170610,
			0.01014513,
		];

		assert_eq!(actual.len(), expected.len());
		for (a, e) in actual.into_iter().zip(expected.into_iter()) {
			assert_abs_diff_eq!(a, e, epsilon = 1e-8);
		}
	});
}

#[test]
fn initial_prices_state_with_empty_genesis() {
	new_test_ext_empty_storage().execute_with(|| {
		let asset = Asset::Btc;

		let updates = FinancialModule::updates(asset);
		let expected_updates = None;

		let prices: Vec<_> = FinancialModule::prices(asset).iter().copied().collect();
		let expected_prices: Vec<FixedNumber> = vec![];

		let prices_cap = FinancialModule::prices(asset).len_cap();
		let expected_price_cap = 0;

		assert_eq!(updates, expected_updates);
		assert_eq!(prices, expected_prices);
		assert_eq!(prices_cap, expected_price_cap);
	});
}

#[test]
fn initial_prices_state_with_nonempty_genesis() {
	new_test_ext().execute_with(|| {
		let asset = Asset::Btc;

		let updates = FinancialModule::updates(asset);
		let expected_updates = None;

		let prices: Vec<_> = FinancialModule::prices(asset).iter().copied().collect();
		let expected_prices: Vec<FixedNumber> = initial_btc_prices()
			.into_iter().map(FixedNumber::from_num).collect();

		let prices_cap = FinancialModule::prices(asset).len_cap();
		let expected_price_cap = PriceCount::get();

		assert_eq!(updates, expected_updates);
		assert_eq!(prices, expected_prices);
		assert_eq!(prices_cap, expected_price_cap);
	});
}

#[test]
fn first_price_with_empty_genesis() {
	new_test_ext_empty_storage().execute_with(|| {
		let asset = Asset::Btc;

		let price = FixedNumber::from_num(9_000);

		let period_start = create_duration(2020, 9, 14, 12, 0, 0);
		let now = create_duration(2020, 9, 14, 12, 31, 0);
		set_now(now);

		assert_ok!(<FinancialModule as OnPriceSet>::on_price_set(asset, price));

		let updates = FinancialModule::updates(asset);
		let expected_updates = Some(PriceUpdate::new(period_start, now, price));

		let prices: Vec<_> = FinancialModule::prices(asset).iter().copied().collect();
		let expected_prices: Vec<FixedNumber> = vec![price];

		let prices_cap = FinancialModule::prices(asset).len_cap();
		let expected_price_cap = PriceCount::get();

		assert_eq!(updates, expected_updates);
		assert_eq!(prices, expected_prices);
		assert_eq!(prices_cap, expected_price_cap);
	});
}

#[test]
fn new_price_is_in_the_past() {
	new_test_ext_empty_storage().execute_with(|| {
		let asset = Asset::Btc;

		let price = FixedNumber::from_num(9_000);

		let period_start = create_duration(2020, 9, 14, 12, 0, 0);
		let now = create_duration(2020, 9, 14, 12, 31, 0);
		set_now(now);

		assert_ok!(<FinancialModule as OnPriceSet>::on_price_set(asset, price));

		let past_price = FixedNumber::from_num(7_300);
		let past_now = create_duration(2020, 9, 13, 0, 3, 0);
		set_now(past_now);

		let result = <FinancialModule as OnPriceSet>::on_price_set(asset, past_price);
		let expected_result: Result<(), DispatchError> = Err(Error::<Test>::PeriodIsInThePast.into());
		assert_eq!(result, expected_result);

		let updates = FinancialModule::updates(asset);
		let expected_updates = Some(PriceUpdate::new(period_start, now, price));

		let prices: Vec<_> = FinancialModule::prices(asset).iter().copied().collect();
		let expected_prices: Vec<FixedNumber> = vec![price];

		let prices_cap = FinancialModule::prices(asset).len_cap();
		let expected_price_cap = PriceCount::get();

		assert_eq!(updates, expected_updates);
		assert_eq!(prices, expected_prices);
		assert_eq!(prices_cap, expected_price_cap);
	});
}

#[test]
fn new_price_is_in_current_period() {
	new_test_ext_empty_storage().execute_with(|| {
		let asset = Asset::Btc;

		let price = FixedNumber::from_num(9_000);

		let period_start = create_duration(2020, 9, 14, 12, 0, 0);
		let now = create_duration(2020, 9, 14, 12, 31, 0);
		set_now(now);

		assert_ok!(<FinancialModule as OnPriceSet>::on_price_set(asset, price));

		let same_period_price = FixedNumber::from_num(9_100);
		let same_period_now = create_duration(2020, 9, 14, 12, 48, 0);
		set_now(same_period_now);

		assert_ok!(<FinancialModule as OnPriceSet>::on_price_set(asset, same_period_price));

		let updates = FinancialModule::updates(asset);
		let expected_updates = Some(PriceUpdate::new(
			period_start,
			same_period_now,
			same_period_price
		));

		let prices: Vec<_> = FinancialModule::prices(asset).iter().copied().collect();
		let expected_prices: Vec<FixedNumber> = vec![price];

		let prices_cap = FinancialModule::prices(asset).len_cap();
		let expected_price_cap = PriceCount::get();

		assert_eq!(updates, expected_updates);
		assert_eq!(prices, expected_prices);
		assert_eq!(prices_cap, expected_price_cap);
	});
}

#[test]
fn new_price_is_in_next_period() {
	new_test_ext_empty_storage().execute_with(|| {
		let asset = Asset::Btc;

		let price = FixedNumber::from_num(9_000);

		let now = create_duration(2020, 9, 14, 12, 31, 0);
		set_now(now);

		assert_ok!(<FinancialModule as OnPriceSet>::on_price_set(asset, price));

		let next_period_start = create_duration(2020, 9, 14, 13, 0, 0);
		let next_period_price = FixedNumber::from_num(9_100);
		let next_period_now = create_duration(2020, 9, 14, 13, 27, 0);
		set_now(next_period_now);

		assert_ok!(<FinancialModule as OnPriceSet>::on_price_set(asset, next_period_price));

		let updates = FinancialModule::updates(asset);
		let expected_updates = Some(PriceUpdate::new(
			next_period_start,
			next_period_now,
			next_period_price,
		));

		let prices: Vec<_> = FinancialModule::prices(asset).iter().copied().collect();
		let expected_prices: Vec<FixedNumber> = vec![price, next_period_price];

		let prices_cap = FinancialModule::prices(asset).len_cap();
		let expected_price_cap = PriceCount::get();

		assert_eq!(updates, expected_updates);
		assert_eq!(prices, expected_prices);
		assert_eq!(prices_cap, expected_price_cap);
	});
}

#[test]
fn new_price_is_a_few_periods_ahead() {
	new_test_ext_empty_storage().execute_with(|| {
		let asset = Asset::Btc;

		let price = FixedNumber::from_num(9_000);

		let now = create_duration(2020, 9, 14, 12, 31, 0);
		set_now(now);

		assert_ok!(<FinancialModule as OnPriceSet>::on_price_set(asset, price));

		let new_period_start = create_duration(2020, 9, 14, 16, 0, 0);
		let new_period_price = FixedNumber::from_num(9_100);
		let new_period_now = create_duration(2020, 9, 14, 16, 27, 0);
		set_now(new_period_now);

		assert_ok!(<FinancialModule as OnPriceSet>::on_price_set(asset, new_period_price));

		let updates = FinancialModule::updates(asset);
		let expected_updates = Some(PriceUpdate::new(
			new_period_start,
			new_period_now,
			new_period_price,
		));

		let prices: Vec<_> = FinancialModule::prices(asset).iter().copied().collect();
		let expected_prices: Vec<FixedNumber> = vec![
			price,
			price,
			price,
			price,
			new_period_price,
		];

		let prices_cap = FinancialModule::prices(asset).len_cap();
		let expected_price_cap = PriceCount::get();

		assert_eq!(updates, expected_updates);
		assert_eq!(prices, expected_prices);
		assert_eq!(prices_cap, expected_price_cap);
	});
}

#[test]
fn new_price_is_many_periods_ahead() {
	new_test_ext_empty_storage().execute_with(|| {
		let asset = Asset::Btc;

		let price = FixedNumber::from_num(9_000);

		let now = create_duration(2020, 9, 14, 12, 31, 0);
		set_now(now);

		assert_ok!(<FinancialModule as OnPriceSet>::on_price_set(asset, price));

		let new_period_start = create_duration(2020, 10, 3, 22, 0, 0);
		let new_period_price = FixedNumber::from_num(9_100);
		let new_period_now = create_duration(2020, 10, 3, 22, 27, 0);
		set_now(new_period_now);

		assert_ok!(<FinancialModule as OnPriceSet>::on_price_set(asset, new_period_price));

		let updates = FinancialModule::updates(asset);
		let expected_updates = Some(PriceUpdate::new(
			new_period_start,
			new_period_now,
			new_period_price,
		));

		let prices: Vec<_> = FinancialModule::prices(asset).iter().copied().collect();
		let mut expected_prices: Vec<FixedNumber> = vec![price; PriceCount::get() as usize - 1];
		expected_prices.push(new_period_price);

		let prices_cap = FinancialModule::prices(asset).len_cap();
		let expected_price_cap = PriceCount::get();

		assert_eq!(updates, expected_updates);
		assert_eq!(prices, expected_prices);
		assert_eq!(prices_cap, expected_price_cap);
	});
}

#[test]
fn wrong_now_in_the_distant_future() {
	new_test_ext_empty_storage().execute_with(|| {
		let asset = Asset::Btc;

		let price = FixedNumber::from_num(9_000);

		let period_start = create_duration(2020, 9, 14, 12, 0, 0);
		let now = create_duration(2020, 9, 14, 12, 31, 0);
		set_now(now);

		assert_ok!(<FinancialModule as OnPriceSet>::on_price_set(asset, price));

		let distant_future_price = FixedNumber::from_num(9_100);
		let distant_future_now = Duration::from_secs(u64::MAX);
		set_now(distant_future_now);

		let result = <FinancialModule as OnPriceSet>::on_price_set(asset, distant_future_price);
		let expected_result: Result<(), DispatchError> = Err(Error::<Test>::Overflow.into());
		assert_eq!(result, expected_result);

		let updates = FinancialModule::updates(asset);
		let expected_updates = Some(PriceUpdate::new(
			period_start,
			now,
			price,
		));

		let prices: Vec<_> = FinancialModule::prices(asset).iter().copied().collect();
		let expected_prices: Vec<FixedNumber> = vec![price];

		let prices_cap = FinancialModule::prices(asset).len_cap();
		let expected_price_cap = PriceCount::get();

		assert_eq!(updates, expected_updates);
		assert_eq!(prices, expected_prices);
		assert_eq!(prices_cap, expected_price_cap);
	});
}

#[test]
fn new_price_has_arravied_while_prices_countainer_is_full() {
	new_test_ext_empty_storage().execute_with(|| {
		let asset = Asset::Btc;

		let prices: Vec<_> = vec![
			1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
		].into_iter().map(FixedNumber::from_num).collect();
 
		let mut now = create_duration(2020, 9, 14, 12, 31, 0);
		set_now(now);

		for &p in prices.iter() {
			let result = <FinancialModule as OnPriceSet>::on_price_set(asset, p);

			assert_ok!(result);

			now = now + Duration::from_secs(60 * 60);
			set_now(now);
		}

		let new_period_start = create_duration(2020, 9, 14, 22, 0, 0);
		let new_period_price = FixedNumber::from_num(11);

		assert_ok!(<FinancialModule as OnPriceSet>::on_price_set(asset, new_period_price));

		let updates = FinancialModule::updates(asset);
		let expected_updates = Some(PriceUpdate::new(
			new_period_start,
			now,
			new_period_price,
		));

		let actual_prices: Vec<_> = FinancialModule::prices(asset).iter().copied().collect();
		let mut expected_prices: Vec<_> = prices[1..].iter().copied().collect();
		expected_prices.push(new_period_price);

		let prices_cap = FinancialModule::prices(asset).len_cap();
		let expected_price_cap = PriceCount::get();

		assert_eq!(updates, expected_updates);
		assert_eq!(actual_prices, expected_prices);
		assert_eq!(prices_cap, expected_price_cap);
	});
}
