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

use super::*;
use crate::mock::*;
use crate::{CalcReturnType, CalcVolatilityType, Error, Financial, PriceUpdate};
use approx::assert_abs_diff_eq;
use chrono::prelude::*;
use financial_primitives::OnPriceSet;
use frame_support::assert_ok;
use frame_support::dispatch::DispatchError;
use sp_std::ops::Range;
use substrate_fixed::traits::LossyInto;

fn create_duration(
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
) -> Duration {
    let timestamp = Utc
        .ymd(year, month, day)
        .and_hms(hour, minute, second)
        .timestamp();
    Duration { secs: timestamp as u64, nanos: 0 }
}

#[cfg(test)]
mod price_period {
    use super::super::*;
    use super::create_duration;

    #[test]
    fn info_for_neighbour_periods() {
        let period_start = create_duration(2020, 9, 14, 12, 31, 0);
        let now = create_duration(2020, 9, 14, 13, 2, 0);

        // period is one hour
        let period = PricePeriod(60);

        let actual = get_curr_period_info(&period.into(), period_start.into(), now.into());
        let expected = Ok((create_duration(2020, 9, 14, 13, 0, 0), 1));

        assert_eq!(actual, expected);
    }

    #[test]
    fn info_for_distant_periods() {
        let period_start = create_duration(2020, 9, 14, 12, 31, 0);
        let now = create_duration(2020, 9, 15, 7, 2, 0);

        // period is one hour
        let period = PricePeriod(60);

        let actual = get_curr_period_info(&period, period_start, now);
        let expected = Ok((create_duration(2020, 9, 15, 7, 0, 0), 19));

        assert_eq!(actual, expected);
    }

    #[test]
    fn invalid_period_start() {
        let period_start = create_duration(2020, 9, 14, 12, 31, 0);
        let now = create_duration(2020, 9, 14, 12, 42, 0);

        // period is one hour
        let period = PricePeriod(60);

        let actual = get_period_change(&period, Some(period_start), now);
        let expected = Err(PricePeriodChangeError::InvalidPeriodStart);

        assert_eq!(actual, expected);
    }

    #[test]
    fn period_remains_unchanged() {
        let period_start = create_duration(2020, 9, 14, 12, 0, 0);
        let now = create_duration(2020, 9, 14, 12, 42, 0);

        // period is one hour
        let period = PricePeriod(60);

        let actual = get_period_change(&period, Some(period_start), now);
        let expected = Ok(PricePeriodChange {
            period_start,
            action: PricePeriodAction::RemainsUnchanged,
        });

        assert_eq!(actual, expected);
    }

    #[test]
    fn period_changed_slightly() {
        let period_start = create_duration(2020, 9, 14, 12, 0, 0);
        let now = create_duration(2020, 9, 14, 13, 2, 0);

        // period is one hour
        let period = PricePeriod(60);

        let actual = get_period_change(&period, Some(period_start), now);
        let expected = Ok(PricePeriodChange {
            period_start: create_duration(2020, 9, 14, 13, 0, 0),
            action: PricePeriodAction::StartedNew(0),
        });

        assert_eq!(actual, expected);
    }

    #[test]
    fn period_changed_significantly() {
        let period_start = create_duration(2020, 9, 14, 12, 0, 0);
        let now = create_duration(2020, 9, 15, 7, 2, 0);

        // period is one hour
        let period = PricePeriod(60);

        let actual = get_period_change(&period, Some(period_start), now);
        let expected = Ok(PricePeriodChange {
            period_start: create_duration(2020, 9, 15, 7, 0, 0),
            action: PricePeriodAction::StartedNew(18),
        });

        assert_eq!(actual, expected);
    }

    #[test]
    fn period_is_in_the_past() {
        let period_start = create_duration(2020, 9, 14, 12, 0, 0);
        let now = create_duration(2020, 9, 12, 16, 27, 10);

        // period is one hour
        let period = PricePeriod(60);

        let actual = get_period_change(&period, Some(period_start), now);
        let expected = Err(PricePeriodChangeError::PeriodIsInThePast);

        assert_eq!(actual, expected);
    }

    #[test]
    fn zero_period() {
        let period_start = create_duration(2020, 9, 14, 12, 0, 0);
        let now = create_duration(2020, 9, 15, 7, 2, 0);

        // period is one hour
        let period = PricePeriod(0);

        let actual = get_period_change(&period, Some(period_start), now);
        let expected = Err(PricePeriodChangeError::DivisionByZero);

        assert_eq!(actual, expected);
    }

    #[test]
    fn elapsed_period_count_is_too_large() {
        let period_start = create_duration(2020, 9, 14, 12, 0, 0);
        let now = Duration::from_secs(u64::MAX);

        // period is one hour
        let period = PricePeriod(60);

        let actual = get_period_change(&period, Some(period_start), now);
        let expected = Err(PricePeriodChangeError::Overflow);

        assert_eq!(actual, expected);
    }
}

#[cfg(test)]
mod new_prices {
    use crate::*;

    #[test]
    fn some_last_price_no_empty_periods() {
        let actual = get_new_prices::<u32>(555, 123, 0, 5);
        let expected = Ok(vec![123]);

        assert_eq!(actual, expected);
    }

    #[test]
    fn some_last_price_some_empty_periods_less_than_max() {
        let actual = get_new_prices::<u32>(555, 123, 3, 5);
        let expected = Ok(vec![555, 555, 555, 123]);

        assert_eq!(actual, expected);
    }

    #[test]
    fn some_last_price_some_empty_periods_equal_to_max() {
        let actual = get_new_prices::<u32>(555, 123, 4, 5);
        let expected = Ok(vec![555, 555, 555, 555, 123]);

        assert_eq!(actual, expected);
    }

    #[test]
    fn some_last_price_some_empty_periods_slightly_more_than_max() {
        let actual = get_new_prices::<u32>(555, 123, 5, 5);
        let expected = Ok(vec![555, 555, 555, 555, 123]);

        assert_eq!(actual, expected);
    }

    #[test]
    fn some_last_price_some_empty_periods_significantly_more_than_max() {
        let actual = get_new_prices::<u32>(555, 123, 27, 5);
        let expected = Ok(vec![555, 555, 555, 555, 123]);

        assert_eq!(actual, expected);
    }

    #[test]
    fn max_periods_is_zero() {
        let actual = get_new_prices::<u32>(555, 123, 27, 0);
        let expected = Err(GetNewPricesError::Overflow);

        assert_eq!(actual, expected);
    }

    #[test]
    fn max_periods_is_one() {
        let actual = get_new_prices::<u32>(555, 123, 3, 1);
        let expected = Ok(vec![123]);

        assert_eq!(actual, expected);
    }
}

#[cfg(test)]
mod index_range {
    use crate::*;

    #[test]
    fn empty_range() {
        let actual = get_index_range(&(3..3), &(0..1));
        let expected = Err(MathError::InvalidArgument);

        assert_eq!(actual, expected);
    }

    #[test]
    fn empty_intersaction() {
        let actual = get_index_range(&(0..5), &(4..4));
        let expected = Ok(0..0);

        assert_eq!(actual, expected);
    }

    #[test]
    fn intersaction_not_included_completely() {
        let actual = get_index_range(&(0..5), &(4..6));
        let expected = Err(MathError::InvalidArgument);

        assert_eq!(actual, expected);
    }
}

#[cfg(test)]
mod range_intersection {
    use crate::*;

    #[test]
    fn one_range() {
        let r = 0..10;
        let v: Vec<_> = vec![r.clone()];
        let actual = get_range_intersection(v.iter()).unwrap();
        let expected = r;

        assert_eq!(actual, expected);
    }

    #[test]
    fn empty_range() {
        let r = 0..0;
        let v: Vec<_> = vec![r.clone()];
        let actual = get_range_intersection(v.iter()).unwrap();
        let expected = r;

        assert_eq!(actual, expected);
    }

    #[test]
    fn empty_iterator() {
        let v: Vec<Range<i32>> = vec![];
        let actual = get_range_intersection(v.iter());
        let expected = Err(MathError::NotEnoughPoints);

        assert_eq!(actual, expected);
    }

    #[test]
    fn not_intersected_ranges() {
        let v: Vec<_> = vec![0..10, 11..12];
        let actual = get_range_intersection(v.iter());
        let expected = Err(MathError::NotEnoughPoints);

        assert_eq!(actual, expected);
    }

    #[test]
    fn two_intersected_ranges() {
        let v: Vec<_> = vec![0..10, 9..12];
        let actual = get_range_intersection(v.iter()).unwrap();
        let expected = 9..10;

        assert_eq!(actual, expected);
    }

    #[test]
    fn three_intersected_ranges() {
        let v: Vec<_> = vec![0..11, 9..12, 8..10];
        let actual = get_range_intersection(v.iter()).unwrap();
        let expected = 9..10;

        assert_eq!(actual, expected);
    }

    #[test]
    fn two_ranges_one_empty() {
        let v: Vec<_> = vec![0..10, 9..9];
        let actual = get_range_intersection(v.iter());
        let expected = Err(MathError::NotEnoughPoints);

        assert_eq!(actual, expected);
    }
}

#[cfg(test)]
mod stateful_calculations {
    use super::super::*;
    use super::{btc_prices, eos_prices};
    use crate::mock::FixedNumber;
    use approx::assert_abs_diff_eq;
    use frame_support::assert_ok;
    use substrate_fixed::traits::LossyInto;

    macro_rules! ret_tests {
        ($($name:ident: $value:expr,)*) => {
        $(
            #[test]
            fn $name() {
                let (return_type, prices, expected) = $value;
                let actual = Ret::<FixedNumber>::new(&prices, return_type);

                assert_ok!(&actual);
                let actual = actual.unwrap().ret;

                assert_eq!(actual.len(), expected.len());
                for (a, e) in actual.into_iter().zip(expected.into_iter()) {
                    let a: f64 = a.lossy_into();
                    assert_abs_diff_eq!(a, e, epsilon = 1e-8);
                }
            }
        )*
        }
    }

    ret_tests! {
        btc_regular_ret: (CalcReturnType::Regular, btc_prices(), vec![
            0.04390906,
            0.01631017,
            0.00252155,
            0.01452191,
            0.01506927,
            0.00147006,
            0.12731809,
            -0.01619013,
            0.02381692,
            0.01396878,
            -0.01013840,
            0.00170610,
            0.01014513,
            0.02951104,
            0.07366250,
            -0.01093803,
            -0.02527465,
            -0.08729193,
            -0.01765902,
            0.02356251,
            0.05287195,
            0.05002487,
            -0.04166136,
            0.00523252,
            0.03132448,
            0.00577308,
            0.00025394,
            -0.02117989,
            -0.04633214,
        ]),
        eos_regular_ret: (CalcReturnType::Regular, eos_prices(), vec![
            0.01908397,
            0.01872659,
            0.00000000,
            0.00735294,
            0.00364964,
            0.01090909,
            0.08633094,
            -0.06291391,
            0.02120141,
            0.02076125,
            -0.03728814,
            -0.02112676,
            -0.00359712,
            -0.02888087,
            0.02602230,
            0.00000000,
            0.00000000,
            -0.11231884,
            -0.01632653,
            0.01244813,
            0.03688525,
            0.03162055,
            -0.00766284,
            0.01544402,
            -0.00380228,
            0.01526718,
            -0.01127820,
            -0.01140684,
            -0.05000000,
        ]),
        btc_log_ret: (CalcReturnType::Log, btc_prices(), vec![
            0.042972378,
            0.016178588,
            0.002518380,
            0.014417479,
            0.014956852,
            0.001468981,
            0.119841444,
            -0.016322624,
            0.023537722,
            0.013872113,
            -0.010190141,
            0.001704649,
            0.010094014,
            0.029083976,
            0.071075698,
            -0.010998288,
            -0.025599536,
            -0.091339192,
            -0.017816804,
            0.023289199,
            0.051521618,
            0.048813845,
            -0.042554076,
            0.005218879,
            0.030843883,
            0.005756484,
            0.000253911,
            -0.021407402,
            -0.047439819,
        ]),
        eos_log_ret: (CalcReturnType::Log, eos_prices(), vec![
            0.018904155,
            0.018553408,
            0.000000000,
            0.007326040,
            0.003642991,
            0.010850016,
            0.082805904,
            -0.064980120,
            0.020979790,
            0.020548668,
            -0.038001118,
            -0.021353124,
            -0.003603608,
            -0.029306127,
            0.025689486,
            0.000000000,
            0.000000000,
            -0.119142655,
            -0.016461277,
            0.012371292,
            0.036221263,
            0.031130919,
            -0.007692346,
            0.015325970,
            -0.003809528,
            0.015151805,
            -0.011342277,
            -0.011472401,
            -0.051293294,
        ]),
    }

    macro_rules! vol_tests {
        ($($name:ident: $value:expr,)*) => {
        $(
            #[test]
            fn $name() {
                let (return_type, vol_cor_type, prices, expected) = $value;
                let ret = Ret::<FixedNumber>::new(&prices, return_type);

                assert_ok!(&ret);
                let ret = ret.unwrap();

                let actual = Vol::<FixedNumber>::new(&ret, vol_cor_type);

                assert_ok!(&actual);
                let actual: f64 = actual.unwrap().vol.lossy_into();

                assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
            }
        )*
        }
    }

    vol_tests! {
        btc_regular_regular_vol: (CalcReturnType::Regular, CalcVolatilityType::Regular, btc_prices(), 0.03957419747),
        eos_regular_regular_vol: (CalcReturnType::Regular, CalcVolatilityType::Regular, eos_prices(), 0.03545408186),
        btc_log_regular_vol: (CalcReturnType::Log, CalcVolatilityType::Regular, btc_prices(), 0.03897272754),
        eos_log_regular_vol: (CalcReturnType::Log, CalcVolatilityType::Regular, eos_prices(), 0.03608986052),
        btc_regular_exp_vol: (CalcReturnType::Regular, CalcVolatilityType::Exponential(36), btc_prices(), 0.03780489568156),
        eos_regular_exp_vol: (CalcReturnType::Regular, CalcVolatilityType::Exponential(36), eos_prices(), 0.03163066694536),
        btc_log_exp_vol: (CalcReturnType::Log, CalcVolatilityType::Exponential(36), btc_prices(), 0.03755128719775),
        eos_log_exp_vol: (CalcReturnType::Log, CalcVolatilityType::Exponential(36), eos_prices(), 0.03234405779893),
    }

    macro_rules! cor_tests {
        ($($name:ident: $value:expr,)*) => {
        $(
            #[test]
            fn $name() {
                let (return_type, vol_cor_type, prices1, prices2, expected) = $value;

                let ret1 = Ret::<FixedNumber>::new(&prices1, return_type);
                let ret2 = Ret::<FixedNumber>::new(&prices2, return_type);

                assert_ok!(&ret1);
                let ret1 = ret1.unwrap();
                assert_ok!(&ret2);
                let ret2 = ret2.unwrap();

                let vol1 = Vol::<FixedNumber>::new(&ret1, vol_cor_type);
                let vol2 = Vol::<FixedNumber>::new(&ret2, vol_cor_type);

                assert_ok!(&vol1);
                let vol1 = vol1.unwrap();
                assert_ok!(&vol2);
                let vol2 = vol2.unwrap();

                let actual = cor(&ret1, &vol1, &ret2, &vol2);

                assert_ok!(&actual);
                let actual: f64 = actual.unwrap().lossy_into();

                assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
            }
        )*
        }
    }

    cor_tests! {
        btc_eos_regular_regular_cor: (CalcReturnType::Regular, CalcVolatilityType::Regular, btc_prices(), eos_prices(), 0.83272869396),
        btc_eos_log_regular_cor: (CalcReturnType::Log, CalcVolatilityType::Regular, btc_prices(), eos_prices(), 0.83127977977),
        btc_eos_regular_exp_cor: (CalcReturnType::Regular, CalcVolatilityType::Exponential(36), btc_prices(), eos_prices(), 0.8417527336731),
        btc_eos_log_exp_cor: (CalcReturnType::Log, CalcVolatilityType::Exponential(36), btc_prices(), eos_prices(), 0.8414762672002),
    }
}

#[test]
fn calc_linear_return_for_btc_using_only_genesis() {
    new_test_ext().execute_with(|| {
        let actual: Vec<f64> =
            <FinancialModule as Financial>::calc_return(CalcReturnType::Regular, Asset::Btc)
                .unwrap()
                .into_iter()
                .map(|x| x.lossy_into())
                .collect();
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
        let actual: Vec<f64> =
            <FinancialModule as Financial>::calc_return(CalcReturnType::Log, Asset::Btc)
                .unwrap()
                .into_iter()
                .map(|x| x.lossy_into())
                .collect();
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
        let actual: Vec<f64> =
            <FinancialModule as Financial>::calc_return(CalcReturnType::Regular, Asset::Eos)
                .unwrap()
                .into_iter()
                .map(|x| x.lossy_into())
                .collect();
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
        let actual: Vec<f64> =
            <FinancialModule as Financial>::calc_return(CalcReturnType::Log, Asset::Eos)
                .unwrap()
                .into_iter()
                .map(|x| x.lossy_into())
                .collect();
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
        let prices: Vec<_> = vec![8_988.60, 8_897.47, 8_912.65, 9_003.07]
            .into_iter()
            .map(FixedNumber::from_num)
            .collect();

        let mut now = create_duration(2020, 9, 14, 12, 31, 0);
        set_now(now);

        for p in prices.into_iter() {
            let result = <FinancialModule as OnPriceSet>::on_price_set(Asset::Btc, p);

            assert_ok!(result);

            now = now + Duration::from_secs(60 * 60);
            set_now(now);
        }

        let actual: Vec<f64> =
            <FinancialModule as Financial>::calc_return(CalcReturnType::Regular, Asset::Btc)
                .unwrap()
                .into_iter()
                .map(|x| x.lossy_into())
                .collect();
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

        let log = FinancialModule::price_logs(asset);
        let expected_log = None;

        assert_eq!(updates, expected_updates);
        assert_eq!(log, expected_log);
    });
}

#[test]
fn initial_prices_state_with_nonempty_genesis() {
    new_test_ext().execute_with(|| {
        let asset = Asset::Btc;

        let updates = FinancialModule::updates(asset);
        let expected_updates = None;

        let log = FinancialModule::price_logs(asset).unwrap();
        let prices: Vec<_> = log.prices.iter().copied().collect();
        let expected_prices: Vec<FixedNumber> = initial_btc_prices()
            .into_iter()
            .map(FixedNumber::from_num)
            .collect();

        let log = FinancialModule::price_logs(asset).unwrap();
        let prices_cap = log.prices.len_cap();
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

        let log = FinancialModule::price_logs(asset).unwrap();
        let prices: Vec<_> = log.prices.iter().copied().collect();
        let expected_prices: Vec<FixedNumber> = vec![price];

        let prices_cap = log.prices.len_cap();
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
        let expected_result: Result<(), DispatchError> =
            Err(Error::<Test>::PeriodIsInThePast.into());
        assert_eq!(result, expected_result);

        let updates = FinancialModule::updates(asset);
        let expected_updates = Some(PriceUpdate::new(period_start, now, price));

        let log = FinancialModule::price_logs(asset).unwrap();
        let prices: Vec<_> = log.prices.iter().copied().collect();
        let expected_prices: Vec<FixedNumber> = vec![price];

        let prices_cap = log.prices.len_cap();
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

        assert_ok!(<FinancialModule as OnPriceSet>::on_price_set(
            asset,
            same_period_price
        ));

        let updates = FinancialModule::updates(asset);
        let expected_updates = Some(PriceUpdate::new(
            period_start,
            same_period_now,
            same_period_price,
        ));

        let log = FinancialModule::price_logs(asset).unwrap();
        let prices: Vec<_> = log.prices.iter().copied().collect();
        let expected_prices: Vec<FixedNumber> = vec![price];

        let prices_cap = log.prices.len_cap();
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

        assert_ok!(<FinancialModule as OnPriceSet>::on_price_set(
            asset,
            next_period_price
        ));

        let updates = FinancialModule::updates(asset);
        let expected_updates = Some(PriceUpdate::new(
            next_period_start,
            next_period_now,
            next_period_price,
        ));

        let log = FinancialModule::price_logs(asset).unwrap();
        let prices: Vec<_> = log.prices.iter().copied().collect();
        let expected_prices: Vec<FixedNumber> = vec![price, next_period_price];

        let prices_cap = log.prices.len_cap();
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

        assert_ok!(<FinancialModule as OnPriceSet>::on_price_set(
            asset,
            new_period_price
        ));

        let updates = FinancialModule::updates(asset);
        let expected_updates = Some(PriceUpdate::new(
            new_period_start,
            new_period_now,
            new_period_price,
        ));

        let log = FinancialModule::price_logs(asset).unwrap();
        let prices: Vec<_> = log.prices.iter().copied().collect();
        let expected_prices: Vec<FixedNumber> = vec![price, price, price, price, new_period_price];

        let prices_cap = log.prices.len_cap();
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

        assert_ok!(<FinancialModule as OnPriceSet>::on_price_set(
            asset,
            new_period_price
        ));

        let updates = FinancialModule::updates(asset);
        let expected_updates = Some(PriceUpdate::new(
            new_period_start,
            new_period_now,
            new_period_price,
        ));

        let log = FinancialModule::price_logs(asset).unwrap();
        let prices: Vec<_> = log.prices.iter().copied().collect();
        let mut expected_prices: Vec<FixedNumber> = vec![price; PriceCount::get() as usize - 1];
        expected_prices.push(new_period_price);

        let prices_cap = log.prices.len_cap();
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
        let expected_updates = Some(PriceUpdate::new(period_start, now, price));

        let log = FinancialModule::price_logs(asset).unwrap();
        let prices: Vec<_> = log.prices.iter().copied().collect();
        let expected_prices: Vec<FixedNumber> = vec![price];

        let prices_cap = log.prices.len_cap();
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

        let prices: Vec<_> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            .into_iter()
            .map(FixedNumber::from_num)
            .collect();

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

        assert_ok!(<FinancialModule as OnPriceSet>::on_price_set(
            asset,
            new_period_price
        ));

        let updates = FinancialModule::updates(asset);
        let expected_updates = Some(PriceUpdate::new(new_period_start, now, new_period_price));

        let log = FinancialModule::price_logs(asset).unwrap();
        let actual_prices: Vec<_> = log.prices.iter().copied().collect();
        let mut expected_prices: Vec<_> = prices[1..].iter().copied().collect();
        expected_prices.push(new_period_price);

        let prices_cap = log.prices.len_cap();
        let expected_price_cap = PriceCount::get();

        assert_eq!(updates, expected_updates);
        assert_eq!(actual_prices, expected_prices);
        assert_eq!(prices_cap, expected_price_cap);
    });
}

fn btc_prices() -> Vec<FixedNumber> {
    vec![
        7_117.21, 7_429.72, 7_550.90, 7_569.94, 7_679.87, 7_795.60, 7_807.06, 8_801.04, 8_658.55,
        8_864.77, 8_988.60, 8_897.47, 8_912.65, 9_003.07, 9_268.76, 9_951.52, 9_842.67, 9_593.90,
        8_756.43, 8_601.80, 8_804.48, 9_269.99, 9_733.72, 9_328.20, 9_377.01, 9_670.74, 9_726.57,
        9_729.04, 9_522.98, 9_081.76,
    ]
    .into_iter()
    .map(FixedNumber::from_num)
    .collect()
}

fn eos_prices() -> Vec<FixedNumber> {
    vec![
        2.62, 2.67, 2.72, 2.72, 2.74, 2.75, 2.78, 3.02, 2.83, 2.89, 2.95, 2.84, 2.78, 2.77, 2.69,
        2.76, 2.76, 2.76, 2.45, 2.41, 2.44, 2.53, 2.61, 2.59, 2.63, 2.62, 2.66, 2.63, 2.6, 2.47,
    ]
    .into_iter()
    .map(FixedNumber::from_num)
    .collect()
}

#[test]
fn calc_correlation_different_price_periods_results_in_periods_intersection() {
    new_test_ext().execute_with(|| {
        let prices: Vec<_> = vec![8_988.60, 8_897.47]
            .into_iter()
            .map(FixedNumber::from_num)
            .collect();

        let mut now = create_duration(2020, 9, 14, 12, 31, 0);
        set_now(now);

        for p in prices.into_iter() {
            let result = <FinancialModule as OnPriceSet>::on_price_set(Asset::Btc, p);

            assert_ok!(result);

            now = now + Duration::from_secs(60 * 60);
            set_now(now);
        }

        let actual = <FinancialModule as Financial>::calc_corr(
            CalcReturnType::Regular,
            CalcVolatilityType::Regular,
            Asset::Btc,
            Asset::Eos,
        );
        assert_ok!(&actual);
        let (_, actual_period) = actual.unwrap();
        let expected_period = Range {
            start: create_duration(2020, 9, 14, 4, 0, 0),
            end: create_duration(2020, 9, 14, 12, 0, 0),
        };

        assert_eq!(actual_period, expected_period);
    });
}

#[cfg(test)]
mod calc_vol {
    use crate::*;
    use approx::assert_abs_diff_eq;
    use frame_support::assert_ok;
    use mock::{alternative_test_ext, new_test_ext, Asset, FinancialModule, Test};
    use substrate_fixed::traits::LossyInto;

    #[test]
    fn calc_vol_btc_regular_regular() {
        new_test_ext().execute_with(|| {
            let actual = <FinancialModule as Financial>::calc_vol(
                CalcReturnType::Regular,
                CalcVolatilityType::Regular,
                Asset::Btc,
            );
            assert_ok!(&actual);
            let actual = actual.unwrap();
            let actual: f64 = actual.lossy_into();
            let expected = 0.04163212889;
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
        });
    }

    #[test]
    fn calc_vol_eos_regular_regular() {
        new_test_ext().execute_with(|| {
            let actual = <FinancialModule as Financial>::calc_vol(
                CalcReturnType::Regular,
                CalcVolatilityType::Regular,
                Asset::Eos,
            );
            assert_ok!(&actual);
            let actual = actual.unwrap();
            let actual: f64 = actual.lossy_into();
            let expected = 0.0380004985;
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
        });
    }

    #[test]
    fn calc_vol_btc_log_regular() {
        new_test_ext().execute_with(|| {
            let actual = <FinancialModule as Financial>::calc_vol(
                CalcReturnType::Log,
                CalcVolatilityType::Regular,
                Asset::Btc,
            );
            assert_ok!(&actual);
            let actual = actual.unwrap();
            let actual: f64 = actual.lossy_into();
            let expected = 0.03932894975;
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
        });
    }

    #[test]
    fn calc_vol_eos_log_regular() {
        new_test_ext().execute_with(|| {
            let actual = <FinancialModule as Financial>::calc_vol(
                CalcReturnType::Log,
                CalcVolatilityType::Regular,
                Asset::Eos,
            );
            assert_ok!(&actual);
            let actual = actual.unwrap();
            let actual: f64 = actual.lossy_into();
            let expected = 0.03764429688;
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
        });
    }

    #[test]
    fn calc_vol_btc_regular_exp() {
        new_test_ext().execute_with(|| {
            let actual = <FinancialModule as Financial>::calc_vol(
                CalcReturnType::Regular,
                CalcVolatilityType::Exponential(36),
                Asset::Btc,
            );
            assert_ok!(&actual);
            let actual = actual.unwrap();
            let actual: f64 = actual.lossy_into();
            let expected = 0.0294825719888474;
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
        });
    }

    #[test]
    fn calc_vol_btc_log_exp() {
        new_test_ext().execute_with(|| {
            let actual = <FinancialModule as Financial>::calc_vol(
                CalcReturnType::Log,
                CalcVolatilityType::Exponential(36),
                Asset::Btc,
            );
            assert_ok!(&actual);
            let actual = actual.unwrap();
            let actual: f64 = actual.lossy_into();
            let expected = 0.0282782458740207;
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
        });
    }

    #[test]
    fn calc_vol_eos_regular_exp() {
        new_test_ext().execute_with(|| {
            let actual = <FinancialModule as Financial>::calc_vol(
                CalcReturnType::Regular,
                CalcVolatilityType::Exponential(36),
                Asset::Eos,
            );
            assert_ok!(&actual);
            let actual = actual.unwrap();
            let actual: f64 = actual.lossy_into();
            let expected = 0.0246077778994953;
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
        });
    }

    #[test]
    fn calc_vol_eos_log_exp() {
        new_test_ext().execute_with(|| {
            let actual = <FinancialModule as Financial>::calc_vol(
                CalcReturnType::Log,
                CalcVolatilityType::Exponential(36),
                Asset::Eos,
            );
            assert_ok!(&actual);
            let actual = actual.unwrap();
            let actual: f64 = actual.lossy_into();
            let expected = 0.0245077247817629;
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
        });
    }

    #[test]
    fn calc_vol_asset_without_prices() {
        new_test_ext().execute_with(|| {
            let actual = <FinancialModule as Financial>::calc_vol(
                CalcReturnType::Log,
                CalcVolatilityType::Exponential(36),
                Asset::Usd,
            );
            let expected: Result<<mock::Test as Config>::Price, DispatchError> =
                Err(Error::<Test>::NotEnoughPoints.into());

            assert_eq!(actual, expected);
        });
    }

    #[test]
    fn calc_vol_regular_asset_with_big_prices() {
        new_test_ext().execute_with(|| {
            let actual = <FinancialModule as Financial>::calc_vol(
                CalcReturnType::Log,
                CalcVolatilityType::Regular,
                Asset::Eq,
            );
            assert_ok!(&actual);
            let actual = actual.unwrap();
            let actual: f64 = actual.lossy_into();
            let expected = 0.0;

            assert_eq!(actual, expected);
        });
    }

    #[test]
    fn calc_vol_regular_asset_with_prices_cause_log_error() {
        alternative_test_ext().execute_with(|| {
            let actual = <FinancialModule as Financial>::calc_vol(
                CalcReturnType::Log,
                CalcVolatilityType::Regular,
                Asset::Eq,
            );
            let expected: Result<<mock::Test as Config>::Price, DispatchError> =
                Err(Error::<Test>::Transcendental.into());

            assert_eq!(actual, expected);
        });
    }

    #[test]
    fn calc_vol_exponential_asset_with_big_prices() {
        new_test_ext().execute_with(|| {
            let actual = <FinancialModule as Financial>::calc_vol(
                CalcReturnType::Log,
                CalcVolatilityType::Exponential(36),
                Asset::Eq,
            );
            assert_ok!(&actual);
            let actual = actual.unwrap();
            let actual: f64 = actual.lossy_into();
            let expected = 0.0;

            assert_eq!(actual, expected);
        });
    }

    #[test]
    fn calc_vol_regular_asset_with_zero_prices() {
        new_test_ext().execute_with(|| {
            let actual = <FinancialModule as Financial>::calc_vol(
                CalcReturnType::Log,
                CalcVolatilityType::Regular,
                Asset::Eth,
            );
            let expected: Result<<mock::Test as Config>::Price, DispatchError> =
                Err(Error::<Test>::DivisionByZero.into());

            assert_eq!(actual, expected);
        });
    }

    #[test]
    fn calc_vol_exponential_asset_with_zero_prices() {
        new_test_ext().execute_with(|| {
            let actual = <FinancialModule as Financial>::calc_vol(
                CalcReturnType::Log,
                CalcVolatilityType::Exponential(36),
                Asset::Eth,
            );
            let expected: Result<<mock::Test as Config>::Price, DispatchError> =
                Err(Error::<Test>::DivisionByZero.into());

            assert_eq!(actual, expected);
        });
    }
}

#[cfg(test)]
mod calc_corr {
    use crate::*;
    use approx::assert_abs_diff_eq;
    use frame_support::assert_ok;
    use mock::{alternative_test_ext, new_test_ext, Asset, FinancialModule, Test};
    use substrate_fixed::traits::LossyInto;

    #[test]
    fn calc_corr_btc_eos_regular_regular() {
        new_test_ext().execute_with(|| {
            let actual = <FinancialModule as Financial>::calc_corr(
                CalcReturnType::Regular,
                CalcVolatilityType::Regular,
                Asset::Btc,
                Asset::Eos,
            );
            assert_ok!(&actual);
            let (actual, _) = actual.unwrap();
            let actual: f64 = actual.lossy_into();
            let expected = 0.8836901525;
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
        });
    }

    #[test]
    fn calc_corr_btc_eos_log_regular() {
        new_test_ext().execute_with(|| {
            let actual = <FinancialModule as Financial>::calc_corr(
                CalcReturnType::Log,
                CalcVolatilityType::Regular,
                Asset::Btc,
                Asset::Eos,
            );
            assert_ok!(&actual);
            let (actual, _) = actual.unwrap();
            let actual: f64 = actual.lossy_into();
            let expected = 0.87585873751;
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
        });
    }

    #[test]
    fn calc_corr_btc_eos_regular_exp() {
        new_test_ext().execute_with(|| {
            let actual = <FinancialModule as Financial>::calc_corr(
                CalcReturnType::Regular,
                CalcVolatilityType::Exponential(36),
                Asset::Btc,
                Asset::Eos,
            );
            assert_ok!(&actual);
            let (actual, _) = actual.unwrap();
            let actual: f64 = actual.lossy_into();
            let expected = 0.866920411417432;
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
        });
    }

    #[test]
    fn calc_corr_btc_eos_log_exponential() {
        new_test_ext().execute_with(|| {
            let actual = <FinancialModule as Financial>::calc_corr(
                CalcReturnType::Log,
                CalcVolatilityType::Exponential(36),
                Asset::Btc,
                Asset::Eos,
            );
            assert_ok!(&actual);
            let (actual, _) = actual.unwrap();
            let actual: f64 = actual.lossy_into();
            let expected = 0.860312211443148;
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
        });
    }

    #[test]
    fn calc_corr_asset_without_prices() {
        new_test_ext().execute_with(|| {
            let actual = <FinancialModule as Financial>::calc_corr(
                CalcReturnType::Log,
                CalcVolatilityType::Exponential(36),
                Asset::Btc,
                Asset::Usd,
            );

            let expected: Result<(<mock::Test as Config>::Price, Range<Duration>), DispatchError> =
                Err(Error::<Test>::NotEnoughPoints.into());

            assert_eq!(actual, expected);
        });
    }

    #[test]
    fn calc_corr_regular_asset_with_big_prices() {
        new_test_ext().execute_with(|| {
            let actual = <FinancialModule as Financial>::calc_corr(
                CalcReturnType::Log,
                CalcVolatilityType::Regular,
                Asset::Btc,
                Asset::Eq,
            );
            assert_ok!(&actual);
            let (actual, _) = actual.unwrap();
            let actual: f64 = actual.lossy_into();
            let expected = 0.0;

            assert_eq!(actual, expected);
        });
    }

    #[test]
    fn calc_corr_exponential_asset_with_big_prices() {
        new_test_ext().execute_with(|| {
            let actual = <FinancialModule as Financial>::calc_corr(
                CalcReturnType::Log,
                CalcVolatilityType::Exponential(36),
                Asset::Btc,
                Asset::Eq,
            );
            assert_ok!(&actual);
            let (actual, _) = actual.unwrap();
            let actual: f64 = actual.lossy_into();
            let expected = 0.0;

            assert_eq!(actual, expected);
        });
    }

    #[test]
    fn calc_corr_exponential_asset_with_prices_cause_log_error() {
        alternative_test_ext().execute_with(|| {
            let actual = <FinancialModule as Financial>::calc_corr(
                CalcReturnType::Log,
                CalcVolatilityType::Exponential(36),
                Asset::Btc,
                Asset::Eq,
            );
            let expected: Result<(<mock::Test as Config>::Price, Range<Duration>), DispatchError> =
                Err(Error::<Test>::Transcendental.into());

            assert_eq!(actual, expected);
        });
    }

    #[test]
    fn calc_corr_regular_asset_with_zero_prices() {
        new_test_ext().execute_with(|| {
            let actual = <FinancialModule as Financial>::calc_corr(
                CalcReturnType::Log,
                CalcVolatilityType::Regular,
                Asset::Btc,
                Asset::Eth,
            );
            let expected: Result<(<mock::Test as Config>::Price, Range<Duration>), DispatchError> =
                Err(Error::<Test>::DivisionByZero.into());

            assert_eq!(actual, expected);
        });
    }

    #[test]
    fn calc_corr_exponential_asset_with_zero_prices() {
        new_test_ext().execute_with(|| {
            let actual = <FinancialModule as Financial>::calc_corr(
                CalcReturnType::Log,
                CalcVolatilityType::Exponential(36),
                Asset::Btc,
                Asset::Eth,
            );
            let expected: Result<(<mock::Test as Config>::Price, Range<Duration>), DispatchError> =
                Err(Error::<Test>::DivisionByZero.into());

            assert_eq!(actual, expected);
        });
    }
}

#[cfg(test)]
mod calc_portf {
    use crate::*;
    use approx::assert_abs_diff_eq;
    use frame_support::assert_ok;
    use mock::{new_test_ext_btc_eos_only, set_balance, Asset, FinancialModule, FixedNumber};
    use substrate_fixed::traits::LossyInto;

    #[test]
    fn calc_portf_vol_btc_eos_regular_regular() {
        new_test_ext_btc_eos_only().execute_with(|| {
            set_balance(333, Asset::Btc, FixedNumber::from_num(10));
            set_balance(333, Asset::Eos, FixedNumber::from_num(10000));

            let actual = <FinancialModule as Financial>::calc_portf_vol(
                CalcReturnType::Regular,
                CalcVolatilityType::Regular,
                333,
            );
            assert_ok!(&actual);
            let actual = actual.unwrap();
            let actual: f64 = actual.lossy_into();
            let expected = 0.03989302699122;
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
        });
    }

    #[test]
    fn calc_portf_var_btc_eos_regular_regular() {
        new_test_ext_btc_eos_only().execute_with(|| {
            set_balance(333, Asset::Btc, FixedNumber::from_num(10));
            set_balance(333, Asset::Eos, FixedNumber::from_num(10000));

            let actual = <FinancialModule as Financial>::calc_portf_var(
                CalcReturnType::Regular,
                CalcVolatilityType::Regular,
                333,
                5,
            );
            assert_ok!(&actual);
            let actual = actual.unwrap();
            let actual: f64 = actual.lossy_into();
            let expected = 0.17744728149940;
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
        });
    }
}

#[cfg(test)]
mod calc_rv {
    use crate::*;

    use approx::assert_abs_diff_eq;
    use frame_support::assert_ok;
    use mock::{alternative_test_ext, Asset, FinancialModule, Test};
    use substrate_fixed::traits::LossyInto;

    #[test]
    fn calc_rv_btc_regular() {
        alternative_test_ext().execute_with(|| {
            let actual =
                <FinancialModule as Financial>::calc_rv(CalcReturnType::Regular, 36, Asset::Btc);
            assert_ok!(&actual);
            let actual = actual.unwrap();
            let actual: f64 = actual.lossy_into();
            let expected = 0.0383157575663070;
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
        });
    }

    #[test]
    fn calc_rv_btc_log() {
        alternative_test_ext().execute_with(|| {
            let actual =
                <FinancialModule as Financial>::calc_rv(CalcReturnType::Log, 36, Asset::Btc);
            assert_ok!(&actual);
            let actual = actual.unwrap();
            let actual: f64 = actual.lossy_into();
            let expected = 0.044161576895487;
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
        });
    }

    #[test]
    fn calc_rv_eos_regular() {
        alternative_test_ext().execute_with(|| {
            let actual =
                <FinancialModule as Financial>::calc_rv(CalcReturnType::Regular, 36, Asset::Eos);
            assert_ok!(&actual);
            let actual = actual.unwrap();
            let actual: f64 = actual.lossy_into();
            let expected = 0.0280740346113347;
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
        });
    }

    #[test]
    fn calc_rv_eos_log() {
        alternative_test_ext().execute_with(|| {
            let actual =
                <FinancialModule as Financial>::calc_rv(CalcReturnType::Log, 36, Asset::Eos);
            assert_ok!(&actual);
            let actual = actual.unwrap();
            let actual: f64 = actual.lossy_into();
            let expected = 0.028657030760318;
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
        });
    }

    #[test]
    fn calc_rv_asset_without_prices() {
        alternative_test_ext().execute_with(|| {
            let actual =
                <FinancialModule as Financial>::calc_rv(CalcReturnType::Regular, 36, Asset::Usd);
            let expected: Result<<mock::Test as Config>::Price, DispatchError> =
                Err(Error::<Test>::NotEnoughPoints.into());
            assert_eq!(actual, expected);
        });
    }

    #[test]
    fn calc_rv_asset_regular_with_small_prices() {
        alternative_test_ext().execute_with(|| {
            let actual =
                <FinancialModule as Financial>::calc_rv(CalcReturnType::Regular, 36, Asset::Dot);
            let expected: Result<<mock::Test as Config>::Price, DispatchError> =
                Err(Error::<Test>::Overflow.into());
            assert_eq!(actual, expected);
        });
    }

    #[test]
    fn calc_rv_asset_regular_with_big_prices() {
        alternative_test_ext().execute_with(|| {
            let actual =
                <FinancialModule as Financial>::calc_rv(CalcReturnType::Regular, 36, Asset::Eq);
            let expected: Result<<mock::Test as Config>::Price, DispatchError> =
                Err(Error::<Test>::Overflow.into());
            assert_eq!(actual, expected);
        });
    }

    #[test]
    fn calc_rv_asset_regular_with_zero_prices() {
        alternative_test_ext().execute_with(|| {
            let actual =
                <FinancialModule as Financial>::calc_rv(CalcReturnType::Regular, 36, Asset::Eth);
            let expected: Result<<mock::Test as Config>::Price, DispatchError> =
                Err(Error::<Test>::DivisionByZero.into());
            assert_eq!(actual, expected);
        });
    }

    #[test]
    fn calc_rv_asset_log_with_zero_prices() {
        alternative_test_ext().execute_with(|| {
            let actual =
                <FinancialModule as Financial>::calc_rv(CalcReturnType::Log, 36, Asset::Eth);
            let expected: Result<<mock::Test as Config>::Price, DispatchError> =
                Err(Error::<Test>::DivisionByZero.into());
            assert_eq!(actual, expected);
        });
    }

    #[test]
    fn calc_rv_asset_log_with_small_prices_cause_log_error() {
        alternative_test_ext().execute_with(|| {
            let actual =
                <FinancialModule as Financial>::calc_rv(CalcReturnType::Log, 36, Asset::Dot);
            let expected: Result<<mock::Test as Config>::Price, DispatchError> =
                Err(Error::<Test>::Transcendental.into());
            assert_eq!(actual, expected);
        });
    }

    #[test]
    fn calc_rv_asset_log_with_big_prices_cause_log_error() {
        alternative_test_ext().execute_with(|| {
            let actual =
                <FinancialModule as Financial>::calc_rv(CalcReturnType::Log, 36, Asset::Eq);
            let expected: Result<<mock::Test as Config>::Price, DispatchError> =
                Err(Error::<Test>::Transcendental.into());
            assert_eq!(actual, expected);
        });
    }
}
