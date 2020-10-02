# Financial Pallet

## Overview

Equilibrium's financial pallet is an open-source substrate module that subscribes to external price feed/oracle, gathers asset prices and calculates financial metrics based on the information collected.  

## Documentation

For the detailed documentation please refer to [doc/README.md](./doc/README.md).

## Prepare Build Environment

Make sure you have done all steps described in [Installation page](https://substrate.dev/docs/en/knowledgebase/getting-started/) of the Substrate Developer Hub.

## Run Tests

Clone this repository first

```bash
git clone https://github.com/equilibrium-eosdt/equilibrium-financial-pallet
cd equilibrium-financial-pallet
```

To run tests simply type:

```bash
cargo test
```

Here are full list of unit tests for the financial pallet:

```
test capvec::tests::test_last_empty
test capvec::tests::test_last_non_empty
test capvec::tests::test_empty
test capvec::tests::test_capped
test capvec::tests::test_not_capped
test tests::calc_return::calc_log_return_x1_is_zero
test tests::calc_return::calc_log_return_x1_is_negative
test tests::calc_return::calc_log_return_valid
test tests::calc_return::calc_return_valid
test tests::calc_return::calc_return_vec_empty
test tests::calc_return::calc_return_vec_one_item
test tests::calc_return::calc_return_vec_linear_valid
test tests::new_prices::max_periods_is_zero
test tests::calc_return::calc_return_x1_is_zero
test tests::new_prices::no_last_price_no_empty_periods
test tests::calc_return::calc_return_vec_log_valid
test tests::new_prices::no_last_price_some_empty_periods
test tests::new_prices::some_last_price_no_empty_periods
test tests::new_prices::some_last_price_some_empty_periods_equal_to_max
test tests::new_prices::some_last_price_some_empty_periods_significantly_more_than_max
test tests::new_prices::some_last_price_some_empty_periods_less_than_max
test tests::price_period::elapsed_period_count_is_too_large
test tests::price_period::info_for_neighbour_periods
test tests::new_prices::some_last_price_some_empty_periods_slightly_more_than_max
test tests::price_period::invalid_period_start
test tests::price_period::period_changed_significantly
test tests::price_period::period_changed_slightly
test tests::price_period::period_id_are_different_for_different_periods
test tests::price_period::period_id_are_same_within_period
test tests::price_period::period_is_in_the_past
test tests::price_period::period_remains_unchanged
test tests::price_period::zero_period
test tests::price_period::info_for_distant_periods
test tests::calc_linear_return_for_eos_using_only_genesis
test tests::calc_linear_return_for_btc_using_only_genesis
test tests::calc_log_return_for_btc_using_only_genesis
test tests::calc_log_return_for_eos_using_only_genesis
test tests::calc_linear_return_for_btc_using_some_oracle_prices
```