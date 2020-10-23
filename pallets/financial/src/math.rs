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

use core::ops::{AddAssign, BitOrAssign, ShlAssign};
use substrate_fixed::traits::{Fixed, FixedSigned, ToFixed};
use substrate_fixed::transcendental::{ln, sqrt};
use substrate_fixed::types::I9F23;

// Type of constants for transcendental operations declared in substrate_fixed crate
pub type ConstType = I9F23;

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum MathError {
    Overflow,
    DivisionByZero,
    Transcendental,
    NotEnoughPoints,
    InvalidArgument,
}

pub type MathResult<T> = Result<T, MathError>;

/// Indicates what returns will be used in calculations of volatilities, correlations, and value at
/// risk: `Regular` or `Log` returns.
///
/// The choice of return type also governs the method for Value at Risk (VAR) calculation:
/// * `Regular` type should be used when arithmetic returns are used and are assumed to be normally
/// distributed;
/// * `Log` normal type should be used when geometric returns (log returns) are used
/// and are assumed to be normally distributed.
///
/// We suggest using the latter approach, as it doesn't
/// lead to losses greater than a portfolio value unlike the normal VaR.
#[derive(Copy, Clone)]
pub enum CalcReturnType {
    /// Regular returns.
    Regular,

    /// Log returns.
    Log,
}

/// Indicates the method for calculating volatility: `Regular` or `Exponential`.
#[derive(Copy, Clone)]
pub enum CalcVolatilityType {
    /// Regular type is a standard statistical approach of calculating standard deviation of
    /// returns using simple average.
    Regular,

    /// Exponentially weighted type gives more weight to most recent data given the decay value or
    /// period of exponentially weighted moving average.
    Exponential(u32),
}

/// Indicates the method for calculating correlations: `Regular` or `Exponential`.
///
/// * `Regular` type is a standard statistical approach of calculating Pearson correlation coefficient between
/// two return series;
/// * `Exponential` type uses exponentially weighted moving average
/// to give more weights to most recent observations of pair of returns.
///
/// Note: correlations on lower periods are more susceptible to data noise and reflect actual
/// dependencies between returns poorly. For robustness we recommend to use daily or weekly periods
/// for correlations.
#[derive(Copy, Clone)]
pub enum CalcCorrelationType {
    /// Regular correlation type.
    Regular,
    /// Exponentially weighted correlation type.
    Exponential(u32),
}

pub fn calc_return<F: Fixed>(x1: F, x2: F) -> Result<F, MathError> {
    let ratio = x2.checked_div(x1).ok_or(MathError::DivisionByZero)?;

    ratio.checked_sub(F::from_num(1)).ok_or(MathError::Overflow)
}

pub fn calc_log_return<F>(x1: F, x2: F) -> Result<F, MathError>
where
    F: FixedSigned + PartialOrd<ConstType> + From<ConstType>,
    F::Bits: Copy + ToFixed + AddAssign + BitOrAssign + ShlAssign,
{
    let ratio = x2.checked_div(x1).ok_or(MathError::DivisionByZero)?;
    ln(ratio).map_err(|_| MathError::Transcendental)
}

pub fn calc_return_exp_vola<F: Fixed>(x1: F, x2: F) -> Result<F, MathError> {
    x2.checked_sub(x1).ok_or(MathError::Overflow)
}

pub fn calc_return_func<F>(return_type: CalcReturnType) -> impl Fn(F, F) -> MathResult<F> + Copy
where
    F: FixedSigned + PartialOrd<ConstType> + From<ConstType>,
    F::Bits: Copy + ToFixed + AddAssign + BitOrAssign + ShlAssign,
{
    match return_type {
        CalcReturnType::Regular => calc_return,
        CalcReturnType::Log => calc_log_return,
    }
}

pub fn calc_return_func_exp_vola<F>(
    return_type: CalcReturnType,
) -> impl Fn(F, F) -> MathResult<F> + Copy
where
    F: FixedSigned + PartialOrd<ConstType> + From<ConstType>,
    F::Bits: Copy + ToFixed + AddAssign + BitOrAssign + ShlAssign,
{
    match return_type {
        CalcReturnType::Regular => calc_return_exp_vola,
        CalcReturnType::Log => calc_log_return,
    }
}

pub fn calc_return_iter<'a, T: Copy, F: 'a + Fn(T, T) -> MathResult<T>>(
    items: &'a [T],
    return_func: F,
) -> impl Iterator<Item = MathResult<T>> + 'a {
    items.windows(2).map(move |x| return_func(x[0], x[1]))
}

pub fn decay<F: Fixed>(ewma_length: u32) -> MathResult<F> {
    let nom = F::from_num(2);
    let denom = F::from_num(1)
        .checked_add(F::from_num(ewma_length))
        .ok_or(MathError::Overflow)?;

    nom.checked_div(denom).ok_or(MathError::DivisionByZero)
}

fn recurrent_ewma_step<F: Fixed>(
    diff: MathResult<F>,
    decay: F,
    prev_var: MathResult<F>,
) -> MathResult<F> {
    let diff = diff?;
    let prev_var = prev_var?;

    let a = diff.checked_mul(decay).ok_or(MathError::Overflow)?;
    let b = F::from_num(1)
        .checked_sub(decay)
        .ok_or(MathError::Overflow)?;
    let c = b.checked_mul(prev_var).ok_or(MathError::Overflow)?;

    a.checked_add(c).ok_or(MathError::Overflow)
}

pub fn last_recurrent_ewma<F: Fixed, I: Iterator<Item = MathResult<F>>>(
    diffs: I,
    decay: F,
) -> MathResult<F> {
    diffs
        .fold(None, move |state, diff| {
            if let Some(prev_var) = state {
                let var = recurrent_ewma_step(diff, decay, prev_var);
                Some(var)
            } else {
                Some(diff)
            }
        })
        .ok_or(MathError::NotEnoughPoints)
        .and_then(|x| x)
}

pub fn sum<'a, F: 'a + Fixed, I: 'a + Iterator<Item = MathResult<F>>>(
    mut iter: I,
) -> MathResult<F> {
    Ok(iter.try_fold(F::from_num(0), |acc, x| {
        let x = x?;
        acc.checked_add(x).ok_or(MathError::Overflow)
    })?)
}

pub fn mean<F: Fixed>(items: &[F]) -> MathResult<F> {
    let sum = sum(items.iter().map(|&x| Ok(x)))?;
    sum.checked_div(F::from_num(items.len()))
        .ok_or(MathError::DivisionByZero)
}

pub fn demeaned<'a, F: 'a + Fixed, I: 'a + Iterator<Item = &'a F>>(
    iter: I,
    mean: F,
) -> impl Iterator<Item = MathResult<F>> + 'a {
    iter.map(move |&x| x.checked_sub(mean).ok_or(MathError::Overflow))
}

pub fn squared<'a, F: 'a + Fixed, I: 'a + Iterator<Item = MathResult<F>>>(
    iter: I,
) -> impl Iterator<Item = MathResult<F>> + 'a {
    iter.map(|x| x.and_then(|y| y.checked_mul(y).ok_or(MathError::Overflow)))
}

pub fn regular_vola<F: Fixed>(n: usize, sum: F) -> MathResult<F> {
    let a = n.checked_sub(1).ok_or(MathError::Overflow)?;
    let b = F::from_num(1)
        .checked_div(F::from_num(a))
        .ok_or(MathError::DivisionByZero)?;
    let c = b.checked_mul(sum).ok_or(MathError::Overflow)?;
    Ok(c)
}

pub fn exp_vola<F>(return_type: CalcReturnType, var: F, price: F) -> MathResult<F>
where
    F: Fixed + PartialOrd<ConstType>,
{
    let a: F = sqrt(var).map_err(|_| MathError::Transcendental)?;
    match return_type {
        CalcReturnType::Regular => a.checked_div(price).ok_or(MathError::DivisionByZero),
        CalcReturnType::Log => Ok(a),
    }
}

pub fn mul<'a, F: 'a + Fixed, I: 'a + Iterator<Item = F>>(
    iter1: I,
    iter2: I,
) -> impl Iterator<Item = MathResult<F>> + 'a {
    iter1
        .zip(iter2)
        .map(|(x1, x2)| x1.checked_mul(x2).ok_or(MathError::Overflow))
}

pub fn regular_corr<F: Fixed>(n: usize, sum: F, vol1: F, vol2: F) -> MathResult<F> {
    let a = n.checked_sub(1).ok_or(MathError::Overflow)?;
    let b = F::from_num(1)
        .checked_div(F::from_num(a))
        .ok_or(MathError::DivisionByZero)?;
    let c = b.checked_mul(sum).ok_or(MathError::Overflow)?;
    let d = c.checked_div(vol1).ok_or(MathError::DivisionByZero)?;
    let e = d.checked_div(vol2).ok_or(MathError::DivisionByZero)?;
    Ok(e)
}

pub fn exp_corr<F: Fixed>(covariance: F, vol1: F, vol2: F) -> MathResult<F> {
    let d = covariance
        .checked_div(vol1)
        .ok_or(MathError::DivisionByZero)?;
    let e = d.checked_div(vol2).ok_or(MathError::DivisionByZero)?;
    Ok(e)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use frame_support::assert_ok;
    use substrate_fixed::traits::LossyInto;
    use substrate_fixed::types::I64F64;

    pub type FixedNumber = I64F64;

    #[test]
    fn squared_example() {
        let diffs: Vec<_> = vec![
            312.51, 121.18, 19.04, 109.93, 115.73, 11.46, 993.98, -142.49, 206.22, 123.83, -91.13,
            15.18, 90.42, 265.69, 682.76, -108.85, -248.77, -837.47, -154.63, 202.68, 465.51,
            463.73, -405.52, 48.81, 293.73, 55.83, 2.47, -206.06, -441.22,
        ]
        .into_iter()
        .map(|x| Ok(FixedNumber::from_num(x)))
        .collect();

        let actual: MathResult<Vec<_>> = squared(diffs.into_iter()).collect();
        assert_ok!(&actual);
        let actual: Vec<f64> = actual
            .unwrap()
            .into_iter()
            .map(|x| x.lossy_into())
            .collect();

        let expected: Vec<_> = vec![
            97662.5001000001,
            14684.5923999999,
            362.5216000000,
            12084.6049000001,
            13393.4329000001,
            131.3316000000,
            987996.2404000010,
            20303.4001000005,
            42526.6884000005,
            15333.8689000000,
            8304.6769000002,
            230.4324000000,
            8175.7764000000,
            70591.1761000003,
            466161.2176000000,
            11848.3225000001,
            61886.5129000002,
            701356.0008999990,
            23910.4369000003,
            41079.1824000001,
            216699.5601000000,
            215045.5129000000,
            164446.4703999990,
            2382.4161000000,
            86277.3128999997,
            3116.9889000000,
            6.1009000000,
            42460.7236000005,
            194675.0883999990,
        ];
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.into_iter().zip(expected.into_iter()) {
            assert_abs_diff_eq!(a, e, epsilon = 1e-8);
        }
    }

    #[test]
    fn last_recurrent_ewma_example() {
        let squared_diffs: Vec<_> = vec![
            97662.50010000010000,
            14684.59239999990000,
            362.52159999999900,
            12084.60490000010000,
            13393.43290000010000,
            131.33160000000100,
            987996.24040000100000,
            20303.40010000050000,
            42526.68840000050000,
            15333.86890000000000,
            8304.67690000019000,
            230.43240000000900,
            8175.77640000001000,
            70591.17610000030000,
            466161.21760000000000,
            11848.32250000010000,
            61886.51290000020000,
            701356.00089999900000,
            23910.43690000030000,
            41079.18240000010000,
            216699.56010000000000,
            215045.51290000000000,
            164446.47039999900000,
            2382.41609999995000,
            86277.31289999970000,
            3116.98889999999000,
            6.10090000000575,
            42460.72360000050000,
            194675.08839999900000,
        ]
        .into_iter()
        .map(|x| Ok(FixedNumber::from_num(x)))
        .collect();
        let decay = FixedNumber::from_num(0.0540540540540541);

        let actual = last_recurrent_ewma(squared_diffs.into_iter(), decay);
        assert_ok!(&actual);
        let actual: f64 = actual.unwrap().lossy_into();

        let expected = 114954.3177623650;
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
    }

    #[test]
    fn recurrent_ewma_step_example() {
        let diff = Ok(FixedNumber::from_num(14684.59239999990000));
        let decay = FixedNumber::from_num(0.0540540540540541);
        let prev_var = Ok(FixedNumber::from_num(97662.5001000001));
        let actual = recurrent_ewma_step(diff, decay, prev_var);
        assert_ok!(&actual);
        let actual: f64 = actual.unwrap().lossy_into();
        let expected = 93177.2077918920;

        assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
    }

    #[test]
    fn exp_vola_expample() {
        let variance = FixedNumber::from_num(114954.3177623650);
        let price = FixedNumber::from_num(9_081.76);
        let actual = exp_vola(CalcReturnType::Regular, variance, price);
        assert_ok!(actual);
        let actual: FixedNumber = actual.unwrap();
        let actual: f64 = actual.lossy_into();
        let expected = 0.0373329770530381;

        assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
    }

    #[cfg(test)]
    mod recurrent_ewma_step {
        use super::super::*;
        use super::FixedNumber;
        use approx::assert_abs_diff_eq;
        use frame_support::assert_ok;
        use substrate_fixed::traits::LossyInto;

        #[test]
        fn recurrent_ewma_step_diff_is_zero() {
            let diff = Ok(FixedNumber::from_num(0));
            let decay = FixedNumber::from_num(0.054);
            let prev_var = Ok(FixedNumber::from_num(9762.51));
            let actual = recurrent_ewma_step(diff, decay, prev_var);
            assert_ok!(&actual);
            let actual: f64 = actual.unwrap().lossy_into();
            let expected = 9235.33446;

            assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
        }

        #[test]
        fn recurrent_ewma_step_mul_overflow() {
            let diff = Ok(FixedNumber::max_value());
            let decay = FixedNumber::from_num(2);
            let prev_var = Ok(FixedNumber::from_num(9762.51));
            let actual = recurrent_ewma_step(diff, decay, prev_var);
            let expected = Err(MathError::Overflow);

            assert_eq!(actual, expected);
        }

        #[test]
        fn recurrent_ewma_step_add_overflow() {
            let diff = Ok(FixedNumber::from_num(2));
            let decay = FixedNumber::max_value();
            let prev_var = Ok(FixedNumber::from_num(9762.51));
            let actual = recurrent_ewma_step(diff, decay, prev_var);
            let expected = Err(MathError::Overflow);

            assert_eq!(actual, expected);
        }

        #[test]
        fn recurrent_ewma_step_valid_negative() {
            let diff = Ok(FixedNumber::from_num(-0.051));
            let decay = FixedNumber::from_num(-0.054);
            let prev_var = Ok(FixedNumber::from_num(-976.51));
            let actual = recurrent_ewma_step(diff, decay, prev_var);
            assert_ok!(&actual);
            let actual: f64 = actual.unwrap().lossy_into();
            let expected = -1029.238786;

            assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
        }

        #[test]
        fn recurrent_ewma_step_valid_zero() {
            let diff = Ok(FixedNumber::from_num(0.0));
            let decay = FixedNumber::from_num(0.0);
            let prev_var = Ok(FixedNumber::from_num(0.0));
            let actual = recurrent_ewma_step(diff, decay, prev_var);
            assert_ok!(&actual);
            let actual: f64 = actual.unwrap().lossy_into();
            let expected = 0.0;

            assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
        }

        #[test]
        fn recurrent_ewma_step_example() {
            let diff = Ok(FixedNumber::from_num(14684.59239999990000));
            let decay = FixedNumber::from_num(0.0540540540540541);
            let prev_var = Ok(FixedNumber::from_num(97662.5001000001));
            let actual = recurrent_ewma_step(diff, decay, prev_var);
            assert_ok!(&actual);
            let actual: f64 = actual.unwrap().lossy_into();
            let expected = 93177.2077918920;

            assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
        }
    }

    #[cfg(test)]
    mod decay {
        use super::super::*;
        use super::FixedNumber;
        use approx::assert_abs_diff_eq;
        use frame_support::assert_ok;
        use substrate_fixed::traits::LossyInto;

        #[test]
        fn decay_zero() {
            let actual = decay(0);
            assert_ok!(actual);
            let actual: FixedNumber = actual.unwrap();
            let actual: f64 = actual.lossy_into();
            let expected = 2.0;

            assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
        }

        #[test]
        fn decay_simple() {
            let actual = decay(1);
            assert_ok!(actual);
            let actual: FixedNumber = actual.unwrap();
            let actual: f64 = actual.lossy_into();
            let expected = 1.0;

            assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
        }

        #[test]
        fn decay_expample() {
            let actual = decay(36);
            assert_ok!(actual);
            let actual: FixedNumber = actual.unwrap();
            let actual: f64 = actual.lossy_into();
            let expected = 0.05405405405;

            assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
        }
    }

    #[cfg(test)]
    mod calc_return {
        use super::super::*;
        use super::FixedNumber;
        use approx::assert_abs_diff_eq;
        use frame_support::assert_ok;
        use substrate_fixed::traits::LossyInto;

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
        fn calc_return_iter_empty() {
            let prices: Vec<FixedNumber> = Vec::new();
            let func = calc_return_func(CalcReturnType::Regular);

            let actual: MathResult<Vec<FixedNumber>> = calc_return_iter(&prices, func).collect();
            let expected = Ok(vec![]);

            assert_eq!(actual, expected);
        }

        #[test]
        fn calc_return_vec_one_item() {
            let prices: Vec<FixedNumber> =
                vec![1.5].into_iter().map(FixedNumber::from_num).collect();
            let func = calc_return_func(CalcReturnType::Regular);

            let actual: MathResult<Vec<FixedNumber>> = calc_return_iter(&prices, func).collect();
            let expected = Ok(vec![]);

            assert_eq!(actual, expected);
        }

        #[test]
        fn calc_return_vec_regular_valid() {
            let prices: Vec<FixedNumber> = vec![
                7_117.21, 7_429.72, 7_550.90, 7_569.94, 7_679.87, 7_795.60, 7_807.06, 8_801.04,
                8_658.55, 8_864.77,
            ]
            .into_iter()
            .map(FixedNumber::from_num)
            .collect();
            let func = calc_return_func(CalcReturnType::Regular);

            let actual: MathResult<Vec<FixedNumber>> = calc_return_iter(&prices, func).collect();
            assert_ok!(&actual);
            let actual: Vec<f64> = actual
                .unwrap()
                .into_iter()
                .map(|x| x.lossy_into())
                .collect();

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
                7_117.21, 7_429.72, 7_550.90, 7_569.94, 7_679.87, 7_795.60, 7_807.06, 8_801.04,
                8_658.55, 8_864.77,
            ]
            .into_iter()
            .map(FixedNumber::from_num)
            .collect();
            let func = calc_return_func(CalcReturnType::Log);

            let actual: MathResult<Vec<FixedNumber>> = calc_return_iter(&prices, func).collect();
            assert_ok!(&actual);
            let actual: Vec<f64> = actual
                .unwrap()
                .into_iter()
                .map(|x| x.lossy_into())
                .collect();

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
}
