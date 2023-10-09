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

#![cfg_attr(not(feature = "std"), no_std)]

use codec::{Decode, Encode};
use sp_std::ops::Range;
use sp_std::prelude::Vec;

enum OneOfThree<T1, T2, T3> {
    First(T1),
    Second(T2),
    Third(T3),
}

fn split_range(
    range: &Range<usize>,
    head: usize,
    length: usize,
) -> OneOfThree<(), Range<usize>, (Range<usize>, Range<usize>)> {
    assert!(
        range.start <= length,
        "range start index {} out of range for CapVec of length {}",
        range.start,
        length
    );
    assert!(
        range.end <= length,
        "range end index {} out of range for CapVec of length {}",
        range.start,
        length
    );

    if range.is_empty() {
        OneOfThree::First(())
    } else {
        let upper_start = range.start + head;
        let upper_end = range.end + head;

        if upper_start < length && upper_end <= length {
            // Range fits into the upper part
            OneOfThree::Second(upper_start..upper_end)
        } else {
            let lower_start = upper_start % length;
            let lower_end = upper_end % length;

            if lower_start < head && lower_end <= head {
                // Range fits into the lower part
                OneOfThree::Second(lower_start..lower_end)
            } else {
                // Range located in both upper and lower parts
                OneOfThree::Third((upper_start..length, 0..lower_end))
            }
        }
    }
}

/// Vector-like data structure which has maximum length.
///
/// `CapVec` can not contain more than `len_cap` elements. If `CapVec` already contains `len_cap`
/// elements and new one is pushed to the end, then the element at the beginning is simultaneously removed.
#[derive(Encode, Decode, Clone, Default, PartialEq, Eq, Debug, scale_info::TypeInfo)]
pub struct CapVec<T> {
    head_index: u32,
    len_cap: u32,
    items: Vec<T>,
}

use sp_std::fmt::Debug;
impl<T> CapVec<T> {
    /// Constructs a new, empty `CapVec<T>` with the specified maximum length.
    ///
    /// # Examples
    /// ```
    /// # use financial_primitives::capvec::CapVec;
    /// let mut items = CapVec::<u64>::new(2);
    /// ```
    pub fn new(length: u32) -> CapVec<T> {
        CapVec {
            head_index: 0,
            len_cap: length,
            items: Vec::new(),
        }
    }

    /// Appends an element to the back of the `CapVec`.
    ///
    /// If after appending collection's length is equal to `len_cap` then first element is removed
    /// automatically from the front.
    ///
    /// # Examples
    /// ```
    /// # use financial_primitives::capvec::CapVec;
    /// let mut items = CapVec::<u64>::new(2);
    /// items.push(1);
    /// ```
    pub fn push(&mut self, item: T) {
        if self.items.len() < (self.len_cap as usize) {
            self.items.push(item);
        } else {
            self.items[self.head_index as usize] = item;
            self.head_index = (self.head_index + 1) % (self.items.len() as u32);
        }
    }

    /// Returns an iterator over items.
    ///
    /// # Examples
    /// ```
    /// # use financial_primitives::capvec::CapVec;
    /// let mut items = CapVec::<u64>::new(2);
    /// items.push(1);
    /// items.push(2);
    /// items.push(3);
    /// let items_vec: Vec<_> = items.iter().copied().collect();
    /// assert_eq!(items_vec, vec![2, 3]);
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        let head = self.head_index as usize;
        let first_part = self.items[head..].iter();
        let last_part = self.items[..head].iter();

        first_part.chain(last_part)
    }

    /// Returns an iterator over specified range of items.
    ///
    /// # Examples
    /// ```
    /// # use financial_primitives::capvec::CapVec;
    /// let mut items = CapVec::<u64>::new(4);
    /// items.push(1);
    /// items.push(2);
    /// items.push(3);
    /// items.push(4);
    /// let items_vec: Vec<_> = items.iter_range(&(1..3)).copied().collect();
    /// assert_eq!(items_vec, vec![2, 3]);
    /// ```
    pub fn iter_range(&self, range: &Range<usize>) -> impl Iterator<Item = &T> {
        let ranges = split_range(range, self.head_index as usize, self.items.len());
        let (range1, range2) = match ranges {
            OneOfThree::First(_) => ((0..0), (0..0)),
            OneOfThree::Second(r) => (r, (0..0)),
            OneOfThree::Third((r1, r2)) => (r1, r2),
        };

        let first_part = self.items[range1].iter();
        let last_part = self.items[range2].iter();

        first_part.chain(last_part)
    }

    /// Returns maximum length of the collection.
    ///
    /// # Examples
    /// ```
    /// # use financial_primitives::capvec::CapVec;
    /// let mut items = CapVec::<u64>::new(2);
    /// assert_eq!(items.len_cap(), 2);
    /// ```
    pub fn len_cap(&self) -> u32 {
        self.len_cap
    }

    /// Returns current length of the collection.
    ///
    /// # Examples
    /// ```
    /// # use financial_primitives::capvec::CapVec;
    /// let mut items = CapVec::<u64>::new(5);
    /// items.push(1);
    /// items.push(2);
    /// assert_eq!(items.len(), 2);
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns the last element, or `None` if it is empty.
    ///
    /// # Examples
    /// ```
    /// # use financial_primitives::capvec::CapVec;
    /// let mut items = CapVec::<u64>::new(2);
    /// assert_eq!(items.last(), None);
    /// ```
    /// ```
    /// # use financial_primitives::capvec::CapVec;
    /// let mut items = CapVec::<u64>::new(2);
    /// items.push(1);
    /// items.push(2);
    /// items.push(3);
    /// assert_eq!(items.last(), Some(&3));
    /// ```
    pub fn last(&self) -> Option<&T> {
        let len = self.items.len();

        if len == 0 {
            None
        } else if len < (self.len_cap as usize) {
            Some(&self.items[len - 1])
        } else {
            let index = (self.len_cap + self.head_index - 1) % (self.len_cap);
            Some(&self.items[index as usize])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let capvec = CapVec::<u32>::new(5);

        let actual: Vec<_> = capvec.iter().cloned().collect();
        let expected: Vec<u32> = vec![];

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_not_capped() {
        let mut capvec = CapVec::<u32>::new(5);
        capvec.push(1);
        capvec.push(2);
        capvec.push(3);

        let actual: Vec<_> = capvec.iter().cloned().collect();
        let expected: Vec<u32> = vec![1, 2, 3];

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_capped() {
        let mut capvec = CapVec::<u32>::new(5);
        capvec.push(1);
        capvec.push(2);
        capvec.push(3);
        capvec.push(4);
        capvec.push(5);
        capvec.push(6);

        let actual: Vec<_> = capvec.iter().cloned().collect();
        let expected: Vec<u32> = vec![2, 3, 4, 5, 6];

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_last_empty() {
        let capvec = CapVec::<u32>::new(5);

        let actual = capvec.last();
        let expected = None;

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_last_non_empty() {
        let mut capvec = CapVec::<u32>::new(5);
        capvec.push(1);
        capvec.push(2);
        capvec.push(3);
        capvec.push(4);
        capvec.push(5);
        capvec.push(6);

        let actual = capvec.last();
        let expected = Some(&6);

        assert_eq!(actual, expected);
    }

    fn from_vec<T: Debug>(cap: u32, items: Vec<T>) -> CapVec<T> {
        let mut capvec = CapVec::<T>::new(cap);
        for item in items.into_iter() {
            capvec.push(item);
        }

        capvec
    }

    #[test]
    fn test_iter_range_full_range() {
        let capvec = from_vec(5, vec![1, 2, 3, 4, 5]);

        let actual: Vec<_> = capvec.iter_range(&(0..5)).copied().collect();
        let expected = vec![1, 2, 3, 4, 5];

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_iter_range_empty_range() {
        let capvec = from_vec(5, vec![1, 2, 3, 4, 5]);

        let actual: Vec<_> = capvec.iter_range(&(0..0)).copied().collect();
        let expected: Vec<i32> = vec![];

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_iter_range_first_part_range() {
        let capvec = from_vec(6, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);

        let actual: Vec<_> = capvec.iter_range(&(1..3)).copied().collect();
        let expected = vec![5, 6];

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_iter_range_second_part_range() {
        let capvec = from_vec(6, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);

        let actual: Vec<_> = capvec.iter_range(&(4..6)).copied().collect();
        let expected = vec![8, 9];

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_iter_range_both_parts_range() {
        let capvec = from_vec(6, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);

        let actual: Vec<_> = capvec.iter_range(&(2..5)).copied().collect();
        let expected = vec![6, 7, 8];

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_iter_range_small_range_start_on_zero() {
        let capvec = from_vec(5, vec![1, 2, 3, 4, 5]);

        let actual: Vec<_> = capvec.iter_range(&(1..3)).copied().collect();
        let expected = vec![2, 3];

        assert_eq!(actual, expected);
    }

    #[test]
    #[should_panic(expected = "index out of bounds: the len is 0 but the index is 0")]
    fn test_zero_cap() {
        let mut capvec = CapVec::<u32>::new(0);
        capvec.push(1);
    }
}
