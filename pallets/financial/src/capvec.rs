#![cfg_attr(not(feature = "std"), no_std)]

use frame_support::codec::{Decode, Encode};
use sp_std::prelude::Vec;

#[derive(Encode, Decode, Clone, Default)]
pub struct CapVec<T> {
    head_index: u32,
    len_cap: u32,
	items: Vec<T>,
}

impl <T> CapVec<T> {
	pub fn new(length: u32) -> CapVec<T> {
		CapVec {
            head_index: 0,
            len_cap: length,
			items: Vec::new(),
		}
	}

	pub fn push(&mut self, item: T) {
        if self.items.len() < (self.len_cap as usize) {
            self.items.push(item);
        } else {
            self.items[self.head_index as usize] = item;
            self.head_index = (self.head_index + 1) % (self.items.len() as u32);
        }
	}

	pub fn iter(&self) -> impl Iterator<Item = &T> {
		let head = self.head_index as usize;
		let first_part = self.items[head..].iter();
		let last_part = self.items[..head].iter();

		first_part.chain(last_part)
    }
    
    pub fn len_cap(&self) -> u32 {
        self.len_cap
    }

    pub fn last(&self) -> Option<& T> {
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
}