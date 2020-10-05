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
use frame_support::{assert_ok};
use financial::common::Asset;

#[test]
fn set_price_is_visible_through_storage() {
	new_test_ext().execute_with(|| {
		assert_ok!(OracleModule::set_price(Origin::signed(1), Asset::Btc, 42));
		assert_eq!(OracleModule::price_points(Asset::Btc), 42);
	});
}