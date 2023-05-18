use std::str::FromStr;
use std::env;

type FixedNumber = substrate_fixed::types::I64F64;

fn main() {
    let args: Vec<String> = env::args().collect();
    let str = &args[1];
    let cleaned_str = str
        .replace(",", "")
        .replace(".", "");

    let raw: i128 = i128::from_str(&cleaned_str).unwrap();
    let num = FixedNumber::from_bits(raw);

    println!("{:?}", num);
}
