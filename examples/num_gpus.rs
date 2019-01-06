#[macro_use]
extern crate mxnet_rs;

use mxnet_rs::ndarray::Context;


fn main() {
    println!("{}", Context::num_gpus());
    check_call!(0);
}