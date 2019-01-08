use mxnet_sys::AtomicSymbolCreator;
use std::collections::HashMap;
use std::ptr;

pub struct Operator {
    handle: AtomicSymbolCreator,
}

impl Operator {
    pub fn new(operator_name: &str) -> Operator {
        let creator: AtomicSymbolCreator = ptr::null_mut();
        Operator { handle: creator }
    }
}
