use mxnet_sys::{
    AtomicSymbolCreator, MXSymbolGetAtomicSymbolInfo, MXSymbolListAtomicSymbolCreators,
    NNGetOpHandle, NNListAllOpNames, OpHandle,
};
use std::collections::HashMap;
use std::ffi::{c_void, CStr};
use std::ptr;
use std::slice;

pub const OP_NAME_PREFIX_LIST: &[&'static str; 5] =
    &["_contrib_", "_linalg_", "_sparse_", "_image_", "_random_"];

#[macro_export]
macro_rules! check_call {
    ($mx_call:expr) => {
        unsafe {
            if $mx_call != 0 {
                panic!(std::ffi::CStr::from_ptr(mxnet_sys::MXGetLastError())
                    .to_str()
                    .unwrap());
            }
        }
    };
}





#[cfg(test)]
mod tests {
    use super::*;


}
