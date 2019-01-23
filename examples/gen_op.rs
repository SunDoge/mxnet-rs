#[macro_use]
extern crate mxnet_rs;

use codegen::Scope;
use mxnet_sys::*;
use std::collections::HashMap;
use std::ffi;
use std::ptr;

fn main() {
    let mut scope = Scope::new();

    let mut plist = ptr::null_mut();
    let mut size = 0;
    check_call!(MXListAllOpNames(&mut size, &mut plist));
    let mut op_names: Vec<&ffi::CStr> = Vec::new();
    // [TODO] macro for it.
    for i in 0..size as isize {
        op_names.push(unsafe { ffi::CStr::from_ptr(*plist.offset(i)) });
    }
    println!("{:?}", op_names);
    let mut module_op = scope.new_module("op");
    let mut modele_internal = scope.new_module("internal");

    let mut submodule_dict = HashMap::new();

    for op_name_prefix in mxnet_rs::base::OP_NAME_PREFIX_LIST {
        submodule_dict.insert(
            op_name_prefix.to_string(),
            op_name_prefix[1..op_name_prefix.len()].to_string(),
        );
    }

    for name in op_names {
        let mut hdl = ptr::null_mut();
        check_call!(NNGetOpHandle(name.as_ptr(), &mut hdl));
        
    }
}
