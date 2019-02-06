#[macro_use]
extern crate mxnet_rs;

use codegen::Scope;
use mxnet_sys::*;
use std::collections::HashMap;
use std::ffi;
use std::ptr;
use std::slice;

fn main() {
    init_op_module("ndarray", make_ndarray_function);
}

fn init_op_module(module_name: &str, make_op_func: fn(NDArrayHandle, &str)) {
    let mut scope = Scope::new();

    let mut plist = ptr::null_mut();
    let mut size = 0;
    check_call!(MXListAllOpNames(&mut size, &mut plist));
    let mut op_names: Vec<&ffi::CStr> = Vec::new();
    // [TODO] macro for it.
    for i in 0..size as isize {
        op_names.push(unsafe { ffi::CStr::from_ptr(*plist.offset(i)) });
    }
    // println!("{:?}", op_names);
    let mut module_op = scope.new_module("op");
    let mut modele_internal = scope.new_module("internal");

    let mut submodule_dict = HashMap::new();

    for op_name_prefix in mxnet_rs::base::OP_NAME_PREFIX_LIST {
        submodule_dict.insert(
            op_name_prefix.to_string(),
            op_name_prefix[1..op_name_prefix.len() - 1].to_string(),
        );
    }

    for name in op_names {
        let mut hdl = ptr::null_mut();
        check_call!(NNGetOpHandle(name.as_ptr(), &mut hdl));

        let name_str = name.to_str().unwrap();

        let op_name_prefix = get_op_name_prefix(name_str);
        let mut module_name_local = module_name.to_string();

        let mut func_name = "";
        let mut cur_module = "";
        if op_name_prefix.len() > 0 {
            if op_name_prefix != "_random_" || name_str.ends_with("_like") {
                func_name = &name_str[op_name_prefix.len()..];
                cur_module = &submodule_dict[&op_name_prefix];
                module_name_local = format!(
                    "{}.{}",
                    module_name,
                    &op_name_prefix[1..op_name_prefix.len() - 1]
                );
            } else {
                func_name = name_str;
                cur_module = "internal";
            }
        } else if name_str.starts_with("_") {
            func_name = name_str;
            cur_module = "internal";
        } else {
            func_name = name_str;
            cur_module = "op";
        }

        let function = make_op_func(hdl, func_name);

        if op_name_prefix == "_contrib_" {
            let mut hdl = ptr::null_mut();
            check_call!(NNGetOpHandle(name.as_ptr(), &mut hdl));
            func_name = &name_str[op_name_prefix.len()..];
            let function = make_op_func(hdl, func_name);
        }

        // println!("{}::{}", cur_module, func_name);
    }
}

fn get_op_name_prefix(op_name: &str) -> String {
    for prefix in mxnet_rs::base::OP_NAME_PREFIX_LIST {
        if op_name.starts_with(prefix) {
            return prefix.to_string();
        }
    }
    "".to_string()
}

fn generate_ndarray_function_code(handle: NDArrayHandle, func_name: &str) -> (String, String) {
    let mut real_name = ptr::null();
    let mut desc = ptr::null();
    let mut num_args = 0;
    let mut arg_names = ptr::null_mut();
    let mut arg_descs = ptr::null_mut();
    let mut arg_types = ptr::null_mut();
    let mut key_var_num_args = ptr::null();
    let mut ret_type = ptr::null();

    check_call!(MXSymbolGetAtomicSymbolInfo(
        handle,
        &mut real_name,
        &mut desc,
        &mut num_args,
        &mut arg_names,
        &mut arg_types,
        &mut arg_descs,
        &mut key_var_num_args,
        &mut ret_type
    ));

    let narg = num_args as usize;

    let arg_names = unsafe { slice::from_raw_parts(arg_names, narg) }
        .iter()
        .map(|item| {
            unsafe { ffi::CStr::from_ptr(*item) }
                .to_string_lossy()
                .into_owned()
        })
        .collect();

    let arg_types = unsafe { slice::from_raw_parts(arg_types, narg) }
        .iter()
        .map(|item| {
            unsafe { ffi::CStr::from_ptr(*item) }
                .to_string_lossy()
                .into_owned()
        })
        .collect();

    let key_var_num_args = unsafe { ffi::CStr::from_ptr(key_var_num_args) }
        .to_string_lossy()
        .into_owned();

    let ret_type = if ret_type.is_null() {
        "".to_owned()
    } else {
        unsafe { ffi::CStr::from_ptr(ret_type) }
            .to_string_lossy()
            .into_owned()
    };

    let doc_str = build_ndarray_doc(
        func_name,
        unsafe { ffi::CStr::from_ptr(desc) }.to_str().unwrap(),
        &arg_names,
        &arg_types,
        &Vec::new(),
        &key_var_num_args,
        &ret_type,
    );

    let mut dtype_name: Option<String> = None;
    let mut arr_name: Option<String> = None;
    let mut ndsignature: Vec<String> = Vec::new();
    let mut signature: Vec<String> = Vec::new();
    let mut ndarg_names: Vec<String> = Vec::new();

    for (name, atype) in arg_names.iter().zip(arg_types) {
        println!("{}: {}", name, atype);
        // break;
        if name == "dtype" {
            // println!("{}: {}", name, atype);
            dtype_name = Some(name.to_owned());
            signature.push(format!("{}: &str", name));
        } else if atype.starts_with("NDArray") || atype.starts_with("Symbol") {
            assert_eq!(arr_name, None);

            if atype.ends_with("[]") {
                // ndsignature.push(format!("{}: "))
            }
        }
        // break;
    }
    println!("{:?}", signature);
    (doc_str.clone(), doc_str)
}

fn make_ndarray_function(handle: NDArrayHandle, func_name: &str) {
    let (code, doc_str) = generate_ndarray_function_code(handle, func_name);
    // println!("{}", code);
    // println!("{}", doc_str);
}

fn build_ndarray_doc(
    func_name: &str,
    desc: &str,
    arg_names: &Vec<String>,
    arg_types: &Vec<String>,
    arg_desc: &Vec<String>,
    key_var_num_args: &str,
    ret_type: &str,
) -> String {
    let doc_str = format!("{}", desc);
    doc_str
}
