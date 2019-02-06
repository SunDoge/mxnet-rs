#[macro_use]
extern crate mxnet_rs;

use codegen::{Function, Module, Scope};
use mxnet_sys::*;
use std::collections::HashMap;
use std::ffi;
use std::ptr;
use std::slice;

fn main() {
    init_op_module("ndarray", make_ndarray_function);
}

// module_name决定写入哪个module
// make_op_func是一个具名函数，不是一个closure
fn init_op_module(module_name: &str, make_op_func: fn(NDArrayHandle, &str, &str) -> Function) {
    // Top-level scope
    let mut scope = Scope::new();

    // Get all Op names
    let mut plist = ptr::null_mut();
    let mut size = 0;
    check_call!(MXListAllOpNames(&mut size, &mut plist));
    let mut op_names: Vec<&ffi::CStr> = Vec::new();
    // [TODO] macro for it.
    for i in 0..size as isize {
        op_names.push(unsafe { ffi::CStr::from_ptr(*plist.offset(i)) });
    }

    // println!("{:?}", op_names);
    let mut module_op = Module::new("op");
    let mut module_internal = Module::new("internal");

    // Store submodule name, _name_ => name
    let mut submodule_dict = HashMap::new();

    for op_name_prefix in mxnet_rs::base::OP_NAME_PREFIX_LIST {
        submodule_dict.insert(
            op_name_prefix.to_string(),
            // op_name_prefix[1..op_name_prefix.len() - 1].to_string(),
            Module::new(&op_name_prefix[1..op_name_prefix.len() - 1]),
        );
    }

    println!("submodule_dict: {:#?}", submodule_dict);

    for name in op_names {
        // Get func handle
        let mut hdl = ptr::null_mut();
        check_call!(NNGetOpHandle(name.as_ptr(), &mut hdl));

        // CStr => str
        let name_str = name.to_str().unwrap();

        // Get op name prefix
        let op_name_prefix = get_op_name_prefix(name_str);
        println!("op_name_prefix: {}", op_name_prefix);

        let mut module_name_local = module_name.to_string();

        let mut func_name = "";
        let mut cur_module = &mut Module::new("");

        if op_name_prefix.len() > 0 {
            if op_name_prefix != "_random_" || name_str.ends_with("_like") {
                func_name = &name_str[op_name_prefix.len()..];
                cur_module = submodule_dict.get_mut(&op_name_prefix).unwrap();
                module_name_local = format!(
                    "{}.{}",
                    module_name,
                    &op_name_prefix[1..op_name_prefix.len() - 1]
                );
            } else {
                func_name = name_str;
                cur_module = &mut module_internal;
            }
        } else if name_str.starts_with("_") {
            func_name = name_str;
            cur_module = &mut module_internal;
        } else {
            func_name = name_str;
            cur_module = &mut module_op;
        }

        let function = make_op_func(hdl, name_str, func_name);
        cur_module.scope().push_fn(function);

        scope.push_module(cur_module.clone());
        println!("{}", scope.to_string());

        // if op_name_prefix == "_contrib_" {
        //     let mut hdl = ptr::null_mut();
        //     check_call!(NNGetOpHandle(name.as_ptr(), &mut hdl));
        //     func_name = &name_str[op_name_prefix.len()..];
        //     function = make_op_func(hdl, name_str, func_name);
        // }

        break;
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

// For NDArray
fn generate_ndarray_function_code(
    handle: NDArrayHandle,
    name: &str,
    func_name: &str,
) -> (String, String) {
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

    println!("ret_type: {}", ret_type);

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
    // println!("{:?}", signature);
    (doc_str.clone(), doc_str)
}

fn make_ndarray_function(handle: NDArrayHandle, name: &str, func_name: &str) -> Function {
    let (code, doc_str) = generate_ndarray_function_code(handle, name, func_name);
    let mut f = Function::new(func_name);
    f.vis("pub").ret("NDArray");
    f
    // println!("{}", code);
    // println!("{}", doc_str);
}

// Rust is strong-typed.
fn build_ndarray_doc(
    _func_name: &str,
    desc: &str,
    _arg_names: &Vec<String>,
    _arg_types: &Vec<String>,
    _arg_desc: &Vec<String>,
    _key_var_num_args: &str,
    _ret_type: &str,
) -> String {
    let doc_str = format!("{}", desc);
    doc_str
}
