use mxnet_sys::{
    AtomicSymbolCreator, MXSymbolGetAtomicSymbolInfo, MXSymbolListAtomicSymbolCreators,
    NNGetOpHandle, NNListAllOpNames, OpHandle,
};
use std::collections::HashMap;
use std::ffi::CStr;
use std::ptr;

lazy_static! {
    static ref OP_MAP: OpMap = OpMap::new();
}

pub struct OpMap {
    symbol_creators: HashMap<String, AtomicSymbolCreator>,
    op_handles: HashMap<String, OpHandle>,
}

unsafe impl Sync for OpMap {}

impl OpMap {
    pub fn new() -> OpMap {
        let mut op_map = OpMap {
            symbol_creators: HashMap::new(),
            op_handles: HashMap::new(),
        };

        let mut num_symbol_creators = 0;
        let mut symbol_creators = ptr::null_mut();
        check_call!(MXSymbolListAtomicSymbolCreators(
            &mut num_symbol_creators,
            &mut symbol_creators
        ));
        // let num_symbol_creators = num_symbol_creators as usize;
        let symbol_creators =
            unsafe { std::slice::from_raw_parts(symbol_creators, num_symbol_creators as usize) };

        // for i in 0..num_symbol_creators {
        for symbol_creator in symbol_creators {
            let mut name = ptr::null();
            let mut description = ptr::null();
            let mut num_args = 0;
            let mut arg_names = ptr::null_mut();
            let mut arg_descriptions = ptr::null_mut();
            let mut arg_type_infos = ptr::null_mut();
            let mut key_var_num_args = ptr::null();
            let mut return_type = ptr::null();

            check_call!(MXSymbolGetAtomicSymbolInfo(
                // symbol_creators[i],
                *symbol_creator,
                &mut name,
                &mut description,
                &mut num_args,
                &mut arg_names,
                &mut arg_type_infos,
                &mut arg_descriptions,
                &mut key_var_num_args,
                &mut return_type
            ));

            op_map.symbol_creators.insert(
                unsafe { CStr::from_ptr(name) }.to_string_lossy().into(),
                // symbol_creators[i],
                *symbol_creator,
            );
        }

        let mut num_ops = 0;
        let mut op_names = ptr::null_mut();
        check_call!(NNListAllOpNames(&mut num_ops, &mut op_names));
        let op_names = unsafe { std::slice::from_raw_parts(op_names, num_ops as usize) };
        for op_name in op_names {
            let mut handle = ptr::null_mut();
            check_call!(NNGetOpHandle(*op_name, &mut handle));
            op_map.op_handles.insert(
                unsafe { CStr::from_ptr(*op_name) }.to_string_lossy().into(),
                handle,
            );
        }

        op_map
    }

    pub fn get_symbol_creator(&self, name: &str) -> AtomicSymbolCreator {
        *self
            .symbol_creators
            .get(name)
            .unwrap_or(&self.get_op_handle(name))
    }

    pub fn get_op_handle(&self, name: &str) -> OpHandle {
        self.op_handles[name]
    }
}

pub struct Operator {
    handle: AtomicSymbolCreator,
}

impl Operator {
    pub fn new(operator_name: &str) -> Operator {
        let handle = OP_MAP.get_symbol_creator(operator_name);
        Operator { handle }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_op_map() {
        let op_map = OpMap::new();
        let _add = op_map.get_op_handle("_add");
    }

    fn create_operator() {
        let _operator = Operator::new("_add");
    }
}
