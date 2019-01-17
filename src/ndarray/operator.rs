use crate::ndarray::NDArray;
use mxnet_sys::{
    AtomicSymbolCreator, MXImperativeInvoke, MXSymbolGetAtomicSymbolInfo,
    MXSymbolListAtomicSymbolCreators, NDArrayHandle, NNGetOpHandle, NNListAllOpNames, OpHandle,
    SymbolHandle,
};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::ptr;
use std::slice;

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
            unsafe { slice::from_raw_parts(symbol_creators, num_symbol_creators as usize) };

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
                unsafe { CStr::from_ptr(name) }.to_str().unwrap().to_owned(),
                // symbol_creators[i],
                *symbol_creator,
            );
        }

        let mut num_ops = 0;
        let mut op_names = ptr::null_mut();
        check_call!(NNListAllOpNames(&mut num_ops, &mut op_names));
        let op_names = unsafe { slice::from_raw_parts(op_names, num_ops as usize) };
        for op_name in op_names {
            let mut handle = ptr::null_mut();
            check_call!(NNGetOpHandle(*op_name, &mut handle));
            op_map.op_handles.insert(
                unsafe { CStr::from_ptr(*op_name) }
                    .to_str()
                    .unwrap()
                    .to_owned(),
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
    params_desc: HashMap<String, String>,
    variable_params: bool,
    params: HashMap<String, String>,
    input_symbols: Vec<SymbolHandle>,
    input_ndarrays: Vec<NDArrayHandle>,
    input_keys: Vec<String>,
    arg_names: Vec<String>,
    handle: AtomicSymbolCreator,
}

impl Operator {
    pub fn new(operator_name: &str) -> Operator {
        let handle = OP_MAP.get_symbol_creator(operator_name);

        // I have no idea why this piece of code is repeated
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
            handle,
            &mut name,
            &mut description,
            &mut num_args,
            &mut arg_names,
            &mut arg_type_infos,
            &mut arg_descriptions,
            &mut key_var_num_args,
            &mut return_type
        ));

        let arg_names = unsafe { slice::from_raw_parts(arg_names, num_args as usize) }
            .iter()
            .map(|name| {
                unsafe { CStr::from_ptr(*name) }
                    .to_str()
                    .unwrap()
                    .to_owned()
            })
            .collect();

        Operator {
            params_desc: HashMap::new(),
            variable_params: false,
            params: HashMap::new(),
            input_symbols: Vec::new(),
            input_ndarrays: Vec::new(),
            input_keys: Vec::new(),
            arg_names,
            handle,
        }
    }

    pub fn invoke(&mut self) -> Vec<NDArray> {
        let mut output_handles = Vec::new();
        self.invoke_with_handles(&mut output_handles);
        let mut outputs = Vec::new();
        for handle in &output_handles {
            outputs.push(NDArray::from(*handle));
        }
        outputs
    }

    pub fn invoke_with(&mut self, output: &mut NDArray) {
        let mut output_handles = vec![output.handle()];
        self.invoke_with_handles(&mut output_handles);
    }

    pub fn invoke_with_handles(&mut self, output_handles: &mut Vec<NDArrayHandle>) {
        if self.input_keys.len() > 0 {
            assert_eq!(self.input_keys.len(), self.input_ndarrays.len());
        }

        let mut param_keys = Vec::new();
        let mut param_values = Vec::new();

        for (key, value) in &self.params {
            param_keys.push(CString::new(key.as_str()).unwrap().as_ptr());
            param_values.push(CString::new(value.as_str()).unwrap().as_ptr());
        }

        let num_inputs = self.input_ndarrays.len() as i32;

        let mut num_outputs = output_handles.len() as i32;

        let mut outputs_receiver = if num_outputs > 0 {
            output_handles.as_mut_ptr()
        } else {
            ptr::null_mut()
        };

        check_call!(MXImperativeInvoke(
            self.handle,
            num_inputs,
            self.input_ndarrays.as_mut_ptr(),
            &mut num_outputs,
            &mut outputs_receiver,
            param_keys.len() as i32,
            param_keys.as_mut_ptr(),
            param_values.as_mut_ptr(),
        ));

        if output_handles.len() > 0 {
            return;
        }

        let handles = unsafe { slice::from_raw_parts(outputs_receiver, num_outputs as usize) };
        for handle in handles {
            output_handles.push(*handle);
        }
    }

    pub fn push_input(&mut self, ndarray: &NDArray) -> &mut Self {
        self.input_ndarrays.push(ndarray.handle());
        self
    }

    pub fn set_input(&mut self, name: &str, ndarray: &NDArray) -> &mut Self {
        self.input_keys.push(name.to_owned());
        self.input_ndarrays.push(ndarray.handle());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ndarray;

    #[test]
    fn create_op_map() {
        let op_map = OpMap::new();
        let _add = op_map.get_op_handle("_plus");
    }

    #[test]
    fn create_operator() {
        let a1 = ndarray::NDArrayBuilder::new().data(&[1.0]).create();
        let a2 = ndarray::NDArrayBuilder::new().data(&[1.0]).create();
        let mut a3 = ndarray::NDArray::new();
        Operator::new("_plus")
            .push_input(&a1)
            .push_input(&a2)
            .invoke_with(&mut a3);

        println!("{:?}", a3.size());
    }
}
