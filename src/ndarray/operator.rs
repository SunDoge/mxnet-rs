use super::NDArray;
use crate::base::{GetHandle, OP_MAP};
use crate::symbol::Symbol;
use mxnet_sys::{
    AtomicSymbolCreator, MXImperativeInvoke, MXSymbolCompose, MXSymbolCreateAtomicSymbol,
    MXSymbolGetAtomicSymbolInfo, NDArrayHandle,
};
use std::collections::HashMap;
use std::ffi::{c_void, CStr, CString};
use std::ptr;
use std::slice;

pub struct Operator {
    params_desc: HashMap<String, String>,
    variable_params: bool,
    params: HashMap<String, String>,
    // input_symbols: Vec<SymbolHandle>,
    // input_ndarrays: Vec<NDArrayHandle>,
    inputs: Vec<*mut c_void>,
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
            // input_symbols: Vec::new(),
            // input_ndarrays: Vec::new(),
            inputs: Vec::new(),
            input_keys: Vec::new(),
            arg_names,
            handle,
        }
    }

    // pub fn set_input()

    pub fn create_symbol(&mut self, name: &str) -> Symbol {
        if self.input_keys.len() > 0 {
            assert_eq!(self.input_keys.len(), self.inputs.len());
        }

        let pname = if name.is_empty() {
            ptr::null()
        } else {
            CString::new(name).unwrap().as_ptr()
        };

        let mut symbol_handle = ptr::null_mut();
        let mut input_keys = Vec::new();
        let mut param_keys = Vec::new();
        let mut param_values = Vec::new();

        for (key, value) in &self.params {
            param_keys.push(CString::new(key.as_str()).unwrap().as_ptr());
            param_values.push(CString::new(value.as_str()).unwrap().as_ptr());
        }

        for data in &self.input_keys {
            input_keys.push(CString::new(data.as_str()).unwrap().as_ptr());
        }

        let input_keys_p = if input_keys.len() > 0 {
            input_keys.as_mut_ptr()
        } else {
            ptr::null_mut()
        };

        check_call!(MXSymbolCreateAtomicSymbol(
            self.handle,
            param_keys.len() as u32,
            param_keys.as_mut_ptr(),
            param_values.as_mut_ptr(),
            &mut symbol_handle
        ));

        check_call!(MXSymbolCompose(
            symbol_handle,
            pname,
            self.inputs.len() as u32,
            input_keys_p,
            self.inputs.as_mut_ptr()
        ));

        Symbol::from(symbol_handle)
    }

    pub fn invoke_with(&mut self, output: &mut NDArray) {
        let mut output_handles = vec![output.handle()];
        self.invoke_with_handles(&mut output_handles);
    }

    pub fn invoke_with_handles(&mut self, output_handles: &mut Vec<NDArrayHandle>) {
        if self.input_keys.len() > 0 {
            assert_eq!(self.input_keys.len(), self.inputs.len());
        }

        let mut param_keys = Vec::new();
        let mut param_values = Vec::new();

        for (key, value) in &self.params {
            param_keys.push(CString::new(key.as_str()).unwrap().as_ptr());
            param_values.push(CString::new(value.as_str()).unwrap().as_ptr());
        }

        let num_inputs = self.inputs.len() as i32;

        let mut num_outputs = output_handles.len() as i32;

        let mut outputs_receiver = if num_outputs > 0 {
            output_handles.as_mut_ptr()
        } else {
            ptr::null_mut()
        };

        check_call!(MXImperativeInvoke(
            self.handle,
            num_inputs,
            self.inputs.as_mut_ptr(),
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

    pub fn push_input(&mut self, value: &impl GetHandle) -> &mut Self {
        self.inputs.push(value.handle());
        self
    }

    pub fn set_input(&mut self, name: &str, value: &impl GetHandle) -> &mut Self {
        self.input_keys.push(name.to_owned());
        self.inputs.push(value.handle());
        self
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

    pub fn set_param(&mut self, name: &str, value: &impl ToString) -> &mut Self {
        let value_str = value.to_string();
        self.params.insert(name.to_owned(), value_str);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ndarray;

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
