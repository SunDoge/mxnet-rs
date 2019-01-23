use super::ndarray::NDArray;
use super::op_map::OP_MAP;
use super::symbol::Symbol;
use mxnet_sys::{
    AtomicSymbolCreator, MXImperativeInvoke, MXSymbolCompose, MXSymbolCreateAtomicSymbol,
    MXSymbolGetAtomicSymbolInfo, NDArrayHandle,
};
use std::collections::HashMap;
use std::ffi::{c_void, CStr, CString};
use std::ptr;
use std::slice;

pub trait GetHandle {
    fn handle(&self) -> *mut c_void;
}

#[derive(Debug)]
pub struct Operator {
    // params_desc: HashMap<String, String>,
    // variable_params: bool,

    // Params has to store CString, or a memory error arise.
    params: HashMap<CString, CString>,
    // Param index
    index: usize,
    // input_symbols: Vec<SymbolHandle>,
    // input_ndarrays: Vec<NDArrayHandle>,
    inputs: Vec<*mut c_void>,
    input_keys: Vec<CString>,
    arg_names: Vec<CString>,
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
            .map(|name| unsafe { CStr::from_ptr(*name).to_owned() })
            .collect();

        Operator {
            // params_desc: HashMap::new(),
            // variable_params: false,
            params: HashMap::new(),
            index: 0,
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
            param_keys.push(key.as_ptr());
            param_values.push(value.as_ptr());
        }

        for data in &self.input_keys {
            input_keys.push(data.as_ptr());
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
            param_keys.push(key.as_ptr());
            param_values.push(value.as_ptr());
        }

        // println!("{:?}", self.params);

        let num_inputs = self.inputs.len() as i32;

        let mut num_outputs = output_handles.len() as i32;

        let mut outputs_receiver = if num_outputs > 0 {
            output_handles.as_mut_ptr()
        } else {
            ptr::null_mut()
        };

        // println!("Before call");

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

        // println!("After call");

        if output_handles.len() > 0 {
            // println!("return if has output handles");
            return;
        }

        let handles = unsafe { slice::from_raw_parts(outputs_receiver, num_outputs as usize) };
        for handle in handles {
            output_handles.push(*handle);
        }
    }

    pub fn push_input(&mut self, value: &impl GetHandle) -> &mut Self {
        self.inputs.push(value.handle());
        self.index += 1;
        self
    }

    pub fn set_input(&mut self, name: &str, value: &impl GetHandle) -> &mut Self {
        self.input_keys.push(CString::new(name).unwrap());
        self.inputs.push(value.handle());
        self.index += 1;
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
        // println!("{}:{}", name, &value_str);
        self.params.insert(
            CString::new(name).unwrap(),
            CString::new(value_str).unwrap(),
        );
        self
    }

    // It seems never used.
    pub fn set_param_at(&mut self, pos: usize, value: &impl ToString) -> &mut Self {
        let value_str = value.to_string();
        self.params.insert(
            self.arg_names[pos].clone(),
            CString::new(value_str).unwrap(),
        );
        self
    }

    pub fn push_param(&mut self, value: &impl ToString) -> &mut Self {
        self.set_param_at(self.index, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ndarray;
    use crate::symbol;

    #[test]
    fn create_operator() {
        let a1 = ndarray::NDArrayBuilder::new().data(&[1.0]).create();
        let a2 = ndarray::NDArrayBuilder::new().data(&[1.0]).create();
        let mut a3 = ndarray::NDArray::new();
        // Operator::new("_plus")
        //     .push_input(&a1)
        //     .push_input(&a2)
        //     .invoke_with(&mut a3);

        // let s1 = symbol::Symbol::new("data");
        // let op = Operator::new("_PlusScalar")
        //     .push_input(&s1)
        //     .set_param("scalar", &1.0)
        //     .create_symbol("");

        let op = Operator::new("_plus_scalar")
            .push_input(&a1)
            .set_param_at(1, &0.1)
            .invoke_with(&mut a3);
        // println!("{:?}", op);
        // println!("{:?}", op.params);

        println!("{:?}", a3.size());
    }
}
