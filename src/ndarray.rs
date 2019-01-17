pub mod operator;

use crate::context::Context;
use mxnet_sys::{
    MXNDArrayCreate, MXNDArrayCreateNone, MXNDArrayFree, MXNDArrayGetDType, MXNDArrayGetShape,
    MXNDArraySyncCopyFromCPU, NDArrayHandle,
};
use std::ffi::c_void;
use std::mem;
use std::{ptr, slice};

// pub enum DType {
//     None = -1,
//     F32 = 0,
//     F64 = 1,
//     F16 = 2,
//     U8 = 3,
//     I32 = 4,
//     I8 = 5,
//     I64 = 6,
// }

pub struct NDArray {
    handle: NDArrayHandle,
}

impl NDArray {
    pub fn new() -> NDArray {
        let mut handle = ptr::null_mut();
        check_call!(MXNDArrayCreateNone(&mut handle));
        NDArray { handle }
    }
}

impl From<NDArrayHandle> for NDArray {
    fn from(handle: NDArrayHandle) -> NDArray {
        NDArray { handle }
    }
}

/// Properties
impl NDArray {
    pub fn size(&self) -> u32 {
        self.raw_shape().iter().fold(1, |acc, x| acc * *x)
    }

    pub fn shape(&self) -> Vec<u32> {
        let ret_slice = self.raw_shape();
        let mut ret = Vec::with_capacity(ret_slice.len());
        for i in ret_slice {
            ret.push(*i);
        }
        ret
    }

    // Not sure if we want a enum instead of i32.
    // Not sure if we realy want dtype.
    pub fn dtype(&self) -> i32 {
        self.raw_dtype()
    }

    pub fn handle(&self) -> NDArrayHandle {
        self.handle
    }
}

/// Private
impl NDArray {
    fn raw_shape(&self) -> &[u32] {
        let mut out_dim = 0;
        let mut out_pdata = ptr::null();

        check_call!(MXNDArrayGetShape(self.handle, &mut out_dim, &mut out_pdata));
        unsafe { slice::from_raw_parts(out_pdata, out_dim as usize) }
    }

    fn raw_dtype(&self) -> i32 {
        let mut mx_dtype = 0;
        check_call!(MXNDArrayGetDType(self.handle, &mut mx_dtype));
        mx_dtype
    }
}

impl Drop for NDArray {
    fn drop(&mut self) {
        unsafe {
            MXNDArrayFree(self.handle);
        }
    }
}

pub struct NDArrayBuilder {
    data: Vec<f32>,
    shape: Vec<u32>,
    context: Context,
    delay_alloc: bool,
}

impl NDArrayBuilder {
    pub fn new() -> NDArrayBuilder {
        NDArrayBuilder {
            data: Vec::new(),
            shape: Vec::new(),
            context: Default::default(),
            delay_alloc: true,
        }
    }

    pub fn data(&mut self, data: &[f32]) -> &mut Self {
        self.data = data.to_vec();
        // Set data as 1-D array.
        self.shape = vec![data.len() as u32];
        self
    }

    pub fn shape(&mut self, shape: &[u32]) -> &mut Self {
        self.shape = shape.to_vec();
        self
    }

    pub fn context(&mut self, context: Context) -> &mut Self {
        self.context = context;
        self
    }

    pub fn delay_alloc(&mut self, delay_alloc: bool) -> &mut Self {
        self.delay_alloc = delay_alloc;
        self
    }

    pub fn create(&self) -> NDArray {
        let mut handle = ptr::null_mut();

        check_call!(MXNDArrayCreate(
            self.shape.as_ptr(),
            self.shape.len() as u32,
            self.context.device_type() as i32,
            self.context.device_id() as i32,
            (self.data.is_empty() && self.delay_alloc) as i32,
            &mut handle
        ));

        if !self.data.is_empty() {
            check_call!(MXNDArraySyncCopyFromCPU(
                handle,
                self.data.as_ptr() as *const c_void,
                self.data.len()
            ));
        }

        NDArray { handle }
    }
}

pub fn ones() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ndarray_create_none() {
        let array = NDArray::new();
        assert_eq!(array.shape(), vec![]);
        assert_eq!(array.size(), 1);
    }

    #[test]
    fn ndarray_builder() {
        let _a1 = NDArrayBuilder::new().data(&[1.0]).create();
        let _a2 = NDArrayBuilder::new()
            .data(&[1.0, 2.0])
            .shape(&[2, 1])
            .create();
    }
}
