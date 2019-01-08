pub mod operator;

use mxnet_sys::{
    MXNDArrayCreateNone, MXNDArrayFree, MXNDArrayGetDType, MXNDArrayGetShape,
    NDArrayHandle,
};
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

/// Properties
impl NDArray {
    pub fn size(&self) -> usize {
        self.raw_shape().iter().fold(1, |acc, x| acc * *x)
    }

    pub fn shape(&self) -> Vec<usize> {
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
}

/// Private
impl NDArray {
    fn raw_shape(&self) -> &[usize] {
        let mut out_dim = 0;
        let mut out_pdata = ptr::null();

        check_call!(MXNDArrayGetShape(self.handle, &mut out_dim, &mut out_pdata));
        unsafe { slice::from_raw_parts(out_pdata as *const usize, out_dim as usize) }
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
}
