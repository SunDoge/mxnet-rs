use mxnet_sys::{MXNDArrayCreateNone, MXNDArrayFree, MXNDArrayGetShape, NDArrayHandle};
use std::{ptr, slice};

pub enum DeviceType {
    CPU = 1,
    GPU = 2,
    CPUPinned = 3,
}

pub struct Context {
    device_type: DeviceType,
    device_id: i32,
}

impl Context {
    pub fn new(device_type: DeviceType, device_id: i32) -> Context {
        Context {
            device_type,
            device_id,
        }
    }

    pub fn device_type(&self) -> &DeviceType {
        &self.device_type
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    pub fn gpu(device_id: i32) -> Context {
        Context::new(DeviceType::GPU, device_id)
    }

    pub fn cpu() -> Context {
        Context::new(DeviceType::CPU, 0)
    }
}

pub struct NDArray {
    handle: NDArrayHandle,
}

impl NDArray {
    pub fn new() -> NDArray {
        let mut handle = ptr::null_mut();
        unsafe {
            assert_eq!(MXNDArrayCreateNone(&mut handle), 0);
        }
        NDArray { handle }
    }

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

    fn raw_shape(&self) -> &[usize] {
        let mut out_dim = 0;
        let mut out_pdata = ptr::null();
        unsafe {
            MXNDArrayGetShape(self.handle, &mut out_dim, &mut out_pdata);
            slice::from_raw_parts(out_pdata as *const usize, out_dim as usize)
        }
    }
}

impl Drop for NDArray {
    fn drop(&mut self) {
        unsafe {
            MXNDArrayFree(self.handle);
        }
    }
}

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
