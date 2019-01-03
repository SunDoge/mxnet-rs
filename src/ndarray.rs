use mxnet_sys::{MXNDArrayCreateNone, NDArrayHandle, MXNDArrayFree};
use std::ptr;

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
            device_id
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
        let _ = NDArray::new();
    }
}