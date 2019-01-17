use mxnet_sys::{MXGetGPUCount, MXGetGPUMemoryInformation64};

pub use DeviceType::*;

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum DeviceType {
    CPU = 1,
    GPU = 2,
    CPUPinned = 3,
    CPUShared = 5,
}

impl From<i32> for DeviceType {
    fn from(int: i32) -> DeviceType {
        match int {
            1 => CPU,
            2 => GPU,
            3 => CPUPinned,
            5 => CPUShared,
            _ => panic!("No DeviceType for {}", int),
        }
    }
}

#[derive(Debug, Copy, Clone)]
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

    pub fn device_type(&self) -> DeviceType {
        self.device_type
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }
}

impl Default for Context {
    fn default() -> Context {
        Context::new(CPU, 0)
    }
}

pub fn gpu(device_id: i32) -> Context {
    Context::new(GPU, device_id)
}

pub fn cpu() -> Context {
    Context::new(CPU, 0)
}

pub fn cpu_pinned() -> Context {
    Context::new(CPUPinned, 0)
}

pub fn num_gpus() -> usize {
    let mut count = 0;
    check_call!(MXGetGPUCount(&mut count));
    count as usize
}

pub fn gpu_memory_info(device_id: i32) -> (u64, u64) {
    let mut free = 0;
    let mut total = 0;
    check_call!(MXGetGPUMemoryInformation64(
        device_id, &mut free, &mut total
    ));
    (free, total)
}
