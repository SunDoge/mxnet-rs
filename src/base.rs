use mxnet_sys::MXGetLastError;
use std::ffi::CStr;

pub type MXResult = Result<(), &'static str>;

pub fn check_call(ret: i32) -> MXResult {
    if ret != 0 {
        unsafe { Err(CStr::from_ptr(MXGetLastError()).to_str().unwrap()) }
    } else {
        Ok(())
    }
}
