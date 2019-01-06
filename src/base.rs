#[macro_export]
macro_rules! check_call {
    ($mx_call:expr) => {
        use mxnet_sys::MXGetLastError;
        use std::ffi::CStr;
        unsafe {
            if $mx_call != 0 {
                panic!(CStr::from_ptr(MXGetLastError()).to_str().unwrap());
            }
        }
    };
}

// fn check_call(ret: i32) {
//     if ret != 0 {
//         unsafe { panic!(CStr::from_ptr(MXGetLastError()).to_str().unwrap()) }
//     }
// }
