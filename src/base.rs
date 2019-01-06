#[macro_export]
macro_rules! check_call {
    ($mx_call:expr) => {
        unsafe {
            if $mx_call != 0 {
                panic!(std::ffi::CStr::from_ptr(mxnet_sys::MXGetLastError())
                    .to_str()
                    .unwrap());
            }
        }
    };
}
