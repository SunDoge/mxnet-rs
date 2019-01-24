#[macro_use]
extern crate mxnet_rs;

use mxnet_rs::context;
use mxnet_sys::*;
use std::ptr;

fn main() {
    println!("num gpus: {}", context::num_gpus());

    // unsafe {
    //     let mut plist = ptr::null_mut();
    //     let mut size = 0;
    //     check_call!(MXListAllOpNames(&mut size, &mut plist));
    //     let pslice = std::slice::from_raw_parts_mut(plist, size as usize);
    //     for p in pslice {
    //         println!("{:?}", std::ffi::CStr::from_ptr(*p));
    //     }
    // }
}
