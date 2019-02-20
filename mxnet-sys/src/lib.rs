#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    #[test]
    fn it_works() {
        unsafe {
            let mut handle: Box<NDArrayHandle> = Box::new(ptr::null_mut());
            assert_eq!(MXNDArrayCreateNone(&mut *handle), 0);
        }
    }
}
