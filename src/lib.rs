#![feature(test)]

#[macro_use]
extern crate enum_str_derive;

#[macro_use]
pub mod base;

pub mod autograd;
pub mod context;
pub mod error;
pub mod ndarray;
pub mod op_map;
pub mod operator;
pub mod symbol;

#[cfg(test)]
mod tests {
    extern crate test;

    use super::*;
    use test::Bencher;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[bench]
    fn cache(b: &mut Bencher) {
        let op_name = std::ffi::CString::new("_plus").unwrap();
        let mut handle = std::ptr::null_mut();
        check_call!(mxnet_sys::NNGetOpHandle(op_name.as_ptr(), &mut handle));
        let mut map = std::collections::HashMap::new();
        map.insert(op_name.clone(), handle);
        b.iter(|| {
            let hdl = map.get(&op_name).unwrap();
            assert_eq!(*hdl, handle);
        });
    }
}
