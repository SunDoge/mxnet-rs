#[macro_use]
extern crate lazy_static;
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
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
