#[macro_use]
extern crate lazy_static;

#[macro_use]
pub mod base;

pub mod autograd;
pub mod context;
pub mod error;
pub mod ndarray;
pub mod operator;
pub mod symbol;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
