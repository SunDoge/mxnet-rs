use crate::base::GetHandle;
use crate::operator::Operator;
use mxnet_sys::{MXSymbolCreateVariable, MXSymbolFree, SymbolHandle};
use std::ffi::{CStr, CString};
use std::ptr;
use std::rc::Rc;

macro_rules! ops {
    (
        $op_name:expr,
        $op_class:ident::$op_method:ident
    ) => {
        impl std::ops::$op_class for Symbol {
            type Output = Symbol;

            fn $op_method(self, rhs: Symbol) -> Symbol {
                Operator::new($op_name)
                    .push_input(&self)
                    .push_input(&rhs)
                    .create_symbol("")
            }
        }

        impl std::ops::$op_class<f32> for Symbol {
            type Output = Symbol;

            fn $op_method(self, rhs: f32) -> Symbol {
                Operator::new(concat!($op_name, "Scalar"))
                    .push_input(&self)
                    .set_param("scalar", &rhs)
                    .create_symbol("")
            }
        }
    };
}

struct SymBlob {
    handle: SymbolHandle,
}

impl SymBlob {
    pub fn new(handle: SymbolHandle) -> SymBlob {
        SymBlob { handle }
    }

    pub fn handle(&self) -> SymbolHandle {
        self.handle
    }
}

impl Drop for SymBlob {
    fn drop(&mut self) {
        check_call!(MXSymbolFree(self.handle()))
    }
}

pub struct Symbol {
    blob: Rc<SymBlob>,
}

ops!("_Plus", Add::add);
ops!("_Minus", Sub::sub);
ops!("_Mul", Mul::mul);
ops!("_Div", Div::div);
ops!("_Mod", Rem::rem);

impl Symbol {
    pub fn new(name: &str) -> Symbol {
        let mut handle = ptr::null_mut();
        check_call!(MXSymbolCreateVariable(
            CString::new(name).unwrap().as_ptr(),
            &mut handle
        ));
        Symbol {
            blob: Rc::new(SymBlob::new(handle)),
        }
    }
}

impl From<SymbolHandle> for Symbol {
    fn from(handle: SymbolHandle) -> Symbol {
        Symbol {
            blob: Rc::new(SymBlob::new(handle)),
        }
    }
}

impl GetHandle for Symbol {
    fn handle(&self) -> SymbolHandle {
        self.blob.handle()
    }
}
