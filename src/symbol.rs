use crate::base::GetHandle;
use mxnet_sys::{MXSymbolCreateVariable, MXSymbolFree, SymbolHandle};
use std::ffi::{CStr, CString};
use std::ptr;
use std::rc::Rc;

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
