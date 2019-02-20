use std::env;

fn main() {
    let mxnet_path = env::var("MXNET_PATH").expect("MXNET_PATH not defined");
    println!("cargo:rustc-env=LD_LIBRARY_PATH={}", mxnet_path);
}