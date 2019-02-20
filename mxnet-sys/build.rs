use std::env;
use std::path::PathBuf;

fn main() {
    let mxnet_path = env::var("MXNET_PATH").expect("MXNET_PATH not defined");
    println!("cargo:rustc-link-search=native={}", mxnet_path);
    // println!("cargo:rustc-env=LD_LIBRARY_PATH={}", mxnet_path);
    // Tell cargo to tell rustc to link the system bzip2
    // shared library.
    println!("cargo:rustc-link-lib=mxnet");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("wrapper.h")
        // max_align_t should be 24usize, but is tested as 32usize.
        // It is not used by other, so ignore it.
        .clang_arg(format!("-I{}", mxnet_path))
        .blacklist_type("max_align_t")
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
