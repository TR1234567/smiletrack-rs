fn main() {
    // Set the path to libtorch
    let libtorch_path = std::env::var("LIBTORCH").unwrap_or_else(|_| {
        let path = std::path::PathBuf::from("libtorch");
        path.to_string_lossy().into_owned()
    });
    
    println!("cargo:rustc-link-search=native={}/lib", libtorch_path);
    println!("cargo:rustc-link-lib=dylib=torch");
    println!("cargo:rustc-link-lib=dylib=c10");
    
    // Add include paths
    println!("cargo:include={}/include", libtorch_path);
    println!("cargo:include={}/include/torch/csrc/api/include", libtorch_path);
} 