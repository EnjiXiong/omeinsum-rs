// build.rs
fn main() {
    #[cfg(feature = "ascend")]
    build_ascend();

    #[cfg(feature = "cuda")]
    {
        // Print version requirement warning
        println!("cargo:warning=cuTENSOR 2.0+ is REQUIRED. Version 1.x uses a different API and will NOT work.");
        println!("cargo:warning=Install cuTENSOR 2.0+: conda install -c nvidia cutensor-cu12 (for CUDA 12)");
        println!("cargo:warning=Or download from: https://developer.nvidia.com/cutensor-downloads");

        let lib_path = if let Ok(path) = std::env::var("CUTENSOR_PATH") {
            println!("cargo:warning=Using CUTENSOR_PATH={}", path);
            path
        } else if let Ok(cuda) = std::env::var("CUDA_PATH") {
            let path = format!("{}/lib64", cuda);
            println!("cargo:warning=Using CUDA_PATH/lib64={}", path);
            path
        } else {
            println!("cargo:warning=Using default path /usr/local/cuda/lib64");
            "/usr/local/cuda/lib64".to_string()
        };

        println!("cargo:rustc-link-search=native={}", lib_path);
        println!("cargo:rustc-link-lib=dylib=cutensor");
        println!("cargo:rerun-if-env-changed=CUTENSOR_PATH");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");

        // Check if library exists (either libcutensor.so or libcutensor.so.2 for cuTENSOR 2.x)
        let lib_file = format!("{}/libcutensor.so", lib_path);
        let lib_file_v2 = format!("{}/libcutensor.so.2", lib_path);
        if !std::path::Path::new(&lib_file).exists() && !std::path::Path::new(&lib_file_v2).exists()
        {
            println!(
                "cargo:warning=libcutensor.so not found at {}. Linking may fail.",
                lib_path
            );
            println!("cargo:warning=Set CUTENSOR_PATH to the directory containing libcutensor.so");
            println!("cargo:warning=For pip-installed cuTENSOR: pip install cutensor-cu12, then create symlink:");
            println!("cargo:warning=  ln -s libcutensor.so.2 $CUTENSOR_PATH/libcutensor.so");
        } else if std::path::Path::new(&lib_file_v2).exists()
            && !std::path::Path::new(&lib_file).exists()
        {
            println!("cargo:warning=Found libcutensor.so.2 but not libcutensor.so - you may need to create a symlink:");
            println!(
                "cargo:warning=  ln -s libcutensor.so.2 {}/libcutensor.so",
                lib_path
            );
        }
    }
}

#[cfg(feature = "ascend")]
fn build_ascend() {
    use std::path::{Path, PathBuf};

    println!("cargo:rerun-if-env-changed=ASCEND_HOME_PATH");
    println!("cargo:rerun-if-env-changed=OME_ASCEND_ENABLE_CAPTURE");
    println!("cargo:rerun-if-changed=native/ascend/ome_ascend.h");
    println!("cargo:rerun-if-changed=native/ascend/ome_ascend.cpp");

    let root = std::env::var_os("ASCEND_HOME_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            panic!("feature `ascend` requires ASCEND_HOME_PATH to point to a CANN installation")
        });
    let include = root.join("include");
    for relative in [
        "acl/acl.h",
        "aclnnop/aclnn_matmul.h",
        "aclnnop/aclnn_add.h",
        "aclnnop/aclnn_sub.h",
        "aclnnop/aclnn_permute.h",
    ] {
        let header = include.join(relative);
        if !header.is_file() {
            panic!("required CANN header is missing: {}", header.display());
        }
    }

    let target_arch =
        std::env::var("CARGO_CFG_TARGET_ARCH").expect("Cargo must provide CARGO_CFG_TARGET_ARCH");
    let candidates = [
        root.join("lib64"),
        root.join(format!("{target_arch}-linux/lib64")),
    ];
    let library_dir = candidates
        .iter()
        .find(|directory| {
            has_shared_library(directory, "ascendcl") && has_shared_library(directory, "nnopbase")
        })
        .unwrap_or_else(|| {
            panic!(
                "no CANN library directory contains both libascendcl.so and \
                 libnnopbase.so; searched {}",
                candidates
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        });

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .include(&include)
        .include("native/ascend")
        .file("native/ascend/ome_ascend.cpp")
        .warnings(true);
    if std::env::var("OME_ASCEND_ENABLE_CAPTURE").as_deref() == Ok("1") {
        build.define("OME_ASCEND_ENABLE_CAPTURE", "1");
    }
    if std::env::var("DEBUG").as_deref() == Ok("true") {
        build.define("OME_ASCEND_DEBUG_DIAGNOSTICS", "1");
    }
    build.compile("ome_ascend");

    println!("cargo:rustc-link-search=native={}", library_dir.display());
    println!("cargo:rustc-link-lib=dylib=ascendcl");
    println!("cargo:rustc-link-lib=dylib=nnopbase");

    fn has_shared_library(directory: &Path, stem: &str) -> bool {
        directory.join(format!("lib{stem}.so")).is_file()
    }
}
