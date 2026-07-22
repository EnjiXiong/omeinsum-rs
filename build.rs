// build.rs
#[cfg(feature = "ascend-tropical")]
fn build_ascend_tropical_kernel(home: &std::path::Path) {
    use std::{env, fs, path::PathBuf, process::Command};

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo must set OUT_DIR"));
    let output = out_dir.join("libomeinsum_tropical_gemm.so");
    if let Some(prebuilt) = env::var_os("ASCEND_TROPICAL_KERNEL") {
        println!(
            "cargo:rerun-if-changed={}",
            PathBuf::from(&prebuilt).display()
        );
        fs::copy(&prebuilt, &output).unwrap_or_else(|error| {
            panic!(
                "failed to copy ASCEND_TROPICAL_KERNEL={} to {}: {error}",
                PathBuf::from(prebuilt).display(),
                output.display()
            )
        });
    } else {
        let ascendc_cmake = [
            home.join("aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake"),
            home.join("x86_64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake"),
            home.join("compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake"),
            home.join("tools/tikcpp/ascendc_kernel_cmake/ascendc.cmake"),
        ]
        .into_iter()
        .find(|path| path.is_file())
        .unwrap_or_else(|| {
            panic!(
                "ascend-tropical requires CANN's ascendc_kernel_cmake or a precompiled \
                 ASCEND_TROPICAL_KERNEL; none was found under ASCEND_HOME_PATH={}",
                home.display()
            )
        });
        let source = env::current_dir()
            .expect("failed to locate the crate root")
            .join("src/backend/ascend/kernels/tropical_gemm.cpp");
        let cmake_source = out_dir.join("ascend-tropical-cmake");
        let cmake_build = out_dir.join("ascend-tropical-build");
        let cmake_install = out_dir.join("ascend-tropical-out");
        fs::create_dir_all(&cmake_source)
            .expect("failed to create Ascend C CMake source directory");
        fs::write(
            cmake_source.join("CMakeLists.txt"),
            format!(
                "cmake_minimum_required(VERSION 3.16)\n\
                 project(omeinsum_tropical_kernel LANGUAGES C CXX)\n\
                 include(\"{}\")\n\
                 ascendc_library(omeinsum_tropical_gemm SHARED \"{}\")\n",
                ascendc_cmake.display(),
                source.display()
            ),
        )
        .expect("failed to write Ascend C CMake project");

        let soc = env::var("ASCEND_SOC_VERSION").unwrap_or_else(|_| "Ascend910_9382".into());
        let status = Command::new("cmake")
            .arg("-S")
            .arg(&cmake_source)
            .arg("-B")
            .arg(&cmake_build)
            .arg(format!("-DASCEND_CANN_PACKAGE_PATH={}", home.display()))
            .arg(format!("-DSOC_VERSION={soc}"))
            .arg("-DCMAKE_BUILD_TYPE=Release")
            .arg(format!(
                "-DCMAKE_INSTALL_PREFIX={}",
                cmake_install.display()
            ))
            .status()
            .unwrap_or_else(|error| panic!("failed to configure Ascend C kernel: {error}"));
        assert!(status.success(), "Ascend C kernel configuration failed");

        let status = Command::new("cmake")
            .arg("--build")
            .arg(&cmake_build)
            .args(["--parallel", "2"])
            .status()
            .unwrap_or_else(|error| panic!("failed to build Ascend C kernel: {error}"));
        assert!(status.success(), "Ascend C tropical kernel build failed");

        let status = Command::new("cmake")
            .arg("--install")
            .arg(&cmake_build)
            .status()
            .unwrap_or_else(|error| panic!("failed to install Ascend C kernel: {error}"));
        assert!(status.success(), "Ascend C tropical kernel install failed");

        let library = cmake_install.join("lib64/libomeinsum_tropical_gemm.so");
        fs::copy(&library, &output).unwrap_or_else(|error| {
            panic!(
                "failed to copy Ascend C library {} to {}: {error}",
                library.display(),
                output.display()
            )
        });
    }
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=dylib=omeinsum_tropical_gemm");
    println!("cargo:rerun-if-changed=src/backend/ascend/kernels/tropical_gemm.cpp");
    println!("cargo:rerun-if-env-changed=ASCEND_TROPICAL_KERNEL");
    println!("cargo:rerun-if-env-changed=ASCEND_SOC_VERSION");
}

fn main() {
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

    #[cfg(any(feature = "ascend", feature = "ascend-tropical"))]
    {
        let home = std::env::var("ASCEND_HOME_PATH")
            .unwrap_or_else(|_| "/usr/local/Ascend/ascend-toolkit/latest".into());
        println!("cargo:rustc-link-search=native={home}/lib64");
        println!("cargo:rustc-link-lib=dylib=ascendcl");
        println!("cargo:rustc-link-lib=dylib=nnopbase");
        println!("cargo:rustc-link-lib=dylib=opapi");
        println!("cargo:rerun-if-env-changed=ASCEND_HOME_PATH");

        #[cfg(feature = "ascend-tropical")]
        build_ascend_tropical_kernel(std::path::Path::new(&home));
    }
}
