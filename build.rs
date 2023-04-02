use std::env;
use std::process::Command;

fn compile_shader(src_path: &str, dst_path: &str) {
    let shaderc_bin_dir = env::var("SHADERC_BIN_DIR").unwrap();
    let output = Command::new(format!("{shaderc_bin_dir}/glslc"))
        .arg(src_path)
        .arg("-o")
        .arg(dst_path)
        .output()
        .unwrap();
    let output = std::str::from_utf8(output.stderr.as_slice()).unwrap();
    if !output.is_empty() {
        panic!("compile glsl failed: {output}");
    }
}

fn main() {
    let input_dir = "src/pipeline/glsl";
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let out_dir = out_dir.to_str().unwrap();
    for entry in std::fs::read_dir(input_dir).unwrap() {
        if let Ok(entry) = entry {
            let path = entry.path();
            if path.is_file() {
                let file_name = path.file_name().unwrap().to_str().unwrap();
                compile_shader(path.to_str().unwrap(), &format!("{out_dir}/{file_name}"))
            }
        }
    }
    println!("cargo:rerun-if-changed={input_dir}");
}
