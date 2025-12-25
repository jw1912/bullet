fn main() {
    #[cfg(not(any(feature = "cuda", feature = "hip", feature = "cpu")))]
    println!("cargo:rustc-cfg=feature=cpu");
}