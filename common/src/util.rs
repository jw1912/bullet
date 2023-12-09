pub fn sigmoid(x: f32, k: f32) -> f32 {
    1. / (1. + (-x * k).exp())
}

pub fn write_to_bin<T, const SIZEOF: usize>(item: &T, output_path: &str, pad: bool) -> std::io::Result<()> {
    use std::io::Write;

    let mut file = std::fs::File::create(output_path)?;

    unsafe {
        let ptr: *const T = item;
        let slice_ptr: *const u8 = std::mem::transmute(ptr);
        let slice = std::slice::from_raw_parts(slice_ptr, SIZEOF);
        file.write_all(slice)?;
    }

    if pad {
        let padding = vec![0u8; (64 - SIZEOF % 64) % 64];
        file.write_all(&padding)?;
    }

    Ok(())
}

pub fn boxed_and_zeroed<T>() -> Box<T> {
    unsafe {
        let layout = std::alloc::Layout::new::<T>();
        let ptr = std::alloc::alloc_zeroed(layout);
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Box::from_raw(ptr.cast())
    }
}