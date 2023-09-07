/// # Safety
/// You need to manually confirm that transmuting between
/// the slices is safe!
pub unsafe fn to_slice_with_lifetime<T, U>(slice: &[T]) -> &[U] {
    let src_size = std::mem::size_of_val(slice);
    let tgt_size = std::mem::size_of::<U>();

    assert!(
        src_size % tgt_size == 0,
        "Target type size does not divide slice size!"
    );

    let len = src_size / tgt_size;
    std::slice::from_raw_parts(slice.as_ptr().cast(), len)
}

pub fn sigmoid(x: f32, k: f32) -> f32 {
    1. / (1. + (-x * k).exp())
}

pub fn write_to_bin<T, const SIZEOF: usize>(item: &T, output_path: &str) -> std::io::Result<()> {
    use std::io::Write;

    let mut file = std::fs::File::create(output_path)?;

    unsafe {
        let ptr: *const T = item;
        let slice_ptr: *const u8 = std::mem::transmute(ptr);
        let slice = std::slice::from_raw_parts(slice_ptr, SIZEOF);
        file.write_all(slice)?;
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