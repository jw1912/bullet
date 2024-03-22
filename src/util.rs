pub fn sigmoid(x: f32, k: f32) -> f32 {
    1. / (1. + (-x * k).exp())
}

pub fn write_to_bin<T>(
    item: &[T],
    size: usize,
    output_path: &str,
    pad: bool,
) -> std::io::Result<()> {
    use std::io::Write;

    let mut file = std::fs::File::create(output_path)?;

    let size = std::mem::size_of::<T>() * size;

    unsafe {
        let slice_ptr: *const u8 = item.as_ptr().cast();
        let slice = std::slice::from_raw_parts(slice_ptr, size);
        file.write_all(slice)?;
    }

    if pad {
        let mut padding = vec![0u8; (64 - size % 64) % 64];

        let chs = [b'b', b'u', b'l', b'l', b'e', b't'];

        for (i, p) in padding.iter_mut().enumerate() {
            *p = chs[i % chs.len()];
        }

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

pub fn to_slice_with_lifetime<T, U>(slice: &[T]) -> &[U] {
    let src_size = std::mem::size_of_val(slice);
    let tgt_size = std::mem::size_of::<U>();

    assert!(
        src_size % tgt_size == 0,
        "Target type size does not divide slice size!"
    );

    let len = src_size / tgt_size;
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast(), len) }
}

pub fn to_slice_with_lifetime_mut<T, U>(slice: &mut [T]) -> &mut [U] {
    let src_size = std::mem::size_of_val(slice);
    let tgt_size = std::mem::size_of::<U>();

    assert!(
        src_size % tgt_size == 0,
        "Target type size does not divide slice size!"
    );

    let len = src_size / tgt_size;
    unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr().cast(), len) }
}
