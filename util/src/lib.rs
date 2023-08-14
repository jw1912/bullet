/// # Safety
/// You need to manually confirm that transmuting between
/// the slices is safe!
pub unsafe fn to_slice_with_lifetime<T, U>(slice: &[T]) -> &[U] {
    use std::mem;

    let src_size = mem::size_of_val(slice);
    let tgt_size = mem::size_of::<U>();

    assert!(src_size % tgt_size == 0, "Target type size does not divide slice size!");

    let len = src_size / tgt_size;
    std::slice::from_raw_parts(slice.as_ptr().cast(), len)
}