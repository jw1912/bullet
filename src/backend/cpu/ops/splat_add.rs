use super::DeviceHandles;

pub unsafe fn splat_add(
    handle: DeviceHandles,
    batch_size: usize,
    tensor_size: usize,
    inp: *const f32,
    out: *mut f32,
) {
    let inp = inp as usize;
    let out = out as usize;

    handle.split_workload(batch_size, |_, idx| {
        let this_out = (out as *mut f32).add(tensor_size * idx);

        for i in 0..tensor_size {
            *this_out.add(i) += *(inp as *const f32).add(i);
        }
    });
}
