use std::{mem, ptr};

use metal_rs::{ComputePipelineState, Device, Function, Library, MTLResourceOptions, MTLSize, NSUInteger};

use crate::loader::Feat;

use super::DeviceHandles;
use crate::backend::cpu;

#[derive(Clone)]
pub(crate) struct Kernels {
    backprop_relu: ComputePipelineState,
    backprop_crelu: ComputePipelineState,
    backprop_screlu: ComputePipelineState,
    activate_relu: ComputePipelineState,
    activate_crelu: ComputePipelineState,
    activate_screlu: ComputePipelineState,
    add_to: ComputePipelineState,
    sigmoid_mpe: ComputePipelineState,
}

impl Kernels {
    pub(crate) fn new(device: &Device, lib: &Library) -> Kernels {
        macro_rules! new_kernel {
            ($device:tt $lib:tt $name:tt) => {
                $device.new_compute_pipeline_state_with_function(&$lib.get_function($name, None).unwrap()).unwrap()
                //$lib.get_function($name, None).unwrap()
            };
        }
        Self {
            backprop_relu: new_kernel!(device lib "backpropReLU"),
            backprop_crelu: new_kernel!(device lib "backpropCReLU"),
            backprop_screlu: new_kernel!(device lib "backpropSCReLU"),
            activate_relu: new_kernel!(device lib "activateReLU"),
            activate_crelu: new_kernel!(device lib "activateCReLU"),
            activate_screlu: new_kernel!(device lib "activateSCReLU"),
            add_to: new_kernel!(device lib "addTo"),
            sigmoid_mpe: new_kernel!(device lib "sigmoidMPE"),
        }
    }
}

pub unsafe fn pairwise_mul(
    handle: &DeviceHandles,
    batch_size: usize,
    input_size: usize,
    output_size: usize,
    inp: *const f32,
    out: *mut f32,
) {
    cpu::ops::pairwise_mul(&handle.cpu, batch_size, input_size, output_size, inp, out)
}

pub unsafe fn backprop_pairwise_mul(
    handle: &DeviceHandles,
    batch_size: usize,
    input_size: usize,
    output_size: usize,
    inp: *const f32,
    out: *mut f32,
) {
    cpu::ops::backprop_pairwise_mul(&handle.cpu, batch_size, input_size, output_size, inp, out)
}

pub unsafe fn splat_mul_matrix_vector(
    handle: &DeviceHandles,
    m: usize,
    n: usize,
    a_ptr: *const f32,
    x_ptr: *const f32,
    y_ptr: *mut f32,
    batch_size: usize,
) {
    cpu::ops::splat_mul_matrix_vector(&handle.cpu, m, n, a_ptr, x_ptr, y_ptr, batch_size)
}

pub unsafe fn splat_mul_matrixt_vector(
    handle: &DeviceHandles,
    m: usize,
    n: usize,
    a_ptr: *const f32,
    y_ptr: *const f32,
    x_ptr: *mut f32,
    batch_size: usize,
) {
    cpu::ops::splat_mul_matrixt_vector(&handle.cpu, m, n, a_ptr, y_ptr, x_ptr, batch_size)
}

pub unsafe fn reduce_add_mul_vector_vectort(
    handle: &DeviceHandles,
    m: usize,
    n: usize,
    y_ptr: *const f32,
    x_ptr: *const f32,
    a_ptr: *mut f32,
    batch_size: usize,
) {
    cpu::ops::reduce_add_mul_vector_vectort(&handle.cpu, m, n, y_ptr, x_ptr, a_ptr, batch_size)
}

pub unsafe fn reduce_add(
    handle: &DeviceHandles,
    a: *const f32,
    batch_size: usize,
    out_size: usize,
    inp: *const f32,
    out: *mut f32,
) {
    cpu::ops::reduce_add(&handle.cpu, a, batch_size, out_size, inp, out)
}

pub unsafe fn select(
    handle: &DeviceHandles,
    batch_size: usize,
    input_size: usize,
    output_size: usize,
    buckets: *const u8,
    inp: *const f32,
    out: *mut f32,
) {
    cpu::ops::select(&handle.cpu, batch_size, input_size, output_size, buckets, inp, out)
}

pub unsafe fn select_backprop(
    handle: &DeviceHandles,
    batch_size: usize,
    input_size: usize,
    output_size: usize,
    buckets: *const u8,
    inp: *const f32,
    out: *mut f32,
) {
    cpu::ops::select_backprop(&handle.cpu, batch_size, input_size, output_size, buckets, inp, out)
}

macro_rules! two_buffer_kernel {
    ($func:ident) => {
        pub unsafe fn $func(handle: &DeviceHandles, size: usize, inp: *const f32, out: *mut f32) {
            objc::rc::autoreleasepool(|| {
                // Create three data buffers each holding data to passed to the device.
                // Buffer 0: Size; Buffer 1: Input Buffer; Buffer 2: Output Buffer
                let siz_buffer = handle.device.new_buffer_with_bytes_no_copy(
                    &size as *const _ as *const std::ffi::c_void,
                    mem::size_of::<usize>() as NSUInteger,
                    MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
                    None,
                );
                let inp_buffer = handle.device.new_buffer_with_bytes_no_copy(
                    inp.cast(),
                    (size * mem::size_of::<f32>()) as NSUInteger,
                    MTLResourceOptions::StorageModeShared,
                    None,
                );
                let out_buffer = handle.device.new_buffer_with_bytes_no_copy(
                    out.cast(),
                    (size * mem::size_of::<f32>()) as NSUInteger,
                    MTLResourceOptions::StorageModeShared,
                    None,
                );

                // Create a new command queue with an encoder for the computation.
                let command_buffer = handle.queue.new_command_buffer();
                let compute_encoder = command_buffer.new_compute_command_encoder();
                compute_encoder.set_compute_pipeline_state(&handle.kernels.$func);
                compute_encoder.set_buffers(0, &[Some(&siz_buffer), Some(&inp_buffer), Some(&out_buffer)], &[0; 3]);

                // Create a simple 1D thread grid for parallel processing.
                // TODO: Possibly improve thread organization? Needs research
                let grid_size = MTLSize::new(size as NSUInteger, 1, 1);
                let threadgroup_size = MTLSize::new(size as NSUInteger, 1, 1);

                compute_encoder.dispatch_threads(grid_size, threadgroup_size);

                // Create the final package and commit it.
                compute_encoder.end_encoding();
                command_buffer.commit();

                // Wait till all computations finish.
                command_buffer.wait_until_completed();
            });
        }
    };
}

two_buffer_kernel!(backprop_relu);
two_buffer_kernel!(backprop_crelu);
two_buffer_kernel!(backprop_screlu);

two_buffer_kernel!(activate_relu);
two_buffer_kernel!(activate_crelu);
two_buffer_kernel!(activate_screlu);

two_buffer_kernel!(add_to);

pub unsafe fn sigmoid_mpe(
    handle: &DeviceHandles,
    buffer_size: usize,
    outputs: *mut f32,
    results: *const f32,
    errors: *mut f32,
    power: f32,
) {
    cpu::ops::sigmoid_mpe(&handle.cpu, buffer_size, outputs, results, errors, power)
}

pub unsafe fn sparse_affine_forward(
    handle: &DeviceHandles,
    batch_size: usize,
    max_input_size: usize,
    output_size: usize,
    weights: *const f32,
    biases: *const f32,
    inputs: *const Feat,
    outputs: *mut f32,
) {
    cpu::ops::sparse_affine_forward(
        &handle.cpu,
        batch_size,
        max_input_size,
        output_size,
        weights,
        biases,
        inputs,
        outputs,
    )
}

pub unsafe fn sparse_affine_backward(
    handle: &DeviceHandles,
    batch_size: usize,
    max_active_inputs: usize,
    input_size: usize,
    output_size: usize,
    weights_grad: *mut f32,
    biases_grad: *mut f32,
    inputs: *const Feat,
    errors: *const f32,
    output: *const f32,
    ft_reg: f32,
) {
    cpu::ops::sparse_affine_backward(
        &handle.cpu,
        batch_size,
        max_active_inputs,
        input_size,
        output_size,
        weights_grad,
        biases_grad,
        inputs,
        errors,
        output,
        ft_reg,
    )
}

pub unsafe fn single_sparse_affine_forward(
    handle: &DeviceHandles,
    batch_size: usize,
    max_active_inputs: usize,
    output_size: usize,
    weights: *const f32,
    biases: *const f32,
    inputs: *const Feat,
    outputs: *mut f32,
) {
    cpu::ops::single_sparse_affine_forward(
        &handle.cpu,
        batch_size,
        max_active_inputs,
        output_size,
        weights,
        biases,
        inputs,
        outputs,
    )
}

pub unsafe fn single_sparse_affine_backward(
    handle: &DeviceHandles,
    batch_size: usize,
    max_active_inputs: usize,
    input_size: usize,
    output_size: usize,
    weights_grad: *mut f32,
    biases_grad: *mut f32,
    inputs: *const Feat,
    errors: *const f32,
    output: *const f32,
    ft_reg: f32,
) {
    cpu::ops::single_sparse_affine_backward(
        &handle.cpu,
        batch_size,
        max_active_inputs,
        input_size,
        output_size,
        weights_grad,
        biases_grad,
        inputs,
        errors,
        output,
        ft_reg,
    )
}

pub unsafe fn splat_add(handle: &DeviceHandles, batch_size: usize, tensor_size: usize, inp: *const f32, out: *mut f32) {
    cpu::ops::splat_add(&handle.cpu, batch_size, tensor_size, inp, out)
}

pub unsafe fn update_weights(
    handle: &DeviceHandles,
    network_size: usize,
    decay: f32,
    beta1: f32,
    beta2: f32,
    min_weight: f32,
    max_weight: f32,
    adj: f32,
    rate: f32,
    network: *mut f32,
    momentum: *mut f32,
    velocity: *mut f32,
    gradients: *const f32,
) {
    cpu::ops::update_weights(
        &handle.cpu,
        network_size,
        decay,
        beta1,
        beta2,
        min_weight,
        max_weight,
        adj,
        rate,
        network,
        momentum,
        velocity,
        gradients,
    )
}
