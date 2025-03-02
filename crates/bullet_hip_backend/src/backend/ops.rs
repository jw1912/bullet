#[rustfmt::skip]
#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
#[allow(non_camel_case_types)]
#[link(name = "kernels", kind = "static")]
extern "C" {
    pub fn activateReLU(size: usize, inp: *const f32, out: *mut f32);
    pub fn activateCReLU(size: usize, inp: *const f32, out: *mut f32);
    pub fn activateSCReLU(size: usize, inp: *const f32, out: *mut f32);
    pub fn activateSqrReLU(size: usize, inp: *const f32, out: *mut f32);
    pub fn activateSigmoid(size: usize, inp: *const f32, out: *mut f32);
    pub fn activateSquare(size: usize, inp: *const f32, out: *mut f32);
    pub fn backpropReLU(size: usize, input: *const f32, output_grad: *const f32, input_grad: *mut f32);
    pub fn backpropCReLU(size: usize, input: *const f32, output_grad: *const f32, input_grad: *mut f32);
    pub fn backpropSCReLU(size: usize, input: *const f32, output_grad: *const f32, input_grad: *mut f32);
    pub fn backpropSqrReLU(size: usize, input: *const f32, output_grad: *const f32, input_grad: *mut f32);
    pub fn backpropSigmoid(size: usize, output: *const f32, output_grad: *const f32, input_grad: *mut f32);
    pub fn backpropSquare(size: usize, output: *const f32, output_grad: *const f32, input_grad: *mut f32);
    pub fn powerError(bufferSize: usize, inputs: *const f32, results: *const f32, output: *mut f32, power: f32);
    pub fn backpropPowerError(bufferSize: usize, inputs: *const f32, results: *const f32, output_grad: *const f32, input_grads: *mut f32, power: f32);
    pub fn Adam(size: usize, beta1: f32, beta2: f32, adj: f32, rate: f32, denom: bool, network: *mut f32, momentum: *mut f32, velocity: *mut f32, gradients: *const f32);
    pub fn sparse_affine(activation: i32, stride: usize, nnz: usize, m: usize, n: usize, k: usize, bb: bool, a: *const f32, x: *const i32, b: *const f32, y: *mut f32);
    pub fn sparse_affine_backward(activation: i32, stride: usize, nnz: usize, m: usize, n: usize, k: usize, bb: bool, x: *const i32, y: *const f32, yg: *const f32, ag: *mut f32, bg: *mut f32);
    pub fn pairwiseMul(batch_size: usize, output_size: usize, input: *const f32, output: *mut f32);
    pub fn backpropPairwiseMul(batch_size: usize, output_size: usize, input: *const f32, output_grad: *const f32, input_grad: *mut f32);
    pub fn selectForward(batchSize: usize, inputSize: usize, outputSize: usize, buckets: *const i32, inp: *const f32, out: *mut f32);
    pub fn selectBackprop(batch_size: usize, input_size: usize, output_size: usize, buckets: *const i32, output_grad: *const f32, input_grad: *mut f32);
    pub fn sparse_to_dense(rows: usize, cols: usize, max_active: usize, inputs: *const i32, outputs: *mut f32);
    pub fn softmax_across_columns(rows: usize, cols: usize, inp: *const f32, out: *mut f32);
    pub fn crossentropy(size: usize, pred: *const f32, target: *const f32, out: *mut f32);
    pub fn backprop_softmax_cross_entropy(size: usize, softmaxed: *const f32, target: *const f32, out_grad: *const f32, input_grad: *mut f32);
    pub fn softmax_across_columns_masked(max_active: usize, rows: usize, cols: usize, mask: *const i32, inp: *const f32, out: *mut f32);
    pub fn crossentropy_masked(max_active: usize, cols: usize, mask: *const i32, pred: *const f32, target: *const f32, out: *mut f32, err: *mut f32);
    pub fn backprop_softmax_crossentropy_masked(max_active: usize, rows: usize, cols: usize, mask: *const i32, softmaxed: *const f32, target: *const f32, out_grad: *const f32, input_grad: *mut f32);
    pub fn sparse_mask(rows: usize, cols: usize, max_active: usize, inputs: *const f32, masks: *const i32, outputs: *mut f32);
    pub fn sparse_mask_backprop(rows: usize, cols: usize, max_active: usize, output_grads: *const f32, masks: *const i32, input_grads: *mut f32);
    pub fn gather(input_rows: usize, output_rows: usize, cols: usize, inputs: *const f32, indices: *const i32, outputs: *mut f32);
    pub fn gather_backprop(input_rows: usize, output_rows: usize, cols: usize, output_grads: *const f32, indices: *const i32, input_grads: *mut f32);
    pub fn clip(size: usize, params: *mut f32, min_weight: f32, max_weight: f32);
    pub fn scale(size: usize, params: *mut f32, alpha: f32);
}
