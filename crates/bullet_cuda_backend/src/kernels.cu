#define BULLET_CUDA_UTILS
#define BULLET_KERNEL extern "C" __global__ void
#define BULLET_KERNEL_IMPL __device__ __forceinline__ void

typedef float(*OpType)(float);
typedef float(*BinaryOpType)(float, float);

constexpr int MaximumBlocksY = 32768;

__device__ float Identity([[maybe_unused]] float in) { return in; }
__device__ float ReLU(float in) { return in > 0.0F ? in : 0.0F; }
__device__ float CReLU(float in) { return in < 0.0F ? 0.0F : (in > 1.0F ? 1.0F : in); }
__device__ float SCReLU(float in) { return in < 0.0F ? 0.0F : (in > 1.0F ? 1.0F : (in * in)); }
__device__ float SqrReLU(float in) { return in < 0.0F ? 0.0F : (in * in); }
__device__ float sigmoid(float in) { return 1.0F / (1.0F + expf(-in)); }

__device__ float primeIdentity([[maybe_unused]] float in) { return 1.0F; }
__device__ float primeReLU(float in) { return in > 0.0F ? 1.0F : 0.0F; }
__device__ float primeCReLU(float in) { return in > 0.0F && in < 1.0F ? 1.0F : 0.0F; }
__device__ float primeSCReLU(float in) { return in > 0.0F && in < 1.0F ? 2.0F * in : 0.0F; }
__device__ float primeSqrReLU(float in) { return in > 0.0F ? 2.0F * in : 0.0F; }
__device__ float primeSigmoid(float in) {
    const float act = sigmoid(in);
    return act * (1.0F - act);
}

__device__ float primeInvIdentity([[maybe_unused]] float in) { return 1.0F; }
__device__ float primeInvReLU(float in) { return in > 0.0F ? 1.0F : 0.0F; }
__device__ float primeInvCReLU(float in) { return in > 0.0F && in < 1.0F ? 1.0F : 0.0F; }
__device__ float primeInvSCReLU(float in) { return in > 0.0F && in < 1.0F ? 2.0F * sqrtf(in) : 0.0F; }
__device__ float primeInvSqrReLU(float in) { return in > 0.0F ? 2.0F * sqrtf(in) : 0.0F; }
__device__ float primeInvSigmoid(float in) { return in * (1.0F - in); }

#define ACTIVATE(name, op)\
BULLET_KERNEL name(const int size, const float* in, float* out)\
{\
    buffer_operation_kernel<op>(size, in, out);\
}\

#define BACKPROP(name, op)\
BULLET_KERNEL name(const int size, const float* input, const float* output_grad, float* input_grad)\
{\
    buffer_backprop_kernel<op>(size, input, output_grad, input_grad);\
}\

template<OpType op>
BULLET_KERNEL_IMPL buffer_operation_kernel(const int size, const float* in, float* out)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size / 4)
    {
        const float4 a = ((const float4 *)in)[tid];
        ((float4 *)out)[tid] = make_float4(op(a.x), op(a.y), op(a.z), op(a.w));
    }
    else if (4 * tid < size)
    {
        for (int i = 0; i < size - 4 * tid; i++)
        {
            const int idx = 4 * tid + i;
            out[idx] = op(in[idx]);
        }
    }
}

template<OpType op>
BULLET_KERNEL_IMPL buffer_backprop_kernel(const int size, const float* input, const float* output_grad, float* input_grad)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size / 4)
    {
        const float4 this_in = ((const float4 *)input)[tid];
        const float4 this_out_grad = ((const float4 *)output_grad)[tid];
        float4 curr_input_grad = ((const float4 *)input_grad)[tid];

        curr_input_grad.x += op(this_in.x) * this_out_grad.x;
        curr_input_grad.y += op(this_in.y) * this_out_grad.y;
        curr_input_grad.z += op(this_in.z) * this_out_grad.z;
        curr_input_grad.w += op(this_in.w) * this_out_grad.w;

        ((float4 *)input_grad)[tid] = curr_input_grad;
    }
    else if (4 * tid < size)
    {
        for (int i = 0; i < size - 4 * tid; i++)
        {
            const int j = 4 * tid + i;
            input_grad[j] += op(input[j]) * output_grad[j];
        }
    }
}

ACTIVATE(ForwardReluKernel, ReLU)
ACTIVATE(ForwardCreluKernel, CReLU)
ACTIVATE(ForwardScreluKernel, SCReLU)
ACTIVATE(ForwardSqrReluKernel, SqrReLU)
ACTIVATE(ForwardSigmoidKernel, sigmoid)

BACKPROP(BackwardReluKernel, primeReLU)
BACKPROP(BackwardCreluKernel, primeCReLU)
BACKPROP(BackwardScreluKernel, primeSCReLU)
BACKPROP(BackwardSqrReluKernel, primeSqrReLU)
BACKPROP(BackwardSigmoidKernel, primeSigmoid)

BULLET_KERNEL ScaleAssignKernel(const int size, float* params, const float alpha) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size / 4)
    {
        float4 a = ((float4 *)params)[tid];
        
        a.x *= alpha;
        a.y *= alpha;
        a.z *= alpha;
        a.w *= alpha;

        ((float4 *)params)[tid] = a;
    }
    else if (4 * tid < size)
    {
        for (int i = 0; i < size - 4 * tid; i++)
        {
            const int j = 4 * tid + i;
            params[j] *= alpha;
        }
    }
}

BULLET_KERNEL ScaleAddAssignKernel(const int size, const float alpha, float* ap, const float beta, const float* bp) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size / 4)
    {
        float4 a = ((float4 *)ap)[tid];
        const float4 b = ((const float4 *)bp)[tid];
        
        a.x = alpha * a.x + beta * b.x;
        a.y = alpha * a.y + beta * b.y;
        a.z = alpha * a.z + beta * b.z;
        a.w = alpha * a.w + beta * b.w;

        ((float4 *)ap)[tid] = a;
    }
    else if (4 * tid < size)
    {
        for (int i = 0; i < size - 4 * tid; i++)
        {
            const int j = 4 * tid + i;
            ap[j] = alpha * ap[j] + beta * bp[j];
        }
    }
}

BULLET_KERNEL ScaleKernel(const int size, const float alpha, const float* inp, float* out) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size / 4)
    {
        float4 a = ((float4 *)out)[tid];
        const float4 b = ((const float4 *)inp)[tid];
        
        a.x = alpha * b.x;
        a.y = alpha * b.y;
        a.z = alpha * b.z;
        a.w = alpha * b.w;

        ((float4 *)out)[tid] = a;
    }
    else if (4 * tid < size)
    {
        for (int i = 0; i < size - 4 * tid; i++)
        {
            const int j = 4 * tid + i;
            out[j] = alpha * inp[j];
        }
    }
}

BULLET_KERNEL LinearCombKernel(const int size, const float alpha, const float* ap, const float beta, const float* bp, float* cp) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size / 4)
    {
        const float4 a = ((const float4 *)ap)[tid];
        const float4 b = ((const float4 *)bp)[tid];
        float4 c = ((float4 *)cp)[tid];
        
        c.x = alpha * a.x + beta * b.x;
        c.y = alpha * a.y + beta * b.y;
        c.z = alpha * a.z + beta * b.z;
        c.w = alpha * a.w + beta * b.w;

        ((float4 *)cp)[tid] = c;
    }
    else if (4 * tid < size)
    {
        for (int i = 0; i < size - 4 * tid; i++)
        {
            const int j = 4 * tid + i;
            cp[j] = alpha * ap[j] + beta * bp[j];
        }
    }
}

constexpr float Epsilon = 0.00000001F;

BULLET_KERNEL_IMPL adamOp(
    const float beta1,
    const float beta2,
    const float adj,
    const float rate,
    const float decay,
    const float wmin,
    const float wmax,
    const bool denom,
    float* p,
    float* m,
    float* v,
    const float* g)
{
    p[0] *= decay;

    const float grad = adj * g[0];
    m[0] = beta1 * m[0] + (1.0F - beta1) * grad;
    v[0] = beta2 * v[0] + (1.0F - beta2) * grad * grad;

    float val = m[0];
    if (denom) val /= sqrt(v[0]) + Epsilon;
    p[0] -= rate * val;

    p[0] = min(max(p[0], wmin), wmax);
}

BULLET_KERNEL AdamKernel(
    const int size,
    const float beta1,
    const float beta2,
    const float adj,
    const float rate,
    const float decay,
    const float min,
    const float max,
    const bool denom,
    float* network,
    float* momentum,
    float* velocity,
    const float* gradients)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size / 4)
    {
        float4 p = ((float4 *)network)[tid];
        float4 m = ((float4 *)momentum)[tid];
        float4 v = ((float4 *)velocity)[tid];
        const float4 g = ((const float4 *)gradients)[tid];

        adamOp(beta1, beta2, adj, rate, decay, min, max, denom, &p.x, &m.x, &v.x, &g.x);
        adamOp(beta1, beta2, adj, rate, decay, min, max, denom, &p.y, &m.y, &v.y, &g.y);
        adamOp(beta1, beta2, adj, rate, decay, min, max, denom, &p.z, &m.z, &v.z, &g.z);
        adamOp(beta1, beta2, adj, rate, decay, min, max, denom, &p.w, &m.w, &v.w, &g.w);

        ((float4 *)network)[tid] = p;
        ((float4 *)momentum)[tid] = m;
        ((float4 *)velocity)[tid] = v;
    }
    else if (4 * tid < size)
    {
        for (int i = 0; i < size - 4 * tid; i++)
        {
            const int j = 4 * tid + i;
            adamOp(beta1, beta2, adj, rate, decay, min, max, denom, &network[j], &momentum[j], &velocity[j], &gradients[j]);
        }
    }
}

BULLET_KERNEL ClipKernel(const int size, float* params, const float min_weight, const float max_weight) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size / 4)
    {
        float4 a = ((float4 *)params)[tid];
        
        a.x = min(max(a.x, min_weight), max_weight);
        a.y = min(max(a.y, min_weight), max_weight);
        a.z = min(max(a.z, min_weight), max_weight);
        a.w = min(max(a.w, min_weight), max_weight);

        ((float4 *)params)[tid] = a;
    }
    else if (4 * tid < size)
    {
        for (int i = 0; i < size - 4 * tid; i++)
        {
            const int j = 4 * tid + i;
            params[j] = min(max(params[j], min_weight), max_weight);
        }
    }
}

BULLET_KERNEL PairwiseMulKernel(
    const int output_size,
    const int batch_size,
    const float* input,
    float* output)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= output_size * batch_size)
        return;

    const int idxInBatch = tid / output_size;
    const int idxInOutput = tid % output_size;

    const float* thisInp = input + 2 * output_size * idxInBatch + idxInOutput;
    float* thisOut = output + output_size * idxInBatch + idxInOutput;

    thisOut[0] = thisInp[0] * thisInp[output_size];
}

BULLET_KERNEL PairwiseMulBackwardKernel(
    const int output_size,
    const int batch_size,
    const float* input,
    const float* output_grad,
    float* input_grad)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= output_size * batch_size)
        return;

    const int idxInBatch = tid / output_size;
    const int idxInOutput = tid % output_size;

    const float* thisOutputGrad = output_grad + output_size * idxInBatch + idxInOutput;
    
    const int inputOffset = 2 * output_size * idxInBatch + idxInOutput;
    const float* thisInput = input + inputOffset;
    float* thisInputGrad = input_grad + inputOffset;

    const float gradIn = thisOutputGrad[0];

    thisInputGrad[0] += gradIn * thisInput[output_size];
    thisInputGrad[output_size] += gradIn * thisInput[0];
}

BULLET_KERNEL PowerErrorKernel(
    const int size,
    const float* inputs,
    const float* results,
    float* output,
    const float power)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size)
        return;

    output[i] = powf(abs(inputs[i] - results[i]), power);
}

BULLET_KERNEL PowerErrorBackwardKernel(
    const int size,
    const float* inputs,
    const float* results,
    const float* output_grad,
    float* input_grads,
    const float power)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size)
        return;

    const float diff = inputs[i] - results[i];
    const float grad = power * powf(abs(diff), power - 1.0F) * output_grad[i];
    input_grads[i] += diff > 0.0F ? grad : -grad;
}

BULLET_KERNEL SetKernel(float* buf, int size, float val)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) buf[tid] = val;
}

#define SCALAR_KERNEL_FORWARD(name, op)\
BULLET_KERNEL name(const int size, const float alpha, const float* in, float* out)\
{\
    scalar_kernel_forward<op>(size, alpha, in, out);\
}\

#define SCALAR_KERNEL_BACKWARD(name, op)\
BULLET_KERNEL name(const int size, const float alpha, const float* input, const float* output_grad, float* input_grad)\
{\
    scalar_kernel_backward<op>(size, alpha, input, output_grad, input_grad);\
}\

template<BinaryOpType op>
BULLET_KERNEL_IMPL scalar_kernel_forward(const int size, const float alpha, const float* inp, float* out) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size / 4)
    {
        float4 a = ((float4 *)out)[tid];
        const float4 b = ((const float4 *)inp)[tid];
        
        a.x = op(b.x, alpha);
        a.y = op(b.y, alpha);
        a.z = op(b.z, alpha);
        a.w = op(b.w, alpha);

        ((float4 *)out)[tid] = a;
    }
    else if (4 * tid < size)
    {
        for (int i = 0; i < size - 4 * tid; i++)
        {
            const int j = 4 * tid + i;
            out[j] = op(inp[j], alpha);
        }
    }
}

template<BinaryOpType op>
BULLET_KERNEL_IMPL scalar_kernel_backward(const int size, const float alpha, const float* input, const float* output_grad, float* input_grad) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size / 4)
    {
        const float4 this_in = ((const float4 *)input)[tid];
        const float4 this_out_grad = ((const float4 *)output_grad)[tid];
        float4 curr_input_grad = ((const float4 *)input_grad)[tid];

        curr_input_grad.x += op(this_in.x, alpha) * this_out_grad.x;
        curr_input_grad.y += op(this_in.y, alpha) * this_out_grad.y;
        curr_input_grad.z += op(this_in.z, alpha) * this_out_grad.z;
        curr_input_grad.w += op(this_in.w, alpha) * this_out_grad.w;

        ((float4 *)input_grad)[tid] = curr_input_grad;
    }
    else if (4 * tid < size)
    {
        for (int i = 0; i < size - 4 * tid; i++)
        {
            const int j = 4 * tid + i;
            input_grad[j] += op(input[j], alpha) * output_grad[j];
        }
    }
}

__device__ float add(float a, float b) { return a + b; }
__device__ float abs_pow(float a, float b) { return powf(abs(a), b); }
__device__ float abs_pow_backward(float a, float b) {
    const float grad = b * powf(abs(a), b - 1.0F);
    return a > 0.0F ? grad : -grad;
};

SCALAR_KERNEL_FORWARD(AddScalarKernel, add)
SCALAR_KERNEL_FORWARD(AbsPowScalarKernel, abs_pow)
SCALAR_KERNEL_BACKWARD(AbsPowScalarBackwardKernel, abs_pow_backward)

BULLET_KERNEL select(
    const int batch_size,
    const int input_size,
    const int output_size,
    const int* buckets,
    const float* in,
    float* out)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= batch_size * output_size)
        return;

    const int idxInBatch = tid / output_size;
    const int idxInOutput = tid % output_size;

    const int thisBucket = buckets[idxInBatch];

    const float* thisInput = in + input_size * idxInBatch + output_size * thisBucket + idxInOutput;
    float* thisOutput = out + output_size * idxInBatch + idxInOutput;

    thisOutput[0] = thisInput[0];
}

BULLET_KERNEL select_backprop(
    const int batch_size,
    const int input_size,
    const int output_size,
    const int* buckets,
    const float* output_grad,
    float* input_grad)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= batch_size * output_size)
        return;

    const int idxInBatch = tid / output_size;
    const int idxInOutput = tid % output_size;

    const int thisBucket = buckets[idxInBatch];

    const float* thisOutputGrad = output_grad + output_size * idxInBatch + idxInOutput;
    float* thisInputGrad = input_grad + input_size * idxInBatch + output_size * thisBucket + idxInOutput;

    thisInputGrad[0] += thisOutputGrad[0];
}

// it is assumed that we will only be using this on matrixs with small number of columns
// (so the perf won't be terrible)
BULLET_KERNEL softmax(const int rows, const int cols, const float* input, float* output)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= cols)
        return;

    const float* thisColumn = input + rows * tid;
    float* thisOutput = output + rows * tid;

    float maximum = thisColumn[0];

    for (int i = 1; i < rows; i++) {
        maximum = max(maximum, thisColumn[i]);
    }

    float total = 0.0F;

    for (int i = 0; i < rows; i++) {
        const float exp = expf(thisColumn[i] - maximum);
        thisOutput[i] = exp;
        total += exp;
    }

    for (int i = 0; i < rows; i++) {
        thisOutput[i] /= total;
    }
}

BULLET_KERNEL cross_entropy(const int size, const float* pred, const float* target, float* out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size)
        return;

    out[i] = (target[i] == 0.0F) ? 0.0F : -target[i] * logf(pred[i]);
}

BULLET_KERNEL backprop_softmax_cross_entropy(
    const int size,
    const float* softmaxed,
    const float* target,
    const float* out_grad,
    float* input_grad)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size)
        return;

    input_grad[i] += (softmaxed[i] - target[i]) * out_grad[i];
}

BULLET_KERNEL sparse_to_dense(const int rows, const int cols, const int max_active, const int* inputs, float* outputs)
{
    const int elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= cols)
        return;

    const int* thisInput = inputs + max_active * elem;
    float* thisOutput = outputs + rows * elem;

    for (int i = 0; i < max_active; i++) {
        const int inp = thisInput[i];

        if (inp == -1)
            break;

        thisOutput[inp] = 1.0F;
    }
}

#define SPARSE_MATMUL_BWD_KERNEL(name, op)\
BULLET_KERNEL name(\
    const int stride,\
    const int nnz,\
    const int m,\
    const int k,\
    const int* X,\
    const float* Y,\
    const float* Yg,\
    float* Ag)\
{\
    sparse_affine_backward_kernel<op>(stride, nnz, m, k, false, X, Y, Yg, Ag, nullptr);\
}\

#define SPARSE_AFFINE_BWD_KERNEL(name, op)\
BULLET_KERNEL name(\
    const int stride,\
    const int nnz,\
    const int m,\
    const int k,\
    const bool Bb,\
    const int* X,\
    const float* Y,\
    const float* Yg,\
    float* Ag,\
    float* Bg)\
{\
    sparse_affine_backward_kernel<op>(stride, nnz, m, k, Bb, X, Y, Yg, Ag, Bg);\
}\

template<OpType op>
BULLET_KERNEL_IMPL sparse_affine_backward_kernel(
    const int stride,
    const int nnz,
    const int m,
    const int k,
    const bool Bb,
    const int* X,
    const float* Y,
    const float* Yg,
    float* Ag,
    float* Bg)
{
    const int loc = MaximumBlocksY * blockIdx.z + blockIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || loc >= k)
        return;

    const int* tX = X + nnz * loc;
    const int offset = stride * m * loc;

    const float tE = op(Y[offset + row]) * Yg[offset + row];

    if (Bg != nullptr && tE != 0.0F)
    {   
        const int offset2 = Bb ? m * loc : 0;
        atomicAdd(&Bg[offset2 + row], tE);
    }

    for (int i = 0; i < nnz; i++)
    {
        const int j = tX[i];

        if (j == -1)
            break;

        if (tE != 0.0F)
            atomicAdd(&Ag[j * m + row], tE);
    }
}

SPARSE_MATMUL_BWD_KERNEL(SparseMatmulBwd, primeInvIdentity);
SPARSE_MATMUL_BWD_KERNEL(SparseMatmulBwdRelu, primeInvReLU);
SPARSE_MATMUL_BWD_KERNEL(SparseMatmulBwdCrelu, primeInvCReLU);
SPARSE_MATMUL_BWD_KERNEL(SparseMatmulBwdScrelu, primeInvSCReLU);
SPARSE_MATMUL_BWD_KERNEL(SparseMatmulBwdSqrRelu, primeInvSqrReLU);
SPARSE_MATMUL_BWD_KERNEL(SparseMatmulBwdSigmoid, primeInvSigmoid);

SPARSE_AFFINE_BWD_KERNEL(SparseAffineBwd, primeInvIdentity);
SPARSE_AFFINE_BWD_KERNEL(SparseAffineBwdRelu, primeInvReLU);
SPARSE_AFFINE_BWD_KERNEL(SparseAffineBwdCrelu, primeInvCReLU);
SPARSE_AFFINE_BWD_KERNEL(SparseAffineBwdScrelu, primeInvSCReLU);
SPARSE_AFFINE_BWD_KERNEL(SparseAffineBwdSqrRelu, primeInvSqrReLU);
SPARSE_AFFINE_BWD_KERNEL(SparseAffineBwdSigmoid, primeInvSigmoid);
