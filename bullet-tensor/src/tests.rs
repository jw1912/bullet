use crate::{
    create_cublas_handle, panic_if_cuda_error, Activation, Shape, SparseTensor, Tensor, TensorBatch,
};

#[test]
fn tensor_activate() {
    let mut xs = [1.0, -1.0, -0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0];

    let x = TensorBatch::new(Shape::new(1, 3), 3);
    let y = TensorBatch::new(Shape::new(1, 3), 3);

    x.load_from_cpu(&xs);
    TensorBatch::activate(Activation::ReLU, &x, &y);
    y.write_to_cpu(&mut xs);

    assert_eq!(xs, [1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0]);
}

fn cpu_linear<const M: usize, const N: usize, const MN: usize>(
    a: &[f32; MN],
    xs: &[f32],
) -> Vec<f32> {
    assert_eq!(xs.len() % M, 0);
    let mut ys = Vec::new();

    for x in xs.chunks_exact(M) {
        let mut y = [0.0; N];

        for (i, row) in a.chunks(M).enumerate() {
            for j in 0..M {
                y[i] += row[j] * x[j];
            }
        }

        for y1 in y {
            ys.push(y1);
        }
    }

    ys
}

/// Calculates
/// [1.0, 1.0, 0.0] [1.0]   [1.0]
/// [0.0, 1.0, 1.0] [0.0] = [0.0]
///                 [0.0]
///
/// [1.0, 1.0, 0.0] [0.0]   [1.0]
/// [0.0, 1.0, 1.0] [1.0] = [1.0]
///                 [0.0]
///
/// [1.0, 1.0, 0.0] [0.0]   [0.0]
/// [0.0, 1.0, 1.0] [0.0] = [1.0]
///                 [1.0]
///
/// Giving output
/// [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
///
/// Then Calculates
/// [1.0, 0.0] [1.0]   [1.0]
/// [1.0, 1.0] [0.0] = [1.0]
/// [0.0, 1.0]         [0.0]
///
/// [1.0, 0.0] [1.0]   [1.0]
/// [1.0, 1.0] [1.0] = [2.0]
/// [0.0, 1.0]         [1.0]
///
/// [1.0, 0.0] [0.0]   [0.0]
/// [1.0, 1.0] [1.0] = [1.0]
/// [0.0, 1.0]         [1.0]
///
/// Giving output
/// [1.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 1.0]
#[test]
fn tensor_lt() {
    let handle = create_cublas_handle();

    const M: usize = 3;
    const N: usize = 2;
    const MN: usize = 6;
    let a = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
    let xs = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    let ys_cpu = cpu_linear::<M, N, MN>(&a, &xs);

    let ys_gpu = unsafe {
        let mut a_gpu = Tensor::uninit(Shape::new(M, N));
        let xs_gpu = TensorBatch::new(Shape::new(1, M), 3);
        let ys_gpu = TensorBatch::new(Shape::new(1, N), 3);

        a_gpu.calloc();
        a_gpu.load_from_cpu(&a);
        xs_gpu.load_from_cpu(&xs);
        TensorBatch::single_lt(handle, &a_gpu, &xs_gpu, &ys_gpu);

        a_gpu.free();

        let mut ys = [0.0; N * 3];
        ys_gpu.write_to_cpu(&mut ys);

        ys
    };

    let ys = ys_gpu;

    let xs = [1.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 1.0];

    let xs_gpu = unsafe {
        let mut a_gpu = Tensor::uninit(Shape::new(M, N));
        let xs_gpu = TensorBatch::new(Shape::new(1, M), 3);
        let ys_gpu = TensorBatch::new(Shape::new(1, N), 3);

        a_gpu.calloc();
        a_gpu.load_from_cpu(&a);
        ys_gpu.load_from_cpu(&ys);

        TensorBatch::single_tlt(handle, &a_gpu, &ys_gpu, &xs_gpu);

        a_gpu.free();

        let mut xs = [0.0; M * 3];
        xs_gpu.write_to_cpu(&mut xs);

        xs
    };

    panic_if_cuda_error("cuda error!");
    assert_eq!(xs, xs_gpu);
    assert_eq!(ys_cpu, ys_gpu);
}

fn cpu_multi_linear<const M: usize, const N: usize, const MNB: usize>(
    a: &[f32; MNB],
    xs: &[f32],
) -> Vec<f32> {
    assert_eq!(xs.len() % M, 0);
    let mut ys = Vec::new();
    for (x, a) in xs.chunks_exact(M).zip(a.chunks_exact(M * N)) {
        let mut y = [0.0; N];

        for (i, row) in a.chunks(M).enumerate() {
            for j in 0..M {
                y[i] += row[j] * x[j];
            }
        }

        for y1 in y {
            ys.push(y1);
        }
    }

    ys
}

#[test]
fn tensor_multi_lt() {
    let handle = create_cublas_handle();

    const M: usize = 3;
    const N: usize = 2;
    const MN: usize = 6;
    const B: usize = 3;

    let a = [
        1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0,
    ];
    let xs = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    let ys_cpu = cpu_multi_linear::<M, N, { MN * B }>(&a, &xs);

    let ys_gpu = {
        let a_gpu = TensorBatch::new(Shape::new(M, N), B);
        let xs_gpu = TensorBatch::new(Shape::new(1, M), B);
        let ys_gpu = TensorBatch::new(Shape::new(1, N), B);

        a_gpu.load_from_cpu(&a);
        xs_gpu.load_from_cpu(&xs);
        TensorBatch::multi_lt(handle, &a_gpu, &xs_gpu, &ys_gpu);

        let mut ys = [0.0; N * B];
        ys_gpu.write_to_cpu(&mut ys);

        ys
    };

    assert_eq!(ys_cpu, ys_gpu);

    let ys = ys_gpu;

    let xs = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0];

    let xs_gpu = {
        let a_gpu = TensorBatch::new(Shape::new(M, N), B);
        let xs_gpu = TensorBatch::new(Shape::new(1, M), B);
        let ys_gpu = TensorBatch::new(Shape::new(1, N), B);

        a_gpu.load_from_cpu(&a);
        ys_gpu.load_from_cpu(&ys);

        TensorBatch::multi_tlt(handle, &a_gpu, &ys_gpu, &xs_gpu);

        let mut xs = [0.0; M * B];
        xs_gpu.write_to_cpu(&mut xs);

        xs
    };

    panic_if_cuda_error("cuda error!");
    assert_eq!(xs, xs_gpu);
}

#[test]
fn tensor_sparse_affine() {
    const M: usize = 3;
    const N: usize = 2;
    const B: usize = 3;

    let a_t = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0];

    let b = [0.5, -0.5];

    let xs = [0, 1, 2];

    let ys_gpu = unsafe {
        let mut weights = Tensor::uninit(Shape::new(N, M));
        let mut biases = Tensor::uninit(Shape::new(1, N));
        let mut inputs = SparseTensor::uninit(B, M, 1, N);
        let outputs = TensorBatch::new(Shape::new(1, N), B);

        weights.calloc();
        biases.calloc();

        weights.load_from_cpu(&a_t);
        biases.load_from_cpu(&b);

        inputs.load_from_cpu(&xs);

        SparseTensor::affine(&weights, &inputs, &biases, &outputs);

        let mut ys = [0.0; N * B];
        outputs.write_to_cpu(&mut ys);

        weights.free();
        biases.free();

        ys
    };

    let expected = [1.5, -0.5, 1.5, 0.5, 0.5, 0.5];
    assert_eq!(expected, ys_gpu);
}
