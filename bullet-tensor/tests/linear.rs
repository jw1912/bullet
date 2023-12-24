use bullet_tensor::{create_cublas_handle, panic_if_cuda_error, Shape, Tensor, TensorBatch};

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
fn test_tensor_lt() {
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
