use bullet_tensor::{create_cublas_handle, panic_if_cuda_error, Shape, TensorBatch};

fn cpu_linear<const M: usize, const N: usize, const MNB: usize>(
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
fn test_tensor_lt() {
    let handle = create_cublas_handle();

    const M: usize = 3;
    const N: usize = 2;
    const MN: usize = 6;
    const B: usize = 3;

    let a = [
        1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0,
    ];
    let xs = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    let ys_cpu = cpu_linear::<M, N, { MN * B }>(&a, &xs);

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
