use bullet_tensor::{SparseTensor, TensorBatch, Tensor, Shape};

#[test]
fn test_sparse_tensor_affine() {
    const M: usize = 3;
    const N: usize = 2;
    const B: usize = 3;

    let a_t = [
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0,
    ];

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