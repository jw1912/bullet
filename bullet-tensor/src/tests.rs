use bullet_core::Feat;

use crate::{
    create_cublas_handle, panic_if_cuda_error, Activation, Shape, SparseTensor, Tensor, TensorBatch, GpuBuffer,
};

#[test]
fn tensor_activate() {
    let mut xs = [1.0, -1.0, -0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0];

    let x = TensorBatch::new(Shape::new(1, 3), 3);
    let y = TensorBatch::new(Shape::new(1, 3), 3);

    x.load_from_cpu(&xs);
    TensorBatch::activate(3, Activation::ReLU, &x, &y);
    y.write_to_cpu(&mut xs);

    assert_eq!(xs, [1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0]);

    TensorBatch::backprop_activation(3, Activation::CReLU, &y, &x);
    x.write_to_cpu(&mut xs);

    assert_eq!(xs, [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn tensor_lt() {
    let handle = create_cublas_handle();

    const M: usize = 3;
    const N: usize = 2;
    let a = [
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0,
        //1.0, 1.0, 0.0,
        //0.0, 1.0, 1.0,
        ];
    let xs = [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0];

    let ys_cpu = [1.0, 0.0, 1.0, 1.0];

    let ys_gpu = unsafe {
        let mut a_gpu = Tensor::uninit(Shape::new(M, N));
        let xs_gpu = TensorBatch::new(Shape::new(1, M), 3);
        let ys_gpu = TensorBatch::new(Shape::new(1, N), 3);

        a_gpu.calloc();
        a_gpu.load_from_cpu(&a);
        xs_gpu.load_from_cpu(&xs);
        TensorBatch::splat_lt_nn(handle, 2, &a_gpu, &xs_gpu, &ys_gpu);

        a_gpu.free();

        let mut ys = [0.0; N * 2];
        ys_gpu.write_to_cpu(&mut ys);

        ys
    };

    let ys = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0];

    let xs = [1.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 1.0];

    let xs_gpu = unsafe {
        let mut a_gpu = Tensor::uninit(Shape::new(M, N));
        let xs_gpu = TensorBatch::new(Shape::new(1, M), 3);
        let ys_gpu = TensorBatch::new(Shape::new(1, N), 3);

        a_gpu.calloc();
        a_gpu.load_from_cpu(&a);
        ys_gpu.load_from_cpu(&ys);

        TensorBatch::splat_lt_tn(handle, 3, &a_gpu, &ys_gpu, &xs_gpu);

        a_gpu.free();

        let mut xs = [0.0; M * 3];
        xs_gpu.write_to_cpu(&mut xs);

        xs
    };

    panic_if_cuda_error("cuda error!");
    assert_eq!(xs, xs_gpu);
    assert_eq!(ys_cpu, ys_gpu);
}

#[test]
fn tensor_multi_lt() {
    let handle = create_cublas_handle();

    const M: usize = 3;
    const N: usize = 2;
    const B: usize = 3;

    let a = [
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0,
        //1.0, 1.0, 0.0,
        //0.0, 1.0, 1.0,

        0.0, 1.0,
        1.0, 0.0,
        0.0, 1.0,
        //0.0, 1.0, 0.0,
        //1.0, 0.0, 1.0,

        0.0, 0.0,
        1.0, 1.0,
        1.0, 0.0,
        //0.0, 1.0, 1.0,
        //0.0, 1.0, 0.0,
    ];
    let xs = [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        ];

    let ys_cpu = [
        1.0, 0.0,
        1.0, 0.0,
        1.0, 0.0,
    ];

    let ys_gpu = {
        let a_gpu = TensorBatch::new(Shape::new(M, N), B);
        let xs_gpu = TensorBatch::new(Shape::new(1, M), B);
        let ys_gpu = TensorBatch::new(Shape::new(1, N), B);

        a_gpu.load_from_cpu(&a);
        xs_gpu.load_from_cpu(&xs);
        TensorBatch::lt_nn(handle, B, &a_gpu, &xs_gpu, &ys_gpu);

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

        TensorBatch::lt_tn(handle, B, &a_gpu, &ys_gpu, &xs_gpu);

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

    let a_t = [
        //1.0, 1.0, 0.0,
        //0.0, 1.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0,
    ];

    let b = [0.5, -0.5];

    let xs = [Feat::new(0, 0), Feat::new(1, 1), Feat::new(2, 2)];

    unsafe {
        let mut weights = Tensor::uninit(Shape::new(N, M));
        let mut biases = Tensor::uninit(Shape::new(1, N));
        let mut inputs = SparseTensor::uninit(B, M, 1);
        let outputs = TensorBatch::new(Shape::new(1, 2 * N), B);

        weights.calloc();
        biases.calloc();

        weights.load_from_cpu(&a_t);
        biases.load_from_cpu(&b);

        inputs.append(&xs);

        SparseTensor::affine(&weights, &inputs, &biases, &outputs);

        let mut ys = [0.0; N * B * 2];
        outputs.write_to_cpu(&mut ys);

        let expected = [1.5, -0.5, 1.5, -0.5, 1.5, 0.5, 1.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        assert_eq!(expected, ys);

        let mut wg = Tensor::uninit(Shape::new(N, M));
        let mut bg = Tensor::uninit(Shape::new(1, N));

        wg.calloc();
        bg.calloc();

        SparseTensor::affine_backprop(&wg, &inputs, &bg, &outputs);

        let mut wbuf = [0.0; 6];
        wg.write_to_cpu(&mut wbuf);
        let expected = [3.0, -1.0, 3.0, 1.0, 1.0, 1.0];
        assert_eq!(wbuf, expected);

        let mut bbuf = [0.0; 2];
        bg.write_to_cpu(&mut bbuf);
        assert_eq!(bbuf, [7.0, 1.0]);

        weights.free();
        biases.free();
        wg.free();
        bg.free();
    }
}

#[test]
fn tensor_lt_nt() {
    let handle = create_cublas_handle();

    const M: usize = 3;
    const N: usize = 2;
    const B: usize = 3;

    let x = [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        ];

    let y = [
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        ];

    let x_gpu = TensorBatch::new(Shape::new(1, M), B);
    let y_gpu = TensorBatch::new(Shape::new(1, N), B);
    let a_gpu = TensorBatch::new(Shape::new(M, N), B);

    x_gpu.load_from_cpu(&x);
    y_gpu.load_from_cpu(&y);

    TensorBatch::lt_nt(handle, B, &y_gpu, &x_gpu, &a_gpu);

    let mut a = [0.0; M * N * B];
    a_gpu.write_to_cpu(&mut a);

    assert_eq!(
        a,
        [
            1.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            //1.0, 0.0, 0.0,
            //0.0, 0.0, 0.0,

            0.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
            //0.0, 0.0, 0.0,
            //0.0, 1.0, 0.0,

            0.0, 0.0,
            0.0, 0.0,
            1.0, 1.0,
            //0.0, 0.0, 1.0,
            //0.0, 0.0, 1.0,
        ]
    );
}

#[test]
fn tensor_reduce_add() {
    let handle = create_cublas_handle();
    let vecs = [
        1.0, 1.0, 2.0,
        1.0, 0.0, 1.0,
        1.0, 1.0, 3.0,
        1.0, 1.0, 1.0,
    ];

    let inp = TensorBatch::new(Shape::new(1, 3), 7);
    inp.load_from_cpu(&vecs);

    let mut out = unsafe { Tensor::uninit(Shape::new(1, 3)) };
    out.calloc();

    let ones = GpuBuffer::new(1);
    let ones_cpu = [1.0];
    ones.load_from_cpu(&ones_cpu);

    unsafe {
        TensorBatch::reduce_add(handle, &ones, 4, &inp, &out);
    }

    let mut buf = [0.0; 3];
    out.write_to_cpu(&mut buf);
    assert_eq!(buf, [4.0, 3.0, 7.0]);

    unsafe {
        out.free();
    }
}

#[test]
fn tensor_splat_add() {
    let splat = [0.5, -1.0, 1.0];
    let vecs = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.5, 1.0, 1.0];

    let mut inp = unsafe { Tensor::uninit(Shape::new(1, 3)) };
    inp.calloc();
    inp.load_from_cpu(&splat);

    let out = TensorBatch::new(Shape::new(1, 3), 7);
    out.load_from_cpu(&vecs);

    unsafe {
        TensorBatch::splat_add(4, &inp, &out);
    }

    let mut buf = [0.0; 12];
    out.write_to_cpu(&mut buf);
    assert_eq!(buf, [1.5, 0.0, 1.0, 1.5, 0.0, 1.0, 1.5, 0.0, 2.0, 2.0, 0.0, 2.0]);

    unsafe {
        inp.free();
    }
}

#[test]
fn affine() {
    let handle = create_cublas_handle();
    let inps = [1.0, 2.0, -0.5];
    let ws = [
        1.0, 0.0, 1.0,
        0.0, 1.0, 0.0,
        1.0, 0.0, 1.0,
    ];
    let bs = [0.1, 0.2, 0.3];

    unsafe {
        let mut w = Tensor::uninit(Shape::new(3, 3));
        let mut b = Tensor::uninit(Shape::new(1, 3));
        let x = TensorBatch::new(Shape::new(1, 3), 1);
        let y = TensorBatch::new(Shape::new(1, 3), 1);
        let ones = GpuBuffer::new(1);
        ones.load_from_cpu(&[1.0]);

        w.calloc();
        b.calloc();

        w.load_from_cpu(&ws);
        b.load_from_cpu(&bs);

        x.load_from_cpu(&inps);

        TensorBatch::affine(handle, 1, &w, &x, &b, &y);

        let mut buf = [0.0; 3];
        y.write_to_cpu(&mut buf);
        assert_eq!(buf, [0.6, 2.2, 0.8]);

        let mut wg = Tensor::uninit(Shape::new(3, 3));
        let mut bg = Tensor::uninit(Shape::new(1, 3));
        let wi = TensorBatch::new(Shape::new(3, 3), 1);

        wg.calloc();
        bg.calloc();

        TensorBatch::backprop_affine(handle, &ones, 1, &w, &y, &x, &wg, &bg, &wi);

        x.write_to_cpu(&mut buf);
        assert_eq!(buf, [1.4000001, 2.2, 1.4000001]);

        let mut wbuf = [0.0; 9];
        wg.write_to_cpu(&mut wbuf);
        assert_eq!(wbuf, [
            0.6, 2.2, 0.8,
            1.2, 4.4, 1.6,
            -0.3, -1.1, -0.4,
            //0.6, 1.2, -0.3,
            //2.2, 4.4, -1.1,
            //0.8, 1.6, -0.4,
            ]);

        let mut bbuf = [0.0; 3];
        bg.write_to_cpu(&mut bbuf);
        assert_eq!(bbuf, [0.6, 2.2, 0.8]);

        w.free();
        b.free();
        wg.free();
        bg.free();
    }
}

#[test]
fn mse() {
    let out = [1.5, 0.0, 1.0];
    let res = [0.5, 0.5, 0.5];

    let error = GpuBuffer::new(1);

    let x = TensorBatch::new(Shape::new(1, 1), 9);
    x.load_from_cpu(&out);

    let r = TensorBatch::new(Shape::new(1, 1), 9);
    r.load_from_cpu(&res);

    x.sigmoid_mse(3, &r, &error);

    let mut buf = [0.0; 3];
    x.write_to_cpu(&mut buf);

    for (e, (&o, &r)) in buf.iter().zip(out.iter().zip(res.iter())) {
        let sig = 1.0 / (1.0 + (-o).exp());

        let diff = e - (sig - r) * sig * (1.0 - sig);
        assert!(diff.abs() < 0.00001);
    }
}
