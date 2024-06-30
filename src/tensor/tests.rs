use crate::{backend::{DeviceHandles, util}, Activation, loader::Feat};
use super::{Shape, SparseTensor, Tensor, TensorBatch, DeviceBuffer};

#[test]
fn tensor_activate() {
    let handle = DeviceHandles::default();
    let mut xs = [1.0, -1.0, -0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0];

    let x = TensorBatch::new(Shape::new(1, 3), 3);
    let y = TensorBatch::new(Shape::new(1, 3), 3);

    x.load_from_host(&xs);
    util::panic_if_device_error("Error");
    TensorBatch::activate(&handle, 3, Activation::ReLU, &x, &y);
    util::panic_if_device_error("Error");
    y.write_to_host(&mut xs);

    assert_eq!(xs, [1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0]);

    TensorBatch::backprop_activation(&handle, 3, Activation::CReLU, &y, &x);
    util::panic_if_device_error("Error");
    x.write_to_host(&mut xs);

    assert_eq!(xs, [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn tensor_lt() {
    let handle = DeviceHandles::default();

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
        a_gpu.load_from_host(&a);
        xs_gpu.load_from_host(&xs);

        util::panic_if_device_error("Error");
        TensorBatch::splat_mul_matrix_vector(&handle, 2, &a_gpu, &xs_gpu, &ys_gpu);
        util::panic_if_device_error("Error");

        a_gpu.free();

        let mut ys = [0.0; N * 2];
        ys_gpu.write_to_host(&mut ys);

        ys
    };

    let ys = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0];

    let xs = [1.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 1.0];

    let xs_gpu = unsafe {
        let mut a_gpu = Tensor::uninit(Shape::new(M, N));
        let xs_gpu = TensorBatch::new(Shape::new(1, M), 3);
        let ys_gpu = TensorBatch::new(Shape::new(1, N), 3);

        a_gpu.calloc();
        a_gpu.load_from_host(&a);
        ys_gpu.load_from_host(&ys);

        util::panic_if_device_error("Error");
        TensorBatch::splat_mul_matrixt_vector(&handle, 3, &a_gpu, &ys_gpu, &xs_gpu);
        util::panic_if_device_error("Error");

        a_gpu.free();

        let mut xs = [0.0; M * 3];
        xs_gpu.write_to_host(&mut xs);

        xs
    };

    util::panic_if_device_error("cuda error!");
    assert_eq!(xs, xs_gpu);
    assert_eq!(ys_cpu, ys_gpu);
}

#[test]
fn tensor_sparse_affine() {
    let handle = DeviceHandles::default();

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
        let zeros = TensorBatch::new(Shape::new(1, 2 * N), B);

        zeros.load_from_host(&[0.0; 2 * N * B]);

        weights.calloc();
        biases.calloc();

        weights.load_from_host(&a_t);
        biases.load_from_host(&b);

        inputs.append(&xs);

        util::panic_if_device_error("Error");
        SparseTensor::affine(&handle, &weights, &inputs, &biases, &outputs);
        util::panic_if_device_error("Error");

        let mut ys = [0.0; N * B * 2];
        outputs.write_to_host(&mut ys);

        let expected = [1.5, -0.5, 1.5, -0.5, 1.5, 0.5, 1.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        assert_eq!(expected, ys);

        let mut wg = Tensor::uninit(Shape::new(N, M));
        let mut bg = Tensor::uninit(Shape::new(1, N));

        wg.calloc();
        bg.calloc();

        util::panic_if_device_error("Error");
        SparseTensor::affine_backprop(&handle, &wg, &inputs, &bg, &outputs, &zeros, 0.0);
        util::panic_if_device_error("Error");

        let mut wbuf = [0.0; 6];
        wg.write_to_host(&mut wbuf);
        let expected = [3.0, -1.0, 3.0, 1.0, 1.0, 1.0];
        assert_eq!(wbuf, expected);

        let mut bbuf = [0.0; 2];
        bg.write_to_host(&mut bbuf);
        assert_eq!(bbuf, [7.0, 1.0]);

        weights.free();
        biases.free();
        wg.free();
        bg.free();
    }
}

#[test]
fn reduce_add_mul_vector_vectort() {
    let handle = DeviceHandles::default();

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

    unsafe {
        let mut a_gpu = Tensor::uninit(Shape::new(M, N));
        a_gpu.calloc();

        x_gpu.load_from_host(&x);
        y_gpu.load_from_host(&y);

        util::panic_if_device_error("Error");
        TensorBatch::reduce_add_mul_vector_vectort(&handle, B, &y_gpu, &x_gpu, &a_gpu);
        util::panic_if_device_error("Error");

        let mut a = [0.0; M * N];
        a_gpu.write_to_host(&mut a);

        assert_eq!(
            a,
            [
                1.0, 0.0,
                0.0, 1.0,
                1.0, 1.0,
                //1.0, 0.0, 1.0,
                //0.0, 1.0, 1.0,
            ]
        );

        a_gpu.free();
    }
}

#[test]
fn tensor_reduce_add() {
    let handle = DeviceHandles::default();
    let vecs = [
        1.0, 1.0, 2.0,
        1.0, 0.0, 1.0,
        1.0, 1.0, 3.0,
        1.0, 1.0, 1.0,
    ];

    let inp = TensorBatch::new(Shape::new(1, 3), 7);
    inp.load_from_host(&vecs);

    let mut out = unsafe { Tensor::uninit(Shape::new(1, 3)) };
    out.calloc();

    let ones = DeviceBuffer::new(4);
    let ones_cpu = [1.0; 4];
    ones.load_from_host(&ones_cpu);

    util::panic_if_device_error("Error");
    unsafe {
        TensorBatch::reduce_add(&handle, &ones, 4, &inp, &out);
    }
    util::panic_if_device_error("Error");

    let mut buf = [0.0; 3];
    out.write_to_host(&mut buf);
    assert_eq!(buf, [4.0, 3.0, 7.0]);

    unsafe {
        out.free();
    }
}

#[test]
fn tensor_splat_add() {
    let handle = DeviceHandles::default();
    let splat = [0.5, -1.0, 1.0];
    let vecs = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.5, 1.0, 1.0];

    let mut inp = unsafe { Tensor::uninit(Shape::new(1, 3)) };
    inp.calloc();
    inp.load_from_host(&splat);

    let out = TensorBatch::new(Shape::new(1, 3), 7);
    out.load_from_host(&vecs);

    util::panic_if_device_error("Error");
    unsafe {
        TensorBatch::splat_add(&handle, 4, &inp, &out);
    }
    util::panic_if_device_error("Error");

    let mut buf = [0.0; 12];
    out.write_to_host(&mut buf);
    assert_eq!(buf, [1.5, 0.0, 1.0, 1.5, 0.0, 1.0, 1.5, 0.0, 2.0, 2.0, 0.0, 2.0]);

    unsafe {
        inp.free();
    }
}

#[test]
fn affine() {
    let handle = DeviceHandles::default();
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
        let ones = DeviceBuffer::new(1);
        ones.load_from_host(&[1.0]);

        w.calloc();
        b.calloc();

        w.load_from_host(&ws);
        b.load_from_host(&bs);

        x.load_from_host(&inps);

        util::panic_if_device_error("Error");
        TensorBatch::affine(&handle, 1, &w, &x, &b, &y);
        util::panic_if_device_error("Error");

        let mut buf = [0.0; 3];
        y.write_to_host(&mut buf);
        assert_eq!(buf, [0.6, 2.2, 0.8]);

        let mut wg = Tensor::uninit(Shape::new(3, 3));
        let mut bg = Tensor::uninit(Shape::new(1, 3));

        wg.calloc();
        bg.calloc();

        util::panic_if_device_error("Error");
        TensorBatch::backprop_affine(&handle, &ones, 1, &w, &y, &x, &wg, &bg);
        util::panic_if_device_error("Error");

        x.write_to_host(&mut buf);
        assert_eq!(buf, [1.4000001, 2.2, 1.4000001]);

        let mut wbuf = [0.0; 9];
        wg.write_to_host(&mut wbuf);
        assert_eq!(wbuf, [
            0.6, 2.2, 0.8,
            1.2, 4.4, 1.6,
            -0.3, -1.1, -0.4,
            //0.6, 1.2, -0.3,
            //2.2, 4.4, -1.1,
            //0.8, 1.6, -0.4,
            ]);

        let mut bbuf = [0.0; 3];
        bg.write_to_host(&mut bbuf);
        assert_eq!(bbuf, [0.6, 2.2, 0.8]);

        w.free();
        b.free();
        wg.free();
        bg.free();
    }
}

#[test]
fn mse() {
    let handle = DeviceHandles::default();
    let out = [1.5, 0.0, 1.0];
    let res = [0.5, 0.5, 0.5];

    let error = DeviceBuffer::new(1);

    let x = TensorBatch::new(Shape::new(1, 1), 9);
    x.load_from_host(&out);

    let r = TensorBatch::new(Shape::new(1, 1), 9);
    r.load_from_host(&res);

    util::panic_if_device_error("Error");
    x.sigmoid_mpe(&handle, 3, &r, &error, 2.0);
    util::panic_if_device_error("Error");

    let mut buf = [0.0; 3];
    x.write_to_host(&mut buf);

    for (e, (&o, &r)) in buf.iter().zip(out.iter().zip(res.iter())) {
        let sig = 1.0 / (1.0 + (-o).exp());

        let diff = e - (sig - r) * sig * (1.0 - sig);
        assert!(diff.abs() < 0.00001);
    }
}

#[test]
fn select() {
    let handle = DeviceHandles::default();
    let buckets = [0, 1, 2, 1];

    let input = [
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0,
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    ];

    let output = [
        0.0, 1.0, 2.0,
        5.0, 4.0, 3.0,
        6.0, 7.0, 8.0,
        3.0, 4.0, 5.0,
    ];

    let input_gpu = TensorBatch::new(Shape::new(1, 9), 4);
    let output_gpu = TensorBatch::new(Shape::new(1, 3), 4);
    let buckets_gpu = util::calloc::<u8>(4);

    input_gpu.load_from_host(&input);

    util::panic_if_device_error("Error");
    unsafe {
        util::copy_to_device(buckets_gpu, buckets.as_ptr(), 4);
        TensorBatch::select(&handle, 4, buckets_gpu, &input_gpu, &output_gpu);
    }
    util::panic_if_device_error("Error");

    let mut buf = [0.0; 12];
    output_gpu.write_to_host(&mut buf);
    assert_eq!(buf, output);

    util::panic_if_device_error("Error");
    unsafe {
        TensorBatch::select_backprop(&handle, 4, buckets_gpu, &output_gpu, &input_gpu);
    }
    util::panic_if_device_error("Error");

    let expected = [
        0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 5.0, 4.0, 3.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 7.0, 8.0,
        0.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0,
    ];

    let mut buf = [0.0; 36];
    input_gpu.write_to_host(&mut buf);
    assert_eq!(buf, expected);
}
