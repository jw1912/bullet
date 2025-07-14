use std::sync::Arc;

use crate::{
    device::{
        base::{AdamConfig, BaseOperations},
        blas::{BlasOperations, GemmConfig},
        Device, DeviceBuffer,
    },
    graph::{
        ir::{operation::unary::DiffableFromOutput, shape::Shape},
        tensor::rng,
    },
};

use super::{CpuBuffer, CpuThread};

impl CpuThread {
    pub fn compare_linear_comb<D: Device>(device: Arc<D>) {
        for (size, alpha, beta) in [(1027, 2.0, -3.0), (101, 2.0, -3.0), (103, 0.0, 1.0), (1024, 2.0, -3.0)] {
            print!("geam alpha={alpha} beta={beta} size={size}... ");
            display_passed(linear_comb_equal(device.clone(), size, alpha, beta));
        }
    }

    pub fn compare_gemm<D: Device>(device: Arc<D>) {
        for (m, n, k, alpha, trans_a, beta, trans_b) in [
            (13, 17, 34, 2.0, false, -3.0, false),
            (13, 17, 34, 2.0, true, -3.0, true),
            (13, 17, 34, 2.0, false, -3.0, true),
            (13, 17, 34, 2.0, true, -3.0, false),
        ] {
            let shape_a = Shape::new(m, n).maybe_transpose(trans_a);
            let shape_b = Shape::new(n, k).maybe_transpose(trans_b);
            let config = GemmConfig { alpha, beta, shape_a, trans_a, shape_b, trans_b };
            print!("gemm alpha={alpha} beta={beta} shape_a=({shape_a}) shape_b=({shape_b}) trans_a={trans_a} trans_b={trans_b}... ");
            display_passed(gemm_equal(device.clone(), config));
        }
    }

    pub fn compare_gebmm<D: Device>(device: Arc<D>) {
        for (bs, m, n, k, alpha, trans_a, beta, trans_b) in [
            (256, 13, 17, 34, 2.0, false, -3.0, false),
            (256, 13, 17, 34, 2.0, true, -3.0, true),
            (256, 13, 17, 34, 2.0, false, -3.0, true),
            (256, 13, 17, 34, 2.0, true, -3.0, false),
        ] {
            let shape_a = Shape::new(m, n).maybe_transpose(trans_a);
            let shape_b = Shape::new(n, k).maybe_transpose(trans_b);
            let config = GemmConfig { alpha, beta, shape_a, trans_a, shape_b, trans_b };
            print!("gebmm batch_size={bs} alpha={alpha} beta={beta} shape_a=({shape_a}) shape_b=({shape_b}) trans_a={trans_a} trans_b={trans_b}... ");
            display_passed(gebmm_equal(device.clone(), bs, config));
        }
    }

    pub fn compare_activate<D: Device>(device: Arc<D>) {
        for (size, act) in [
            (1027, DiffableFromOutput::ReLU),
            (1027, DiffableFromOutput::CReLU),
            (1027, DiffableFromOutput::SCReLU),
            (1027, DiffableFromOutput::Sigmoid),
            (1027, DiffableFromOutput::SqrReLU),
        ] {
            print!("activation={act:?} size={size} fwd... ");
            display_passed(base_op_equal(device.clone(), size, BaseOp::DiffableFromOutput(act), true));
            print!("activation={act:?} size={size} bwd... ");
            display_passed(base_op_equal(device.clone(), size, BaseOp::DiffableFromOutput(act), false));
        }
    }

    pub fn compare_power_error<D: Device>(device: Arc<D>) {
        for size in [13, 34, 1023, 1027] {
            print!("power_error size={size} fwd... ");
            display_passed(base_op_equal(device.clone(), size, BaseOp::PowerErr, true));
            print!("power_error size={size} bwd... ");
            display_passed(base_op_equal(device.clone(), size, BaseOp::PowerErr, false));
        }
    }

    pub fn compare_pairwise<D: Device>(device: Arc<D>) {
        for size in [14, 34, 1022, 1028] {
            print!("pairwise size={size} fwd... ");
            display_passed(base_op_equal(device.clone(), size, BaseOp::Pairwise, true));
            print!("pairwise size={size} bwd... ");
            display_passed(base_op_equal(device.clone(), size, BaseOp::Pairwise, false));
        }
    }

    pub fn compare_adam<D: Device>(device: Arc<D>) {
        let config = AdamConfig {
            beta1: 0.9,
            beta2: 0.999,
            gradient_factor: 0.1,
            learning_rate: 0.001,
            denom: true,
            decay: 0.5,
            clip: Some((-1.0, 1.0)),
        };
        for size in [13, 34, 1023, 1027] {
            print!("adam size={size}... ");
            display_passed(base_op_equal(device.clone(), size, BaseOp::Adam(config), true));
        }
    }

    pub fn compare_copy_or_add_strided<D: Device>(device: Arc<D>) {
        for add in [false, true] {
            for rows in [1, 7, 64] {
                for cols in [1023, 1027] {
                    print!("copy_or_add_strided add={add} rows={rows} cols={cols}... ");
                    display_passed(copy_strided_equal(device.clone(), add, rows, cols));
                }
            }
        }
    }

    pub fn compare_clip<D: Device>(device: Arc<D>) {
        for size in [1023, 1027] {
            print!("clip size={size}... ");
            display_passed(base_op_equal(device.clone(), size, BaseOp::Clip(-1.98, 1.98), true));
        }
    }

    pub fn compare_add<D: Device>(device: Arc<D>) {
        for size in [1023, 1027] {
            print!("add_scalar size={size}... ");
            display_passed(base_op_equal(device.clone(), size, BaseOp::Add(3.0), true));
        }
    }

    pub fn compare_abs_pow<D: Device>(device: Arc<D>) {
        for size in [1023, 1027] {
            print!("abs_pow_scalar size={size} fwd... ");
            display_passed(base_op_equal(device.clone(), size, BaseOp::AbsPow(2.5), true));
            print!("abs_pow_scalar size={size} bwd... ");
            display_passed(base_op_equal(device.clone(), size, BaseOp::AbsPow(2.5), false));
        }
    }
}

fn linear_comb_equal<D: Device>(device: Arc<D>, size: usize, alpha: f32, beta: f32) -> bool {
    let device = device;
    let cpu = Arc::new(CpuThread);
    let a = rng::vec_f32(size, 1.0, 0.5, false);
    let b = rng::vec_f32(size, 1.0, 0.5, false);

    let acpu = load(cpu.clone(), &a);
    let adev = load(device.clone(), &a);

    let mut bcpu = load(cpu.clone(), &b);
    let mut bdev = load(device.clone(), &b);

    bcpu.linear_comb(size, alpha, beta, &acpu).unwrap();
    bdev.linear_comb(size, alpha, beta, &adev).unwrap();

    approx_equal::<D>(&bcpu, &bdev, 0.001).is_none()
}

fn gemm_equal<D: Device>(device: Arc<D>, config: GemmConfig) -> bool {
    let device = device;
    let cpu = Arc::new(CpuThread);

    let a = rng::vec_f32(config.shape_a.size(), 1.0, 0.5, false);
    let b = rng::vec_f32(config.shape_b.size(), 1.0, 0.5, false);
    let c = rng::vec_f32(config.output_shape().size(), 1.0, 0.5, false);

    let acpu = load(cpu.clone(), &a);
    let bcpu = load(cpu.clone(), &b);
    let adev = load(device.clone(), &a);
    let bdev = load(device.clone(), &b);
    let mut ccpu = load(cpu.clone(), &c);
    let mut cdev = load(device.clone(), &c);

    ccpu.gemm(&config, &acpu, &bcpu).unwrap();
    cdev.gemm(&config, &adev, &bdev).unwrap();

    approx_equal::<D>(&ccpu, &cdev, 0.01).is_none()
}

fn gebmm_equal<D: Device>(device: Arc<D>, batch_size: usize, config: GemmConfig) -> bool {
    let device = device;
    let cpu = Arc::new(CpuThread);
    let a = rng::vec_f32(batch_size * config.shape_a.size(), 1.0, 0.5, false);
    let b = rng::vec_f32(batch_size * config.shape_b.size(), 1.0, 0.5, false);
    let c = rng::vec_f32(batch_size * config.output_shape().size(), 1.0, 0.5, false);

    let acpu = load(cpu.clone(), &a);
    let bcpu = load(cpu.clone(), &b);
    let adev = load(device.clone(), &a);
    let bdev = load(device.clone(), &b);
    let mut ccpu = load(cpu.clone(), &c);
    let mut cdev = load(device.clone(), &c);

    ccpu.gebmm(&config, batch_size, &acpu, &bcpu).unwrap();
    cdev.gebmm(&config, batch_size, &adev, &bdev).unwrap();

    approx_equal::<D>(&ccpu, &cdev, 0.01).is_none()
}

fn copy_strided_equal<D: Device>(device: Arc<D>, add: bool, rows: usize, cols: usize) -> bool {
    let device = device;
    let cpu = Arc::new(CpuThread);
    let a = rng::vec_f32(2 * rows * cols, 1.0, 0.5, false);
    let c = rng::vec_f32(rows * cols, 1.0, 0.5, false);

    let acpu = load(cpu.clone(), &a);
    let adev = load(device.clone(), &a);
    let mut ccpu = load(cpu.clone(), &c);
    let mut cdev = load(device.clone(), &c);

    ccpu.copy_or_add_strided(add, rows, cols, 0, rows, &acpu, rows, 2 * rows).unwrap();
    cdev.copy_or_add_strided(add, rows, cols, 0, rows, &adev, rows, 2 * rows).unwrap();

    approx_equal::<D>(&ccpu, &cdev, 0.001).is_none()
}

enum BaseOp {
    DiffableFromOutput(DiffableFromOutput),
    PowerErr,
    Pairwise,
    Clip(f32, f32),
    Adam(AdamConfig),
    Add(f32),
    AbsPow(f32),
}

fn base_op_equal<D: Device>(device: Arc<D>, size: usize, op: BaseOp, fwd: bool) -> bool {
    let device = device;
    let cpu = Arc::new(CpuThread);
    let a = rng::vec_f32(size * 4, 0.5, 5.5, false);
    let b = rng::vec_f32(size * 4, 0.5, 5.5, false);
    let c = rng::vec_f32(size * 4, 0.5, 5.5, false);

    let acpu = load(cpu.clone(), &a);
    let adev = load(device.clone(), &a);
    let mut bcpu = load(cpu.clone(), &b);
    let mut bdev = load(device.clone(), &b);
    let mut ccpu = load(cpu.clone(), &c);
    let mut cdev = load(device.clone(), &c);

    match op {
        BaseOp::DiffableFromOutput(act) => {
            if fwd {
                ccpu.diffable_from_output_fwd(size, &acpu, act).unwrap();
                cdev.diffable_from_output_fwd(size, &adev, act).unwrap();
            } else {
                ccpu.diffable_from_output_bwd(size, &acpu, &bcpu, act).unwrap();
                cdev.diffable_from_output_bwd(size, &adev, &bdev, act).unwrap();
            }
        }
        BaseOp::Pairwise => {
            if fwd {
                ccpu.pairwise_fwd(size, 4, &acpu).unwrap();
                cdev.pairwise_fwd(size, 4, &adev).unwrap();
            } else {
                ccpu.pairwise_bwd(size, 4, &acpu, &bcpu).unwrap();
                cdev.pairwise_bwd(size, 4, &adev, &bdev).unwrap();
            }
        }
        BaseOp::PowerErr => {
            if fwd {
                ccpu.power_error_fwd(2.0, size, &acpu, &bcpu).unwrap();
                cdev.power_error_fwd(2.0, size, &adev, &bdev).unwrap();
            } else {
                ccpu.power_error_bwd(2.0, size, &acpu, &bcpu, &bcpu).unwrap();
                cdev.power_error_bwd(2.0, size, &adev, &bdev, &bdev).unwrap();
            }
        }
        BaseOp::Clip(a, b) => {
            ccpu.clip(size, a, b).unwrap();
            cdev.clip(size, a, b).unwrap();
        }
        BaseOp::Adam(config) => {
            let d = rng::vec_f32(size, 0.5, 1.5, false);
            let mut dcpu = load(cpu.clone(), &d);
            let mut ddev = load(device.clone(), &d);
            dcpu.adam(&config, size, &acpu, &mut bcpu, &mut ccpu).unwrap();
            ddev.adam(&config, size, &adev, &mut bdev, &mut cdev).unwrap();

            assert!(approx_equal::<D>(&bcpu, &bdev, 0.001).is_none());
            assert!(approx_equal::<D>(&dcpu, &ddev, 0.001).is_none());
        }
        BaseOp::AbsPow(x) => {
            if fwd {
                ccpu.abs_pow_scalar(size, x, &acpu).unwrap();
                cdev.abs_pow_scalar(size, x, &adev).unwrap();
            } else {
                ccpu.abs_pow_scalar_backward(size, x, &acpu, &bcpu).unwrap();
                cdev.abs_pow_scalar_backward(size, x, &adev, &bdev).unwrap();
            }
        }
        BaseOp::Add(x) => {
            ccpu.add_scalar(size, x, &acpu).unwrap();
            cdev.add_scalar(size, x, &adev).unwrap();
        }
    }

    approx_equal::<D>(&ccpu, &cdev, 0.001).is_none()
}

fn approx_equal<D: Device>(a: &CpuBuffer<f32>, b: &D::BufferF32, err: f32) -> Option<usize> {
    let a = write::<CpuThread>(a);
    let b = write::<D>(b);

    if a.len() != b.len() {
        return Some(usize::MAX);
    }

    for (i, (&a, &b)) in a.iter().zip(b.iter()).enumerate() {
        if (a - b).abs() > err {
            print!("a={a} b={b} err={} ", (a - b).abs());
            return Some(i);
        }
    }

    None
}

fn load<D: Device>(device: Arc<D>, a: &[f32]) -> D::BufferF32 {
    let mut buf = D::BufferF32::new(device.clone(), a.len()).unwrap();
    buf.load_from_slice(a).unwrap();
    buf
}

fn write<D: Device>(a: &D::BufferF32) -> Vec<f32> {
    let mut buf = vec![0.0; a.size()];
    a.write_into_slice(&mut buf, a.size()).unwrap();
    buf
}

fn display_passed(pass: bool) {
    if pass {
        println!("\x1b[32;1mpass\x1b[0m");
    } else {
        println!("\x1b[31mfail\x1b[0m");
    }
}
