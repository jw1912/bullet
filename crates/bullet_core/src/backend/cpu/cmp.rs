use std::{sync::Arc, time::Instant};

use crate::backend::{
    device::{
        base::{Activation, AdamConfig, BaseOperations},
        blas::{BlasOperations, GemmConfig, Shape},
        Device, DeviceBuffer,
    },
    tensor::rng,
};

use super::{CpuBuffer, CpuThread};

impl CpuThread {
    pub fn compare_geam<D: Device>(device: Arc<D>) {
        for (size, alpha, a, beta, b) in [
            (1027, 2.0, false, -3.0, false),
            (101, 2.0, true, -3.0, true),
            (103, 2.0, false, -3.0, true),
            (1024, 2.0, true, -3.0, false),
        ] {
            print!("geam alpha={alpha} beta={beta} size={size} a={a} b={b}... ");
            display_passed(geam_equal(device.clone(), size, alpha, a, beta, b));
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
            (1027, Activation::ReLU),
            (1027, Activation::CReLU),
            (1027, Activation::SCReLU),
            (1027, Activation::Sigmoid),
            (1027, Activation::SqrReLU),
            (1027, Activation::Square),
        ] {
            print!("activation={act:?} size={size} fwd... ");
            display_passed(base_op_equal(device.clone(), size, BaseOp::Activate(act), true));
            print!("activation={act:?} size={size} bwd... ");
            display_passed(base_op_equal(device.clone(), size, BaseOp::Activate(act), false));
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

    pub fn compare_adam<D: Device>(device: Arc<D>) {
        let config = AdamConfig::new(0.9, 0.999, 0.1, 0.1, true);
        for size in [13, 34, 1023, 1027] {
            print!("adam size={size}... ");
            display_passed(base_op_equal(device.clone(), size, BaseOp::Adam(config), true));
        }
    }
}

fn geam_equal<D: Device>(device: Arc<D>, size: usize, alpha: f32, a: bool, beta: f32, b: bool) -> bool {
    let device = device;
    let cpu = Arc::new(CpuThread);
    let a = a.then(|| rng::vec_f32(size, 1.0, 0.5, false));
    let b = b.then(|| rng::vec_f32(size, 1.0, 0.5, false));

    let acpu = load_optional(cpu.clone(), &a);
    let bcpu = load_optional(cpu.clone(), &b);

    let adev = load_optional(device.clone(), &a);
    let bdev = load_optional(device.clone(), &b);

    let mut ccpu = CpuBuffer::new(cpu.clone(), size).unwrap();
    let mut cdev = D::BufferF32::new(device.clone(), size).unwrap();

    ccpu.geam(size, alpha, acpu.as_ref(), beta, bcpu.as_ref()).unwrap();
    cdev.geam(size, alpha, adev.as_ref(), beta, bdev.as_ref()).unwrap();

    approx_equal(&write::<CpuThread>(&ccpu), &write::<D>(&cdev)).is_none()
}

fn gemm_equal<D: Device>(device: Arc<D>, config: GemmConfig) -> bool {
    let device = device;
    let cpu = Arc::new(CpuThread);
    let a = rng::vec_f32(config.shape_a.size(), 1.0, 0.5, false);
    let b = rng::vec_f32(config.shape_b.size(), 1.0, 0.5, false);

    let acpu = load(cpu.clone(), &a);
    let bcpu = load(cpu.clone(), &b);

    let adev = load(device.clone(), &a);
    let bdev = load(device.clone(), &b);

    let shape_o = config.output_shape();
    let mut ccpu = CpuBuffer::new(cpu.clone(), shape_o.size()).unwrap();
    let mut cdev = D::BufferF32::new(device.clone(), shape_o.size()).unwrap();

    ccpu.gemm(&config, &acpu, &bcpu).unwrap();
    cdev.gemm(&config, &adev, &bdev).unwrap();

    approx_equal(&write::<CpuThread>(&ccpu), &write::<D>(&cdev)).is_none()
}

fn gebmm_equal<D: Device>(device: Arc<D>, batch_size: usize, config: GemmConfig) -> bool {
    let device = device;
    let cpu = Arc::new(CpuThread);
    let a = rng::vec_f32(batch_size * config.shape_a.size(), 1.0, 0.5, false);
    let b = rng::vec_f32(batch_size * config.shape_b.size(), 1.0, 0.5, false);

    let acpu = load(cpu.clone(), &a);
    let bcpu = load(cpu.clone(), &b);

    let adev = load(device.clone(), &a);
    let bdev = load(device.clone(), &b);

    let size_o = batch_size * config.output_shape().size();
    let mut ccpu = CpuBuffer::new(cpu.clone(), size_o).unwrap();
    let mut cdev = D::BufferF32::new(device.clone(), size_o).unwrap();

    ccpu.gebmm(&config, batch_size, &acpu, &bcpu).unwrap();
    cdev.gebmm(&config, batch_size, &adev, &bdev).unwrap();

    approx_equal(&write::<CpuThread>(&ccpu), &write::<D>(&cdev)).is_none()
}

enum BaseOp {
    Activate(Activation),
    PowerErr,
    Adam(AdamConfig),
}

fn base_op_equal<D: Device>(device: Arc<D>, size: usize, op: BaseOp, fwd: bool) -> bool {
    let device = device;
    let cpu = Arc::new(CpuThread);
    let a = rng::vec_f32(size, 0.5, 1.5, false);
    let b = rng::vec_f32(size, 0.5, 1.5, false);
    let c = rng::vec_f32(size, 0.5, 1.5, false);

    let acpu = load(cpu.clone(), &a);
    let adev = load(device.clone(), &a);
    let mut bcpu = load(cpu.clone(), &b);
    let mut bdev = load(device.clone(), &b);
    let mut ccpu = load(cpu.clone(), &c);
    let mut cdev = load(device.clone(), &c);

    match op {
        BaseOp::Activate(act) => {
            if fwd {
                ccpu.activate_fwd(size, &acpu, act).unwrap();
                cdev.activate_fwd(size, &adev, act).unwrap();
            } else {
                ccpu.activate_bwd(size, &acpu, &bcpu, act).unwrap();
                cdev.activate_bwd(size, &adev, &bdev, act).unwrap();
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
        BaseOp::Adam(config) => {
            let mut dcpu = CpuBuffer::new(cpu.clone(), size).unwrap();
            let mut ddev = D::BufferF32::new(device.clone(), size).unwrap();
            dcpu.adam(&config, size, &acpu, &mut bcpu, &mut ccpu).unwrap();
            ddev.adam(&config, size, &adev, &mut bdev, &mut cdev).unwrap();
        }
    }

    approx_equal(&write::<CpuThread>(&ccpu), &write::<D>(&cdev)).is_none()
}

fn approx_equal(a: &[f32], b: &[f32]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(usize::MAX);
    }

    for (i, (&a, &b)) in a.iter().zip(b.iter()).enumerate() {
        if (a - b).abs() > 0.01 {
            print!("a={a} b={b} err={} ", (a - b).abs());
            return Some(i);
        }
    }

    None
}

fn load_optional<D: Device>(device: Arc<D>, a: &Option<Vec<f32>>) -> Option<D::BufferF32> {
    a.as_ref().map(|a| load(device, a))
}

fn load<D: Device>(device: Arc<D>, a: &[f32]) -> D::BufferF32 {
    let mut buf = D::BufferF32::new(device.clone(), a.len()).unwrap();
    buf.load_from_slice(a);
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
