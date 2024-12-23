#![allow(unused)]

pub struct ReLU;
impl ReLU {
    pub fn activate(x: f32) -> f32 {
        x.max(0.0)
    }

    pub fn prime(x: f32) -> f32 {
        if x < 0.0 {
            0.0
        } else {
            1.0
        }
    }
}

pub struct CReLU;
impl CReLU {
    pub fn activate(x: f32) -> f32 {
        x.clamp(0.0, 1.0)
    }

    pub fn prime(x: f32) -> f32 {
        if !(0.0..1.0).contains(&x) {
            0.0
        } else {
            1.0
        }
    }
}

pub struct SCReLU;
impl SCReLU {
    pub fn activate(x: f32) -> f32 {
        let clamped = x.clamp(0.0, 1.0);
        clamped * clamped
    }

    pub fn prime(x: f32) -> f32 {
        if !(0.0..1.0).contains(&x) {
            0.0
        } else {
            2.0 * x
        }
    }
}

pub struct FastSCReLU;
impl FastSCReLU {
    const FAST_SCRELU_FACTOR: f32 = 255.0 / 256.0;

    pub fn activate(x: f32) -> f32 {
        let squared = x * x * Self::FAST_SCRELU_FACTOR;
        squared.min(1.0)
    }

    pub fn prime(x: f32) -> f32 {
        if !(0.0..1.0).contains(&x) {
            0.0
        } else {
            2.0 * x * Self::FAST_SCRELU_FACTOR
        }
    }
}
