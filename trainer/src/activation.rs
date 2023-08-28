pub trait Activation {
    fn activate(x: f32) -> f32;
    fn activate_prime(x: f32) -> f32;
}

pub struct ReLU;
impl Activation for ReLU {
    fn activate(x: f32) -> f32 {
        x.max(0.0)
    }

    fn activate_prime(x: f32) -> f32 {
        if x < 0.0 {
            0.0
        } else {
            1.0
        }
    }
}

pub struct CReLU;
impl Activation for CReLU {
    fn activate(x: f32) -> f32 {
        x.clamp(0.0, 1.0)
    }

    fn activate_prime(x: f32) -> f32 {
        if !(0.0..1.0).contains(&x) {
            0.0
        } else {
            1.0
        }
    }
}

pub struct SCReLU;
impl Activation for SCReLU {
    fn activate(x: f32) -> f32 {
        let clamped = x.clamp(0.0, 1.0);
        clamped * clamped
    }

    fn activate_prime(x: f32) -> f32 {
        if !(0.0..1.0).contains(&x) {
            0.0
        } else {
            2.0 * x
        }
    }
}
