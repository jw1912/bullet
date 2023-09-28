#![cfg_attr(feature = "simd", feature(stdsimd))]

#[cfg(feature = "simd")]
const _: () = assert!(HIDDEN % 32 == 0, "Must be a multiple of 64 for hand-written SIMD.");

mod accumulator;
mod gradient;
mod optimiser;
mod quantise;
mod simd;

pub use accumulator::Accumulator;
pub use gradient::update_single_grad_cpu;
pub use optimiser::AdamW;
pub use quantise::quantise_and_write;

use common::{rng::Rand, Input, inputs::InputType, OutputBucket, HIDDEN, util::write_to_bin};

pub type NetworkParams = Network<f32>;

pub const NETWORK_SIZE: usize = OUTPUT_BIAS + OutputBucket::NUM;
pub const FEATURE_BIAS: usize = Input::SIZE * HIDDEN;
pub const OUTPUT_WEIGHTS: usize = (Input::SIZE + 1) * HIDDEN;
pub const OUTPUT_BIAS: usize = OUTPUT_WEIGHTS + 2 * OutputBucket::NUM * HIDDEN;

#[derive(Clone)]
#[repr(C)]
pub struct Network<T> {
    weights: [T; NETWORK_SIZE],
}

impl<T: std::ops::AddAssign<T> + Copy> std::ops::AddAssign<&Network<T>> for Network<T> {
    fn add_assign(&mut self, rhs: &Network<T>) {
        for (i, &j) in self.iter_mut().zip(rhs.iter()) {
            *i += j;
        }
    }
}

impl<T> std::ops::Deref for Network<T> {
    type Target = [T; NETWORK_SIZE];
    fn deref(&self) -> &Self::Target {
        &self.weights
    }
}

impl<T> std::ops::DerefMut for Network<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.weights
    }
}

impl<T> Network<T> {
    pub fn new() -> Box<Self> {
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(ptr.cast())
        }
    }
}

impl NetworkParams {
    pub fn random() -> Box<Self> {
        let mut params = NetworkParams::new();
        let mut gen = Rand::new(173645501);

        for param in params[..FEATURE_BIAS].iter_mut() {
            *param = gen.rand(0.01);
        }

        for param in params[OUTPUT_WEIGHTS..OUTPUT_BIAS].iter_mut() {
            *param = gen.rand(0.01);
        }

        params
    }

    pub fn write_to_bin(&self, output_path: &str) -> std::io::Result<()> {
        const SIZEOF: usize = std::mem::size_of::<NetworkParams>();
        write_to_bin::<Self, SIZEOF>(self, output_path, true)
    }

    pub fn load_from_bin(&mut self, path: &str) {
        use std::fs::File;
        use std::io::{Read, BufReader};
        let file = File::open(path).unwrap();
        let reader = BufReader::new(file);

        let mut buf = [0u8; 4];

        for (i, byte) in reader.bytes().enumerate() {
            let idx = i % 4;

            buf[idx] = byte.unwrap();

            if idx == 3 {
                self[i / 4] = f32::from_ne_bytes(buf);
            }
        }
    }
}
