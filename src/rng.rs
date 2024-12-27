use std::time::{SystemTime, UNIX_EPOCH};

use rand::{rngs::ThreadRng, thread_rng};
use rand_distr::{Distribution, Normal, Uniform};

enum Dist {
    Normal(Normal<f32>),
    Uniform(Uniform<f32>),
}

impl Dist {
    fn new(mean: f32, stdev: f32, use_gaussian: bool) -> Self {
        if use_gaussian {
            Self::Normal(Normal::new(mean, stdev).unwrap())
        } else {
            Self::Uniform(Uniform::new(mean - stdev, mean + stdev))
        }
    }

    fn sample(&self, rng: &mut ThreadRng) -> f32 {
        match self {
            Dist::Normal(x) => x.sample(rng),
            Dist::Uniform(x) => x.sample(rng),
        }
    }
}

pub fn vec_f32(length: usize, mean: f32, stdev: f32, use_gaussian: bool) -> Vec<f32> {
    let mut res = Vec::with_capacity(length);

    let mut rng = thread_rng();
    let dist = Dist::new(mean, stdev, use_gaussian);

    for _ in 0..length {
        res.push(dist.sample(&mut rng));
    }

    res
}

pub struct SimpleRand(u64);

impl SimpleRand {
    pub fn with_seed() -> Self {
        let seed = SystemTime::now().duration_since(UNIX_EPOCH).expect("Guaranteed increasing.").as_micros() as u64
            & 0xFFFF_FFFF;

        Self(seed)
    }

    pub fn rng(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}
