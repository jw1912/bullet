use rand::{Rng as _, RngExt, SeedableRng};
use rand_distr::{Distribution, Normal, Uniform};
use rand_xoshiro::Xoroshiro128Plus;

pub use rand;
pub use rand_distr;
pub use rand_xoshiro;

pub struct Rng(Xoroshiro128Plus);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(Xoroshiro128Plus::seed_from_u64(seed))
    }

    pub fn seeded() -> Self {
        Self(rand::make_rng())
    }

    pub fn sample(&mut self) -> u64 {
        self.0.next_u64()
    }

    pub fn sample_bool(&mut self, success_prob: f64) -> bool {
        self.0.random_bool(success_prob)
    }

    pub fn shuffle<T>(&mut self, data: &mut [T]) {
        for i in (0..data.len()).rev() {
            let idx = self.sample() as usize % (i + 1);
            data.swap(idx, i);
        }
    }

    pub fn vec_f32(&mut self, length: usize, mean: f32, stdev: f32, use_gaussian: bool) -> Vec<f32> {
        let mut res = Vec::with_capacity(length);

        let dist = Dist::new(mean, stdev, use_gaussian);

        for _ in 0..length {
            res.push(dist.sample(&mut self.0));
        }

        res
    }
}

enum Dist {
    Normal(Normal<f32>),
    Uniform(Uniform<f32>),
}

impl Dist {
    fn new(mean: f32, stdev: f32, use_gaussian: bool) -> Self {
        if use_gaussian {
            Self::Normal(Normal::new(mean, stdev).unwrap())
        } else {
            Self::Uniform(Uniform::new(mean - stdev, mean + stdev).unwrap())
        }
    }

    fn sample(&self, rng: &mut Xoroshiro128Plus) -> f32 {
        match self {
            Dist::Normal(x) => x.sample(rng),
            Dist::Uniform(x) => x.sample(rng),
        }
    }
}
