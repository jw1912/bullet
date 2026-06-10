use rand_distr::{Distribution, Normal, Uniform};
use rand_xoshiro::Xoroshiro128Plus;

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

pub fn vec_f32(rng: &mut Xoroshiro128Plus, length: usize, mean: f32, stdev: f32, use_gaussian: bool) -> Vec<f32> {
    let mut res = Vec::with_capacity(length);

    let dist = Dist::new(mean, stdev, use_gaussian);

    for _ in 0..length {
        res.push(dist.sample(rng));
    }

    res
}
