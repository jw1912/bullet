use bulletformat::{BulletFormat, ChessBoard};

use crate::{inputs::InputType, Activation, InputFeatures, OutputBuckets, HL_SIZE};

const INPUTS: usize = InputFeatures::SIZE;
const HL: usize = HL_SIZE;
const OUTPUT_BUCKETS: usize = OutputBuckets::NUM;
const MAX_ACTIVE: usize = 32;

pub struct Network {
    l1w: [Accumulator; INPUTS],
    l1b: Accumulator,
    l2w: [[Accumulator; 2]; OUTPUT_BUCKETS],
    l2b: [f32; OUTPUT_BUCKETS],
}

impl Network {
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

    pub fn random() -> Box<Self> {
        let mut params = Self::new();
        let mut rng = Rand::new(173645501);

        for col in params.l1w.iter_mut() {
            for elem in col.0.iter_mut() {
                *elem = rng.rand(0.01);
            }
        }

        for bucket in params.l2w.iter_mut() {
            for col in bucket.iter_mut() {
                for elem in col.0.iter_mut() {
                    *elem = rng.rand(0.01);
                }
            }
        }

        params
    }

    pub fn add(&mut self, other: &Self) {
        for (i, j) in self.l1w.iter_mut().zip(other.l1w.iter()) {
            i.add(j);
        }

        self.l1b.add(&other.l1b);

        for (ibucket, jbucket) in self.l2w.iter_mut().zip(other.l2w.iter()) {
            for (i, j) in ibucket.iter_mut().zip(jbucket.iter()) {
                i.add(j);
            }
        }

        for (i, j) in self.l2b.iter_mut().zip(other.l2b.iter()) {
            *i += *j;
        }
    }

    pub fn update(&mut self, m: &mut Self, v: &mut Self, grad: &Self, decay: f32, adj: f32, rate: f32) {
        for (p, (m, (v, g))) in self.l1w.iter_mut().zip(m.l1w.iter_mut().zip(v.l1w.iter_mut().zip(grad.l1w.iter()))) {
            p.adamw(m, v, g, decay, adj, rate);
        }

        self.l1b.adamw(&mut m.l1b, &mut v.l1b, &grad.l1b, decay, adj, rate);

        for (p, (m, (v, g))) in self.l2w.iter_mut().zip(m.l2w.iter_mut().zip(v.l2w.iter_mut().zip(grad.l2w.iter()))) {
            for (p, (m, (v, g))) in p.iter_mut().zip(m.iter_mut().zip(v.iter_mut().zip(g.iter()))) {
                p.adamw(m, v, g, decay, adj, rate);
            }
        }

        for (p, (m, (v, &g))) in self.l2b.iter_mut().zip(m.l2b.iter_mut().zip(v.l2b.iter_mut().zip(grad.l2b.iter()))) {
            adamw(p, m, v, g, decay, adj, rate);
        }
    }

    pub fn update_single_grad(
        &self,
        grad: &mut Self,
        pos: &ChessBoard,
        blend: f32,
        rscale: f32,
    ) -> f32 {
        let bias = self.l1b;
        let mut accs = [bias; 2];
        let mut activated = [Accumulator([0.0; HL]); 2];
        let mut features = Features::default();
    
        let (eval, bucket) = self.forward(pos, &mut accs, &mut activated, &mut features);
    
        let result = pos.blended_result(blend, rscale);
    
        let sigmoid = sigmoid(eval);
        let err = 2.0 * (sigmoid - result) * sigmoid * (1. - sigmoid);
    
        self.backprop(err, grad, &accs, &activated, &mut features, bucket);

        (sigmoid - result).powi(2)
    }

    fn forward(
        &self,
        pos: &ChessBoard,
        accs: &mut [Accumulator; 2],
        activated: &mut [Accumulator; 2],
        features: &mut Features,
    ) -> (f32, usize) {
        let mut idx = 0;

        let kings = (pos.our_ksq(), pos.opp_ksq());

        for feat in pos.into_iter() {
            let (wfeat, bfeat) = InputFeatures::get_feature_indices(feat, kings);

            features.push(wfeat, bfeat);
            accs[0].add(&self.l1w[wfeat]);
            accs[1].add(&self.l1w[bfeat]);

            OutputBuckets::update_output_bucket(&mut idx, usize::from(feat.0 & 7));
        }

        let bucket = OutputBuckets::get_bucket(idx);

        let mut eval = self.l2b[bucket];

        for i in [0, 1] {
            for j in 0..HL {
                activated[i].0[j] = Activation::activate(accs[i].0[j]);
                eval += activated[i].0[j] * self.l2w[bucket][i].0[j];
            }
        }

        (eval, bucket)
    }

    fn backprop(
        &self,
        err: f32,
        grad: &mut Self,
        accs: &[Accumulator; 2],
        activated: &[Accumulator; 2],
        features: &mut Features,
        bucket: usize,
    ) {
        let mut components = [(0.0, 0.0); HL];

        for (i, component) in components.iter_mut().enumerate() {
            *component = (
                err * self.l2w[bucket][0].0[i] * Activation::prime(accs[0].0[i]),
                err * self.l2w[bucket][1].0[i] * Activation::prime(accs[1].0[i]),
            );

            grad.l1b.0[i] += component.0 + component.1;

            grad.l2w[bucket][0].0[i] += err * activated[0].0[i];
            grad.l2w[bucket][1].0[i] += err * activated[1].0[i];
        }

        for (wfeat, bfeat) in features {
            for (i, component) in components.iter().enumerate() {
                grad.l1w[wfeat].0[i] += component.0;
                grad.l1w[bfeat].0[i] += component.1;
            }
        }

        grad.l2b[bucket] += err;
    }
}

#[derive(Clone, Copy)]
pub struct Accumulator([f32; HL]);

impl Accumulator {
    fn add(&mut self, other: &Self) {
        for (i, &j) in self.0.iter_mut().zip(other.0.iter()) {
            *i += j;
        }
    }

    fn adamw(&mut self, m: &mut Self, v: &mut Self, grad: &Self, decay: f32, adj: f32, rate: f32) {
        for (p, (m, (v, &g))) in self.0.iter_mut().zip(m.0.iter_mut().zip(v.0.iter_mut().zip(grad.0.iter()))) {
            adamw(p, m, v, g, decay, adj, rate);
        }
    }
}

#[derive(Debug)]
pub struct Features {
    features: [(usize, usize); MAX_ACTIVE],
    len: usize,
    consumed: usize,
}

impl Default for Features {
    fn default() -> Self {
        Self {
            features: [(0, 0); MAX_ACTIVE],
            len: 0,
            consumed: 0,
        }
    }
}

impl Features {
    pub fn push(&mut self, wfeat: usize, bfeat: usize) {
        self.features[self.len] = (wfeat, bfeat);
        self.len += 1;
    }
}

impl Iterator for Features {
    type Item = (usize, usize);
    fn next(&mut self) -> Option<Self::Item> {
        if self.consumed == self.len {
            return None;
        }

        let ret = self.features[self.consumed];

        self.consumed += 1;

        Some(ret)
    }
}

pub struct Rand(u32);
impl Default for Rand {
    fn default() -> Self {
        Self(
            (std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("valid")
                .as_nanos()
                & 0xFFFF_FFFF) as u32,
        )
    }
}

impl Rand {
    pub fn new(seed: u32) -> Self {
        Self(seed)
    }

    pub fn rand(&mut self, max: f64) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        ((1. - f64::from(self.0) / f64::from(u32::MAX)) * max) as f32
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn adamw(param: &mut f32, m: &mut f32, v: &mut f32, mut grad: f32, decay: f32, adj: f32, rate: f32) {
    const B1: f32 = 0.9;
    const B2: f32 = 0.999;
    *param *= decay;
    grad *= adj;
    *m = B1 * *m + (1. - B1) * grad;
    *v = B2 * *v + (1. - B2) * grad * grad;
    *param -= rate * *m / (v.sqrt() + 0.000_000_01);
    *param = param.clamp(-1.98, 1.98);
}
