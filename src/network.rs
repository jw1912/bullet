use std::io::{Read, Write};

use bulletformat::{BulletFormat, ChessBoard};

use crate::{inputs::InputType, Activation, InputFeatures, OutputBuckets, HL_SIZE, QA, QB};

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

    pub fn write(&self, writer: &mut impl Write) -> std::io::Result<()> {
        for col in &self.l1w {
            col.write(writer)?;
        }

        self.l1b.write(writer)?;

        for bucket in &self.l2w {
            for col in bucket {
                col.write(writer)?;
            }
        }

        for elem in &self.l2b {
            writer.write_all(&f32::to_le_bytes(*elem))?;
        }

        Ok(())
    }

    pub fn write_quantised(&self, writer: &mut impl Write) -> std::io::Result<()> {
        for col in &self.l1w {
            col.write_quantised(writer, QA)?;
        }

        self.l1b.write_quantised(writer, QA)?;

        for bucket in &self.l2w {
            for col in bucket {
                col.write_quantised(writer, QB)?;
            }
        }

        for elem in &self.l2b {
            let quantised = (*elem * f32::from(QA) * f32::from(QB)) as i16;
            writer.write_all(&i16::to_le_bytes(quantised))?;
        }

        Ok(())
    }

    pub fn read(reader: &mut impl Read) -> std::io::Result<Box<Self>> {
        let mut res = Self::new();
        
        for col in &mut res.l1w {
            col.read_from(reader)?;
        }

        res.l1b.read_from(reader)?;

        for bucket in &mut res.l2w {
            for col in bucket {
                col.read_from(reader)?;
            }
        }

        for elem in &mut res.l2b {
            let mut bytes = [0; 4];
            reader.read_exact(&mut bytes)?;
            *elem = f32::from_le_bytes(bytes);
        }

        Ok(res)
    }

    pub fn random() -> Box<Self> {
        let mut params = Self::new();
        let mut rng = Rand::new(173645501);

        let val = (1.0 / INPUTS as f64).sqrt();

        for col in params.l1w.iter_mut() {
            col.randomise(&mut rng, val);
        }

        params.l1b.randomise(&mut rng, val);

        let val = (1.0 / HL as f64).sqrt();

        for bucket in params.l2w.iter_mut() {
            for col in bucket.iter_mut() {
                col.randomise(&mut rng, val);
            }
        }

        for elem in params.l2b.iter_mut() {
            *elem = rng.rand(val);
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

    fn randomise(&mut self, rng: &mut Rand, max: f64) {
        for elem in self.0.iter_mut() {
            *elem = rng.rand(max);
        }
    }

    fn read_from(&mut self, reader: &mut impl Read) -> std::io::Result<()> {
        for elem in &mut self.0 {
            let mut bytes = [0; 4];
            reader.read_exact(&mut bytes)?;
            *elem = f32::from_le_bytes(bytes);
        }

        Ok(())
    }

    fn write(&self, writer: &mut impl Write) -> std::io::Result<()> {
        for elem in &self.0 {
            writer.write_all(&f32::to_le_bytes(*elem))?;
        }

        Ok(())
    }

    fn write_quantised(&self, writer: &mut impl Write, q: i16) -> std::io::Result<()> {
        for elem in &self.0 {
            let quantised = (*elem * f32::from(q)) as i16;
            writer.write_all(&i16::to_le_bytes(quantised))?;
        }

        Ok(())
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
        (2.0 * (0.5 - f64::from(self.0) / f64::from(u32::MAX)) * max) as f32
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