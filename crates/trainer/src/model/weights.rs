use std::{
    collections::{BTreeMap, btree_map},
    io::{self, Write},
    rc::Rc,
    sync::Arc,
};

use bullet_compiler::{
    model::{InitSettings, Shape},
    tensor::{DType, TValue},
};
use bullet_gpu::{
    buffer::Buffer,
    runtime::{Device, Gpu},
};

use rand_distr::{Distribution, Normal, Uniform};
use rand_xoshiro::{Xoroshiro128Plus, rand_core::SeedableRng};

use crate::model::ModelDefinition;

pub type TensorMap<G> = BTreeMap<String, Arc<Buffer<G>>>;

#[derive(Clone)]
pub struct ShapedTValue {
    pub values: TValue,
    pub shape: Shape,
}

#[derive(Clone)]
pub struct ModelWeights {
    stores: BTreeMap<String, ShapedTValue>,
}

impl ModelWeights {
    pub fn zeroed(defn: &ModelDefinition) -> Self {
        let ir = defn.ir();
        let mut stores = BTreeMap::new();

        for (&id, (name, _)) in ir.weights() {
            let ty = ir.node(id).ty();
            let size = ty.single_size();

            assert!(ty.is_dense() && !ty.is_batched());

            stores.insert(name.clone(), ShapedTValue { values: TValue::F32(vec![0.0; size]), shape: ty.shape() });
        }

        Self { stores }
    }

    pub fn new(defn: &ModelDefinition, rng_seed: u64) -> Self {
        let ir = defn.ir();
        let mut stores = BTreeMap::new();
        let mut rng = Xoroshiro128Plus::seed_from_u64(rng_seed);

        for (&id, (name, init)) in ir.weights() {
            let ty = ir.node(id).ty();
            let size = ty.single_size();

            assert!(ty.is_dense() && !ty.is_batched());

            let init = match init {
                InitSettings::Zeroed => vec![0.0; size],
                InitSettings::Uniform { mean, stdev } => vec_f32(&mut rng, size, *mean, *stdev, false),
                InitSettings::Normal { mean, stdev } => vec_f32(&mut rng, size, *mean, *stdev, true),
                InitSettings::Custom(value) => value.f32().to_vec(),
            };

            stores.insert(name.clone(), ShapedTValue { values: TValue::F32(init), shape: ty.shape() });
        }

        Self { stores }
    }

    pub fn iter(&self) -> btree_map::Iter<'_, String, ShapedTValue> {
        self.stores.iter()
    }

    pub fn get(&self, id: &str) -> &ShapedTValue {
        self.stores.get(id).as_ref().unwrap()
    }

    pub fn set(&mut self, id: &str, value: TValue) -> bool {
        if let Some(val) = self.stores.get_mut(id) {
            if value.size() == val.values.size() && value.dtype() == val.values.dtype() {
                val.values = value;
                return true;
            }
        }

        false
    }

    pub fn write_into(&self, writer: &mut impl io::Write) -> io::Result<()> {
        let mut buf = Vec::new();

        for (id, value) in &self.stores {
            if value.values.dtype() != DType::F32 {
                unimplemented!("Non f32 writing!");
            }

            let byte_buf = utils::write_to_byte_buffer(&value.values, id).unwrap();
            buf.extend_from_slice(&byte_buf);
        }

        writer.write_all(&buf).unwrap();

        Ok(())
    }

    pub fn load_from(&mut self, mut reader: impl io::Read) -> io::Result<()> {
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).unwrap();

        let mut offset = 0;

        while offset < buf.len() {
            let (buffer, id, bytes_read) = utils::read_from_byte_buffer(&buf[offset..]);
            let weights = self.stores.get_mut(&id).expect("No weight with ID found!");

            if weights.values.dtype() != DType::F32 {
                unimplemented!("Non f32 writing!");
            }

            if buffer.len() != weights.values.size() {
                panic!("Invalid buffer size!");
            }

            weights.values = TValue::F32(buffer);

            offset += bytes_read;
        }

        Ok(())
    }

    pub fn to_device<G: Gpu>(&self, device: &Arc<Device<G>>) -> Result<TensorMap<G>, G::Error> {
        self.stores.iter().map(|(x, y)| Buffer::from_host(device, &y.values).map(|buf| (x.clone(), buf))).collect()
    }

    pub fn write_to_device<G: Gpu>(&self, values: &TensorMap<G>) -> Result<(), G::Error> {
        for (id, val) in &self.stores {
            values.get(id).ok_or(format!("No weight \"{id}\"!"))?.copy_from_host(&val.values)?;
        }

        Ok(())
    }

    pub fn load_from_device<G: Gpu>(&mut self, values: &TensorMap<G>) -> Result<(), G::Error> {
        for (id, val) in &mut self.stores {
            val.values = values.get(id).ok_or(format!("No weight \"{id}\"!"))?.to_host()?;
        }

        Ok(())
    }

    pub fn to_quantised_buffer(&self, format: &[SavedFormat], pad: bool) -> io::Result<Vec<u8>> {
        let mut buf = Vec::new();

        for fmt in format {
            buf.extend_from_slice(&fmt.write_to_byte_buffer(self)?);
        }

        if pad {
            let bytes = buf.len() % 64;
            if bytes > 0 {
                let chs = [b'b', b'u', b'l', b'l', b'e', b't'];

                for i in 0..64 - bytes {
                    buf.push(chs[i % chs.len()]);
                }
            }
        }

        Ok(buf)
    }
}

type Transform = Rc<dyn Fn(&ModelWeights, Vec<f32>) -> Vec<f32>>;

#[derive(Clone)]
pub struct SavedFormat {
    custom: Option<Vec<u8>>,
    quant: QuantTarget,
    transforms: Vec<Transform>,
    round: bool,
    id: Option<String>,
}

impl SavedFormat {
    /// Save a custom set of bytes.
    /// This should be used to add a network header, padding, etc.
    pub fn custom(bytes: impl Into<Vec<u8>>) -> Self {
        Self { custom: Some(bytes.into()), ..Self::empty() }
    }

    pub fn get_id(&self) -> Option<String> {
        self.id.clone()
    }

    /// Create a `SavedFormat` that is initialised with the weights from `id`.
    pub fn id(id: &str) -> Self {
        let id = id.to_string();
        Self { id: Some(id.clone()), ..Self::empty() }.transform(move |store, _| {
            let TValue::F32(v) = store.get(&id).values.clone() else { panic!() };
            v
        })
    }

    /// Create an empty `SavedFormat`, where the initial values are empty.
    /// Appropriate for constructing save formats where multiple weights are interleaved.
    pub fn empty() -> Self {
        SavedFormat { custom: None, id: None, quant: QuantTarget::Float, transforms: Vec::new(), round: false }
    }

    /// If quantising, round rather than truncate.
    pub fn round(mut self) -> Self {
        assert!(self.custom.is_none());
        self.round = true;
        self
    }

    /// Write weights quantised by factor `multiplier` as type `T`.
    pub fn quantise<T: Quant>(mut self, multiplier: T::Multiplier) -> Self {
        assert!(self.custom.is_none());
        self.quant = T::to_target(multiplier);
        self
    }

    /// Transpose current values using the shape of the weights from weight `id`.
    /// Panics if this `SavedFormat` was constructed without an associated weight `id`.
    pub fn transpose(self) -> Self {
        let id = self.id.clone().unwrap();
        self.transform(move |graph, weights| Self::transpose_impl(graph.get(&id).shape, &weights))
    }

    /// Transform current values using the provided closure.
    pub fn transform(mut self, f: impl Fn(&ModelWeights, Vec<f32>) -> Vec<f32> + 'static) -> Self {
        assert!(self.custom.is_none());
        self.transforms.push(Rc::new(f));
        self
    }

    pub fn write_to_byte_buffer(&self, graph: &ModelWeights) -> io::Result<Vec<u8>> {
        match &self.custom {
            Some(bytes) => Ok(bytes.clone()),
            None => {
                let mut weights = Vec::new();

                for transform in &self.transforms {
                    weights = transform(graph, weights);
                }

                self.quant.quantise(self.round, &weights)
            }
        }
    }

    pub(crate) fn transpose_impl(shape: Shape, weights: &[f32]) -> Vec<f32> {
        assert_eq!(shape.size(), weights.len());

        let rows = shape.rows();
        let cols = shape.cols();
        let mut new_buf = vec![0.0; shape.size()];

        for i in 0..rows {
            for j in 0..cols {
                new_buf[cols * i + j] = weights[rows * j + i];
            }
        }

        new_buf
    }
}

#[derive(Clone, Copy)]
pub enum QuantTarget {
    Float,
    /// This takes an `i16` because it is common to want to use a quantisation
    /// value of, say, 128 with weights clipped to [-0.99, 0.99]
    I8(i16),
    I16(i16),
    I32(i32),
}

fn round_or_trunc(x: f64, round: bool) -> f64 {
    if round { x.round() } else { x.trunc() }
}

impl QuantTarget {
    pub fn quantise(self, round: bool, buf: &[f32]) -> io::Result<Vec<u8>> {
        let mut quantised = Vec::<u8>::new();

        for &float in buf {
            let to_write = match self {
                Self::Float => float.to_le_bytes().to_vec(),
                Self::I8(q) => {
                    let qf = round_or_trunc(f64::from(q) * f64::from(float), round);
                    let x = qf as i8;

                    if qf != f64::from(x) {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "Failed quantisation from f32 to i8!"));
                    }

                    x.to_le_bytes().to_vec()
                }
                Self::I16(q) => {
                    let qf = round_or_trunc(f64::from(q) * f64::from(float), round);
                    let x = qf as i16;

                    if qf != f64::from(x) {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "Failed quantisation from f32 to i16!"));
                    }

                    x.to_le_bytes().to_vec()
                }
                Self::I32(q) => {
                    let qf = round_or_trunc(f64::from(q) * f64::from(float), round);
                    let x = qf as i32;

                    if qf != f64::from(x) {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "Failed quantisation from f32 to i32!"));
                    }

                    x.to_le_bytes().to_vec()
                }
            };

            quantised.write_all(&to_write)?;
        }

        Ok(quantised)
    }
}

pub trait Quant {
    type Multiplier;

    fn to_target(q: Self::Multiplier) -> QuantTarget;
}

impl Quant for i8 {
    type Multiplier = i16;

    fn to_target(q: Self::Multiplier) -> QuantTarget {
        QuantTarget::I8(q)
    }
}

impl Quant for i16 {
    type Multiplier = i16;

    fn to_target(q: Self::Multiplier) -> QuantTarget {
        QuantTarget::I16(q)
    }
}

impl Quant for i32 {
    type Multiplier = i32;

    fn to_target(q: Self::Multiplier) -> QuantTarget {
        QuantTarget::I32(q)
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

pub fn vec_f32(rng: &mut Xoroshiro128Plus, length: usize, mean: f32, stdev: f32, use_gaussian: bool) -> Vec<f32> {
    let mut res = Vec::with_capacity(length);

    let dist = Dist::new(mean, stdev, use_gaussian);

    for _ in 0..length {
        res.push(dist.sample(rng));
    }

    res
}

pub mod utils {
    use bullet_compiler::tensor::TValue;

    pub fn write_to_byte_buffer(value: &TValue, id: &str) -> std::io::Result<Vec<u8>> {
        use std::io::{Error, ErrorKind, Write};

        let TValue::F32(value) = value else { unimplemented!() };

        if !id.is_ascii() {
            return Err(Error::new(ErrorKind::InvalidInput, "IDs may not contain non-ASCII characters!"));
        }

        if id.contains('\n') {
            return Err(Error::new(ErrorKind::InvalidInput, "IDs may not contain newlines!"));
        }

        let mut id_bytes = id.chars().map(|ch| ch as u8).collect::<Vec<_>>();

        id_bytes.push(b'\n');

        let mut buf = Vec::new();

        buf.write_all(&id_bytes)?;
        buf.write_all(&usize::to_le_bytes(value.len()))?;

        for &val in value {
            buf.write_all(&f32::to_le_bytes(val))?;
        }

        Ok(buf)
    }

    pub fn read_from_byte_buffer(bytes: &[u8]) -> (Vec<f32>, String, usize) {
        const USIZE: usize = std::mem::size_of::<usize>();

        let mut offset = 0;

        let mut id = String::new();
        loop {
            let ch = bytes[offset];
            offset += 1;

            if ch == b'\n' {
                break;
            }

            id.push(char::from(ch));
        }

        let mut single_size = [0u8; USIZE];
        single_size.copy_from_slice(&bytes[offset..offset + USIZE]);
        offset += USIZE;

        let single_size = usize::from_le_bytes(single_size);

        let total_read = offset + single_size * 4;

        let mut values = vec![0.0; single_size];

        for (word, val) in bytes[offset..total_read].chunks_exact(4).zip(values.iter_mut()) {
            let mut buf = [0; 4];
            buf.copy_from_slice(word);
            *val = f32::from_le_bytes(buf);
        }

        (values, id, total_read)
    }
}
