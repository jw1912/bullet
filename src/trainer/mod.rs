mod builder;
mod components;
mod run;
pub mod schedule;

pub use builder::TrainerBuilder;
use components::{Affine, FeatureTransformer, Node, Operation, QuantiseInfo};
use rand_distr::Distribution;
pub use run::{ansi, run, set_cbcs};
use std::io::Write;

use crate::{
    inputs::InputType,
    loader::GpuDataLoader,
    optimiser::Optimiser,
    outputs::OutputBuckets,
    tensor::{self, device_synchronise, DeviceBuffer, DeviceHandles, SparseTensor, TensorBatch},
    util,
};

pub struct Trainer<T, U, O> {
    input_getter: T,
    bucket_getter: U,
    handle: DeviceHandles,
    optimiser: O,
    ft: FeatureTransformer,
    ft_reg: f32,
    nodes: Vec<Node>,
    inputs: SparseTensor,
    results: TensorBatch,
    error_device: DeviceBuffer,
    error: f32,
    error_record: Vec<(usize, usize, f32)>,
    used: usize,
    quantiser: Vec<QuantiseInfo>,
    buckets: *mut u8,
}

impl<T: InputType, U, O: Optimiser> std::fmt::Display for Trainer<T, U, O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inp_size = self.input_getter.inputs();
        let buckets = self.input_getter.buckets();

        if !self.ft.single_perspective {
            write!(f, "(")?;
        }
        write!(f, "{inp_size}")?;

        if buckets > 1 {
            write!(f, "x{buckets}")?;
        }

        if !self.ft.single_perspective {
            write!(f, " -> {})x2", self.nodes[0].outputs.shape().rows() / 2)?;
        } else {
            write!(f, " -> {}", self.nodes[0].outputs.shape().rows())?;
        }

        for (i, node) in self.nodes.iter().enumerate() {
            match node.op {
                Operation::Affine(_) => {}
                _ => continue,
            }

            let rows = node.outputs.shape().rows();

            if i == 0 {
                write!(f, " -> {})x2", rows / 2)?;
            } else {
                write!(f, " -> {rows}")?;
            }
        }
        Ok(())
    }
}

impl<T: InputType, U: OutputBuckets<T::RequiredDataType>, O: Optimiser> Trainer<T, U, O> {
    pub fn set_error_zero(&mut self) {
        self.error = 0.0;
    }

    pub fn save(&self, out_dir: &str, name: String) {
        let path = format!("{out_dir}/{name}");

        std::fs::create_dir(path.as_str()).unwrap_or(());

        self.optimiser.write_to_checkpoint(path.as_str());

        let mut writer = std::io::BufWriter::new(
            std::fs::File::create(format!("{path}/log.txt")).expect("Opening log file failed!"),
        );
        for (superbatch, batch, loss) in &self.error_record {
            writeln!(writer, "superbatch:{superbatch},batch:{batch},loss:{loss}",)
                .expect("Writing to log file failed!");
        }

        if !self.quantiser.is_empty() {
            self.save_quantised(&format!("{path}/{name}.bin"));
        }
    }

    pub fn save_quantised(&self, out_path: &str) {
        let size = self.optimiser.size();
        let mut buf = vec![0.0; size];

        self.optimiser.write_weights_to_host(&mut buf);

        let mut qbuf = vec![0i16; size];
        let mut qiter = self.quantiser.iter().peekable();
        while let Some(&QuantiseInfo { val, start }) = qiter.next() {
            let end = if let Some(QuantiseInfo { start: next_start, .. }) = qiter.peek() { *next_start } else { size };

            for i in start..end {
                let qf = (f64::from(val) * f64::from(buf[i])).trunc();
                let q = qf as i16;
                if f64::from(q) != qf {
                    println!("================= WARNING ================");
                    println!("   An error occured during quantisation:  ");
                    println!("     > Cannot convert \"{qf:.0}\"");
                    println!("   You will need to quantise manually.    ");
                    println!("==========================================");
                    return;
                }
                qbuf[i] = q;
            }
        }

        util::write_to_bin(&qbuf, size, out_path, true).unwrap_or_else(|_| panic!("Writing to [{out_path}] failed!"));
    }

    pub fn set_threads(&mut self, threads: usize) {
        self.handle.set_threads(threads);
        self.error_device = DeviceBuffer::new(threads);
    }

    pub fn load_weights_from_file(&self, path: &str) {
        let network = util::load_from_bin_f32_slice(self.net_size(), path);
        self.optimiser.load_weights_from_host(&network);
    }

    pub fn load_from_checkpoint(&self, path: &str) {
        self.optimiser.load_from_checkpoint(path);
    }

    pub fn set_batch_size(&mut self, batch_size: usize) {
        if !self.buckets.is_null() {
            unsafe { tensor::util::free(self.buckets, self.batch_size()) }
        }
        self.buckets = tensor::util::calloc(batch_size);

        let inp_dim = self.input_getter.size();
        let max_active_inputs = self.input_getter.max_active_inputs();

        unsafe {
            self.inputs = SparseTensor::uninit(batch_size, inp_dim, max_active_inputs);
        }

        self.results = TensorBatch::new(self.results.shape(), batch_size);
        self.ft.outputs = TensorBatch::new(self.ft.outputs.shape(), batch_size);
        self.ft.copy = TensorBatch::new(self.ft.copy.shape(), batch_size);

        for node in &mut self.nodes {
            node.outputs = TensorBatch::new(node.outputs.shape(), batch_size);

            if let Operation::Affine(Affine { ones, .. }) = &mut node.op {
                *ones = DeviceBuffer::new(batch_size);
                ones.load_from_host(&vec![1.0; batch_size]);
            }
        }
    }

    pub fn randomise_weights(&self, init_biases: bool, use_gaussian: bool) {
        use rand::{rngs::ThreadRng, thread_rng};
        use rand_distr::{Normal, Uniform};

        enum Dist {
            Normal(Normal<f32>),
            Uniform(Uniform<f32>),
        }

        impl Dist {
            fn new(stdev: f32, use_gaussian: bool) -> Self {
                if use_gaussian {
                    Self::Normal(Normal::new(0.0, stdev).unwrap())
                } else {
                    Self::Uniform(Uniform::new(-stdev, stdev))
                }
            }

            fn sample(&self, rng: &mut ThreadRng) -> f32 {
                match self {
                    Dist::Normal(x) => x.sample(rng),
                    Dist::Uniform(x) => x.sample(rng),
                }
            }
        }

        let mut network = vec![0.0; self.net_size()];

        let mut rng = thread_rng();

        let ft_wsize = self.ft.weights.num_elements();
        let ft_bsize = self.ft.biases.num_elements();
        let input_size = self.ft.weights.shape().cols();

        let stdev = (1.0 / input_size as f32).sqrt();
        let dist = Dist::new(stdev, use_gaussian);

        for weight in network.iter_mut().take(ft_wsize) {
            *weight = dist.sample(&mut rng);
        }

        let mut offset = ft_wsize;

        if init_biases {
            for weight in network.iter_mut().skip(offset).take(ft_bsize) {
                *weight = dist.sample(&mut rng);
            }
        }

        offset += ft_bsize;

        for Node { op, .. } in &self.nodes {
            if let Operation::Affine(Affine { weights, biases, .. }) = op {
                let wsize = weights.num_elements();
                let bsize = biases.num_elements();
                let input_size = weights.shape().cols();

                let stdev = (1.0 / input_size as f32).sqrt();
                let dist = Dist::new(stdev, use_gaussian);

                for weight in network.iter_mut().skip(offset).take(wsize) {
                    *weight = dist.sample(&mut rng);
                }

                offset += wsize;

                if init_biases {
                    for weight in network.iter_mut().skip(offset).take(bsize) {
                        *weight = dist.sample(&mut rng);
                    }
                }

                offset += bsize;
            }
        }

        self.optimiser.load_weights_from_host(&network);
    }

    pub fn set_ft_reg(&mut self, val: f32) {
        self.ft_reg = val;
    }

    pub fn error(&self) -> f32 {
        self.error
    }

    pub fn input_getter(&self) -> T {
        self.input_getter
    }

    pub fn bucket_getter(&self) -> U {
        self.bucket_getter
    }

    pub fn net_size(&self) -> usize {
        self.optimiser.size()
    }

    pub fn write_weights_to_cpu(&self, buf: &mut [f32]) {
        self.optimiser.write_weights_to_host(buf);
    }

    pub fn clear_data(&mut self) {
        self.used = 0;
        self.inputs.clear();
    }

    pub fn load_data(&mut self, loader: &GpuDataLoader<T, U>) {
        let inputs = loader.inputs();
        let results = loader.results();
        let buckets = loader.buckets();

        unsafe {
            let our = std::slice::from_raw_parts(inputs.as_ptr().cast(), inputs.len());
            self.inputs.append(our);
            self.results.load_from_host(results);

            if U::BUCKETS > 1 {
                let ptr = buckets.as_ptr();
                let amt = buckets.len();
                tensor::util::copy_to_device(self.buckets, ptr, amt);
            }

            self.used += results.len();
        }
    }

    pub fn batch_size(&self) -> usize {
        self.ft.outputs.cap()
    }

    pub fn eval(&mut self, fen: &str) -> f32
    where
        T::RequiredDataType: std::str::FromStr<Err = String>,
    {
        self.clear_data();
        let board = format!("{fen} | 0 | 0.0").parse::<T::RequiredDataType>().expect("Failed to parse position!");
        let mut loader = GpuDataLoader::new(self.input_getter, self.bucket_getter);
        loader.load(&[board], 1, 0.0, 1.0);
        self.load_data(&loader);

        unsafe {
            self.forward();
        }

        tensor::panic_if_device_error("Something went wrong!");

        let mut eval = vec![0.0; self.batch_size()];
        self.nodes.last().expect("Nodes is empty!").outputs.write_to_host(&mut eval);

        self.clear_data();
        eval[0]
    }

    pub fn train_on_batch(
        &mut self,
        rate: f32,
        power: f32,
        superbatch: usize,
        curr_batch: usize,
        params: &O::AdditionalOptimiserParams,
    ) -> bool {
        self.optimiser.zero_gradient();
        self.error_device.set_zero();

        unsafe {
            self.forward();
            self.calc_errors(power);
            self.backprop();
        }

        let mut errors = vec![0.0; self.error_device.size()];
        self.error_device.write_to_host(&mut errors);
        let error = errors.iter().sum::<f32>() / self.inputs.used() as f32;
        self.error += error;
        self.error_record.push((superbatch, curr_batch, error));

        tensor::panic_if_device_error("Something went wrong!");

        if self.error.is_nan() {
            return false;
        }

        let adj = power / self.inputs.used() as f32;
        self.optimiser.update(&self.handle, adj, rate, params);

        device_synchronise();
        true
    }

    /// # Safety
    /// It is undefined behaviour to call this if `our_inputs` is not
    /// properly initialised.
    unsafe fn forward(&self) {
        let batch_size = self.inputs.used();

        if self.ft.single_perspective {
            SparseTensor::single_affine(&self.handle, &self.ft.weights, &self.inputs, &self.ft.biases, &self.ft.outputs);
        } else {
            SparseTensor::affine(&self.handle, &self.ft.weights, &self.inputs, &self.ft.biases, &self.ft.outputs);
        }

        let mut inputs = &self.ft.outputs;
        let mut res_inputs = inputs;
        let mut in_res_block = false;

        for node in &self.nodes {
            // entering residual block
            if !in_res_block && node.in_res_block {
                in_res_block = true;
                res_inputs = inputs;
            }

            // exiting residual block
            if in_res_block && !node.in_res_block {
                in_res_block = false;
                TensorBatch::add_to(&self.handle, batch_size, res_inputs, inputs);
            }

            match &node.op {
                Operation::Activate(activation) => {
                    TensorBatch::activate(&self.handle, batch_size, *activation, inputs, &node.outputs);
                }
                Operation::Affine(Affine { weights, biases, .. }) => {
                    TensorBatch::affine(&self.handle, batch_size, weights, inputs, biases, &node.outputs);
                }
                Operation::Select => TensorBatch::select(&self.handle, batch_size, self.buckets, inputs, &node.outputs),
            }

            inputs = &node.outputs;
        }
    }

    /// # Safety
    /// It is undefined behaviour to call this without previously calling
    /// `self.forward`.
    unsafe fn calc_errors(&self, power: f32) {
        let batch_size = self.inputs.used();
        let output_layer = self.nodes.last().expect("Nodes is empty!");

        assert_eq!(self.results.shape(), output_layer.outputs.shape());

        output_layer.outputs.sigmoid_mpe(&self.handle, batch_size, &self.results, &self.error_device, power);
    }

    /// # Safety
    /// It is undefined behaviour to call this without previously calling
    /// `self.forward` and `self.calc_errors()`, as well as if `our_inputs`
    /// is not properly initialised.
    unsafe fn backprop(&self) {
        let batch_size = self.inputs.used();
        let num_nodes = self.nodes.len();
        device_synchronise();

        let mut res_errors = &self.nodes[num_nodes - 1].outputs;
        let mut in_res_block = false;

        for node in (1..num_nodes).rev() {
            backprop_single(
                &self.handle,
                batch_size,
                &self.nodes[node],
                &self.nodes[node - 1].outputs,
                self.nodes[node - 1].in_res_block,
                self.buckets,
                &mut res_errors,
                &mut in_res_block,
            );
        }

        if self.ft_reg != 0.0 {
            self.ft.copy.copy_from(&self.ft.outputs);
        }

        backprop_single(
            &self.handle,
            batch_size,
            &self.nodes[0],
            &self.ft.outputs,
            false,
            self.buckets,
            &mut res_errors,
            &mut in_res_block,
        );

        if self.ft.single_perspective {
            SparseTensor::single_affine_backprop(
                &self.handle,
                &self.ft.weights_grad,
                &self.inputs,
                &self.ft.biases_grad,
                &self.ft.outputs,
                &self.ft.copy,
                self.ft_reg,
            );
        } else {
            SparseTensor::affine_backprop(
                &self.handle,
                &self.ft.weights_grad,
                &self.inputs,
                &self.ft.biases_grad,
                &self.ft.outputs,
                &self.ft.copy,
                self.ft_reg,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn backprop_single<'a>(
    handle: &DeviceHandles,
    batch_size: usize,
    this_node: &Node,
    inputs: &'a TensorBatch,
    in_res: bool,
    buckets: *const u8,
    res_errors: &mut &'a TensorBatch,
    in_res_block: &mut bool,
) {
    let errors = &this_node.outputs;

    match &this_node.op {
        Operation::Activate(activation) => {
            TensorBatch::backprop_activation(handle, batch_size, *activation, errors, inputs);
        }
        Operation::Affine(Affine { weights: w, weights_grad: wg, biases_grad: bg, ones, .. }) => {
            TensorBatch::backprop_affine(handle, ones, batch_size, w, errors, inputs, wg, bg);
        }
        Operation::Select => TensorBatch::select_backprop(handle, batch_size, buckets, errors, inputs),
    }

    // entering residual block
    if !*in_res_block && in_res {
        *in_res_block = true;
        *res_errors = inputs;
    }

    // exiting residual block
    if *in_res_block && !in_res {
        *in_res_block = false;
        TensorBatch::add_to(handle, batch_size, res_errors, inputs);
    }
}
