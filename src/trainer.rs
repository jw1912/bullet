use bullet_core::{Activation, inputs::InputType, util, GpuDataLoader, Rand};
use bullet_tensor::{
    device_synchronise, DeviceHandles, DeviceBuffer, Optimiser, Shape, SparseTensor,
    Tensor, TensorBatch,
};
use bulletformat::BulletFormat;

use crate::ansi;

struct FeatureTransformer {
    weights: Tensor,
    biases: Tensor,
    weights_grad: Tensor,
    biases_grad: Tensor,
    outputs: TensorBatch,
}

struct Affine {
    weights: Tensor,
    biases: Tensor,
    weights_grad: Tensor,
    biases_grad: Tensor,
    ones: DeviceBuffer,
}

enum Operation {
    Activate(Activation),
    Affine(Affine),
}

struct Node {
    outputs: TensorBatch,
    op: Operation,
}

struct QuantiseInfo {
    val: i32,
    start: usize,
}

pub struct Trainer<T> {
    input_getter: T,
    handle: DeviceHandles,
    optimiser: Optimiser,
    ft: FeatureTransformer,
    nodes: Vec<Node>,
    inputs: SparseTensor,
    results: TensorBatch,
    error_device: DeviceBuffer,
    error: f32,
    used: usize,
    scale: f32,
    quantiser: Vec<QuantiseInfo>,
}

impl<T: InputType> std::fmt::Display for Trainer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inp_size = self.input_getter.inputs();
        let buckets = self.input_getter.buckets();
        write!(f, "({inp_size}")?;
        if buckets > 1 {
            write!(f, "x{buckets}")?;
        }
        write!(f, " -> {})x2", self.nodes[0].outputs.shape().rows() / 2)?;
        for (i, node) in self.nodes.iter().enumerate() {
            if let Operation::Activate(_) = node.op {
                continue;
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

impl<T: InputType> Trainer<T> {
    pub fn display(&self) {
        println!("Arch           : {}", ansi(format!("{self}"), 31));
        println!("Batch Size     : {}", ansi(self.batch_size(), 31));
        println!("Scale          : {}", ansi(format!("{:.0}", self.eval_scale()), 31));
    }

    pub fn save(&self, out_dir: &str, name: String, epoch: usize) {
        let size = self.optimiser.size();

        let mut buf1 = vec![0.0; size];
        let mut buf2 = vec![0.0; size];
        let mut buf3 = vec![0.0; size];

        self.optimiser.write_to_host(&mut buf1, &mut buf2, &mut buf3);

        let path = format!("{out_dir}/{name}-epoch{epoch}");

        std::fs::create_dir(path.as_str()).unwrap_or(());

        util::write_to_bin(&buf1, size, &format!("{path}/params.bin"), false).unwrap();
        util::write_to_bin(&buf2, size, &format!("{path}/momentum.bin"), false).unwrap();
        util::write_to_bin(&buf3, size, &format!("{path}/velocity.bin"), false).unwrap();

        if !self.quantiser.is_empty() {
            self.save_quantised(&format!("{path}/{name}-epoch{epoch}.bin"));
        }
    }

    pub fn save_quantised(&self, out_path: &str) {
        let size = self.optimiser.size();
        let mut buf = vec![0.0; size];

        self.optimiser.write_weights_to_host(&mut buf);

        let mut qbuf = vec![0i16; size];
        let mut qiter = self.quantiser.iter().peekable();
        while let Some(&QuantiseInfo { val, start }) = qiter.next() {
            let end = if let Some(QuantiseInfo {
                start: next_start, ..
            }) = qiter.peek()
            {
                *next_start
            } else {
                size
            };

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

        util::write_to_bin(&qbuf, size, out_path, true).unwrap();
    }

    fn load_from_bin(&self, path: &str) -> Vec<f32> {
        use std::fs::File;
        use std::io::{BufReader, Read};
        let file = File::open(path).unwrap();

        assert_eq!(
            file.metadata().unwrap().len() as usize,
            self.net_size() * std::mem::size_of::<f32>(),
            "Incorrect File Size!"
        );

        let reader = BufReader::new(file);
        let mut res = vec![0.0; self.net_size()];

        let mut buf = [0u8; 4];

        for (i, byte) in reader.bytes().enumerate() {
            let idx = i % 4;

            buf[idx] = byte.unwrap();

            if idx == 3 {
                res[i / 4] = f32::from_ne_bytes(buf);
            }
        }

        res
    }

    pub fn set_threads(&mut self, threads: usize) {
        self.handle.set_threads(threads);
        self.error_device = DeviceBuffer::new(threads);
    }

    pub fn load_weights_from_file(&self, path: &str) {
        let network = self.load_from_bin(path);
        self.optimiser.load_weights_from_host(&network);
    }

    pub fn load_from_checkpoint(&self, path: &str) {
        let network = self.load_from_bin(format!("{path}/params.bin").as_str());
        let momentum = self.load_from_bin(format!("{path}/momentum.bin").as_str());
        let velocity = self.load_from_bin(format!("{path}/velocity.bin").as_str());

        self.optimiser.load_from_cpu(&network, &momentum, &velocity)
    }

    pub fn error(&self) -> f32 {
        self.error
    }

    pub fn input_getter(&self) -> T {
        self.input_getter
    }

    pub fn net_size(&self) -> usize {
        self.optimiser.size()
    }

    pub fn eval_scale(&self) -> f32 {
        self.scale
    }

    pub fn write_weights_to_cpu(&self, buf: &mut [f32]) {
        self.optimiser.write_weights_to_host(buf);
    }

    pub fn clear_data(&mut self) {
        self.used = 0;
        self.inputs.clear();
    }

    pub fn load_data(&mut self, loader: &GpuDataLoader<T>) {
        let inputs = loader.inputs();
        let results = loader.results();

        unsafe {
            let our = std::slice::from_raw_parts(inputs.as_ptr().cast(), inputs.len());
            self.inputs.append(our);
            self.results.load_from_host(results);
            self.used += results.len();
        }
    }

    pub fn batch_size(&self) -> usize {
        self.ft.outputs.cap()
    }

    pub fn eval(&mut self, fen: &str)
    where
        T::RequiredDataType: std::str::FromStr<Err = String>,
    {
        self.clear_data();
        let board = fen.parse::<T::RequiredDataType>().unwrap();
        let mut loader = GpuDataLoader::new(self.input_getter);
        loader.load(&[board], 1, 0.0, self.scale);
        self.load_data(&loader);

        unsafe {
            self.forward();
        }

        bullet_tensor::panic_if_device_error("Something went wrong!");

        let mut eval = vec![0.0; self.batch_size()];
        self.nodes.last().unwrap().outputs.write_to_host(&mut eval);
        println!("FEN: {fen}");
        println!("EVAL: {}", self.scale * eval[0]);

        self.clear_data();
    }

    pub fn train_on_batch(&mut self, decay: f32, rate: f32) {
        self.optimiser.zero_gradient();
        self.error_device.set_zero();

        unsafe {
            self.forward();
            self.calc_errors();
            self.backprop();
        }

        let adj = 2. / self.inputs.used() as f32;
        self.optimiser.update(self.handle, decay, adj, rate);

        let mut errors = vec![0.0; self.error_device.size()];
        self.error_device.write_to_host(&mut errors);
        self.error += errors.iter().sum::<f32>() / self.inputs.used() as f32;

        device_synchronise();
    }

    /// # Safety
    /// It is undefined behaviour to call this if `our_inputs` is not
    /// properly initialised.
    unsafe fn forward(&self) {
        let batch_size = self.inputs.used();

        SparseTensor::affine(
            self.handle,
            &self.ft.weights,
            &self.inputs,
            &self.ft.biases,
            &self.ft.outputs,
        );

        let mut inputs = &self.ft.outputs;

        for node in &self.nodes {
            match &node.op {
                Operation::Activate(activation) => {
                    TensorBatch::activate(self.handle, batch_size, *activation, inputs, &node.outputs);
                }
                Operation::Affine(Affine {
                    weights, biases, ..
                }) => {
                    TensorBatch::affine(
                        self.handle,
                        batch_size,
                        weights,
                        inputs,
                        biases,
                        &node.outputs,
                    );
                }
            }

            inputs = &node.outputs;
        }
    }

    /// # Safety
    /// It is undefined behaviour to call this without previously calling
    /// `self.forward`.
    unsafe fn calc_errors(&self) {
        let batch_size = self.inputs.used();
        let output_layer = self.nodes.last().unwrap();

        assert_eq!(self.results.shape(), output_layer.outputs.shape());

        output_layer
            .outputs
            .sigmoid_mse(self.handle, batch_size, &self.results, &self.error_device);
    }

    /// # Safety
    /// It is undefined behaviour to call this without previously calling
    /// `self.forward` and `self.calc_errors()`, as well as if `our_inputs`
    /// is not properly initialised.
    unsafe fn backprop(&self) {
        let batch_size = self.inputs.used();
        let num_nodes = self.nodes.len();
        device_synchronise();

        for node in (1..num_nodes).rev() {
            backprop_single(
                self.handle,
                batch_size,
                &self.nodes[node],
                &self.nodes[node - 1].outputs,
            );
        }

        backprop_single(self.handle, batch_size, &self.nodes[0], &self.ft.outputs);

        SparseTensor::affine_backprop(
            self.handle,
            &self.ft.weights_grad,
            &self.inputs,
            &self.ft.biases_grad,
            &self.ft.outputs,
        );
    }
}

fn backprop_single(
    handle: DeviceHandles,
    batch_size: usize,
    this_node: &Node,
    inputs: &TensorBatch,
) {
    let errors = &this_node.outputs;

    match &this_node.op {
        Operation::Activate(activation) => {
            TensorBatch::backprop_activation(handle, batch_size, *activation, errors, inputs);
        }
        Operation::Affine(Affine {
            weights: w,
            weights_grad: wg,
            biases_grad: bg,
            ones,
            ..
        }) => unsafe {
            TensorBatch::backprop_affine(handle, ones, batch_size, w, errors, inputs, wg, bg);
        },
    }
}

enum OpType {
    Activate(Activation),
    Affine,
}

struct NodeType {
    size: usize,
    op: OpType,
}

pub struct TrainerBuilder<T> {
    input_getter: T,
    batch_size: usize,
    ft_out_size: usize,
    nodes: Vec<NodeType>,
    quantisations: Vec<i32>,
    size: usize,
    scale: f32,
}

impl<T: InputType> Default for TrainerBuilder<T> {
    fn default() -> Self {
        Self {
            input_getter: T::default(),
            batch_size: 0,
            ft_out_size: 0,
            nodes: Vec::new(),
            quantisations: Vec::new(),
            size: 0,
            scale: 400.0,
        }
    }
}

impl<T: InputType> TrainerBuilder<T> {
    fn get_last_layer_size(&self) -> usize {
        if let Some(node) = self.nodes.last() {
            node.size
        } else {
            2 * self.ft_out_size
        }
    }

    pub fn set_input(mut self, input_getter: T) -> Self {
        self.input_getter = input_getter;
        self
    }

    pub fn set_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn set_eval_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    pub fn set_quantisations(mut self, quants: &[i32]) -> Self {
        self.quantisations = quants.to_vec();
        self
    }

    pub fn ft(mut self, size: usize) -> Self {
        assert!(self.nodes.is_empty());
        self.ft_out_size = size;
        self
    }

    fn add(mut self, size: usize, op: OpType) -> Self {
        self.nodes.push(NodeType { size, op });

        self
    }

    pub fn add_layer(mut self, size: usize) -> Self {
        self.size += (self.get_last_layer_size() + 1) * size;
        self.add(size, OpType::Affine)
    }

    pub fn activate(self, activation: Activation) -> Self {
        let size = self.get_last_layer_size();
        self.add(size, OpType::Activate(activation))
    }
}

impl<T: InputType> TrainerBuilder<T> {
    pub fn build(self) -> Trainer<T> {
        let inp_getter_size = self.input_getter.size();

        let ft_size = (inp_getter_size + 1) * self.ft_out_size;
        let net_size = self.size + ft_size;

        let opt = Optimiser::new(net_size);
        let batch_size = self.batch_size;

        unsafe {
            let ftw_shape = Shape::new(self.ft_out_size, inp_getter_size);
            let ftb_shape = Shape::new(1, self.ft_out_size);

            let mut ft = FeatureTransformer {
                weights: Tensor::uninit(ftw_shape),
                biases: Tensor::uninit(ftb_shape),
                weights_grad: Tensor::uninit(ftw_shape),
                biases_grad: Tensor::uninit(ftb_shape),
                outputs: TensorBatch::new(Shape::new(1, 2 * self.ft_out_size), batch_size),
            };

            let mut offset = 0;
            ft.weights.set_ptr(opt.weights_offset(offset));
            ft.weights_grad.set_ptr(opt.gradients_offset(offset));
            offset += self.ft_out_size * inp_getter_size;

            ft.biases.set_ptr(opt.weights_offset(offset));
            ft.biases_grad.set_ptr(opt.gradients_offset(offset));
            offset += self.ft_out_size;

            let mut nodes = Vec::new();
            let mut inp_size = 2 * self.ft_out_size;

            let mut quantiser = Vec::new();
            let mut qi = 0;
            let mut accq = 1;
            if !self.quantisations.is_empty() {
                quantiser.push(QuantiseInfo {
                    val: self.quantisations[qi],
                    start: 0,
                });
                accq *= self.quantisations[qi];
                qi += 1;
            }

            for NodeType { size, op } in &self.nodes {
                let size = *size;
                let bsh = Shape::new(1, size);

                let op = match op {
                    OpType::Affine => {
                        let wsh = Shape::new(inp_size, size);
                        let ones = DeviceBuffer::new(1);
                        ones.load_from_host(&[1.0]);
                        let mut affine = Affine {
                            weights: Tensor::uninit(wsh),
                            biases: Tensor::uninit(bsh),
                            weights_grad: Tensor::uninit(wsh),
                            biases_grad: Tensor::uninit(bsh),
                            ones,
                        };

                        affine.weights.set_ptr(opt.weights_offset(offset));
                        affine.weights_grad.set_ptr(opt.gradients_offset(offset));

                        if !self.quantisations.is_empty() {
                            quantiser.push(QuantiseInfo {
                                val: self.quantisations[qi],
                                start: offset,
                            });
                        }

                        offset += inp_size * size;

                        affine.biases.set_ptr(opt.weights_offset(offset));
                        affine.biases_grad.set_ptr(opt.gradients_offset(offset));

                        if !self.quantisations.is_empty() {
                            accq *= self.quantisations[qi];
                            quantiser.push(QuantiseInfo {
                                val: accq,
                                start: offset,
                            });
                            qi += 1;
                        }

                        offset += size;

                        Operation::Affine(affine)
                    }
                    OpType::Activate(activation) => Operation::Activate(*activation),
                };

                let outputs = TensorBatch::new(bsh, batch_size);

                nodes.push(Node { outputs, op });

                inp_size = size;
            }

            assert_eq!(
                qi,
                self.quantisations.len(),
                "Incorrectly specified number of quantisations!"
            );
            assert_eq!(offset, net_size);

            let inputs = SparseTensor::uninit(
                batch_size,
                inp_getter_size,
                T::RequiredDataType::MAX_FEATURES,
            );

            let results = TensorBatch::new(Shape::new(1, 1), batch_size);
            let error_device = DeviceBuffer::new(1);

            let mut net = vec![0.0; net_size];
            let mut rng = Rand::default();

            for val in net.iter_mut() {
                *val = rng.rand(0.01);
            }

            opt.load_weights_from_host(&net);

            Trainer {
                input_getter: self.input_getter,
                handle: DeviceHandles::default(),
                optimiser: opt,
                ft,
                nodes,
                inputs,
                results,
                error_device,
                error: 0.0,
                used: 0,
                scale: self.scale,
                quantiser,
            }
        }
    }
}
