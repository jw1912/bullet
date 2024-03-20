use crate::{
    inputs::InputType,
    loader::GpuDataLoader,
    outputs::OutputBuckets,
    tensor::{
        self, device_synchronise, DeviceBuffer, DeviceHandles, Optimiser, Shape, SparseTensor,
        Tensor, TensorBatch,
    },
    util, Activation, Rand,
};

struct FeatureTransformer {
    weights: Tensor,
    biases: Tensor,
    weights_grad: Tensor,
    biases_grad: Tensor,
    single_perspective: bool,
    outputs: TensorBatch,
    copy: TensorBatch,
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
    Select,
    DualActivate,
}

struct Node {
    outputs: TensorBatch,
    op: Operation,
}

struct QuantiseInfo {
    val: i32,
    start: usize,
}

pub struct Trainer<T, U> {
    input_getter: T,
    bucket_getter: U,
    handle: DeviceHandles,
    optimiser: Optimiser,
    ft: FeatureTransformer,
    ft_reg: f32,
    nodes: Vec<Node>,
    inputs: SparseTensor,
    results: TensorBatch,
    error_device: DeviceBuffer,
    error: f32,
    used: usize,
    quantiser: Vec<QuantiseInfo>,
    buckets: *mut u8,
}

impl<T: InputType, U> std::fmt::Display for Trainer<T, U> {
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

impl<T: InputType, U: OutputBuckets<T::RequiredDataType>> Trainer<T, U> {
    pub fn set_error_zero(&mut self) {
        self.error = 0.0;
    }

    pub fn save(&self, out_dir: &str, name: String) {
        let size = self.optimiser.size();

        let mut buf1 = vec![0.0; size];
        let mut buf2 = vec![0.0; size];
        let mut buf3 = vec![0.0; size];

        self.optimiser
            .write_to_host(&mut buf1, &mut buf2, &mut buf3);

        let path = format!("{out_dir}/{name}");

        std::fs::create_dir(path.as_str()).unwrap_or(());

        util::write_to_bin(&buf1, size, &format!("{path}/params.bin"), false)
            .unwrap_or_else(|_| panic!("Writing to [{path}/params.bin] failed!"));
        util::write_to_bin(&buf2, size, &format!("{path}/momentum.bin"), false)
            .unwrap_or_else(|_| panic!("Writing to [{path}/momentum.bin] failed!"));
        util::write_to_bin(&buf3, size, &format!("{path}/velocity.bin"), false)
            .unwrap_or_else(|_| panic!("Writing to [{path}/velocity.bin] failed!"));

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

        util::write_to_bin(&qbuf, size, out_path, true)
            .unwrap_or_else(|_| panic!("Writing to [{out_path}] failed!"));
    }

    fn load_from_bin(&self, path: &str) -> Vec<f32> {
        use std::fs::File;
        use std::io::{BufReader, Read};
        let file = File::open(path).unwrap_or_else(|_| panic!("Invalid File Path: {path}"));

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

    pub fn set_batch_size(&mut self, batch_size: usize) {
        if !self.buckets.is_null() {
            unsafe { tensor::util::free_raw_bytes(self.buckets, self.batch_size()) }
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
        }
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
        let board = format!("{fen} | 0 | 0.0")
            .parse::<T::RequiredDataType>()
            .expect("Failed to parse position!");
        let mut loader = GpuDataLoader::new(self.input_getter, self.bucket_getter);
        loader.load(&[board], 1, 0.0, 1.0);
        self.load_data(&loader);

        unsafe {
            self.forward();
        }

        tensor::panic_if_device_error("Something went wrong!");

        let mut eval = vec![0.0; self.batch_size()];
        self.nodes
            .last()
            .expect("Nodes is empty!")
            .outputs
            .write_to_host(&mut eval);

        self.clear_data();
        eval[0]
    }

    pub fn train_on_batch(&mut self, decay: f32, rate: f32) -> bool {
        self.optimiser.zero_gradient();
        self.error_device.set_zero();

        unsafe {
            self.forward();
            self.calc_errors();
            self.backprop();
        }

        let mut errors = vec![0.0; self.error_device.size()];
        self.error_device.write_to_host(&mut errors);
        self.error += errors.iter().sum::<f32>() / self.inputs.used() as f32;

        if self.error.is_nan() {
            return false;
        }

        let adj = 2. / self.inputs.used() as f32;
        self.optimiser.update(self.handle, decay, adj, rate);

        device_synchronise();
        true
    }

    /// # Safety
    /// It is undefined behaviour to call this if `our_inputs` is not
    /// properly initialised.
    unsafe fn forward(&self) {
        let batch_size = self.inputs.used();

        if self.ft.single_perspective {
            SparseTensor::single_affine(
                self.handle,
                &self.ft.weights,
                &self.inputs,
                &self.ft.biases,
                &self.ft.outputs,
            );
        } else {
            SparseTensor::affine(
                self.handle,
                &self.ft.weights,
                &self.inputs,
                &self.ft.biases,
                &self.ft.outputs,
            );
        }

        let mut inputs = &self.ft.outputs;

        for node in &self.nodes {
            match &node.op {
                Operation::Activate(activation) => {
                    TensorBatch::activate(
                        self.handle,
                        batch_size,
                        *activation,
                        inputs,
                        &node.outputs,
                    );
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
                Operation::DualActivate => {
                    TensorBatch::activate_dual(self.handle, batch_size, inputs, &node.outputs)
                }
                Operation::Select => TensorBatch::select(
                    self.handle,
                    batch_size,
                    self.buckets,
                    inputs,
                    &node.outputs,
                ),
            }

            inputs = &node.outputs;
        }
    }

    /// # Safety
    /// It is undefined behaviour to call this without previously calling
    /// `self.forward`.
    unsafe fn calc_errors(&self) {
        let batch_size = self.inputs.used();
        let output_layer = self.nodes.last().expect("Nodes is empty!");

        assert_eq!(self.results.shape(), output_layer.outputs.shape());

        output_layer.outputs.sigmoid_mse(
            self.handle,
            batch_size,
            &self.results,
            &self.error_device,
        );
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
                self.buckets,
            );
        }

        self.ft.copy.copy_from(&self.ft.outputs);

        backprop_single(
            self.handle,
            batch_size,
            &self.nodes[0],
            &self.ft.outputs,
            self.buckets,
        );

        if self.ft.single_perspective {
            SparseTensor::single_affine_backprop(
                self.handle,
                &self.ft.weights_grad,
                &self.inputs,
                &self.ft.biases_grad,
                &self.ft.outputs,
                &self.ft.copy,
                self.ft_reg,
            );
        } else {
            SparseTensor::affine_backprop(
                self.handle,
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

unsafe fn backprop_single(
    handle: DeviceHandles,
    batch_size: usize,
    this_node: &Node,
    inputs: &TensorBatch,
    buckets: *const u8,
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
        }) => {
            TensorBatch::backprop_affine(handle, ones, batch_size, w, errors, inputs, wg, bg);
        }
        Operation::DualActivate => TensorBatch::backprop_dual(handle, batch_size, errors, inputs),
        Operation::Select => {
            TensorBatch::select_backprop(handle, batch_size, buckets, errors, inputs)
        }
    }
}

enum OpType {
    Activate(Activation),
    Affine,
    DualActivate,
}

struct NodeType {
    size: usize,
    op: OpType,
}

pub struct TrainerBuilder<T, U> {
    input_getter: T,
    bucket_getter: U,
    ft_out_size: usize,
    nodes: Vec<NodeType>,
    quantisations: Vec<i32>,
    single_perspective: bool,
    size: usize,
}

impl<T: InputType, U: OutputBuckets<T::RequiredDataType>> Default for TrainerBuilder<T, U> {
    fn default() -> Self {
        Self {
            input_getter: T::default(),
            bucket_getter: U::default(),
            ft_out_size: 0,
            nodes: Vec::new(),
            quantisations: Vec::new(),
            single_perspective: false,
            size: 0,
        }
    }
}

impl<T: InputType, U: OutputBuckets<T::RequiredDataType>> TrainerBuilder<T, U> {
    fn get_last_layer_size(&self) -> usize {
        if let Some(node) = self.nodes.last() {
            node.size
        } else {
            self.ft_out_size * if self.single_perspective { 1 } else { 2 }
        }
    }

    pub fn single_perspective(mut self) -> Self {
        if !self.nodes.is_empty() {
            panic!("You need to set 'single_perspective' before adding any layers!");
        }
        self.single_perspective = true;
        self
    }

    pub fn input(mut self, input_getter: T) -> Self {
        self.input_getter = input_getter;
        self
    }

    pub fn output_buckets(mut self, bucket_getter: U) -> Self {
        self.bucket_getter = bucket_getter;
        self
    }

    pub fn quantisations(mut self, quants: &[i32]) -> Self {
        self.quantisations = quants.to_vec();
        self
    }

    pub fn feature_transformer(mut self, size: usize) -> Self {
        assert!(self.nodes.is_empty());
        self.ft_out_size = size;
        self
    }

    fn add(mut self, size: usize, op: OpType) -> Self {
        self.nodes.push(NodeType { size, op });

        self
    }

    pub fn add_layer(mut self, size: usize) -> Self {
        self.size += (self.get_last_layer_size() + 1) * size * U::BUCKETS;
        self.add(size, OpType::Affine)
    }

    pub fn activate(self, activation: Activation) -> Self {
        let size = self.get_last_layer_size();
        self.add(size, OpType::Activate(activation))
    }

    /// Apply both CReLU and SCReLU, concat results
    pub fn dual_activate(self) -> Self {
        let size = 2 * self.get_last_layer_size();
        self.add(size, OpType::DualActivate)
    }

    pub fn build(self) -> Trainer<T, U> {
        let inp_getter_size = self.input_getter.size();
        let max_active_inputs = self.input_getter.max_active_inputs();

        let buckets = U::BUCKETS;

        let ft_size = (inp_getter_size + 1) * self.ft_out_size;
        let net_size = self.size + ft_size;

        let opt = Optimiser::new(net_size);
        let batch_size = 1;
        let mul = if self.single_perspective { 1 } else { 2 };

        unsafe {
            let ftw_shape = Shape::new(self.ft_out_size, inp_getter_size);
            let ftb_shape = Shape::new(1, self.ft_out_size);
            let fto_shape = Shape::new(1, mul * self.ft_out_size);

            let mut ft = FeatureTransformer {
                weights: Tensor::uninit(ftw_shape),
                biases: Tensor::uninit(ftb_shape),
                weights_grad: Tensor::uninit(ftw_shape),
                biases_grad: Tensor::uninit(ftb_shape),
                single_perspective: self.single_perspective,
                outputs: TensorBatch::new(fto_shape, batch_size),
                copy: TensorBatch::new(fto_shape, batch_size),
            };

            let mut offset = 0;
            ft.weights.set_ptr(opt.weights_offset(offset));
            ft.weights_grad.set_ptr(opt.gradients_offset(offset));
            offset += self.ft_out_size * inp_getter_size;

            ft.biases.set_ptr(opt.weights_offset(offset));
            ft.biases_grad.set_ptr(opt.gradients_offset(offset));
            offset += self.ft_out_size;

            let mut nodes = Vec::new();
            let mut inp_size = mul * self.ft_out_size;

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

                match op {
                    OpType::Affine => {
                        let raw_size = size * buckets;
                        let wsh = Shape::new(inp_size, raw_size);
                        let bsh = Shape::new(1, raw_size);

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

                        offset += inp_size * raw_size;

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

                        offset += raw_size;

                        let outputs = TensorBatch::new(bsh, batch_size);
                        nodes.push(Node {
                            outputs,
                            op: Operation::Affine(affine),
                        });

                        if buckets > 1 {
                            nodes.push(Node {
                                outputs: TensorBatch::new(Shape::new(1, size), batch_size),
                                op: Operation::Select,
                            });
                        }
                    }
                    OpType::Activate(activation) => {
                        let bsh = Shape::new(1, size);
                        let outputs = TensorBatch::new(bsh, batch_size);
                        nodes.push(Node {
                            outputs,
                            op: Operation::Activate(*activation),
                        });
                    }
                    OpType::DualActivate => {
                        let bsh = Shape::new(1, size);
                        let outputs = TensorBatch::new(bsh, batch_size);
                        nodes.push(Node {
                            outputs,
                            op: Operation::DualActivate,
                        });
                    }
                };

                inp_size = size;
            }

            assert_eq!(
                qi,
                self.quantisations.len(),
                "Incorrectly specified number of quantisations!"
            );
            assert_eq!(offset, net_size);

            let inputs = SparseTensor::uninit(batch_size, inp_getter_size, max_active_inputs);

            let results = TensorBatch::new(Shape::new(1, 1), batch_size);
            let error_device = DeviceBuffer::new(1);

            let mut net = vec![0.0; net_size];
            let mut rng = Rand::default();

            for (i, val) in net.iter_mut().enumerate() {
                *val = rng.rand(if i < ft_size { 0.01 } else { 0.1 });
            }

            opt.load_weights_from_host(&net);

            Trainer {
                input_getter: self.input_getter,
                bucket_getter: self.bucket_getter,
                handle: DeviceHandles::default(),
                optimiser: opt,
                ft,
                ft_reg: 1.0 / 4194304.0,
                nodes,
                inputs,
                results,
                error_device,
                error: 0.0,
                used: 0,
                quantiser,
                buckets: tensor::util::calloc(batch_size),
            }
        }
    }
}
