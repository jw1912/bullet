use std::marker::PhantomData;

use bulletformat::BulletFormat;
use bullet_core::{inputs::InputType, Rand, util, GpuDataLoader};
use bullet_tensor::{
    cublasHandle_t, device_synchronise, Activation, GpuBuffer, Optimiser, SparseTensor,
    Tensor, TensorBatch, create_cublas_handle, Shape,
};

struct FeatureTransormer<T> {
    marker: PhantomData<T>,
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
    weights_intermediate: TensorBatch,
    ones: GpuBuffer,
}

enum Operation {
    Activate(Activation),
    Affine(Affine),
}

struct Node {
    outputs: TensorBatch,
    op: Operation,
}

pub struct Trainer<T> {
    handle: cublasHandle_t,
    optimiser: Optimiser,
    ft: FeatureTransormer<T>,
    nodes: Vec<Node>,
    inputs: SparseTensor,
    results: TensorBatch,
    error: GpuBuffer,
    used: usize,
    scale: f32,
}

impl<T: InputType> std::fmt::Display for Trainer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inp_size = T::RequiredDataType::INPUTS;
        let buckets = T::BUCKETS;
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

impl<T> Trainer<T>
where T: InputType
{
    pub fn save(&self, name: String, epoch: usize) {
        let size = self.optimiser.size();

        let mut buf1 = vec![0.0; size];
        let mut buf2 = vec![0.0; size];
        let mut buf3 = vec![0.0; size];

        self.optimiser.write_to_cpu(&mut buf1, &mut buf2, &mut buf3);

        let path = format!("checkpoints/{name}-epoch{epoch}");

        std::fs::create_dir(path.as_str()).unwrap_or(());

        util::write_to_bin(&buf1, size, &format!("{path}/params.bin"), false).unwrap();
        util::write_to_bin(&buf2, size, &format!("{path}/momentum.bin"), false).unwrap();
        util::write_to_bin(&buf3, size, &format!("{path}/velocity.bin"), false).unwrap();
    }

    pub fn load_from_checkpoint(&self, path: &str) {
        let load_from_bin = |name: &str| {
            use std::fs::File;
            use std::io::{Read, BufReader};
            let file = File::open(format!("{path}/{name}.bin")).unwrap();
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
        };

        let network = load_from_bin("params");
        let momentum = load_from_bin("momentum");
        let velocity = load_from_bin("velocity");

        self.optimiser.load_from_cpu(&network, &momentum, &velocity)
    }

    pub fn prep_for_epoch(&mut self) {
        self.error.load_from_cpu(&[0.0]);
        device_synchronise();
    }

    pub fn error(&self) -> f32 {
        device_synchronise();
        let mut buf = [0.0];
        self.error.write_to_cpu(&mut buf);
        buf[0]
    }

    pub fn net_size(&self) -> usize {
        self.optimiser.size()
    }

    pub fn eval_scale(&self) -> f32 {
        self.scale
    }

    pub fn write_weights_to_cpu(&self, buf: &mut [f32]) {
        self.optimiser.write_weights_to_buffer(buf);
    }

    pub fn clear_data(&mut self) {
        self.used = 0;
        self.inputs.clear();
    }

    pub fn load_data(
        &mut self,
        loader: &GpuDataLoader<T>,
    ) {
        let inputs = loader.inputs();
        let results = loader.results();

        unsafe {
            let our = std::slice::from_raw_parts(inputs.as_ptr().cast(), inputs.len());
            self.inputs.append(our);
            self.results.load_from_cpu(results);
            self.used += results.len();
        }
    }

    pub fn batch_size(&self) -> usize {
        self.ft.outputs.cap()
    }

    pub fn train_on_batch(&self, decay: f32, rate: f32) {
        self.optimiser.zero_gradient();

        unsafe {
            self.forward();
            self.calc_errors();
            self.backprop();
        }

        let adj = 2. / self.inputs.used() as f32;
        self.optimiser.update(decay, adj, rate);
        device_synchronise();
    }

    /// # Safety
    /// It is undefined behaviour to call this if `our_inputs` is not
    /// properly initialised.
    unsafe fn forward(&self) {
        let batch_size = self.inputs.used();

        SparseTensor::affine(
            &self.ft.weights,
            &self.inputs,
            &self.ft.biases,
            &self.ft.outputs,
        );

        let mut inputs = &self.ft.outputs;

        for node in &self.nodes {
            match &node.op {
                Operation::Activate(activation) => {
                    TensorBatch::activate(batch_size, *activation, inputs, &node.outputs);
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
            .sigmoid_mse(batch_size, &self.results, &self.error);
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
            &self.ft.weights_grad,
            &self.inputs,
            &self.ft.biases_grad,
            &self.ft.outputs,
        );
    }
}

fn backprop_single(
    handle: cublasHandle_t,
    batch_size: usize,
    this_node: &Node,
    inputs: &TensorBatch,
) {
    let errors = &this_node.outputs;

    match &this_node.op {
        Operation::Activate(activation) => {
            TensorBatch::backprop_activation(batch_size, *activation, errors, inputs);
        }
        Operation::Affine(Affine {
            weights: w,
            weights_grad: wg,
            biases_grad: bg,
            weights_intermediate: wi,
            ones,
            ..
        }) => unsafe {
            TensorBatch::backprop_affine(handle, ones, batch_size, w, errors, inputs, wg, bg, wi);
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
    marker: PhantomData<T>,
    batch_size: usize,
    ft_out_size: usize,
    nodes: Vec<NodeType>,
    size: usize,
    scale: f32,
}

impl<T> Default for TrainerBuilder<T> {
    fn default() -> Self {
        Self {
            marker: PhantomData,
            batch_size: 0,
            ft_out_size: 0,
            nodes: Vec::new(),
            size: 0,
            scale: 400.0,
        }
    }
}

impl<T> TrainerBuilder<T> {
    fn get_last_layer_size(&self) -> usize {
        if let Some(node) = self.nodes.last() {
            node.size
        } else {
            2 * self.ft_out_size
        }
    }

    pub fn set_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn set_eval_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    pub fn ft(mut self, size: usize) -> Self {
        assert!(self.nodes.is_empty());
        self.ft_out_size = size;
        self
    }

    fn add(mut self, size: usize, op: OpType) -> Self {
        self.nodes.push(
            NodeType { size, op }
        );

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
        let ft_size = (T::SIZE + 1) * self.ft_out_size;
        let net_size = self.size + ft_size;

        let opt = Optimiser::new(net_size);
        let batch_size = self.batch_size;

        unsafe {
            let ftw_shape = Shape::new(self.ft_out_size, T::SIZE);
            let ftb_shape = Shape::new(1, self.ft_out_size);

            let mut ft = FeatureTransormer {
                marker: PhantomData,
                weights: Tensor::uninit(ftw_shape),
                biases: Tensor::uninit(ftb_shape),
                weights_grad: Tensor::uninit(ftw_shape),
                biases_grad: Tensor::uninit(ftb_shape),
                outputs: TensorBatch::new(Shape::new(1, 2 * self.ft_out_size), batch_size),
            };

            let mut offset = 0;
            ft.weights.set_ptr(opt.weights_offset(offset));
            ft.weights_grad.set_ptr(opt.gradients_offset(offset));
            offset += self.ft_out_size * T::SIZE;

            ft.biases.set_ptr(opt.weights_offset(offset));
            ft.biases_grad.set_ptr(opt.gradients_offset(offset));
            offset += self.ft_out_size;

            let mut nodes = Vec::new();
            let mut inp_size = 2 * self.ft_out_size;

            for NodeType { size, op } in &self.nodes {
                let size = *size;
                let bsh = Shape::new(1, size);

                let op = match op {
                    OpType::Affine => {
                        let wsh = Shape::new(inp_size, size);
                        let ones = GpuBuffer::new(1);
                        ones.load_from_cpu(&[1.0]);
                        let mut affine = Affine {
                            weights: Tensor::uninit(wsh),
                            biases: Tensor::uninit(bsh),
                            weights_grad: Tensor::uninit(wsh),
                            biases_grad: Tensor::uninit(bsh),
                            weights_intermediate: TensorBatch::new(wsh, batch_size),
                            ones,
                        };

                        affine.weights.set_ptr(opt.weights_offset(offset));
                        affine.weights_grad.set_ptr(opt.gradients_offset(offset));

                        offset += inp_size * size;

                        affine.biases.set_ptr(opt.weights_offset(offset));
                        affine.biases_grad.set_ptr(opt.gradients_offset(offset));

                        offset += size;

                        Operation::Affine(affine)
                    }
                    OpType::Activate(activation) => Operation::Activate(*activation),
                };

                let outputs = TensorBatch::new(bsh, batch_size);

                nodes.push(Node { outputs, op });

                inp_size = size;
            }

            assert_eq!(offset, net_size);

            let our_inputs = SparseTensor::uninit(
                batch_size,
                T::SIZE,
                T::RequiredDataType::MAX_FEATURES,
            );

            let opp_inputs = SparseTensor::uninit(
                batch_size,
                T::SIZE,
                T::RequiredDataType::MAX_FEATURES,
            );

            let results = TensorBatch::new(Shape::new(1, 1), batch_size);
            let error = GpuBuffer::new(1);

            let mut net = vec![0.0; net_size];
            let mut rng = Rand::default();

            for val in net.iter_mut() {
                *val = rng.rand(0.01);
            }

            opt.load_weights_from_cpu(&net);

            Trainer {
                handle: create_cublas_handle(),
                optimiser: opt,
                ft,
                nodes,
                inputs,
                results,
                error,
                used: 0,
                scale: self.scale,
            }
        }
    }
}
