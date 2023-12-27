use std::marker::PhantomData;

use bulletformat::BulletFormat;
use bullet_core::{inputs::InputType, data::BoardCUDA, rng::Rand};
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
    our_inputs: SparseTensor,
    opp_inputs: SparseTensor,
    results: TensorBatch,
    error: GpuBuffer,
    used: usize,
}

impl<T: InputType> std::fmt::Display for Trainer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", T::SIZE)?;
        for node in &self.nodes {
            write!(f, " -> {}", node.outputs.shape().rows())?;
        }
        Ok(())
    }
}

impl<T> Trainer<T> {
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

    pub fn write_weights_to_cpu(&self, buf: &mut [f32]) {
        self.optimiser.write_weights_to_buffer(buf);
    }

    pub fn clear_data(&mut self) {
        self.used = 0;
        self.our_inputs.clear();
        self.opp_inputs.clear();
    }

    pub fn append_data(
        &mut self,
        our_inputs: &[BoardCUDA],
        opp_inputs: &[BoardCUDA],
        results: &[f32]
    ) {
        assert_eq!(our_inputs.len(), opp_inputs.len());
        assert_eq!(opp_inputs.len(), results.len());
        unsafe {
            let our = std::slice::from_raw_parts(our_inputs.as_ptr().cast(), our_inputs.len() * BoardCUDA::len());
            let opp = std::slice::from_raw_parts(opp_inputs.as_ptr().cast(), opp_inputs.len() * BoardCUDA::len());

            self.our_inputs.append(our);
            self.opp_inputs.append(opp);
            self.results.offset_load_from_cpu(results, self.used);
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
            device_synchronise();
        };

        let adj = 2. / self.our_inputs.used() as f32;
        self.optimiser.update(decay, adj, rate);
    }

    /// # Safety
    /// It is undefined behaviour to call this if `our_inputs` is not
    /// properly initialised.
    unsafe fn forward(&self) {
        let batch_size = self.our_inputs.used();

        SparseTensor::affine(
            &self.ft.weights,
            &self.our_inputs,
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
        let batch_size = self.our_inputs.used();
        let output_layer = self.nodes.last().unwrap();
        assert_eq!(output_layer.outputs.shape(), self.results.shape());

        output_layer
            .outputs
            .sigmoid_mse(batch_size, &self.results, &self.error);
    }

    /// # Safety
    /// It is undefined behaviour to call this without previously calling
    /// `self.forward` and `self.calc_errors()`, as well as if `our_inputs`
    /// is not properly initialised.
    unsafe fn backprop(&self) {
        let batch_size = self.our_inputs.used();
        let num_nodes = self.nodes.len();

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
            &self.our_inputs,
            &self.ft.biases_grad,
            &self.ft.outputs,
        );
    }

    pub fn eval(&mut self, fen: &str) {
        let bfmt = fen.parse().unwrap();
        let mut our_inputs = Vec::new();
        let mut opp_inputs = Vec::new();
        let mut results = Vec::new();
        BoardCUDA::push(&bfmt, &mut our_inputs, &mut opp_inputs, &mut results, 0.5, 1.0 / 400.0);
        self.clear_data();
        self.append_data(&our_inputs, &opp_inputs, &results);

        unsafe {
            self.forward();
        }

        let mut out = vec![0.0; self.batch_size()];
        self.nodes.last().unwrap().outputs.write_to_cpu(&mut out);
        self.clear_data();

        println!("{}", out[0]);
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
            ..
        }) => unsafe {
            TensorBatch::backprop_affine(handle, batch_size, w, errors, inputs, wg, bg, wi);
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
}

impl<T> Default for TrainerBuilder<T> {
    fn default() -> Self {
        Self {
            marker: PhantomData,
            batch_size: 0,
            ft_out_size: 0,
            nodes: Vec::new(),
            size: 0,
        }
    }
}

impl<T> TrainerBuilder<T> {
    fn get_last_layer_size(&self) -> usize {
        if let Some(node) = self.nodes.last() {
            node.size
        } else {
            self.ft_out_size
        }
    }

    pub fn set_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
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
                outputs: TensorBatch::new(ftb_shape, batch_size),
            };

            let mut offset = 0;
            ft.weights.set_ptr(opt.weights_offset(offset));
            ft.weights_grad.set_ptr(opt.gradients_offset(offset));
            offset += self.ft_out_size * T::SIZE;

            ft.biases.set_ptr(opt.weights_offset(offset));
            ft.biases_grad.set_ptr(opt.gradients_offset(offset));
            offset += self.ft_out_size;

            let mut nodes = Vec::new();
            let mut inp_size = self.ft_out_size;

            for NodeType { size, op } in &self.nodes {
                let size = *size;
                let bsh = Shape::new(1, size);

                let op = match op {
                    OpType::Affine => {
                        let wsh = Shape::new(inp_size, size);
                        let mut affine = Affine {
                            weights: Tensor::uninit(wsh),
                            biases: Tensor::uninit(bsh),
                            weights_grad: Tensor::uninit(wsh),
                            biases_grad: Tensor::uninit(bsh),
                            weights_intermediate: TensorBatch::new(wsh, batch_size),
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
                self.ft_out_size,
            );

            let opp_inputs = SparseTensor::uninit(
                batch_size,
                T::SIZE,
                T::RequiredDataType::MAX_FEATURES,
                self.ft_out_size,
            );

            let last_layer = nodes.last().unwrap();
            let results = TensorBatch::new(last_layer.outputs.shape(), batch_size);
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
                our_inputs,
                opp_inputs,
                results,
                error,
                used: 0,
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use bullet_core::inputs::Chess768;

    #[test]
    fn train() {
        let mut trainer = TrainerBuilder::<Chess768>::default()
            .set_batch_size(1)
            .ft(32)
            .activate(Activation::ReLU)
            .add_layer(1)
            .build();

        let buf = vec![0.01; trainer.net_size()];

        trainer.optimiser.load_weights_from_cpu(&buf);

        let bfmt = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.5".parse().unwrap();
        let mut our_inputs = Vec::new();
        let mut opp_inputs = Vec::new();
        let mut results = Vec::new();
        BoardCUDA::push(&bfmt, &mut our_inputs, &mut opp_inputs, &mut results, 0.5, 1.0 / 400.0);
        trainer.append_data(&our_inputs, &opp_inputs, &results);

        unsafe {
            trainer.forward();
        }

        let mut out = [0.0];
        trainer.nodes.last().unwrap().outputs.write_to_cpu(&mut out);
        assert!(out[0] - 0.33 < 0.00001);

        unsafe {
            trainer.calc_errors();
        }

        let sig = 1.0 / (1.0 + (-out[0]).exp());
        let err = sig * (1.0 - sig) * (sig - 0.5);

        trainer.nodes.last().unwrap().outputs.write_to_cpu(&mut out);
        device_synchronise();

        assert!(out[0] - err < 0.00001);

        unsafe {
            trainer.backprop();
            let mut outw = Tensor::uninit(Shape::new(2, 1));
            outw.set_ptr(trainer.optimiser.gradients_offset(323 * 32 + 31));

            let mut wbuf = [0.0; 2];
            outw.write_to_cpu(&mut wbuf);
            assert_eq!(wbuf[0], 0.0);
            assert_eq!(wbuf[1], 7.192903e-5);
        }

        trainer.train_on_batch(0.01, 0.001);
        unsafe {
            let mut outw = Tensor::uninit(Shape::new(2, 1));
            outw.set_ptr(trainer.optimiser.gradients_offset(323 * 32 + 31));

            let mut wbuf = [0.0; 2];
            outw.write_to_cpu(&mut wbuf);
            assert_eq!(wbuf[0], 0.0);
            assert_eq!(wbuf[1], 7.192903e-5);
        }
    }
}
