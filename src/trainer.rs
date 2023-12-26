use std::marker::PhantomData;

use bullet_core::inputs::InputType;
use bullet_tensor::{
    cublasHandle_t, device_synchronise, Activation, GpuBuffer, Optimiser, SparseTensor,
    Tensor, TensorBatch, create_cublas_handle, Shape,
};

#[derive(Debug)]
struct FeatureTransormer<T> {
    marker: PhantomData<T>,
    weights: Tensor,
    biases: Tensor,
    weights_grad: Tensor,
    biases_grad: Tensor,
    outputs: TensorBatch,
}

#[derive(Debug)]
struct Affine {
    weights: Tensor,
    biases: Tensor,
    weights_grad: Tensor,
    biases_grad: Tensor,
    weights_intermediate: TensorBatch,
}

#[derive(Debug)]
enum Operation {
    Activate(Activation),
    Affine(Affine),
}

#[derive(Debug)]
struct Node {
    outputs: TensorBatch,
    op: Operation,
}

#[derive(Debug)]
pub struct Trainer<T> {
    handle: cublasHandle_t,
    optimiser: Optimiser,
    ft: FeatureTransormer<T>,
    nodes: Vec<Node>,
}

impl<T> Trainer<T> {
    fn new(
        handle: cublasHandle_t,
        optimiser: Optimiser,
        ft: FeatureTransormer<T>,
        nodes: Vec<Node>,
    ) -> Self {
        Self {
            handle,
            optimiser,
            ft,
            nodes,
        }
    }

    pub fn train_on_batch(
        &self,
        sparse_inputs: &SparseTensor,
        results: &TensorBatch,
        decay: f32,
        rate: f32,
        error: &GpuBuffer,
    ) {
        let batch_size = sparse_inputs.used();

        unsafe {
            self.forward(sparse_inputs);
            self.calc_errors(batch_size, results, error);
            self.backprop(sparse_inputs);
            device_synchronise();
        };

        let adj = 2. / sparse_inputs.used() as f32;
        self.optimiser.update(decay, adj, rate);
    }

    /// # Safety
    /// It is undefined behaviour to call this if `sparse_inputs` is not
    /// properly initialised.
    unsafe fn forward(&self, sparse_inputs: &SparseTensor) {
        let batch_size = sparse_inputs.used();

        SparseTensor::affine(
            &self.ft.weights,
            sparse_inputs,
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
    unsafe fn calc_errors(&self, batch_size: usize, results: &TensorBatch, error: &GpuBuffer) {
        self.nodes
            .last()
            .unwrap()
            .outputs
            .sigmoid_mse(batch_size, results, error);
    }

    /// # Safety
    /// It is undefined behaviour to call this without previously calling
    /// `self.forward` and `self.calc_errors()`, as well as if `sparse_inputs`
    /// is not properly initialised.
    unsafe fn backprop(&self, sparse_inputs: &SparseTensor) {
        let batch_size = sparse_inputs.used();
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
            sparse_inputs,
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
            ..
        }) => unsafe {
            TensorBatch::backprop_affine(handle, batch_size, w, errors, inputs, wg, bg, wi);
        },
    }
}

#[derive(Debug)]
enum OpType {
    Activate(Activation),
    Affine,
}

#[derive(Debug)]
struct NodeType {
    size: usize,
    op: OpType,
}

#[derive(Debug)]
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
            offset += T::SIZE;

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

            Trainer::new(create_cublas_handle(), opt, ft, nodes)
        }
    }
}
