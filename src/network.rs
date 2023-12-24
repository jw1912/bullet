use bullet_tensor::{
    cublasHandle_t, device_synchronise, Activation, Optimiser, SparseTensor, Tensor, TensorBatch,
};

pub struct FeatureTransormer<T> {
    marker: std::marker::PhantomData<T>,
    weights: Tensor,
    biases: Tensor,
    weights_grad: Tensor,
    biases_grad: Tensor,
    outputs: TensorBatch,
}

pub struct Affine {
    weights: Tensor,
    biases: Tensor,
    weights_grad: Tensor,
    biases_grad: Tensor,
    weights_intermediate: TensorBatch,
}

pub enum Operation {
    Activate(Activation),
    Affine(Affine),
}

pub struct Node {
    outputs: TensorBatch,
    op: Operation,
}

pub struct Trainer<T> {
    handle: cublasHandle_t,
    optimiser: Optimiser,
    batch_size: usize,
    ft: FeatureTransormer<T>,
    nodes: Vec<Node>,
}

impl<T> Trainer<T> {
    /// # Safety
    /// It is undefined behaviour to call this if `sparse_inputs` is not
    /// properly initialised.
    pub unsafe fn forward(&self, sparse_inputs: &SparseTensor) {
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
                    TensorBatch::activate(*activation, inputs, &node.outputs);
                }
                Operation::Affine(info) => {
                    TensorBatch::affine(&info.weights, inputs, &info.biases, &node.outputs);
                }
            }

            inputs = &node.outputs;
        }

        device_synchronise();
    }

    /// # Safety
    /// It is undefined behaviour to call this without previously calling,
    /// `self.forward` and `self.calc_errors`, as well as if `sparse_inputs`
    /// is not properly initialised.
    pub unsafe fn backprop(&self, sparse_inputs: &SparseTensor) {
        let num_nodes = self.nodes.len();

        for node in (1..num_nodes).rev() {
            backprop_single(&self.nodes[node], &self.nodes[node - 1].outputs);
        }

        backprop_single(&self.nodes[0], &self.ft.outputs);

        SparseTensor::affine_backprop(
            &self.ft.weights_grad,
            sparse_inputs,
            &self.ft.biases_grad,
            &self.ft.outputs,
        );

        device_synchronise();
    }
}

fn backprop_single(this_node: &Node, inputs: &TensorBatch) {
    match &this_node.op {
        Operation::Activate(activation) => {
            TensorBatch::backprop_activation(*activation, &this_node.outputs, inputs);
        }
        Operation::Affine(info) => {
            TensorBatch::backprop_affine(
                &info.weights,
                &this_node.outputs,
                inputs,
                &info.weights_grad,
                &info.biases_grad,
            );
        }
    }
}
