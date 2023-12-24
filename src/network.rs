use bullet_tensor::{Activation, TensorBatch, cublasHandle_t, Optimiser, SparseTensor, Tensor};

pub struct FeatureTransormer {
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
    AffineTransform(Affine),
}

pub struct Node {
    outputs: TensorBatch,
    op: Operation,
}

pub struct Trainer<T> {
    marker: std::marker::PhantomData<T>,
    handle: cublasHandle_t,
    optimiser: Optimiser,
    batch_size: usize,
    ft: FeatureTransormer,
    nodes: Vec<Node>,
}

impl<T> Trainer<T> {
    /// # Safety
    /// It is undefined behaviour to call this if `sparse_inputs` is not
    /// properly initialised.
    pub unsafe fn forward(&self, sparse_inputs: &SparseTensor) {
        SparseTensor::affine(&self.ft.weights, sparse_inputs, &self.ft.biases, &self.ft.outputs);

        let mut inputs = &self.ft.outputs;

        for node in &self.nodes {
            match &node.op {
                Operation::Activate(activation) => {
                    TensorBatch::activate(*activation, inputs, &node.outputs);
                }
                Operation::AffineTransform(info) => {
                    TensorBatch::single_lt(self.handle, &info.weights, inputs, &node.outputs);
                    //TensorBatch::single_add(&info.biases, &node.outputs);
                }
            }

            inputs = &node.outputs;
        }

        bullet_tensor::device_synchronise();
    }

    /// # Safety
    /// It is undefined behaviour to call this without previously calling,
    /// `self.forward` and `self.calc_errors`, as well as if `sparse_inputs`
    /// is not properly initialised.
    pub unsafe fn backprop(&self, sparse_inputs: &SparseTensor) {
        let num_nodes = self.nodes.len();

        if num_nodes > 1 {
            for node in (1..num_nodes).rev() {
                let this_node = &self.nodes[node];
                let last_node = &self.nodes[node - 1];

                match &this_node.op {
                    Operation::Activate(activation) => {
                        TensorBatch::backprop_activation(*activation, &this_node.outputs, &last_node.outputs);
                    }
                    Operation::AffineTransform(_) => {

                    }
                }
            }
        }

        let first_layer = &self.nodes[0];

        match &first_layer.op {
            Operation::Activate(activation) => {
                TensorBatch::backprop_activation(*activation, &first_layer.outputs, &self.ft.outputs);
            }
            Operation::AffineTransform(_) => {

            }
        }

        SparseTensor::affine_backprop(
            &self.ft.weights_grad,
            sparse_inputs,
            &self.ft.biases_grad,
            &self.ft.outputs,
        );

        bullet_tensor::device_synchronise();
    }
}