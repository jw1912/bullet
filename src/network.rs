use bullet_tensor::{
    cublasHandle_t, device_synchronise, Activation, GpuBuffer, Optimiser, SparseTensor, Tensor,
    TensorBatch,
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
    ft: FeatureTransormer<T>,
    nodes: Vec<Node>,
}

impl<T> Trainer<T> {
    pub fn new(
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
        unsafe {
            self.forward(sparse_inputs);
            self.calc_errors(results, error);
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
                Operation::Affine(Affine {
                    weights, biases, ..
                }) => {
                    TensorBatch::affine(self.handle, weights, inputs, biases, &node.outputs);
                }
            }

            inputs = &node.outputs;
        }
    }

    /// # Safety
    /// It is undefined behaviour to call this without previously calling
    /// `self.forward`.
    unsafe fn calc_errors(&self, results: &TensorBatch, error: &GpuBuffer) {
        self.nodes
            .last()
            .unwrap()
            .outputs
            .sigmoid_mse(results, error);
    }

    /// # Safety
    /// It is undefined behaviour to call this without previously calling
    /// `self.forward` and `self.calc_errors()`, as well as if `sparse_inputs`
    /// is not properly initialised.
    unsafe fn backprop(&self, sparse_inputs: &SparseTensor) {
        let num_nodes = self.nodes.len();

        for node in (1..num_nodes).rev() {
            backprop_single(
                self.handle,
                &self.nodes[node],
                &self.nodes[node - 1].outputs,
            );
        }

        backprop_single(self.handle, &self.nodes[0], &self.ft.outputs);

        SparseTensor::affine_backprop(
            &self.ft.weights_grad,
            sparse_inputs,
            &self.ft.biases_grad,
            &self.ft.outputs,
        );
    }
}

fn backprop_single(handle: cublasHandle_t, this_node: &Node, inputs: &TensorBatch) {
    let errors = &this_node.outputs;

    match &this_node.op {
        Operation::Activate(activation) => {
            TensorBatch::backprop_activation(*activation, errors, inputs);
        }
        Operation::Affine(Affine {
            weights: w,
            weights_grad: wg,
            biases_grad: bg,
            weights_intermediate: wi,
            ..
        }) => unsafe {
            TensorBatch::backprop_affine(handle, w, errors, inputs, wg, bg, wi);
        }
    }
}