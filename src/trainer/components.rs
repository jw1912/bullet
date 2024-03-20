use crate::{
    tensor::{DeviceBuffer, Tensor, TensorBatch},
    Activation,
};

pub(super) struct FeatureTransformer {
    pub weights: Tensor,
    pub biases: Tensor,
    pub weights_grad: Tensor,
    pub biases_grad: Tensor,
    pub single_perspective: bool,
    pub outputs: TensorBatch,
    pub copy: TensorBatch,
}

pub(super) struct Affine {
    pub weights: Tensor,
    pub biases: Tensor,
    pub weights_grad: Tensor,
    pub biases_grad: Tensor,
    pub ones: DeviceBuffer,
}

pub(super) enum Operation {
    Activate(Activation),
    Affine(Affine),
    Select,
}

pub(super) struct Node {
    pub outputs: TensorBatch,
    pub op: Operation,
}

pub(super) struct QuantiseInfo {
    pub val: i32,
    pub start: usize,
}
