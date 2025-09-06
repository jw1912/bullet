use crate::{
    device::{Device, OperationResult},
    graph::{builder::Shape, ir::operation::unary::DiffableFromOutput},
};

pub trait SparseAffineOps: Device {
    #[allow(clippy::too_many_arguments)]
    fn sparse_affine_activate(
        batch_size: usize,
        activation: DiffableFromOutput,
        input_a: &Self::BufferF32,
        shape_a: Shape,
        input_b: &Self::BufferI32,
        input_b_vals: Option<&Self::BufferF32>,
        shape_b: Shape,
        nnz: usize,
        input_c: Option<&Self::BufferF32>,
        input_c_batched: bool,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

    #[allow(clippy::too_many_arguments)]
    fn backprop_sparse_affine_activate(
        batch_size: usize,
        activation: DiffableFromOutput,
        input_a_grad: &mut Self::BufferF32,
        shape_a: Shape,
        input_b: &Self::BufferI32,
        input_b_vals: Option<&Self::BufferF32>,
        shape_b: Shape,
        nnz: usize,
        input_c_grad: Option<&mut Self::BufferF32>,
        input_c_batched: bool,
        outputs: &Self::BufferF32,
        output_grad: &Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;
}
