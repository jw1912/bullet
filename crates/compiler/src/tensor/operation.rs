pub mod autograd;
mod constant;
mod index;
mod linear;
mod pointwise;
mod subgraph;

use std::{any::Any, collections::HashSet, fmt::Debug, rc::Rc};

pub use constant::{Constant, ScalarConstant};
pub use index::{
    broadcast::BroadcastAcrossDimension,
    pad::PadAcrossDimension,
    select::{Select, SelectPad},
    slice::SliceAcrossDimension,
};
pub use linear::{
    matmul::{Matmul, MatrixLayout},
    reduce::{ReduceAcrossDimension, Reduction},
    sparse::{SparseMatmul, SparseMatmulBwd},
};
pub use pointwise::{
    binary::{CABinary, CABinaryOp},
    copy::CopyOp,
    unary::{Unary, UnaryOp},
};
pub use subgraph::SubGraph;

use crate::{
    ir::Operation,
    tensor::{TType, TValue},
};

pub trait OpType: Any + Debug + 'static {
    fn opname(&self) -> String;

    fn inputs(&self) -> Vec<TType>;

    fn outputs(&self) -> Vec<TType>;

    /// Returns true if self is provably equal to other
    fn equals(&self, _other: &Rc<dyn OpType>) -> bool;

    /// Evaluates the operation given concrete inputs and sized
    /// output buffers. Returns false if not available.
    fn evaluate(&self, _inputs: Vec<&TValue>, _outputs: Vec<&mut TValue>) -> bool {
        false
    }

    fn commutating_groups(&self) -> Vec<HashSet<usize>> {
        Vec::new()
    }
}

#[derive(Clone, Debug)]
pub struct TensorOp(pub Rc<dyn OpType>);

impl Operation<TType> for TensorOp {
    fn opname(&self) -> String {
        self.0.opname()
    }

    fn inputs(&self) -> Vec<TType> {
        self.0.inputs()
    }

    fn outputs(&self) -> Vec<TType> {
        self.0.outputs()
    }
}

impl TensorOp {
    pub fn downcast_rc<T: OpType>(input: &Rc<dyn OpType>) -> Option<&T> {
        let op: &dyn Any = input.as_ref();
        op.downcast_ref::<T>()
    }

    pub fn downcast<T: OpType>(&self) -> Option<&T> {
        Self::downcast_rc::<T>(&self.0)
    }

    pub fn is_input(&self) -> bool {
        self.downcast::<Input>().is_some()
    }
}

#[derive(Debug)]
pub struct Input(pub TType);

impl OpType for Input {
    fn opname(&self) -> String {
        format!("leaf<{:?}>", self.0)
    }

    fn inputs(&self) -> Vec<TType> {
        Vec::new()
    }

    fn outputs(&self) -> Vec<TType> {
        vec![self.0]
    }

    fn equals(&self, _: &Rc<dyn OpType>) -> bool {
        false
    }
}
