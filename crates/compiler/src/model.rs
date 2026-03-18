pub mod lower;
pub mod operations;

use std::{collections::BTreeSet, fmt, rc::Rc};

use crate::{
    ir::{IR, IRError, NodeId, Operation, TypeSystem},
    tensor::{DType, TensorIR},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    Sparse(usize),
    Dense(DType),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MType {
    batch: bool,
    rows: usize,
    cols: usize,
    layout: Layout,
}

pub trait ModelOperation: 'static + fmt::Debug {
    fn opname(&self) -> String;

    fn inputs(&self) -> Vec<MType>;

    fn outputs(&self) -> Vec<MType>;

    fn lower(&self, batch_size: usize, tensor: &mut TensorIR, inputs: Vec<NodeId>) -> Result<Vec<NodeId>, IRError>;

    fn gradient(&self, output_grads: Vec<NodeId>) -> Result<Vec<Option<NodeId>>, IRError>;
}

#[derive(Clone, Debug)]
pub struct ModelOp(pub Rc<dyn ModelOperation>);

impl Operation<MType> for ModelOp {
    fn opname(&self) -> String {
        self.0.opname()
    }

    fn inputs(&self) -> Vec<MType> {
        self.0.inputs()
    }

    fn outputs(&self) -> Vec<MType> {
        self.0.outputs()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Model;
impl TypeSystem for Model {
    type Type = MType;
    type OpData = ModelOp;
}

#[derive(Clone, Debug, Default)]
pub struct ModelIR {
    ir: IR<Model>,
    outputs: BTreeSet<NodeId>,
}

//impl ModelIR {
//    pub fn add_weight(&mut self, rows: usize, cols: usize) -> NodeId {
//        self.ir.add_op([], data).unwrap()[0]
//    }
//
//    pub fn add_input(&mut self, rows: usize, cols: usize) -> NodeId {
//        self.ir.add_op([], data).unwrap()[0]
//    }
//}
