pub mod manager;
pub mod node;

use std::{cell::RefCell, rc::Rc};

use manager::IrManager;
pub use node::ProgramNode;

use crate::{
    common::{DType, Size},
    elementwise::ElementwiseNode,
    ir::{node::IrNodeId, ops::IrOperation},
    program::Program,
};

#[derive(Default)]
pub struct ProgramBuilder {
    ir: Rc<RefCell<IrManager>>,
}

impl ProgramBuilder {
    fn new_node<'a>(&'a self, node: IrNodeId) -> ProgramNode<'a> {
        ProgramNode::new(self, node)
    }

    fn add_op<'a>(&'a self, op: impl IrOperation) -> Vec<ProgramNode<'a>> {
        let outs = self.ir.borrow_mut().add_op(op).unwrap();
        outs.into_iter().map(|out| self.new_node(out)).collect()
    }

    pub fn add_leaf<'a>(&'a self, size: impl Into<Size>, dtype: DType) -> ProgramNode<'a> {
        let node = self.ir.borrow_mut().add_leaf(size, dtype).unwrap();
        self.new_node(node)
    }

    pub fn elementwise<'a, F, const M: usize, const N: usize>(
        &'a self,
        inputs: [ProgramNode<'a>; M],
        f: F,
    ) -> [ProgramNode<'a>; N]
    where
        for<'b> F: Fn([ElementwiseNode<'b>; M]) -> [ElementwiseNode<'b>; N],
    {
        self.ir
            .borrow_mut()
            .modify(|x| x.add_elementwise(inputs.map(|y| y.node()), |x| Some(f(x))))
            .unwrap()
            .map(|id| ProgramNode::new(self, id))
    }

    pub fn display_ir(&self) {
        println!("{}", self.ir.borrow())
    }

    pub fn build<'a>(&'a self, returns: impl AsRef<[ProgramNode<'a>]>) -> Program {
        let mut ir = self.ir.borrow_mut();

        for ret in returns.as_ref() {
            ir.register_output(ret.node()).unwrap();
        }

        ir.eliminate_dead_ops().unwrap();

        ir.lower().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_usage() {
        let builder = ProgramBuilder::default();

        let x = builder.add_leaf(8, DType::F32);
        let a = builder.add_leaf(8, DType::F32);
        let b = builder.add_leaf(8, DType::F32);

        let y = a * x + b;

        let _program = builder.build([y]);
    }
}
