mod generate;
mod ir;
mod operations;
pub(crate) mod transforms;
mod write;

use bullet_compiler::{
    ir::NodeId,
    tensor::{IRTrace, OpType, TType, TValue, TensorOp, operation::SubGraph},
};

pub use ir::PointwiseIR;

use crate::runtime::DeviceProps;

#[derive(Clone, Debug)]
pub struct FusedPointwise {
    sub: SubGraph,
    ir: PointwiseIR,
    vectorised: bool,
}

impl FusedPointwise {
    fn new(sub: SubGraph, props: &DeviceProps) -> Result<Option<Self>, IRTrace> {
        let maybe_ir = generate::generate(&sub, props)?;
        Ok(maybe_ir.map(|(ir, vectorised)| Self { sub, ir, vectorised }))
    }

    fn from_op(op: TensorOp, inputs: &[NodeId], props: &DeviceProps) -> Result<Option<(Self, Vec<NodeId>)>, IRTrace> {
        let (graph, inputs) = SubGraph::from_op(op, inputs)?;
        Ok(Self::new(graph, props)?.map(|x| (x, inputs)))
    }
}

impl OpType for FusedPointwise {
    fn opname(&self) -> String {
        let src = format!("{:?}", format!("{}", self.sub.internal_graph()));
        let src = src.strip_prefix("\"irgraph").unwrap().strip_suffix('"').unwrap().replace("\\n", "\\l");
        format!("{}Kernel\\n{src}\\l", if self.vectorised { "Vectorised " } else { "" })
    }

    fn inputs(&self) -> Vec<TType> {
        self.sub.inputs()
    }

    fn outputs(&self) -> Vec<TType> {
        self.sub.outputs()
    }

    fn evaluate(&self, inputs: Vec<&TValue>, outputs: Vec<&mut TValue>) -> bool {
        self.sub.evaluate(inputs, outputs)
    }
}
