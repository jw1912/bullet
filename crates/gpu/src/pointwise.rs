mod generate;
mod ir;
mod operations;
mod write;

use std::collections::HashSet;

use bullet_compiler::tensor::{
    IRTrace, OpType, TType, TValue, TensorIR, TensorOp,
    operation::SubGraph,
    transform::{IRTransform, modify::AddOperation},
};

pub use ir::PointwiseIR;

#[derive(Debug)]
pub struct FusedPointwise {
    sub: SubGraph,
    ir: PointwiseIR,
}

impl FusedPointwise {
    pub fn new(sub: SubGraph) -> Result<Option<Self>, IRTrace> {
        let maybe_ir = generate::generate(&sub)?;
        Ok(maybe_ir.map(|ir| Self { sub, ir }))
    }

    pub fn from_op(op: TensorOp) -> Result<Option<Self>, IRTrace> {
        Self::new(SubGraph::from_op(op)?)
    }
}

impl OpType for FusedPointwise {
    fn opname(&self) -> String {
        "fused-pointwise".into()
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LowerPointwise;
impl IRTransform for LowerPointwise {
    fn apply(&self, ir: &mut TensorIR) -> Result<(), IRTrace> {
        // lower individual ops to FusedPointwise
        for op in ir.operations() {
            if let Some(pntwise) = FusedPointwise::from_op(op.data().clone())? {
                let add = AddOperation::new(op.inputs(), Ok(TensorOp::new(pntwise)));
                ir.replace_op(op.id(), add)?;
            }
        }

        // perform fusions of the FusedPointwise where possible
        let mut failed = HashSet::new();
        let mut success = true;
        'outer: while success {
            success = false;

            let ops = ir.ordered_operations()?;
            let ops = ops.into_iter().filter(|op| op.data().downcast::<FusedPointwise>().is_some()).collect::<Vec<_>>();

            for (i, op_i) in ops.iter().enumerate() {
                for op_j in ops.iter().skip(i + 1) {
                    if failed.contains(&(op_i.id(), op_j.id())) {
                        continue;
                    }

                    // `op_i` comes before `op_j` in topo ordering so know that if there is a
                    // dependency then `op_j` is dependent on `op_i` we can only fuse `op_i`
                    // and `op_j` if there does not exist an in between op that is dependent
                    // on `op_i` and is depended upon by `op_j`
                    if ir.is_immediate_dependent_op(op_i.id(), op_j.id())? {
                        continue 'outer;
                    }

                    failed.insert((op_i.id(), op_j.id()));
                }
            }
        }

        // lower fused pointwise to KernelSrc ops
        for op in ir.operations() {
            if let Some(pntwise) = op.data().downcast::<FusedPointwise>() {
                let src = unsafe { pntwise.ir.lower(format!("kernel{}", op.id().inner()))? };

                for (&i1, &i2) in op.data().inputs().iter().zip(src.inputs.iter()) {
                    if i1 != i2 {
                        return Err("Mismatched input types!".into());
                    }
                }

                for (&o1, &o2) in op.data().outputs().iter().zip(src.outputs.iter()) {
                    if o1 != o2 {
                        return Err("Mismatched output types!".into());
                    }
                }

                let add = AddOperation::new(op.inputs(), Ok(TensorOp::new(src)));
                ir.replace_op(op.id(), add)?;
            }
        }

        Ok(())
    }
}
