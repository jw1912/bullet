use acyclib::{
    dag::NodeId,
    device::{cpu::CpuMarker, function::UnaryOp, operation::DiffableFromOutput, tensor::Shape},
    graph::{
        builder::GraphBuilder,
        ir::{
            BackendMarker, GraphIR, GraphIRError, GraphIRMethods,
            operation::unary::Unary,
            passes::{GraphIRSimplePass, downcast},
        },
    },
};

fn main() {
    let builder = GraphBuilder::<CpuMarker>::default();

    let input = builder.new_dense_input("input", Shape::new(1, 1));
    let crelu = input.crelu();
    let _screlu = crelu.abs_pow(2.0);

    let mut ir = builder.ir();

    println!("{}", ir.formatted().unwrap());

    ir.apply_pass(FuseCreluWithSquare).unwrap();

    println!("{}", ir.formatted().unwrap());
}

#[derive(Debug)]
pub struct FuseCreluWithSquare;

impl<B: BackendMarker> GraphIRSimplePass<B> for FuseCreluWithSquare {
    fn try_pass_on_node(&self, ir: &mut GraphIR<B>, target: NodeId) -> Result<bool, GraphIRError> {
        let op = ir.get(target)?.op();

        // Is this node a result of a `Square` operation?
        if let Some(Unary { input: parent, op: UnaryOp::AbsPow(2.0) }) = downcast(op) {
            let parent = ir.get(parent.idx)?;

            // If the parent is used by other nodes, it needs to be
            // computed anyway, so no benefit in performing the fusion.
            if parent.children() == 1 {
                // Is the parent node a result of a `CReLU` operation?
                if let Some(Unary { input, op: UnaryOp::DiffableFromOutput(DiffableFromOutput::CReLU) }) =
                    downcast(parent.op())
                {
                    // We started with
                    //   parent = CReLU(input)
                    //   target = Square(parent)
                    // Now we replace to get
                    //   parent = CReLU(input)
                    //   target = SCReLU(input)
                    // And dead node elimination will remove `parent`
                    ir.replace(target, Unary { input, op: UnaryOp::DiffableFromOutput(DiffableFromOutput::SCReLU) })?;
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }
}
