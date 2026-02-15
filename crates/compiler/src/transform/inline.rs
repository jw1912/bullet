use std::collections::HashMap;

use crate::{
    ir::{IR, IRTrace},
    operation::{CopyOp, SubGraph},
    transform::{IRTransform, eliminate::EliminateCopies},
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InlineSubgraphs;

impl IRTransform for InlineSubgraphs {
    fn apply(&self, ir: &mut IR) -> Result<(), IRTrace> {
        loop {
            let mut count = 0;

            for op in ir.operations() {
                if let Some(subgraph) = op.downcast::<SubGraph>() {
                    let mut map = HashMap::new();

                    for (&inp, &int_inp) in op.inputs().iter().zip(subgraph.internal_inputs()) {
                        let new_inp = ir.add_op([inp], Ok::<_, IRTrace>(CopyOp(ir.get_node(inp)?.ty())))?[0];
                        map.insert(int_inp, new_inp);
                    }

                    let int_graph = subgraph.internal_graph();

                    for op_id in int_graph.topo_order_ops()? {
                        let int_op = int_graph.get_op(op_id)?;

                        if int_op.is_input() {
                            continue;
                        }

                        let op_inputs: Vec<_> = int_op.inputs().iter().map(|i| *map.get(i).unwrap()).collect();
                        let op_outputs = ir.add_dyn_op(op_inputs, Ok(int_op.op().clone()))?;

                        for (&out, &int_out) in op_outputs.iter().zip(int_op.outputs()) {
                            map.insert(int_out, out);
                        }
                    }

                    for (&old_out, int_out) in op.outputs().iter().zip(subgraph.internal_outputs()) {
                        let new_out = *map.get(int_out).unwrap();
                        ir.swap_outputs(new_out, old_out)?;
                        ir.replace_input(old_out, new_out)?;
                    }

                    ir.remove_op(op.id())?;

                    count += 1;
                }
            }

            if count == 0 {
                return ir.transform(EliminateCopies);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::frontend::{DType, IRBuilder, TType};

    use super::*;

    #[test]
    fn test_axby() -> Result<(), IRTrace> {
        let mut ir = IR::default();

        let ttype = TType::new(1, DType::F32);

        let a = ir.add_input(ttype);
        let b = ir.add_input(ttype);
        let x = ir.add_input(ttype);

        let builder = IRBuilder::default();
        let sub_a = builder.add_input(ttype.size(), ttype.dtype());
        let sub_b = builder.add_input(ttype.size(), ttype.dtype());
        let sub_x = builder.add_input(ttype.size(), ttype.dtype());
        let sub_y = ((sub_a * sub_x)? + sub_b)?;
        let graph = builder.build([sub_x, sub_y]).graph().clone();
        let subgraph =
            SubGraph::new(graph, vec![sub_a.node(), sub_b.node(), sub_x.node()], vec![sub_x.node(), sub_y.node()]);
        let [x, y] = ir.add_op([a, b, x], subgraph)?[..] else { panic!() };

        ir.register_output(x);
        ir.register_output(y);

        ir.transform(InlineSubgraphs)?;

        Ok(())
    }
}
