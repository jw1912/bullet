mod generate;
mod ir;
mod operations;
mod write;

use std::collections::{HashMap, HashSet};

use bullet_compiler::{
    ir::{NodeId, Op},
    tensor::{
        IRTrace, OpType, TType, TValue, Tensor, TensorIR, TensorOp,
        operation::{ScalarConstant, SubGraph},
        transform::{IRTransform, eliminate::EliminateUnusedOperations, inline::InlineSubgraphs, modify::AddOperation},
    },
};

pub use ir::PointwiseIR;

#[derive(Clone, Debug)]
pub struct FusedPointwise {
    sub: SubGraph,
    ir: PointwiseIR,
    vectorised: bool,
}

impl FusedPointwise {
    pub fn new(sub: SubGraph) -> Result<Option<Self>, IRTrace> {
        let maybe_ir = generate::generate(&sub)?;
        Ok(maybe_ir.map(|(ir, vectorised)| Self { sub, ir, vectorised }))
    }

    pub fn from_op(op: TensorOp, inputs: &[NodeId]) -> Result<Option<(Self, Vec<NodeId>)>, IRTrace> {
        let (graph, inputs) = SubGraph::from_op(op, inputs)?;
        Ok(Self::new(graph)?.map(|x| (x, inputs)))
    }
}

impl OpType for FusedPointwise {
    fn opname(&self) -> String {
        let src = format!("{:?}", format!("{}", self.sub.internal_graph()));
        let src = src.strip_prefix("\"irgraph").unwrap().strip_suffix('"').unwrap().replace("\\n", "\\l");
        format!("{}Fused Kernel\\n{src}\\l", if self.vectorised { "Vectorised " } else { "" })
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
        // separate out all `ScalarConst`s, as otherwise we end up
        // materialising them in kernel A and passing to kernel B,
        // rather than handling internally for each
        for op in ir.operations() {
            for &input in op.inputs() {
                if let Some(&ScalarConstant(value, size)) = ir.parent_op(input)? {
                    let new_scalar = ir.add_scalar(value, size);
                    ir.ir_mut().replace_single_input(op.id(), new_scalar, input)?;
                }
            }
        }

        ir.transform(EliminateUnusedOperations)?;

        // lower individual ops to FusedPointwise
        for op in ir.operations() {
            if let Some((pntwise, inputs)) = FusedPointwise::from_op(op.data().clone(), op.inputs()).unwrap() {
                let add = AddOperation::new(inputs, Ok(TensorOp::new(pntwise)));
                ir.replace_op(op.id(), add)?;
            }
        }

        // perform fusions of the FusedPointwise where possible
        let mut failed = HashSet::new();
        let mut costs = HashMap::new();

        loop {
            let mut candidates = HashMap::new();

            let ops = ir.ordered_operations()?;
            let ops = ops
                .into_iter()
                .filter(|op| {
                    if let Some(pntwise) = op.data().downcast::<FusedPointwise>() {
                        costs.insert(op.id(), pntwise.ir.estimate_memory_cost().unwrap());
                        true
                    } else {
                        false
                    }
                })
                .collect::<Vec<_>>();

            for (i, op_i) in ops.iter().enumerate() {
                'inner: for op_j in ops.iter().skip(i + 1) {
                    if failed.contains(&(op_i.id(), op_j.id())) {
                        continue;
                    }

                    // `op_i` comes before `op_j` in topo ordering so know that if there is a
                    // dependency then `op_j` is dependent on `op_i` we can only fuse `op_i`
                    // and `op_j` if there does not exist an in between op that is dependent
                    // on `op_i` and is depended upon by `op_j`
                    if ir.is_immediate_dependent_op(op_i.id(), op_j.id())? {
                        let (subgraph, inputs, outputs) = fuse_subgraphs(ir, op_i, op_j)?;
                        if let Some(pntwise) = FusedPointwise::new(subgraph.clone())? {
                            let new_cost = pntwise.ir.estimate_memory_cost()?;
                            let old_cost =
                                costs.get(&op_i.id()).unwrap().dominator_sum(*costs.get(&op_j.id()).unwrap());

                            if new_cost.is_le(old_cost) {
                                let saving = if new_cost.var_power() != old_cost.var_power() {
                                    Some(old_cost)
                                } else if new_cost.factor() != old_cost.factor() {
                                    Some(old_cost - new_cost)
                                } else {
                                    None
                                };
                                candidates.insert((op_i.id(), op_j.id()), (pntwise, inputs, outputs, new_cost, saving));
                                continue 'inner;
                            }
                        }
                    }

                    failed.insert((op_i.id(), op_j.id()));
                }
            }

            if candidates.is_empty() {
                break;
            } else {
                let first = candidates.iter().next().unwrap();
                let mut max_saving = first.1.4;
                let mut argmin = *first.0;

                for (arg, (_, _, _, _, saving)) in &candidates {
                    if match (max_saving, saving) {
                        (Some(x), Some(y)) => x.is_le(*y),
                        (None, Some(_)) => true,
                        (_, None) => false,
                    } {
                        max_saving = *saving;
                        argmin = *arg;
                    }
                }

                let (pntwise, inputs, outputs, cost, _) = candidates.get(&argmin).cloned().unwrap();
                let new_outputs = ir.add_op(inputs, Ok::<_, IRTrace>(pntwise))?;
                costs.insert(ir.get_parent_op(new_outputs[0])?, cost);

                for (new, old) in new_outputs.into_iter().zip(outputs) {
                    ir.swap_outputs(new, old)?;
                }

                ir.remove_op(argmin.1)?;
                ir.remove_op(argmin.0)?;
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

fn fuse_subgraphs(
    ir: &TensorIR,
    op_i: &Op<Tensor>,
    op_j: &Op<Tensor>,
) -> Result<(SubGraph, Vec<NodeId>, Vec<NodeId>), IRTrace> {
    let sub_i = &op_i.data().downcast::<FusedPointwise>().unwrap().sub;
    let sub_j = &op_j.data().downcast::<FusedPointwise>().unwrap().sub;

    let mut new_sub = TensorIR::default();

    let mut map = HashMap::new();
    let mut inputs_i = Vec::new();
    for &input in op_i.inputs() {
        let new_input = new_sub.add_input(ir.get_node(input)?.ty());
        map.insert(input, new_input);
        inputs_i.push(new_input);
    }

    let mut total_outputs = Vec::new();

    let op_j_set = op_j.inputs().iter().cloned().collect::<HashSet<_>>();

    let outputs_i = new_sub.add_op(inputs_i, Ok::<_, IRTrace>(sub_i.clone()))?;
    for (&out, &new_out) in op_i.outputs().iter().zip(outputs_i.iter()) {
        map.insert(out, new_out);

        if !op_j_set.contains(&out) || ir.get_node(out)?.children() > 1 {
            total_outputs.push(out);
            new_sub.register_output(new_out);
        }
    }

    let mut total_inputs = op_i.inputs().to_vec();

    let mut inputs_j = Vec::new();
    for &input in op_j.inputs() {
        if let Some(&new_input) = map.get(&input) {
            inputs_j.push(new_input);
        } else {
            let new_input = new_sub.add_input(ir.get_node(input)?.ty());
            map.insert(input, new_input);
            inputs_j.push(new_input);
            total_inputs.push(input);
        }
    }

    let outputs_j = new_sub.add_op(inputs_j, Ok::<_, IRTrace>(sub_j.clone()))?;
    for (&out, &new_out) in op_j.outputs().iter().zip(outputs_j.iter()) {
        map.insert(out, new_out);
        total_outputs.push(out);
        new_sub.register_output(new_out);
    }

    new_sub.transform(InlineSubgraphs)?;

    let subgraph = SubGraph::new(
        new_sub,
        total_inputs.iter().map(|x| *map.get(x).unwrap()).collect(),
        total_outputs.iter().map(|x| *map.get(x).unwrap()).collect(),
    )?;

    Ok((subgraph, total_inputs, total_outputs))
}
