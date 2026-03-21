use std::collections::{BTreeMap, BTreeSet};

use bullet_compiler::{
    ir::{NodeId, OpId},
    tensor::{
        IRTrace, TensorIR, TensorOp,
        operation::SubGraph,
        transform::{IRTransform, inline::InlineSubgraphs, modify::AddOperation},
    },
};

use crate::{pointwise::FusedPointwise, runtime::DeviceProps};

/// Lower individual ops to FusedPointwise
#[derive(Clone, Debug)]
pub struct LowerPointwise(pub(crate) DeviceProps);
impl IRTransform for LowerPointwise {
    fn apply(&self, ir: &mut TensorIR) -> Result<(), IRTrace> {
        for op in ir.operations() {
            if let Some((pntwise, inputs)) = FusedPointwise::from_op(op.data().clone(), op.inputs(), &self.0).unwrap() {
                let add = AddOperation::new(inputs, Ok(TensorOp::new(pntwise)));
                ir.replace_op(op.id(), add)?;
            }
        }

        Ok(())
    }
}

/// Fuse pointwise operations greedily
#[derive(Clone, Debug)]
pub struct FusePointwise(pub(crate) DeviceProps);
impl IRTransform for FusePointwise {
    fn apply(&self, ir: &mut TensorIR) -> Result<(), IRTrace> {
        let mut cache = BTreeMap::new();
        let mut failed = BTreeSet::new();
        let mut costs = ir
            .operations()
            .into_iter()
            .filter_map(|op| {
                op.data()
                    .downcast::<FusedPointwise>()
                    .map(|pntwise| (op.id(), pntwise.ir.estimate_memory_cost().unwrap()))
            })
            .collect::<BTreeMap<_, _>>();

        loop {
            let mut candidates = BTreeSet::new();

            let ops = ir
                .ordered_operations()?
                .into_iter()
                .filter_map(|op| op.data().downcast::<FusedPointwise>().map(|_| op.id()))
                .collect::<Vec<_>>();

            for (i, &op_i) in ops.iter().enumerate() {
                'inner: for &op_j in ops.iter().skip(i + 1) {
                    if failed.contains(&(op_i, op_j)) {
                        continue;
                    }

                    if cache.contains_key(&(op_i, op_j)) {
                        candidates.insert((op_i, op_j));
                        continue 'inner;
                    }

                    // `op_i` comes before `op_j` in topo ordering so know that if there is a
                    // dependency then `op_j` is dependent on `op_i` we can only fuse `op_i`
                    // and `op_j` if there does not exist an in between op that is dependent
                    // on `op_i` and is depended upon by `op_j`
                    if ir.is_immediate_dependent_op(op_i, op_j)? || !ir.is_dependent_op(op_j, op_i)? {
                        let (subgraph, inputs, outputs) = fuse_subgraphs(ir, op_i, op_j)?;
                        if let Some(pntwise) = FusedPointwise::new(subgraph.clone(), &self.0)? {
                            let new_cost = pntwise.ir.estimate_memory_cost()?;
                            let old_cost = costs.get(&op_i).unwrap().dominator_sum(*costs.get(&op_j).unwrap());

                            if new_cost.is_le(old_cost) {
                                let saving = if new_cost.var_power() != old_cost.var_power() {
                                    Some(old_cost)
                                } else if new_cost.factor() != old_cost.factor() {
                                    Some(old_cost - new_cost)
                                } else {
                                    None
                                };

                                cache.insert((op_i, op_j), (pntwise, inputs, outputs, new_cost, saving));
                                candidates.insert((op_i, op_j));
                                continue 'inner;
                            }
                        }
                    }

                    failed.insert((op_i, op_j));
                }
            }

            if candidates.is_empty() {
                break;
            } else {
                let mut argmin = *candidates.iter().next().unwrap();
                let mut max_saving = cache.get(&argmin).unwrap().4;

                for arg in candidates {
                    let (_, _, _, _, saving) = cache.get(&arg).unwrap();

                    if match (max_saving, saving) {
                        (Some(x), Some(y)) => x.is_le(*y),
                        (None, Some(_)) => true,
                        (_, None) => false,
                    } {
                        max_saving = *saving;
                        argmin = arg;
                    }
                }

                let (pntwise, inputs, outputs, cost, _) = cache.get(&argmin).cloned().unwrap();
                let new_outputs = ir.add_op(inputs, Ok::<_, IRTrace>(pntwise))?;
                costs.insert(ir.get_parent_op(new_outputs[0])?, cost);

                for (new, old) in new_outputs.into_iter().zip(outputs) {
                    ir.swap_outputs(new, old)?;
                }

                ir.remove_op(argmin.1)?;
                ir.remove_op(argmin.0)?;

                costs.remove(&argmin.0);
                costs.remove(&argmin.1);
                cache.remove(&argmin);
            }
        }

        Ok(())
    }
}

/// Codegen FusedPointwise ops to KernelSrc
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CodegenPointwise;
impl IRTransform for CodegenPointwise {
    fn apply(&self, ir: &mut TensorIR) -> Result<(), IRTrace> {
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

fn fuse_subgraphs(ir: &TensorIR, op_i: OpId, op_j: OpId) -> Result<(SubGraph, Vec<NodeId>, Vec<NodeId>), IRTrace> {
    let op_i = ir.get_op(op_i)?;
    let op_j = ir.get_op(op_j)?;

    let sub_i = &op_i.data().downcast::<FusedPointwise>().unwrap().sub;
    let sub_j = &op_j.data().downcast::<FusedPointwise>().unwrap().sub;

    let mut new_sub = TensorIR::default();

    let mut map = BTreeMap::new();
    let mut inputs_i = Vec::new();
    for &input in op_i.inputs() {
        let new_input = new_sub.add_input(ir.get_node(input)?.ty());
        map.insert(input, new_input);
        inputs_i.push(new_input);
    }

    let mut total_outputs = Vec::new();

    let op_j_set = op_j.inputs().iter().cloned().collect::<BTreeSet<_>>();

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
