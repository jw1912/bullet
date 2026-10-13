use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt,
    rc::Rc,
};

use crate::{
    ir::{NodeId, OpId},
    tensor::{
        IRTrace, TensorIR, TensorOp,
        transform::{IRTransform, autograd::Gradients},
    },
};

/// Returns true if an operation is cheap enough to be recomputed in the backward pass
pub type RecomputePolicy = Rc<dyn Fn(&TensorOp) -> bool>;

/// Min-cut optimal recomputation, as described in
/// https://dev-discuss.pytorch.org/t/min-cut-optimal-recomputation-i-e-activation-checkpointing-with-aotautograd/467
///
/// The forward pass is everything required to compute the forward outputs recorded in `gradients`, and
/// the backward pass is everything else required to compute the graph outputs. Chooses the set of forward
/// nodes to save for the backward pass that minimises the bytes read/written between passes, subject to only
/// recomputing operations allowed by `policy`, then rewrites the backward pass to recompute everything else
/// it needs from the saved nodes.
///
/// N.B. Recomputed operations are duplicates of forward operations, so common subexpression elimination
/// will undo this transform.
pub struct MinCutRecompute {
    gradients: Gradients,
    policy: RecomputePolicy,
    saved: Rc<RefCell<BTreeSet<NodeId>>>,
}

impl MinCutRecompute {
    pub fn new(
        gradients: &Gradients,
        policy: impl Fn(&TensorOp) -> bool + 'static,
    ) -> (Self, Rc<RefCell<BTreeSet<NodeId>>>) {
        let saved = Rc::<RefCell<BTreeSet<NodeId>>>::default();
        (Self { gradients: gradients.clone(), policy: Rc::new(policy), saved: saved.clone() }, saved)
    }

    /// Operations required for the graph outputs but not for the forward outputs
    pub fn backward_pass(ir: &TensorIR, forward_outputs: &[NodeId]) -> Result<BTreeSet<OpId>, IRTrace> {
        if forward_outputs.is_empty() {
            return Err("No forward pass recorded, has `TakeGradient` been applied?".into());
        }

        for &node in forward_outputs {
            if ir.get_node(node).is_err() {
                return Err(format!("Forward output {node:?} was removed, register it as an output!").into());
            }
        }

        let forward = ancestor_ops(ir, forward_outputs.iter().copied())?;
        let all = ancestor_ops(ir, ir.outputs.iter().copied())?;
        Ok(all.difference(&forward).copied().collect())
    }
}

impl fmt::Debug for MinCutRecompute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MinCutRecompute").finish_non_exhaustive()
    }
}

impl IRTransform for MinCutRecompute {
    fn apply(&self, ir: &mut TensorIR) -> Result<(), IRTrace> {
        let forward_outputs = self.gradients.forward_outputs();
        let region = Self::backward_pass(ir, &forward_outputs)?;
        let cost = TransferCost(forward_outputs.into_iter().collect());

        *self.saved.borrow_mut() = recompute_region(ir, &region, &*self.policy, &cost)?;

        Ok(())
    }
}

/// Nodes that are always in memory only need to be read in the
/// backward pass, anything else needs to be written and then read
struct TransferCost(BTreeSet<NodeId>);

impl RecomputeCost for TransferCost {
    fn save(&self, ir: &TensorIR, node: NodeId) -> Result<u64, IRTrace> {
        let ty = ir.get_node(node)?.ty();
        let bytes = (ty.size().get() * ty.dtype().bytes()) as u64;
        let persistent = ir.is_input(node)? || ir.is_constant(node)? || self.0.contains(&node);
        Ok(if persistent { bytes } else { 2 * bytes })
    }
}

fn ancestor_ops(ir: &TensorIR, nodes: impl Iterator<Item = NodeId>) -> Result<BTreeSet<OpId>, IRTrace> {
    let mut ops = BTreeSet::new();
    let mut stack: Vec<_> = nodes.collect();

    while let Some(node) = stack.pop() {
        let op = ir.get_op(ir.get_parent_op(node)?)?;
        if ops.insert(op.id()) {
            stack.extend(op.inputs().iter().copied());
        }
    }

    Ok(ops)
}

pub trait RecomputeCost {
    /// Cost of materialising `node` for the region
    fn save(&self, ir: &TensorIR, node: NodeId) -> Result<u64, IRTrace>;

    /// Cost of recomputing `node` by reading directly from its inputs, if this is not the cost of
    /// those inputs, e.g. a slice only reads part of its input. Its inputs must be in memory
    fn load(&self, _ir: &TensorIR, _node: NodeId) -> Result<Option<u64>, IRTrace> {
        Ok(None)
    }
}

/// Rewrites `region` to recompute the values it reads from outside itself, wherever that is cheaper
/// than materialising them, returning the nodes that are still read from outside.
///
/// The choice is a minimum cut, where `cost` is the cost of materialising a node for the region and
/// only operations allowed by `policy` may be recomputed. The region must be convex, i.e. no path may
/// leave and then reenter it.
pub fn recompute_region(
    ir: &mut TensorIR,
    region: &BTreeSet<OpId>,
    policy: &dyn Fn(&TensorOp) -> bool,
    cost: &dyn RecomputeCost,
) -> Result<BTreeSet<NodeId>, IRTrace> {
    let mut required = BTreeSet::new();
    for &id in region {
        for &input in ir.get_op(id)?.inputs() {
            if !region.contains(&ir.get_parent_op(input)?) {
                required.insert(input);
            }
        }
    }

    // only ancestors reachable through recomputable operations can affect the cut
    let mut index = BTreeMap::new();
    let mut fixed = BTreeSet::new();
    let mut loads = BTreeMap::new();
    let mut stack: Vec<_> = required.iter().copied().collect();

    while let Some(node) = stack.pop() {
        if index.contains_key(&node) {
            continue;
        }

        index.insert(node, index.len());

        let op = ir.get_op(ir.get_parent_op(node)?)?;

        if region.contains(&op.id()) {
            return Err("Recompute region is not convex!".into());
        }

        if op.data().is_input() || !policy(op.data()) {
            fixed.insert(node);
        } else if let Some(load) = cost.load(ir, node)? {
            loads.insert(node, load);
        } else {
            stack.extend(op.inputs().iter().copied());
        }
    }

    // each node is split into an `in` and `out` vertex,
    // joined by an edge weighted by the cost of saving the node
    let vin = |node: NodeId| 2 * index[&node] + 2;
    let vout = |node: NodeId| 2 * index[&node] + 3;

    let mut graph = FlowGraph::new(2 * index.len() + 2);

    for &node in index.keys() {
        graph.add_edge(vin(node), vout(node), cost.save(ir, node)?);

        if fixed.contains(&node) {
            graph.add_edge(FlowGraph::SOURCE, vin(node), FlowGraph::INF);
        } else if let Some(&load) = loads.get(&node) {
            graph.add_edge(FlowGraph::SOURCE, vin(node), load);
        } else {
            for &input in ir.get_op(ir.get_parent_op(node)?)?.inputs() {
                graph.add_edge(vout(input), vin(node), FlowGraph::INF);
            }
        }
    }

    for &node in &required {
        graph.add_edge(vout(node), FlowGraph::SINK, FlowGraph::INF);
    }

    let mut remat = Rematerialise {
        source_side: graph.min_cut(),
        index,
        map: BTreeMap::new(),
        saved: BTreeSet::new(),
        order: Vec::new(),
    };

    for &node in &required {
        remat.get(ir, node)?;
    }

    for &id in region {
        let inputs: BTreeSet<_> = ir.get_op(id)?.inputs().iter().copied().collect();
        for old in inputs {
            if let Some(&new) = remat.map.get(&old) {
                ir.ir_mut().replace_single_input(id, new, old)?;
            }
        }
    }

    // originals that were only needed by the region are now dead
    for &id in remat.order.iter().rev() {
        let mut dead = true;
        for &out in ir.get_op(id)?.outputs() {
            dead &= ir.get_node(out)?.children() == 0 && !ir.is_output(out);
        }

        if dead {
            ir.remove_op(id)?;
        }
    }

    ir.debug_check_edit()?;

    Ok(remat.saved)
}

/// Walks back from the required nodes until hitting the cut, duplicating operations along the way
struct Rematerialise {
    source_side: Vec<bool>,
    index: BTreeMap<NodeId, usize>,
    map: BTreeMap<NodeId, NodeId>,
    saved: BTreeSet<NodeId>,
    order: Vec<OpId>,
}

impl Rematerialise {
    fn get(&mut self, ir: &mut TensorIR, node: NodeId) -> Result<NodeId, IRTrace> {
        if let Some(&new) = self.map.get(&node) {
            return Ok(new);
        }

        // inputs to loaded nodes are already in memory
        let Some(&i) = self.index.get(&node) else {
            self.saved.insert(node);
            return Ok(node);
        };

        if self.source_side[2 * i + 3] {
            return Err("Required node not available in recompute region!".into());
        }

        if self.source_side[2 * i + 2] {
            self.saved.insert(node);
            return Ok(node);
        }

        let op = ir.get_op(ir.get_parent_op(node)?)?.clone();
        let inputs = op.inputs().iter().map(|&input| self.get(ir, input)).collect::<Result<Vec<_>, _>>()?;
        let outputs = ir.add_dyn_op(inputs, Ok(op.data().clone()))?;

        self.map.extend(op.outputs().iter().copied().zip(outputs));
        self.order.push(op.id());

        Ok(self.map[&node])
    }
}

/// Dinic's algorithm, edge `e` has reverse edge `e ^ 1`
struct FlowGraph {
    edges: Vec<(usize, u64)>,
    adj: Vec<Vec<usize>>,
}

impl FlowGraph {
    const SOURCE: usize = 0;
    const SINK: usize = 1;
    const INF: u64 = u64::MAX / 4;

    fn new(vertices: usize) -> Self {
        Self { edges: Vec::new(), adj: vec![Vec::new(); vertices] }
    }

    fn add_edge(&mut self, from: usize, to: usize, capacity: u64) {
        self.adj[from].push(self.edges.len());
        self.edges.push((to, capacity));
        self.adj[to].push(self.edges.len());
        self.edges.push((from, 0));
    }

    fn levels(&self) -> Vec<Option<usize>> {
        let mut levels = vec![None; self.adj.len()];
        let mut queue = VecDeque::from([Self::SOURCE]);
        levels[Self::SOURCE] = Some(0);

        while let Some(v) = queue.pop_front() {
            for &e in &self.adj[v] {
                let (to, capacity) = self.edges[e];
                if capacity > 0 && levels[to].is_none() {
                    levels[to] = levels[v].map(|l| l + 1);
                    queue.push_back(to);
                }
            }
        }

        levels
    }

    fn augment(&mut self, v: usize, limit: u64, levels: &[Option<usize>], next: &mut [usize]) -> u64 {
        if v == Self::SINK {
            return limit;
        }

        while let Some(&e) = self.adj[v].get(next[v]) {
            let (to, capacity) = self.edges[e];

            if capacity > 0 && levels[to] == levels[v].map(|l| l + 1) {
                let flow = self.augment(to, limit.min(capacity), levels, next);

                if flow > 0 {
                    self.edges[e].1 -= flow;
                    self.edges[e ^ 1].1 += flow;
                    return flow;
                }
            }

            next[v] += 1;
        }

        0
    }

    /// Saturates the graph, returning which vertices are on the source side of the minimum cut
    fn min_cut(&mut self) -> Vec<bool> {
        loop {
            let levels = self.levels();

            if levels[Self::SINK].is_none() {
                return levels.iter().map(Option::is_some).collect();
            }

            let mut next = vec![0; self.adj.len()];
            while self.augment(Self::SOURCE, Self::INF, &levels, &mut next) > 0 {}
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{
        DType, OpType, TType, TValue,
        operation::{CABinary, CABinaryOp, Constant, CopyOp, ScalarConstant, Unary, UnaryOp},
        transform::{
            autograd::{LowerForward, TakeGradient},
            inline::InlineSubgraphs,
        },
    };

    use super::*;

    /// Placeholder for the rest of the graph, takes the forward
    /// output and produces its gradient for the backward pass
    #[derive(Debug)]
    struct RestOfGraph(TType);

    impl OpType for RestOfGraph {
        fn opname(&self) -> String {
            "rest-of-graph".to_string()
        }

        fn inputs(&self) -> Vec<TType> {
            vec![self.0]
        }

        fn outputs(&self) -> Vec<TType> {
            vec![self.0]
        }

        fn evaluate(&self, inputs: Vec<&TValue>, mut outputs: Vec<&mut TValue>) -> bool {
            for idx in 0..inputs[0].size() {
                outputs[0].write(idx, inputs[0].read(idx));
            }

            true
        }
    }

    fn pointwise(op: &TensorOp) -> bool {
        op.downcast::<UnaryOp>().is_some()
            || op.downcast::<CABinaryOp>().is_some()
            || op.downcast::<CopyOp>().is_some()
            || op.downcast::<Constant>().is_some()
            || op.downcast::<ScalarConstant>().is_some()
    }

    fn count<T: OpType>(ir: &TensorIR) -> usize {
        ir.operations().iter().filter(|op| op.data().downcast::<T>().is_some()).count()
    }

    struct JointGraph {
        ir: TensorIR,
        gradients: Gradients,
        seeds: BTreeMap<NodeId, TValue>,
    }

    /// Constructs the joint forward + backward graph of `f`
    fn joint_graph(
        inputs: &[Vec<f32>],
        f: impl Fn(&mut TensorIR, &[NodeId]) -> Result<NodeId, IRTrace>,
    ) -> Result<JointGraph, IRTrace> {
        let mut ir = TensorIR::default();

        let nodes: Vec<_> = inputs.iter().map(|x| ir.add_input(TType::new(x.len(), DType::F32))).collect();
        let y = f(&mut ir, &nodes)?;
        let tangent = ir.add_op([y], Ok::<_, IRTrace>(RestOfGraph(ir.get_node(y)?.ty())))?[0];

        let (transform, grads) = TakeGradient::new(ir.get_parent_op(y)?, [tangent]);
        ir.transform(transform)?;

        ir.register_output(y);
        for x in &nodes {
            ir.register_output(grads.get(*x).unwrap());
        }

        ir.transform(LowerForward)?;
        ir.transform(InlineSubgraphs)?;
        ir.optimise()?;

        let seeds = nodes.iter().zip(inputs).map(|(&n, x)| (n, TValue::F32(x.clone()))).collect();

        Ok(JointGraph { ir, gradients: grads, seeds })
    }

    /// Applies the transform, checking that it saves `expected` and preserves semantics
    fn check_recompute(
        graph: JointGraph,
        policy: impl Fn(&TensorOp) -> bool + 'static,
        expected: impl Fn(&TensorIR) -> Result<BTreeSet<NodeId>, IRTrace>,
    ) -> Result<TensorIR, IRTrace> {
        let JointGraph { mut ir, gradients, seeds } = graph;
        ir.debug = true;

        let before = ir.evaluate(seeds.clone())?.unwrap();
        let expected = expected(&ir)?;

        let (transform, saved) = MinCutRecompute::new(&gradients, policy);
        ir.transform(transform)?;

        assert_eq!(*saved.borrow(), expected, "{ir}");
        assert_eq!(ir.evaluate(seeds)?.unwrap(), before);

        // backward pass only reads saved forward nodes
        let region = MinCutRecompute::backward_pass(&ir, &gradients.forward_outputs())?;
        for &id in &region {
            for input in ir.get_op(id)?.inputs() {
                assert!(region.contains(&ir.get_parent_op(*input)?) || saved.borrow().contains(input), "{ir}");
            }
        }

        Ok(ir)
    }

    fn parent_of(ir: &TensorIR, node: NodeId, input: usize) -> Result<NodeId, IRTrace> {
        Ok(ir.get_op(ir.get_parent_op(node)?)?.inputs()[input])
    }

    fn rest_of_graph_input(ir: &TensorIR) -> NodeId {
        let ops = ir.operations();
        ops.iter().find(|op| op.data().downcast::<RestOfGraph>().is_some()).unwrap().inputs()[0]
    }

    #[test]
    fn cos_cos_saves_input() -> Result<(), IRTrace> {
        let xs = vec![-0.9, -0.3, 0.2, 0.7];
        let graph = joint_graph(&[xs], |ir, x| {
            let y = ir.add_unary(x[0], Unary::Cos)?;
            ir.add_unary(y, Unary::Cos)
        })?;

        let x = *graph.seeds.keys().next().unwrap();
        let ir = check_recompute(graph, pointwise, |_| Ok(BTreeSet::from([x])))?;

        // forward pass is unchanged, and the backward pass recomputes cos(x) for its own use
        // and cos(cos(x)) for the rest of the graph, as the latter is cheaper than reading it
        assert_eq!(count::<UnaryOp>(&ir), 2 + 2 + 2);

        Ok(())
    }

    #[test]
    fn cos_cos_respects_policy() -> Result<(), IRTrace> {
        let xs = vec![-0.9, -0.3, 0.2, 0.7];
        let graph = joint_graph(&[xs], |ir, x| {
            let y = ir.add_unary(x[0], Unary::Cos)?;
            ir.add_unary(y, Unary::Cos)
        })?;

        let x = *graph.seeds.keys().next().unwrap();
        let no_cos = |op: &TensorOp| pointwise(op) && op.downcast::<UnaryOp>().is_none_or(|op| op.op() != Unary::Cos);

        // all values either side of cos are needed in the backward pass, but cos cannot be recomputed
        check_recompute(graph, no_cos, |ir| {
            let y = rest_of_graph_input(ir);
            Ok(BTreeSet::from([x, parent_of(ir, y, 0)?, y]))
        })?;

        Ok(())
    }

    #[test]
    fn sum_cos_cos_saves_sum() -> Result<(), IRTrace> {
        let inputs = [vec![0.1, 0.2], vec![0.3, -0.4], vec![-0.5, 0.6], vec![0.7, 0.8]];
        let graph = joint_graph(&inputs, |ir, x| {
            let ab = ir.add_binary(x[0], x[1], CABinary::Add)?;
            let abc = ir.add_binary(ab, x[2], CABinary::Add)?;
            let abcd = ir.add_binary(abc, x[3], CABinary::Add)?;
            let y = ir.add_unary(abcd, Unary::Cos)?;
            ir.add_unary(y, Unary::Cos)
        })?;

        // save the sum rather than either the inputs or both inputs to cos
        let sum = |ir: &TensorIR| parent_of(ir, parent_of(ir, rest_of_graph_input(ir), 0)?, 0);
        check_recompute(graph, pointwise, |ir| Ok(BTreeSet::from([sum(ir)?])))?;

        Ok(())
    }

    /// `cos(cos(x)) * cos(cos(a + b + c + d))`
    fn composed_graph() -> Result<JointGraph, IRTrace> {
        let inputs = [vec![-0.9, 0.7], vec![0.1, 0.2], vec![0.3, -0.4], vec![-0.5, 0.6], vec![0.7, 0.8]];
        joint_graph(&inputs, |ir, x| {
            let cx = ir.add_unary(x[0], Unary::Cos)?;
            let ccx = ir.add_unary(cx, Unary::Cos)?;

            let ab = ir.add_binary(x[1], x[2], CABinary::Add)?;
            let abc = ir.add_binary(ab, x[3], CABinary::Add)?;
            let abcd = ir.add_binary(abc, x[4], CABinary::Add)?;
            let cs = ir.add_unary(abcd, Unary::Cos)?;
            let ccs = ir.add_unary(cs, Unary::Cos)?;

            ir.add_binary(ccx, ccs, CABinary::Mul)
        })
    }

    /// Inputs and outputs of all cos operations
    fn cos_nodes(ir: &TensorIR) -> (BTreeSet<NodeId>, BTreeSet<NodeId>) {
        let ops = ir.operations();
        let cos = ops.iter().filter(|op| op.data().downcast::<UnaryOp>().is_some_and(|op| op.op() == Unary::Cos));
        cos.map(|op| (op.inputs()[0], op.outputs()[0])).unzip()
    }

    #[test]
    fn composed_saves_input_and_sum() -> Result<(), IRTrace> {
        // save x and the sum, recomputing everything else
        check_recompute(composed_graph()?, pointwise, |ir| {
            let (inputs, outputs) = cos_nodes(ir);
            let saved: BTreeSet<_> = inputs.difference(&outputs).copied().collect();
            assert_eq!(saved.len(), 2);
            Ok(saved)
        })?;

        Ok(())
    }

    #[test]
    fn composed_respects_policy() -> Result<(), IRTrace> {
        let no_cos = |op: &TensorOp| pointwise(op) && op.downcast::<UnaryOp>().is_none_or(|op| op.op() != Unary::Cos);

        // every value either side of a cos is needed, but the sum is still cheaper to save than its inputs
        check_recompute(composed_graph()?, no_cos, |ir| {
            let (inputs, outputs) = cos_nodes(ir);
            let saved: BTreeSet<_> = inputs.union(&outputs).copied().collect();
            assert_eq!(saved.len(), 6);
            Ok(saved)
        })?;

        Ok(())
    }

    #[test]
    fn requires_recorded_forward_pass() -> Result<(), IRTrace> {
        let mut ir = TensorIR::default();
        let x = ir.add_input(TType::new(4, DType::F32));
        let y = ir.add_unary(x, Unary::Cos)?;
        ir.register_output(y);

        let (transform, _) = MinCutRecompute::new(&Gradients::default(), pointwise);
        assert!(ir.transform(transform).is_err());

        Ok(())
    }
}
