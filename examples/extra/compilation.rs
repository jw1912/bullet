use acyclib::{
    dag::NodeId,
    device::{
        cpu::{CpuError, CpuMarker, CpuThread},
        function::{self, DeviceFunction},
        tensor::Shape,
    },
    graph::{
        Graph, GraphNodeIdTy,
        builder::GraphBuilder,
        ir::{
            BackendMarker, GraphIR, GraphIRError,
            node::AnnotatedNode,
            operation::{GraphIROperationBase, GraphIROperationCompilable, GraphIROperationError, util},
        },
    },
};

fn main() -> Result<(), CpuError> {
    let builder = GraphBuilder::default();

    // inputs
    let stm = builder.new_sparse_input("stm", Shape::new(768, 1), 32);
    let nstm = builder.new_sparse_input("nstm", Shape::new(768, 1), 32);
    let targets = builder.new_dense_input("targets", Shape::new(1, 1));

    // trainable weights
    let l0 = builder.new_affine("l0", 768, 512);
    let l1a = builder.new_affine("l1a", 256, 1);
    let l1b = builder.new_affine("l1b", 256, 1);

    // inference
    let stm_subnet = l0.forward(stm).crelu().pairwise_mul().pairwise_mul();
    let ntm_subnet = l0.forward(nstm).crelu().pairwise_mul().pairwise_mul();
    let hl = stm_subnet.concat(ntm_subnet);

    let a = l1a.forward(hl).annotated_node();
    let b = l1b.forward(hl).annotated_node();
    let out = builder.apply(MyCustomAdd { a, b });

    let pred = out.sigmoid();
    pred.squared_error(targets);

    // build graph
    let graph = builder.build(CpuThread);

    println!();
    println!("Forward Pass Code");
    println!();
    graph.display_function_code("forward").unwrap();

    println!();
    println!("Backward Pass Code");
    println!();
    graph.display_function_code("backward").unwrap();

    graph.get_last_device_error()
}

#[derive(Debug)]
struct MyCustomAdd {
    a: AnnotatedNode,
    b: AnnotatedNode,
}

impl<B: BackendMarker> GraphIROperationBase<B> for MyCustomAdd {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.a, self.b]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_same_batching(ir, &[&self.a, &self.b])?;
        util::check_dense_eq(ir, &self.a, true)?;
        util::check_dense_eq(ir, &self.b, true)?;

        if self.a.shape == self.b.shape {
            Ok(self.a.shape)
        } else {
            Err(GraphIRError::Op(GraphIROperationError::MismatchedInputShapes(vec![self.a.shape, self.b.shape])))
        }
    }
}

impl GraphIROperationCompilable<CpuMarker> for MyCustomAdd {
    fn forward_pass(&self, graph: &Graph<CpuThread>, output_node: NodeId) -> DeviceFunction<CpuThread> {
        let a = graph.get_ref(self.a.idx, GraphNodeIdTy::Values);
        let b = graph.get_ref(self.b.idx, GraphNodeIdTy::Values);
        let output = graph.get_ref(output_node, GraphNodeIdTy::Values);

        let mut func = DeviceFunction::default();

        func.push(function::MaybeUpdateBatchSize { input: a.clone(), output: output.clone() });
        func.push(function::LinearCombination { input: a, input_mul: 1.0, output: output.clone(), output_mul: 0.0 });
        func.push(function::LinearCombination { input: b, input_mul: 1.0, output, output_mul: 1.0 });

        func
    }

    fn backward_pass(&self, graph: &Graph<CpuThread>, output_node: NodeId) -> DeviceFunction<CpuThread> {
        let input = graph.get_ref(output_node, GraphNodeIdTy::Gradients);

        let mut func = DeviceFunction::default();

        if let Some(output) = graph.maybe_get_ref(self.a.idx, GraphNodeIdTy::Gradients) {
            func.push(function::MaybeUpdateBatchSize { input: input.clone(), output: output.clone() });
            func.push(function::LinearCombination { input: input.clone(), input_mul: 1.0, output, output_mul: 1.0 });
        }

        if let Some(output) = graph.maybe_get_ref(self.b.idx, GraphNodeIdTy::Gradients) {
            func.push(function::MaybeUpdateBatchSize { input: input.clone(), output: output.clone() });
            func.push(function::LinearCombination { input, input_mul: 1.0, output, output_mul: 1.0 });
        }

        func
    }
}
