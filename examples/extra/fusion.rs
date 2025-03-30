use bullet_core::{
    backend::cpu::{CpuError, CpuThread},
    graph::{
        builder::{GraphBuilder, InitSettings, Shape},
        ir::args::GraphIRCompileArgs,
        Graph,
    },
};

fn main() -> Result<(), CpuError> {
    let graph = build_network(768, 32, 8, 512);
    graph.get_last_device_error()
}

fn build_network(num_inputs: usize, max_active: usize, num_buckets: usize, hl: usize) -> Graph<CpuThread> {
    let mut builder = GraphBuilder::default();

    // inputs
    let stm = builder.new_sparse_input("stm", Shape::new(num_inputs, 1), max_active);
    let nstm = builder.new_sparse_input("nstm", Shape::new(num_inputs, 1), max_active);
    let targets = builder.new_dense_input("targets", Shape::new(1, 1));
    let buckets = builder.new_sparse_input("buckets", Shape::new(num_buckets, 1), 1);

    // trainable weights
    let l0 = builder.new_affine("l0", num_inputs, hl);
    let l1 = builder.new_affine("l1", hl, num_buckets * 16);
    let l2 = builder.new_affine("l2", 30, num_buckets * 32);
    let l3 = builder.new_affine("l3", 32, num_buckets);
    let pst = builder.new_weights("pst", Shape::new(1, num_inputs), InitSettings::Zeroed);

    // inference
    let stm_subnet = l0.forward(stm).crelu().pairwise_mul();
    let ntm_subnet = l0.forward(nstm).crelu().pairwise_mul();
    let mut out = stm_subnet.concat(ntm_subnet);

    out = l1.forward(out).select(buckets);

    let skip_neuron = out.slice_rows(15, 16);
    out = out.slice_rows(0, 15);

    out = out.concat(out.abs_pow(2.0));
    out = out.crelu();

    out = l2.forward(out).select(buckets).screlu();
    out = l3.forward(out).select(buckets);

    let pst_out = pst.matmul(stm) - pst.matmul(nstm);
    out = out + skip_neuron + pst_out;

    let pred = out.sigmoid();
    pred.squared_error(targets);

    // build graph
    builder.set_compile_args(GraphIRCompileArgs::default().fancy_ir_display(1.0));
    builder.build(CpuThread)
}
