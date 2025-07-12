use bullet_core::{
    cpu::{CpuError, CpuThread},
    graph::{
        builder::{GraphBuilder, Shape},
        ir::BackendMarker,
    },
};

#[derive(Clone, Copy, Default)]
pub struct Marker;
impl BackendMarker for Marker {
    type Backend = CpuThread;
}

fn main() -> Result<(), CpuError> {
    let builder = GraphBuilder::<Marker>::default();

    // inputs
    let stm = builder.new_sparse_input("stm", Shape::new(768, 1), 32);
    let nstm = builder.new_sparse_input("nstm", Shape::new(768, 1), 32);
    let targets = builder.new_dense_input("targets", Shape::new(1, 1));

    // trainable weights
    let l0 = builder.new_affine("l0", 768, 512);
    let l1 = builder.new_affine("l1", 256, 1);

    // inference
    let stm_subnet = l0.forward(stm).crelu().pairwise_mul().pairwise_mul();
    let ntm_subnet = l0.forward(nstm).crelu().pairwise_mul().pairwise_mul();
    let out = l1.forward(stm_subnet.concat(ntm_subnet));
    let pred = out.sigmoid();
    pred.squared_error(targets);

    // build graph
    let graph = builder.build(CpuThread);

    graph.get_last_device_error()
}
