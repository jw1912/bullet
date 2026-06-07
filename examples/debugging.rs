use std::fs::File;

use bullet_compiler::model::ModelBuilder;
use bullet_gpu::runtime::Device;
use bullet_lib::{
    game::inputs::{Chess768, SparseInputType},
    nn::ExecutionContext,
};
use bullet_trainer::model::Model;

fn main() {
    let hl_size = 128;

    let inputs = Chess768;
    let num_inputs = inputs.num_inputs();
    let nnz = inputs.max_active();

    let (ir, (output, hidden)) = {
        let builder = ModelBuilder::default();

        let stm = builder.new_sparse_input("stm", (num_inputs, 1), nnz);
        let ntm = builder.new_sparse_input("ntm", (num_inputs, 1), nnz);

        // weights
        let l0 = builder.new_affine("l0", 768, hl_size);
        let l1 = builder.new_affine("l1", 2 * hl_size, 1);

        // inference
        let stm_hidden = l0.forward(stm).screlu();
        let ntm_hidden = l0.forward(ntm).screlu();
        let hidden_layer = stm_hidden.concat(ntm_hidden);
        let output = l1.forward(hidden_layer);

        (builder.ir().clone(), (output.node(), stm_hidden.node()))
    };

    let device = Device::<ExecutionContext>::new(0).unwrap();
    let mut model = Model::new(ir, device, None, [(output, "output".into()), (hidden, "hidden".into())]);

    let file = File::open("checkpoints/simple-10/optimiser_state/weights.bin").unwrap();
    model.load_weights_from(file).unwrap();
}
