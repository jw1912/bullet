use bullet_lib::{
    default::{
        inputs::{self, SparseInputType},
        loader, outputs, Trainer,
    },
    NetworkBuilder,
    lr,
    optimiser::{AdamWOptimiser, AdamWParams},
    save::{Layout, QuantTarget, SavedFormat},
    wdl, Activation, ExecutionContext, Graph, LocalSettings, Node, Shape, TrainingSchedule,
    TrainingSteps,
};

fn main() {
    let inputs = inputs::Chess768;
    let hl = 512;
    let num_inputs = inputs.num_inputs();

    let (graph, output_node) = build_network(num_inputs, hl);

    let mut trainer = Trainer::<AdamWOptimiser, inputs::Chess768, outputs::Single>::new(
        graph,
        output_node,
        AdamWParams::default(),
        inputs::Chess768,
        outputs::Single,
        vec![
            SavedFormat::new("pst", QuantTarget::I16(255), Layout::Normal),
            SavedFormat::new("l0w", QuantTarget::I16(255), Layout::Normal),
            SavedFormat::new("l0b", QuantTarget::I16(255), Layout::Normal),
            SavedFormat::new("l1w", QuantTarget::I16(64), Layout::Normal),
            SavedFormat::new("l1b", QuantTarget::I16(64 * 255), Layout::Normal),
        ],
        false,
    );

    let schedule = TrainingSchedule {
        net_id: "test".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: 240,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.0 },
        lr_scheduler: lr::StepLR { start: 0.001, gamma: 0.3, step: 60 },
        save_rate: 150,
    };

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 512 };

    let data_loader = loader::DirectSequentialDataLoader::new(&["data/baseline.data"]);

    trainer.run(&schedule, &settings, &data_loader);

    let eval = 400.0 * trainer.eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.0");
    println!("Eval: {eval:.3}cp");
}

fn build_network(inputs: usize, hl: usize) -> (Graph, Node) {
    let builder = NetworkBuilder::default();

    // inputs
    let stm = builder.new_input("stm", Shape::new(inputs, 1));
    let nstm = builder.new_input("nstm", Shape::new(inputs, 1));
    let targets = builder.new_input("targets", Shape::new(1, 1));

    // trainable weights
    let l0 = builder.new_affine("l0", inputs, hl);
    let l1 = builder.new_affine("l1", hl * 2, 1);
    let pst = builder.new_weights("pst", Shape::new(1, inputs));

    // inference
    let mut out = l0.forward_sparse_dual_with_activation(stm, nstm, Activation::SCReLU);
    out = l1.forward(out);
    out += pst * stm;
    out.mse(targets);

    // graph, output node
    let output_node = out.node();
    (builder.build(ExecutionContext::default()), output_node)
}
