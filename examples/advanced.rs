use bullet_lib::{
    default::{
        inputs::{self, InputType},
        loader, outputs, QuantTarget, Trainer,
    },
    lr, operations,
    optimiser::{AdamWOptimiser, AdamWParams},
    wdl, Activation, ExecutionContext, Graph, GraphBuilder, LocalSettings, Node, Shape, TrainingSchedule,
    TrainingSteps,
};

fn main() {
    let inputs = inputs::Chess768;
    let hl = 512;
    let num_inputs = inputs.size();

    let (mut graph, output_node) = build_network(num_inputs, hl);

    graph.get_weights_mut("l0w").seed_random(0.0, 1.0 / (num_inputs as f32).sqrt(), true);
    graph.get_weights_mut("l0b").seed_random(0.0, 1.0 / (num_inputs as f32).sqrt(), true);
    graph.get_weights_mut("l1w").seed_random(0.0, 1.0 / (2.0 * hl as f32).sqrt(), true);
    graph.get_weights_mut("l1b").seed_random(0.0, 1.0 / (2.0 * hl as f32).sqrt(), true);

    let mut trainer = Trainer::<AdamWOptimiser, inputs::Chess768, outputs::Single>::new(
        graph,
        output_node,
        AdamWParams::default(),
        inputs::Chess768,
        outputs::Single,
        vec![
            ("l0w".to_string(), QuantTarget::I16(255)),
            ("l0b".to_string(), QuantTarget::I16(255)),
            ("l1w".to_string(), QuantTarget::I16(64)),
            ("l1b".to_string(), QuantTarget::I16(64 * 255)),
            ("pst".to_string(), QuantTarget::I16(255)),
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
    let mut builder = GraphBuilder::default();

    // inputs
    let stm = builder.create_input("stm", Shape::new(inputs, 1));
    let nstm = builder.create_input("nstm", Shape::new(inputs, 1));
    let targets = builder.create_input("targets", Shape::new(1, 1));

    // trainable weights
    let l0w = builder.create_weights("l0w", Shape::new(hl, inputs));
    let l0b = builder.create_weights("l0b", Shape::new(hl, 1));
    let l1w = builder.create_weights("l1w", Shape::new(1, hl * 2));
    let l1b = builder.create_weights("l1b", Shape::new(1, 1));
    let pst = builder.create_weights("pst", Shape::new(1, inputs));

    // inference
    let l1 = operations::sparse_affine_dual_with_activation(&mut builder, l0w, stm, nstm, l0b, Activation::SCReLU);
    let l2 = operations::affine(&mut builder, l1w, l1, l1b);
    let psqt = operations::matmul(&mut builder, pst, stm);
    let predicted = operations::add(&mut builder, l2, psqt);

    let sigmoided = operations::activate(&mut builder, predicted, Activation::Sigmoid);
    operations::mse(&mut builder, sigmoided, targets);

    // graph, output node
    (builder.build(ExecutionContext::default()), predicted)
}
