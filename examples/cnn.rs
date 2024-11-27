use bullet_lib::{
    inputs, loader, lr, operations,
    optimiser::{self, AdamWOptimiser, AdamWParams},
    outputs, wdl, Activation, ConvolutionDescription, ExecutionContext, Graph, GraphBuilder, LocalSettings, Node,
    QuantTarget, Shape, Trainer, TrainingSchedule, TrainingSteps,
};

fn main() {
    let mut trainer = make_trainer();

    let schedule = TrainingSchedule {
        net_id: "cnn".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 128,
            start_superbatch: 1,
            end_superbatch: 20,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.75 },
        lr_scheduler: lr::StepLR { start: 0.001, gamma: 0.1, step: 8 },
        save_rate: 10,
    };

    let optimiser_params =
        optimiser::AdamWParams { decay: 0.01, beta1: 0.9, beta2: 0.999, min_weight: -0.99, max_weight: 0.99 };

    trainer.set_optimiser_params(optimiser_params);

    let settings = LocalSettings { threads: 8, test_set: None, output_directory: "checkpoints", batch_queue_size: 256 };

    let data_loader = loader::DirectSequentialDataLoader::new(&["data/bestmove-q.data"]);

    trainer.run(&schedule, &settings, &data_loader);
}

fn make_trainer() -> Trainer<AdamWOptimiser, inputs::Chess768, outputs::Single> {
    let channels = [8, 4, 2, 1];

    let (mut graph, output_node) = build_network(&channels);

    let mut save = Vec::new();

    let mut input_channels = 12;

    for (i, &size) in channels.iter().enumerate() {
        let stdev = 1.0 / ((64 * input_channels) as f32).sqrt();
        graph.get_weights_mut(&format!("l{i}f")).seed_random(0.0, stdev, true);
        graph.get_weights_mut(&format!("l{i}b")).seed_random(0.0, stdev, true);
        save.push((format!("l{i}f"), QuantTarget::Float));
        save.push((format!("l{i}b"), QuantTarget::Float));
        input_channels = size;
    }

    let stdev = 1.0 / ((64 * input_channels) as f32).sqrt();
    graph.get_weights_mut("ow").seed_random(0.0, stdev, true);
    graph.get_weights_mut("ob").seed_random(0.0, stdev, true);
    save.push(("ow".to_string(), QuantTarget::Float));
    save.push(("ob".to_string(), QuantTarget::Float));

    Trainer::new(graph, output_node, AdamWParams::default(), inputs::Chess768, outputs::Single, save, true)
}

fn build_network(channels: &[usize]) -> (Graph, Node) {
    let mut builder = GraphBuilder::default();

    // inputs
    let mut stm = builder.create_input("stm", Shape::new(768, 1));
    let targets = builder.create_input("targets", Shape::new(3, 1));

    let mut input_channels = 12;

    for (i, &output_channels) in channels.iter().enumerate() {
        let filters = builder.create_weights(&format!("l{i}f"), Shape::new(9, input_channels * output_channels));
        let bias = builder.create_weights(&format!("l{i}b"), Shape::new(64 * output_channels, 1));

        let conv_desc = ConvolutionDescription::new(
            Shape::new(8, 8),
            input_channels,
            output_channels,
            Shape::new(3, 3),
            (1, 1),
            Shape::new(1, 1),
        );

        stm = operations::convolution(&mut builder, filters, stm, conv_desc);
        stm = operations::add(&mut builder, stm, bias);
        stm = operations::activate(&mut builder, stm, Activation::ReLU);

        input_channels = output_channels;
    }

    let output_weights = builder.create_weights("ow", Shape::new(3, 64 * input_channels));
    let output_bias = builder.create_weights("ob", Shape::new(3, 1));
    let predicted = operations::affine(&mut builder, output_weights, stm, output_bias);

    operations::softmax_crossentropy_loss(&mut builder, predicted, targets);

    // graph, output node
    (builder.build(ExecutionContext::default()), predicted)
}
