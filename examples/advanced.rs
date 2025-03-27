use bullet_lib::{
    nn::{
        optimiser::{AdamWOptimiser, AdamWParams},
        ExecutionContext, Graph, GraphCompileArgs, InitSettings, NetworkBuilder, Node, Shape,
    },
    trainer::{
        default::{inputs, loader, outputs, Trainer},
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
};

type InputFeatures = inputs::Chess768;
const INPUTS: usize = 768;

fn main() {
    let inputs = InputFeatures::default();

    let (graph, output_node) = build_lora(128, 4);

    println!("Number of weights: {}", graph.get_num_params());

    let mut trainer = Trainer::<AdamWOptimiser, _, _>::new(
        graph,
        output_node,
        AdamWParams::default(),
        inputs,
        outputs::Single,
        vec![],
        false,
    );

    let schedule = TrainingSchedule {
        net_id: "test".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 1024,
            start_superbatch: 1,
            end_superbatch: 10,
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

#[allow(unused)]
fn build_simple(hl: usize) -> (Graph, Node) {
    let mut builder = NetworkBuilder::default();

    let stm = builder.new_sparse_input("stm", Shape::new(INPUTS, 1), 32);
    let nstm = builder.new_sparse_input("nstm", Shape::new(INPUTS, 1), 32);
    let targets = builder.new_dense_input("targets", Shape::new(1, 1));

    let l0 = builder.new_affine("l0", INPUTS, hl);
    let l1 = builder.new_affine("l1", 2 * hl, 1);

    let stm_subnet = l0.forward(stm).crelu();
    let ntm_subnet = l0.forward(nstm).crelu();
    let out = l1.forward(stm_subnet.concat(ntm_subnet));
    let pred = out.sigmoid();
    pred.squared_error(targets);

    let output_node = out.node();
    (builder.build(ExecutionContext::default()), output_node)
}

#[allow(unused)]
fn build_lora(hl: usize, rank: usize) -> (Graph, Node) {
    let mut builder = NetworkBuilder::default();

    let stm = builder.new_sparse_input("stm", Shape::new(INPUTS, 1), 32);
    let nstm = builder.new_sparse_input("nstm", Shape::new(INPUTS, 1), 32);
    let targets = builder.new_dense_input("targets", Shape::new(1, 1));

    let l0x = builder.new_weights("l0x", Shape::new(hl, rank), InitSettings::Normal { mean: 0.0, stdev: 0.1 });
    let l0y = builder.new_weights("l0y", Shape::new(rank, INPUTS), InitSettings::Normal { mean: 0.0, stdev: 0.1 });
    let l0b = builder.new_weights("l0b", Shape::new(hl, 1), InitSettings::Zeroed);
    let l1 = builder.new_affine("l1", 2 * hl, 1);

    let l0 = l0x.matmul(l0y);
    let stm_subnet = (l0.matmul(stm) + l0b).crelu();
    let ntm_subnet = (l0.matmul(nstm) + l0b).crelu();
    let out = l1.forward(stm_subnet.concat(ntm_subnet));
    let pred = out.sigmoid();
    pred.squared_error(targets);

    let output_node = out.node();
    builder.set_compile_args(GraphCompileArgs::default().emit_ir());
    (builder.build(ExecutionContext::default()), output_node)
}
