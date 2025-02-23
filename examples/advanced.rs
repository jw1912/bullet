use bullet_lib::{
    nn::{
        optimiser::{AdamWOptimiser, AdamWParams},
        Activation, ExecutionContext, Graph, InitSettings, NetworkBuilder, Node, Shape,
    },
    trainer::{
        default::{inputs, loader, outputs, Trainer},
        save::{Layout, QuantTarget, SavedFormat},
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
};

type InputFeatures = inputs::Chess768;
type OutputBuckets = outputs::MaterialCount<8>;
const HL_SIZE: usize = 512;

fn main() {
    let inputs = InputFeatures::default();
    let output_buckets = OutputBuckets::default();

    let num_inputs = <InputFeatures as inputs::SparseInputType>::num_inputs(&inputs);
    let max_active = <InputFeatures as inputs::SparseInputType>::max_active(&inputs);
    let num_buckets = <OutputBuckets as outputs::OutputBuckets<_>>::BUCKETS;

    let (graph, output_node) = build_network(num_inputs, max_active, num_buckets, HL_SIZE);

    let mut trainer = Trainer::<AdamWOptimiser, _, _>::new(
        graph,
        output_node,
        AdamWParams::default(),
        inputs,
        output_buckets,
        vec![
            SavedFormat::new("pst", QuantTarget::I16(255), Layout::Normal),
            SavedFormat::new("l0w", QuantTarget::I16(255), Layout::Normal),
            SavedFormat::new("l0b", QuantTarget::I16(255), Layout::Normal),
            SavedFormat::new("l1w", QuantTarget::I16(64), Layout::Normal),
            SavedFormat::new("l1b", QuantTarget::I16(64 * 255), Layout::Normal),
            SavedFormat::new("l2w", QuantTarget::Float, Layout::Normal),
            SavedFormat::new("l2b", QuantTarget::Float, Layout::Normal),
            SavedFormat::new("l3w", QuantTarget::Float, Layout::Normal),
            SavedFormat::new("l3b", QuantTarget::Float, Layout::Normal),
        ],
        false,
    );

    let schedule = TrainingSchedule {
        net_id: "test".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 1024,
            start_superbatch: 1,
            end_superbatch: 1,
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

fn build_network(num_inputs: usize, max_active: usize, num_buckets: usize, hl: usize) -> (Graph, Node) {
    let builder = NetworkBuilder::default();

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
    let mut out = l0.forward_sparse_dual_with_activation(stm, nstm, Activation::CReLU);

    out = out.pairwise_mul_post_affine_dual();
    out = l1.forward(out).select(buckets);

    let skip_neuron = out.slice_rows(15, 16);
    out = out.slice_rows(0, 15);

    out = out.concat(out.activate(Activation::Square));
    out = out.activate(Activation::CReLU);

    out = l2.forward(out).select(buckets).activate(Activation::SCReLU);
    out = l3.forward(out).select(buckets);

    let pst_out = pst.matmul(stm) - pst.matmul(nstm);
    out = out + skip_neuron + pst_out;

    let pred = out.activate(Activation::Sigmoid);
    pred.mse(targets);

    // graph, output node
    let output_node = out.node();
    (builder.build(ExecutionContext::default()), output_node)
}
