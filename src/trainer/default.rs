mod builder;
pub mod cutechess;
mod loader;
pub mod testing;

pub use builder::{Loss, TrainerBuilder};
pub use loader::DefaultDataPreparer;

use loader::DefaultDataLoader;

use std::collections::HashSet;

use diffable::Node;

use crate::{
    inputs::InputType,
    loader::DataLoader,
    logger,
    lr::LrScheduler,
    optimiser::Optimiser,
    outputs::{self, OutputBuckets},
    trainer::NetworkTrainer,
    wdl::WdlScheduler,
    Graph, LocalSettings, TrainingSchedule,
};

pub struct Trainer<Opt, Inp, Out = outputs::Single> {
    optimiser: Opt,
    input_getter: Inp,
    output_getter: Out,
    output_node: Node,
    perspective: bool,
    output_buckets: bool,
}

impl<Opt: Optimiser, Inp: InputType, Out: OutputBuckets<Inp::RequiredDataType>> NetworkTrainer
    for Trainer<Opt, Inp, Out>
{
    type Optimiser = Opt;
    type PreparedData = DefaultDataPreparer<Inp, Out>;

    fn load_batch(&mut self, prepared: &Self::PreparedData) -> usize {
        let batch_size = prepared.batch_size;

        let graph = self.optimiser.graph_mut();

        let input = &prepared.stm;
        graph.get_input_mut("stm").load_sparse_from_slice(input.shape, input.max_active, &input.value);

        if self.perspective {
            let input = &prepared.nstm;
            graph.get_input_mut("nstm").load_sparse_from_slice(input.shape, input.max_active, &input.value);
        }

        if self.output_buckets {
            let input = &prepared.buckets;
            graph.get_input_mut("buckets").load_sparse_from_slice(input.shape, input.max_active, &input.value);
        }

        graph.get_input_mut("targets").load_dense_from_slice(prepared.targets.shape, &prepared.targets.value);

        batch_size
    }

    fn optimiser(&self) -> &Self::Optimiser {
        &self.optimiser
    }

    fn optimiser_mut(&mut self) -> &mut Self::Optimiser {
        &mut self.optimiser
    }
}

impl<Opt: Optimiser, Inp: InputType, Out: OutputBuckets<Inp::RequiredDataType>> Trainer<Opt, Inp, Out> {
    pub fn new(graph: Graph, output_node: Node, params: Opt::Params, input_getter: Inp, output_getter: Out) -> Self {
        let inputs = graph.input_ids();
        let inputs = inputs.iter().map(String::as_str).collect::<HashSet<_>>();

        assert!(inputs.contains("stm"), "Graph does not contain stm inputs!");
        assert!(inputs.contains("targets"), "Graph does not contain targets!");

        let perspective = inputs.contains("nstm");
        let output_buckets = inputs.contains("buckets");
        let expected = 2 + usize::from(perspective) + usize::from(output_buckets);

        if inputs.len() != expected {
            println!("WARNING: The network graph contains an unexpected number of inputs!")
        };

        Self {
            optimiser: Opt::new(graph, params),
            input_getter,
            output_getter,
            output_node,
            perspective,
            output_buckets,
        }
    }

    pub fn run<D: DataLoader<Inp::RequiredDataType>, LR: LrScheduler, WDL: WdlScheduler>(
        &mut self,
        schedule: &TrainingSchedule<LR, WDL>,
        settings: &LocalSettings,
        data_loader: &D,
    ) {
        logger::clear_colours();
        println!("{}", logger::ansi("Beginning Training", "34;1"));
        schedule.display();
        settings.display();
        let preparer = DefaultDataLoader::new(self.input_getter, self.output_getter, schedule.eval_scale, data_loader.clone());

        self.train_custom(&preparer, schedule, settings, |_, _, _, _| {});
    }

    pub fn eval(&mut self, fen: &str) -> f32
    where
        Inp::RequiredDataType: std::str::FromStr<Err = String>,
    {
        let pos = format!("{fen} | 0 | 0.0").parse::<Inp::RequiredDataType>().unwrap();

        let prepared = DefaultDataPreparer::prepare(self.input_getter, self.output_getter, &[pos], 1, 1.0, 1.0);

        self.load_batch(&prepared);
        self.optimiser.graph_mut().forward();

        let eval = self.optimiser.graph().get_node(self.output_node);

        let mut val = vec![0.0; eval.values.dense().allocated_size()];
        eval.values.dense().write_to_slice(&mut val);
        val[0]
    }

    pub fn set_optimiser_params(&mut self, params: Opt::Params) {
        self.optimiser.set_params(params);
    }
}
