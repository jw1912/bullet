use std::collections::HashSet;

use diffable::Node;

use crate::{
    inputs::InputType,
    loader::{DataLoader, DefaultDataLoader},
    lr::LrScheduler,
    optimiser::Optimiser,
    outputs::{self, OutputBuckets},
    wdl::WdlScheduler,
    Graph, LocalSettings, TrainingSchedule,
};

use super::{DefaultDataPreparer, NetworkTrainer};

pub struct Trainer<Opt, Inp, Out = outputs::Single> {
    optimiser: Opt,
    input_getter: Inp,
    output_getter: Out,
    output_node: Node,
}

impl<Opt: Optimiser, Inp: InputType, Out: OutputBuckets<Inp::RequiredDataType>> NetworkTrainer
    for Trainer<Opt, Inp, Out>
{
    type Optimiser = Opt;
    type PreparedData = DefaultDataPreparer<Inp, Out>;

    fn load_batch(&mut self, prepared: &Self::PreparedData) -> usize {
        let batch_size = prepared.batch_size;

        let graph = self.optimiser.graph_mut();

        for (id, input) in [("stm", &prepared.stm), ("nstm", &prepared.nstm)] {
            graph.get_input_mut(id).values.load_sparse_from_slice(input.shape, input.max_active, &input.value);
        }

        if Out::BUCKETS > 1 {
            let input = &prepared.buckets;
            graph.get_input_mut("buckets").values.load_sparse_from_slice(input.shape, input.max_active, &input.value);
        }

        graph.get_input_mut("results").values.load_from_slice(prepared.results.shape, &prepared.results.value);

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
        assert!(inputs.contains("nstm"), "Graph does not contain nstm inputs!");
        assert!(inputs.contains("results"), "Graph does not contain targets!");

        let expected = if Out::BUCKETS > 1 {
            assert!(inputs.contains("buckets"), "Graph does not contain output buckets!");
            4
        } else {
            3
        };

        assert_eq!(inputs.len(), expected, "Graph contains wrong number of inputs!");

        Self { optimiser: Opt::new(graph, params), input_getter, output_getter, output_node }
    }

    pub fn train<D: DataLoader<Inp::RequiredDataType>, LR: LrScheduler, WDL: WdlScheduler>(
        &mut self,
        data_loader: D,
        schedule: &TrainingSchedule<LR, WDL>,
        settings: &LocalSettings,
    ) {
        let preparer = DefaultDataLoader::new(
            self.input_getter.clone(),
            self.output_getter.clone(),
            schedule.eval_scale,
            data_loader,
        );

        self.train_custom(&preparer, schedule, settings, |_, _, _, _| {});
    }
}
