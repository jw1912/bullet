use std::marker::PhantomData;

use bullet_core::graph::{builder::Shape, ir::args::GraphIRCompileArgs};

use crate::{
    game::{
        inputs::SparseInputType,
        outputs::{OutputBuckets, Single},
    },
    nn::{optimiser::OptimiserType, NetworkBuilder, NetworkBuilderNode},
    trainer::save::SavedFormat,
    ExecutionContext,
};

use super::ValueTrainer;

pub struct ValueTrainerBuilder<O, I, P, Out, L> {
    input_getter: Option<I>,
    saved_format: Option<Vec<SavedFormat>>,
    optimiser: Option<O>,
    compile_args: Option<GraphIRCompileArgs>,
    perspective: PhantomData<P>,
    output_buckets: Out,
    loss_fn: Option<L>,
    factorised: Vec<String>,
    wdl_output: bool,
}

impl<O, I, L> Default for ValueTrainerBuilder<O, I, SinglePerspective, NoOutputBuckets, L> {
    fn default() -> Self {
        Self {
            input_getter: None,
            saved_format: None,
            optimiser: None,
            compile_args: None,
            perspective: PhantomData,
            output_buckets: NoOutputBuckets,
            loss_fn: None,
            wdl_output: false,
            factorised: Vec::new(),
        }
    }
}

impl<O, I, P, Out, L> ValueTrainerBuilder<O, I, P, Out, L>
where
    I: SparseInputType,
    O: OptimiserType,
    L: for<'a> Fn(Nbn<'a>, Nbn<'a>) -> Nbn<'a>,
{
    pub fn inputs(mut self, inputs: I) -> Self {
        assert!(self.input_getter.is_none(), "Inputs already set!");
        self.input_getter = Some(inputs);
        self
    }

    pub fn optimiser(mut self, optimiser: O) -> Self {
        assert!(self.optimiser.is_none(), "Optimiser already set!");
        self.optimiser = Some(optimiser);
        self
    }

    pub fn compile_args(mut self, args: GraphIRCompileArgs) -> Self {
        self.compile_args = Some(args);
        self
    }

    pub fn wdl_output(mut self) -> Self {
        self.wdl_output = true;
        self
    }

    pub fn save_format(mut self, fmt: &[SavedFormat]) -> Self {
        assert!(self.saved_format.is_none(), "Save format already set!");
        self.saved_format = Some(fmt.to_vec());
        self
    }

    pub fn loss_fn(mut self, f: L) -> Self {
        assert!(self.loss_fn.is_none(), "Loss function already set!");
        self.loss_fn = Some(f);
        self
    }

    pub fn mark_input_factorised(mut self, list: &[&str]) -> Self {
        for id in list {
            self.factorised.push(id.to_string());
        }

        self
    }

    fn build_internal<F>(self, f: F) -> ValueTrainer<O::Optimiser, I, Out::Inner>
    where
        F: for<'a> Fn(usize, usize, &'a NetworkBuilder) -> NetworkBuilderNode<'a>,
        Out: Bucket,
        Out::Inner: OutputBuckets<I::RequiredDataType>,
    {
        let input_getter = self.input_getter.expect("Need to set inputs!");
        let saved_format = self.saved_format.expect("Need to set save format!");
        let buckets = self.output_buckets.inner();
        let loss = self.loss_fn.expect("Loss function not specified!");

        let inputs = input_getter.num_inputs();
        let nnz = input_getter.max_active();

        let mut builder = NetworkBuilder::default();

        if let Some(args) = self.compile_args {
            builder.set_compile_args(args);
        }

        let output_size = if self.wdl_output { 3 } else { 1 };
        let targets = builder.new_dense_input("targets", Shape::new(output_size, 1));
        let out = f(inputs, nnz, &builder);
        let output_node = out.node();

        let _ = loss(out, targets);
        let graph = builder.build(ExecutionContext::default());

        let mut trainer =
            ValueTrainer::new(graph, output_node, Default::default(), input_getter, buckets, saved_format, false);

        let factorised = self.factorised.iter().map(String::as_str).collect::<Vec<_>>();
        if !factorised.is_empty() {
            trainer.mark_weights_as_input_factorised(&factorised);
        }

        trainer
    }
}

pub trait Bucket {
    type Inner;

    fn inner(self) -> Self::Inner;
}

pub struct NoOutputBuckets;
impl Bucket for NoOutputBuckets {
    type Inner = Single;

    fn inner(self) -> Self::Inner {
        Single
    }
}

pub struct OutputBucket<T>(T);
impl<T> Bucket for OutputBucket<T> {
    type Inner = T;

    fn inner(self) -> Self::Inner {
        self.0
    }
}

pub struct SinglePerspective;
pub struct DualPerspective;

impl<O, I, Out, L> ValueTrainerBuilder<O, I, SinglePerspective, Out, L>
where
    I: SparseInputType,
    O: OptimiserType,
{
    pub fn single_perspective(self) -> Self {
        self
    }

    pub fn dual_perspective(self) -> ValueTrainerBuilder<O, I, DualPerspective, Out, L> {
        ValueTrainerBuilder {
            input_getter: self.input_getter,
            saved_format: self.saved_format,
            optimiser: self.optimiser,
            compile_args: self.compile_args,
            perspective: PhantomData,
            output_buckets: self.output_buckets,
            loss_fn: self.loss_fn,
            factorised: self.factorised,
            wdl_output: self.wdl_output,
        }
    }
}

impl<O, I, P, L> ValueTrainerBuilder<O, I, P, NoOutputBuckets, L>
where
    I: SparseInputType,
    O: OptimiserType,
{
    pub fn output_buckets<Out: OutputBuckets<I::RequiredDataType>>(
        self,
        buckets: Out,
    ) -> ValueTrainerBuilder<O, I, P, OutputBucket<Out>, L> {
        assert!(Out::BUCKETS > 1, "The output bucket type must have more than 1 bucket!");

        ValueTrainerBuilder {
            input_getter: self.input_getter,
            saved_format: self.saved_format,
            optimiser: self.optimiser,
            compile_args: self.compile_args,
            perspective: self.perspective,
            output_buckets: OutputBucket(buckets),
            loss_fn: self.loss_fn,
            factorised: self.factorised,
            wdl_output: self.wdl_output,
        }
    }
}

type Nb<'a> = &'a NetworkBuilder;
type Nbn<'a> = NetworkBuilderNode<'a>;

impl<O, I, L> ValueTrainerBuilder<O, I, SinglePerspective, NoOutputBuckets, L>
where
    I: SparseInputType,
    O: OptimiserType,
    L: for<'a> Fn(Nbn<'a>, Nbn<'a>) -> Nbn<'a>,
{
    pub fn build<F>(self, f: F) -> ValueTrainer<O::Optimiser, I, Single>
    where
        F: for<'a> Fn(Nb<'a>, Nbn<'a>) -> Nbn<'a>,
    {
        self.build_internal(|inputs, nnz, builder| {
            let stm = builder.new_sparse_input("stm", Shape::new(inputs, 1), nnz);
            f(builder, stm)
        })
    }
}

impl<O, I, L> ValueTrainerBuilder<O, I, DualPerspective, NoOutputBuckets, L>
where
    I: SparseInputType,
    O: OptimiserType,
    L: for<'a> Fn(Nbn<'a>, Nbn<'a>) -> Nbn<'a>,
{
    pub fn build<F>(self, f: F) -> ValueTrainer<O::Optimiser, I, Single>
    where
        F: for<'a> Fn(Nb<'a>, Nbn<'a>, Nbn<'a>) -> Nbn<'a>,
    {
        self.build_internal(|inputs, nnz, builder| {
            let stm = builder.new_sparse_input("stm", Shape::new(inputs, 1), nnz);
            let ntm = builder.new_sparse_input("nstm", Shape::new(inputs, 1), nnz);
            f(builder, stm, ntm)
        })
    }
}

impl<O, I, Out, L> ValueTrainerBuilder<O, I, SinglePerspective, OutputBucket<Out>, L>
where
    I: SparseInputType,
    O: OptimiserType,
    Out: OutputBuckets<I::RequiredDataType>,
    L: for<'a> Fn(Nbn<'a>, Nbn<'a>) -> Nbn<'a>,
{
    pub fn build<F>(self, f: F) -> ValueTrainer<O::Optimiser, I, Out>
    where
        F: for<'a> Fn(Nb<'a>, Nbn<'a>, Nbn<'a>) -> Nbn<'a>,
    {
        self.build_internal(|inputs, nnz, builder| {
            let stm = builder.new_sparse_input("stm", Shape::new(inputs, 1), nnz);
            let buckets = builder.new_sparse_input("buckets", Shape::new(Out::BUCKETS, 1), 1);
            f(builder, stm, buckets)
        })
    }
}

impl<O, I, Out, L> ValueTrainerBuilder<O, I, DualPerspective, OutputBucket<Out>, L>
where
    I: SparseInputType,
    O: OptimiserType,
    Out: OutputBuckets<I::RequiredDataType>,
    L: for<'a> Fn(Nbn<'a>, Nbn<'a>) -> Nbn<'a>,
{
    pub fn build<F>(self, f: F) -> ValueTrainer<O::Optimiser, I, Out>
    where
        F: for<'a> Fn(Nb<'a>, Nbn<'a>, Nbn<'a>, Nbn<'a>) -> Nbn<'a>,
    {
        self.build_internal(|inputs, nnz, builder| {
            let stm = builder.new_sparse_input("stm", Shape::new(inputs, 1), nnz);
            let ntm = builder.new_sparse_input("nstm", Shape::new(inputs, 1), nnz);
            let buckets = builder.new_sparse_input("buckets", Shape::new(Out::BUCKETS, 1), 1);
            f(builder, stm, ntm, buckets)
        })
    }
}
