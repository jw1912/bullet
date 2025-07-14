use std::marker::PhantomData;

use bullet_core::{graph::builder::Shape, optimiser::Optimiser, trainer::Trainer};

use crate::{
    game::{inputs::SparseInputType, outputs::OutputBuckets},
    nn::{optimiser::OptimiserType, BackendMarker, NetworkBuilder, NetworkBuilderNode},
    trainer::save::SavedFormat,
    value::ValueTrainerState,
    ExecutionContext,
};

use super::{ValueTrainer, B};

type Wgt<I> = fn(&<I as SparseInputType>::RequiredDataType) -> f32;
type LossFn = for<'a> fn(Nbn<'a>, Nbn<'a>) -> Nbn<'a>;

pub struct ValueTrainerBuilder<O, I: SparseInputType, P, Out> {
    input_getter: Option<I>,
    saved_format: Option<Vec<SavedFormat>>,
    optimiser: Option<O>,
    perspective: PhantomData<P>,
    output_buckets: Out,
    blend_getter: B<I>,
    weight_getter: Option<Wgt<I>>,
    loss_fn: Option<LossFn>,
    factorised: Vec<String>,
    wdl_output: bool,
    use_win_rate_model: bool,
}

impl<O, I> Default for ValueTrainerBuilder<O, I, SinglePerspective, NoOutputBuckets>
where
    I: SparseInputType,
{
    fn default() -> Self {
        Self {
            input_getter: None,
            saved_format: None,
            optimiser: None,
            perspective: PhantomData,
            output_buckets: NoOutputBuckets,
            blend_getter: |_, wdl| wdl,
            weight_getter: None,
            loss_fn: None,
            wdl_output: false,
            use_win_rate_model: false,
            factorised: Vec::new(),
        }
    }
}

impl<O, I, P, Out> ValueTrainerBuilder<O, I, P, Out>
where
    I: SparseInputType,
    O: OptimiserType,
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

    pub fn wdl_output(mut self) -> Self {
        self.wdl_output = true;
        self
    }

    pub fn save_format(mut self, fmt: &[SavedFormat]) -> Self {
        assert!(self.saved_format.is_none(), "Save format already set!");
        self.saved_format = Some(fmt.to_vec());
        self
    }

    pub fn loss_fn(mut self, f: LossFn) -> Self {
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

    pub fn wdl_adjust_function(mut self, f: B<I>) -> Self {
        self.blend_getter = f;
        self
    }

    pub fn datapoint_weight_function(mut self, f: Wgt<I>) -> Self {
        assert!(self.weight_getter.is_none(), "Position weight function alrady set!");
        self.weight_getter = Some(f);
        self
    }

    pub fn use_win_rate_model(mut self) -> Self {
        self.use_win_rate_model = true;
        self
    }

    fn build_custom_internal<F>(self, f: F) -> ValueTrainer<O::Optimiser, I, Out::Inner>
    where
        F: for<'a> Fn(usize, usize, Nbn<'a>, Nb<'a>) -> (Nbn<'a>, Nbn<'a>),
        Out: Bucket,
        Out::Inner: OutputBuckets<I::RequiredDataType>,
    {
        let input_getter = self.input_getter.expect("Need to set inputs!");
        let saved_format = self.saved_format.expect("Need to set save format!");
        let buckets = self.output_buckets.inner();

        let inputs = input_getter.num_inputs();
        let nnz = input_getter.max_active();

        let builder = NetworkBuilder::default();

        let output_size = if self.wdl_output { 3 } else { 1 };
        let targets = builder.new_dense_input("targets", Shape::new(output_size, 1));
        let (out, loss) = f(inputs, nnz, targets, &builder);

        if self.weight_getter.is_some() {
            let entry_weights = builder.new_dense_input("entry_weights", Shape::new(1, 1));
            let _ = entry_weights * loss;
        }

        let output_node = out.node();
        let graph = builder.build(ExecutionContext::default());

        ValueTrainer(Trainer {
            optimiser: Optimiser::new(graph, Default::default()).unwrap(),
            state: ValueTrainerState {
                input_getter: input_getter.clone(),
                output_getter: buckets,
                blend_getter: self.blend_getter,
                weight_getter: self.weight_getter,
                output_node,
                use_win_rate_model: self.use_win_rate_model,
                wdl: self.wdl_output,
                saved_format: saved_format.clone(),
            },
        })
    }

    fn build_internal<F>(self, f: F) -> ValueTrainer<O::Optimiser, I, Out::Inner>
    where
        F: for<'a> Fn(usize, usize, Nb<'a>) -> Nbn<'a>,
        Out: Bucket,
        Out::Inner: OutputBuckets<I::RequiredDataType>,
    {
        let loss = self.loss_fn.expect("Loss function not specified!");

        self.build_custom_internal(|inputs, nnz, targets, builder| {
            let out = f(inputs, nnz, builder);

            let raw_loss = loss(out, targets);

            assert_eq!(raw_loss.node().shape, Shape::new(1, 1));

            (out, raw_loss)
        })
    }
}

pub trait Bucket {
    type Inner;

    fn inner(self) -> Self::Inner;
}

#[derive(Clone, Copy, Default)]
pub struct NoOutputBuckets;

impl Bucket for NoOutputBuckets {
    type Inner = Self;

    fn inner(self) -> Self::Inner {
        self
    }
}

impl<T: 'static> OutputBuckets<T> for NoOutputBuckets {
    const BUCKETS: usize = 1;

    fn bucket(&self, _: &T) -> u8 {
        0
    }
}

pub struct OutputBucket<T>(pub T);
impl<T> Bucket for OutputBucket<T> {
    type Inner = T;

    fn inner(self) -> Self::Inner {
        self.0
    }
}

pub struct SinglePerspective;
pub struct DualPerspective;

impl<O, I, Out> ValueTrainerBuilder<O, I, SinglePerspective, Out>
where
    I: SparseInputType,
    O: OptimiserType,
{
    pub fn single_perspective(self) -> Self {
        self
    }

    pub fn dual_perspective(self) -> ValueTrainerBuilder<O, I, DualPerspective, Out> {
        ValueTrainerBuilder {
            input_getter: self.input_getter,
            saved_format: self.saved_format,
            optimiser: self.optimiser,
            perspective: PhantomData,
            output_buckets: self.output_buckets,
            blend_getter: self.blend_getter,
            weight_getter: self.weight_getter,
            loss_fn: self.loss_fn,
            factorised: self.factorised,
            wdl_output: self.wdl_output,
            use_win_rate_model: self.use_win_rate_model,
        }
    }
}

impl<O, I, P> ValueTrainerBuilder<O, I, P, NoOutputBuckets>
where
    I: SparseInputType,
    O: OptimiserType,
{
    pub fn output_buckets<Out: OutputBuckets<I::RequiredDataType>>(
        self,
        buckets: Out,
    ) -> ValueTrainerBuilder<O, I, P, OutputBucket<Out>> {
        assert!(Out::BUCKETS > 1, "The output bucket type must have more than 1 bucket!");

        ValueTrainerBuilder {
            input_getter: self.input_getter,
            saved_format: self.saved_format,
            optimiser: self.optimiser,
            perspective: self.perspective,
            output_buckets: OutputBucket(buckets),
            blend_getter: self.blend_getter,
            weight_getter: self.weight_getter,
            loss_fn: self.loss_fn,
            factorised: self.factorised,
            wdl_output: self.wdl_output,
            use_win_rate_model: self.use_win_rate_model,
        }
    }
}

type Nb<'a> = &'a NetworkBuilder<BackendMarker>;
type Nbn<'a> = NetworkBuilderNode<'a, BackendMarker>;

impl<O, I> ValueTrainerBuilder<O, I, SinglePerspective, NoOutputBuckets>
where
    I: SparseInputType,
    O: OptimiserType,
{
    pub fn build<F>(self, f: F) -> ValueTrainer<O::Optimiser, I, NoOutputBuckets>
    where
        F: for<'a> Fn(Nb<'a>, Nbn<'a>) -> Nbn<'a>,
    {
        self.build_internal(|inputs, nnz, builder| {
            let stm = builder.new_sparse_input("stm", Shape::new(inputs, 1), nnz);
            f(builder, stm)
        })
    }
}

impl<O, I> ValueTrainerBuilder<O, I, DualPerspective, NoOutputBuckets>
where
    I: SparseInputType,
    O: OptimiserType,
{
    pub fn build<F>(self, f: F) -> ValueTrainer<O::Optimiser, I, NoOutputBuckets>
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

impl<O, I, Out> ValueTrainerBuilder<O, I, SinglePerspective, OutputBucket<Out>>
where
    I: SparseInputType,
    O: OptimiserType,
    Out: OutputBuckets<I::RequiredDataType>,
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

impl<O, I, Out> ValueTrainerBuilder<O, I, DualPerspective, OutputBucket<Out>>
where
    I: SparseInputType,
    O: OptimiserType,
    Out: OutputBuckets<I::RequiredDataType>,
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

impl<O, I> ValueTrainerBuilder<O, I, SinglePerspective, NoOutputBuckets>
where
    I: SparseInputType,
    O: OptimiserType,
{
    pub fn build_custom<F>(self, f: F) -> ValueTrainer<O::Optimiser, I, NoOutputBuckets>
    where
        F: for<'a> Fn(Nb<'a>, Nbn<'a>, Nbn<'a>) -> (Nbn<'a>, Nbn<'a>),
    {
        assert!(self.loss_fn.is_none(), "Can't specify loss function separately!");
        self.build_custom_internal(|inputs, nnz, targets, builder| {
            let stm = builder.new_sparse_input("stm", Shape::new(inputs, 1), nnz);
            f(builder, stm, targets)
        })
    }
}

impl<O, I> ValueTrainerBuilder<O, I, DualPerspective, NoOutputBuckets>
where
    I: SparseInputType,
    O: OptimiserType,
{
    pub fn build_custom<F>(self, f: F) -> ValueTrainer<O::Optimiser, I, NoOutputBuckets>
    where
        F: for<'a> Fn(Nb<'a>, (Nbn<'a>, Nbn<'a>), Nbn<'a>) -> (Nbn<'a>, Nbn<'a>),
    {
        assert!(self.loss_fn.is_none(), "Can't specify loss function separately!");
        self.build_custom_internal(|inputs, nnz, targets, builder| {
            let stm = builder.new_sparse_input("stm", Shape::new(inputs, 1), nnz);
            let ntm = builder.new_sparse_input("nstm", Shape::new(inputs, 1), nnz);
            f(builder, (stm, ntm), targets)
        })
    }
}

impl<O, I, Out> ValueTrainerBuilder<O, I, SinglePerspective, OutputBucket<Out>>
where
    I: SparseInputType,
    O: OptimiserType,
    Out: OutputBuckets<I::RequiredDataType>,
{
    pub fn build_custom<F>(self, f: F) -> ValueTrainer<O::Optimiser, I, Out>
    where
        F: for<'a> Fn(Nb<'a>, (Nbn<'a>, Nbn<'a>), Nbn<'a>) -> (Nbn<'a>, Nbn<'a>),
    {
        assert!(self.loss_fn.is_none(), "Can't specify loss function separately!");
        self.build_custom_internal(|inputs, nnz, targets, builder| {
            let stm = builder.new_sparse_input("stm", Shape::new(inputs, 1), nnz);
            let buckets = builder.new_sparse_input("buckets", Shape::new(Out::BUCKETS, 1), 1);
            f(builder, (stm, buckets), targets)
        })
    }
}

impl<O, I, Out> ValueTrainerBuilder<O, I, DualPerspective, OutputBucket<Out>>
where
    I: SparseInputType,
    O: OptimiserType,
    Out: OutputBuckets<I::RequiredDataType>,
{
    pub fn build_custom<F>(self, f: F) -> ValueTrainer<O::Optimiser, I, Out>
    where
        F: for<'a> Fn(Nb<'a>, (Nbn<'a>, Nbn<'a>, Nbn<'a>), Nbn<'a>) -> (Nbn<'a>, Nbn<'a>),
    {
        assert!(self.loss_fn.is_none(), "Can't specify loss function separately!");
        self.build_custom_internal(|inputs, nnz, targets, builder| {
            let stm = builder.new_sparse_input("stm", Shape::new(inputs, 1), nnz);
            let ntm = builder.new_sparse_input("nstm", Shape::new(inputs, 1), nnz);
            let buckets = builder.new_sparse_input("buckets", Shape::new(Out::BUCKETS, 1), 1);
            f(builder, (stm, ntm, buckets), targets)
        })
    }
}
