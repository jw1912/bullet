use crate::{
    logger, operations,
    optimiser::{self, Optimiser, OptimiserType},
    rng,
    tensor::Operation,
    trainer::default::{quant::QuantTarget, AdditionalTrainerInputs},
    Activation, ExecutionContext, GraphBuilder, Shape,
};

use super::{
    inputs::InputType,
    outputs::{self, OutputBuckets},
    Trainer,
};

#[derive(Clone, Copy, Debug)]
pub enum Loss {
    None,
    SigmoidMSE,
    SigmoidMPE(f32),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OpType {
    Activate(Activation),
    Affine,
    PairwiseMul,
}

struct NodeType {
    size: usize,
    op: OpType,
}

pub struct TrainerBuilder<T, U = outputs::Single, O = optimiser::AdamW> {
    input_getter: T,
    bucket_getter: U,
    ft_out_size: usize,
    nodes: Vec<NodeType>,
    quantisations: Option<Vec<i16>>,
    perspective: bool,
    loss: Loss,
    optimiser: O,
}

impl<T: InputType, U: OutputBuckets<T::RequiredDataType>, O: OptimiserType> Default for TrainerBuilder<T, U, O> {
    fn default() -> Self {
        Self {
            input_getter: T::default(),
            bucket_getter: U::default(),
            ft_out_size: 0,
            nodes: Vec::new(),
            quantisations: None,
            perspective: true,
            loss: Loss::None,
            optimiser: O::default(),
        }
    }
}

impl<T: InputType, U: OutputBuckets<T::RequiredDataType>, O: OptimiserType> TrainerBuilder<T, U, O> {
    fn get_last_layer_size(&self) -> usize {
        if let Some(node) = self.nodes.last() {
            node.size
        } else {
            self.ft_out_size * if self.perspective { 2 } else { 1 }
        }
    }

    /// Makes the first layer single-perspective.
    pub fn single_perspective(mut self) -> Self {
        if !self.nodes.is_empty() {
            panic!("You need to set 'single_perspective' before adding any layers!");
        }
        self.perspective = false;
        self
    }

    /// Sets the optimiser.
    pub fn optimiser(mut self, optimiser: O) -> Self {
        self.optimiser = optimiser;
        self
    }

    /// Sets the input featureset.
    pub fn input(mut self, input_getter: T) -> Self {
        self.input_getter = input_getter;
        self
    }

    /// Sets the output buckets.
    pub fn output_buckets(mut self, bucket_getter: U) -> Self {
        self.bucket_getter = bucket_getter;
        self
    }

    /// Provide a list of quantisations.
    pub fn quantisations(mut self, quants: &[i16]) -> Self {
        self.quantisations = Some(quants.to_vec());
        self
    }

    /// Sets the size of the feature-transformer.
    /// Must be done before all other layers.
    pub fn feature_transformer(mut self, size: usize) -> Self {
        assert!(self.nodes.is_empty());
        self.ft_out_size = size;
        self
    }

    fn add(mut self, size: usize, op: OpType) -> Self {
        assert_ne!(
            self.ft_out_size,
            0,
            "You must start the network with a feature transformer to transform the sparse inputs into a dense embedding!"
        );
        self.nodes.push(NodeType { size, op });

        self
    }

    /// Performs an affine transform without output size `size`.
    pub fn add_layer(self, size: usize) -> Self {
        let mut two_in_a_row = false;

        if let Some(layer) = self.nodes.last() {
            if layer.op == OpType::Affine {
                two_in_a_row = true;
            }
        } else {
            two_in_a_row = true;
        }

        if two_in_a_row {
            panic!(
                "Two affine transforms in a row is equivalent to a single affine transform! This is clearly erronous!"
            );
        }

        self.add(size, OpType::Affine)
    }

    pub fn loss_fn(mut self, loss: Loss) -> Self {
        self.loss = loss;
        self
    }

    /// Reduces a layer of size `2N` to one of size `N` by splitting it in half
    /// and performing the elementwise product of the two halves.
    pub fn add_pairwise_mul(self) -> Self {
        let ll_size = self.get_last_layer_size();
        assert_eq!(ll_size % 2, 0, "You can only perform paiwise mul on a layer with an even number of neurons!",);
        let size = ll_size / 2;
        self.add(size, OpType::PairwiseMul)
    }

    /// Applies the given activation function.
    pub fn activate(self, activation: Activation) -> Self {
        let size = self.get_last_layer_size();
        self.add(size, OpType::Activate(activation))
    }

    pub fn build(self) -> Trainer<O::Optimiser, T, U> {
        let mut builder = GraphBuilder::default();

        let output_buckets = U::BUCKETS > 1;

        let input_size = self.input_getter.size();
        let input_shape = Shape::new(input_size, 1);
        let targets = builder.create_input("targets", Shape::new(1, 1));

        let buckets =
            if output_buckets { Some(builder.create_input("buckets", Shape::new(U::BUCKETS, 1))) } else { None };

        let mut still_in_ft = true;

        let mut saved_format = Vec::new();

        if self.ft_out_size % 8 != 0 {
            logger::set_colour("31");
            println!("==================================");
            println!("  Feature transformer size = {}", self.ft_out_size);
            println!("     is not a multiple of 8.");
            println!("     Why are you doing this?");
            println!("        Please seek help.");
            println!("==================================");
            logger::clear_colours();
        }

        let l0w = builder.create_weights("l0w", Shape::new(self.ft_out_size, input_size));
        let l0b = builder.create_weights("l0b", Shape::new(self.ft_out_size, 1));

        let mut net_quant = 1i16;

        let mut push_saved_format = |layer: usize| {
            let w = format!("l{layer}w");
            let b = format!("l{layer}b");

            if let Some(quants) = &self.quantisations {
                net_quant *= quants[layer];
                saved_format.push((w, QuantTarget::I16(quants[layer])));
                saved_format.push((b, QuantTarget::I16(net_quant)));
            } else {
                saved_format.push((w, QuantTarget::Float));
                saved_format.push((b, QuantTarget::Float));
            }
        };

        let input_buckets = self.input_getter.buckets();
        let mut ft_desc = if input_buckets > 1 {
            format!("{}x{input_buckets} -> {}", self.input_getter.inputs(), self.ft_out_size)
        } else {
            format!("{} -> {}", self.input_getter.inputs(), self.ft_out_size)
        };

        if self.perspective {
            ft_desc = format!("({ft_desc})x2");
        }

        push_saved_format(0);

        let mut out = builder.create_input("stm", input_shape);

        assert!(self.nodes.len() > 1, "Require at least 2 nodes for a working arch!");

        let (skip, activation) = if self.perspective {
            if let NodeType { op: OpType::Activate(act), .. } = self.nodes[0] {
                (1, act)
            } else {
                (0, Activation::Identity)
            }
        } else {
            (0, Activation::Identity)
        };

        out = if self.perspective {
            let ntm = builder.create_input("nstm", input_shape);
            builder.create_result_of_operation(Operation::SparseAffineDual(activation), &[l0w, out, ntm, l0b])
        } else {
            operations::affine(&mut builder, l0w, out, l0b)
        };

        let mut layer = 1;

        let mut layer_sizes = Vec::new();

        let mut prev_size = self.ft_out_size * if self.perspective { 2 } else { 1 };

        for &NodeType { size, op } in self.nodes.iter().skip(skip) {
            match op {
                OpType::Activate(activation) => {
                    out = operations::activate(&mut builder, out, activation);
                }
                OpType::Affine => {
                    still_in_ft = false;
                    let raw_size = size * U::BUCKETS;

                    let w = builder.create_weights(&format!("l{layer}w"), Shape::new(raw_size, prev_size));
                    let b = builder.create_weights(&format!("l{layer}b"), Shape::new(raw_size, 1));

                    push_saved_format(layer);

                    layer += 1;

                    out = operations::affine(&mut builder, w, out, b);
                    prev_size = size;

                    layer_sizes.push(size);

                    if let Some(buckets) = buckets {
                        out = operations::select(&mut builder, out, buckets);
                    }
                }
                OpType::PairwiseMul => {
                    if still_in_ft && self.perspective {
                        out = builder.create_result_of_operation(Operation::PairwiseMul(true), &[out]);
                    } else {
                        out = operations::pairwise_mul(&mut builder, out);
                    }

                    prev_size /= 2;
                }
            }
        }

        assert!(!still_in_ft);

        let predicted = operations::activate(&mut builder, out, Activation::Sigmoid);

        match self.loss {
            Loss::None => panic!("No loss function specified!"),
            Loss::SigmoidMSE => operations::mse(&mut builder, predicted, targets),
            Loss::SigmoidMPE(power) => operations::mpe(&mut builder, predicted, targets, power),
        };

        let ctx = ExecutionContext::default();
        let graph = builder.build(ctx);

        let mut output_desc = format!("{}", layer_sizes[0]);

        for size in layer_sizes.iter().skip(1) {
            output_desc.push_str(&format!(" -> {size}"));
        }

        if output_buckets {
            if layer_sizes.len() == 1 {
                output_desc = format!("{output_desc}x{}", U::BUCKETS);
            } else {
                output_desc = format!("({output_desc})x{}", U::BUCKETS);
            }
        }

        let mut trainer = Trainer {
            optimiser: O::Optimiser::new(graph, Default::default()),
            input_getter: self.input_getter,
            output_getter: self.bucket_getter,
            output_node: out,
            additional_inputs: AdditionalTrainerInputs {
                nstm: self.perspective,
                output_buckets,
                wdl: false,
                dense_inputs: false,
            },
            saved_format,
            arch_description: Some(format!("{ft_desc} -> {output_desc}")),
        };

        let graph = trainer.optimiser.graph_mut();

        for l in 0..layer {
            let w = graph.get_weights_mut(&format!("l{l}w"));
            let shape = w.values.shape();
            let stdev = 1.0 / (shape.cols() as f32).sqrt();

            let wv = rng::vec_f32(w.values.shape().size(), 0.0, stdev, true);
            w.load_from_slice(&wv);
            let wb = rng::vec_f32(w.values.shape().rows(), 0.0, stdev, true);
            graph.get_weights_mut(&format!("l{l}b")).load_from_slice(&wb);
        }

        trainer
    }
}
