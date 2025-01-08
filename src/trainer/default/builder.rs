use crate::{
    default::{Layout, SavedFormat},
    logger, operations,
    optimiser::{self, Optimiser, OptimiserType},
    rng,
    tensor::{Operation, SparseMatrix},
    trainer::save::QuantTarget,
    Activation, ExecutionContext, GraphBuilder, Shape,
};

use super::{
    inputs::SparseInputType,
    outputs::{self, OutputBuckets},
    AdditionalTrainerInputs, Trainer,
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
    input_getter: Option<T>,
    bucket_getter: U,
    ft_out_size: usize,
    nodes: Vec<NodeType>,
    quantisations: Option<Vec<QuantTarget>>,
    perspective: bool,
    loss: Loss,
    optimiser: O,
    psqt_subnet: bool,
    allow_transpose: bool,
}

impl<T: SparseInputType, U: OutputBuckets<T::RequiredDataType>, O: OptimiserType> Default for TrainerBuilder<T, U, O> {
    fn default() -> Self {
        Self {
            input_getter: None,
            bucket_getter: U::default(),
            ft_out_size: 0,
            nodes: Vec::new(),
            quantisations: None,
            perspective: true,
            loss: Loss::None,
            optimiser: O::default(),
            psqt_subnet: false,
            allow_transpose: true,
        }
    }
}

impl<T: SparseInputType, U: OutputBuckets<T::RequiredDataType>, O: OptimiserType> TrainerBuilder<T, U, O> {
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
        assert!(self.input_getter.is_none(), "Cannot set the input features more than once!");
        self.input_getter = Some(input_getter);
        self
    }

    /// Sets the output buckets.
    pub fn output_buckets(mut self, bucket_getter: U) -> Self {
        self.bucket_getter = bucket_getter;
        self
    }

    /// Provide a list of quantisations.
    pub fn quantisations(mut self, quants: &[i16]) -> Self {
        assert!(self.quantisations.is_none(), "Quantisations already set!");
        self.quantisations = Some(quants.iter().map(|&x| QuantTarget::I16(x)).collect());
        self
    }

    /// Provide a list of quantisations.
    pub fn advanced_quantisations(mut self, quants: &[QuantTarget]) -> Self {
        assert!(self.quantisations.is_none(), "Quantisations already set!");
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

    pub fn disallow_transpose_in_quantised_network(mut self) -> Self {
        self.allow_transpose = false;
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

    /// Adds a PSQT subnet directly from inputs to output.
    /// The PSQT weights will be placed **before** all other network weights.
    pub fn psqt_subnet(mut self) -> Self {
        self.psqt_subnet = true;
        self
    }

    pub fn build(self) -> Trainer<O::Optimiser, T, U> {
        let mut builder = GraphBuilder::default();

        let output_buckets = U::BUCKETS > 1;

        let input_getter = self.input_getter.expect("Need to set the input features!");

        let input_size = input_getter.num_inputs();
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

        //let input_buckets = self.input_getter.buckets();
        let mut ft_desc = format!("{} -> {}", input_getter.shorthand(), self.ft_out_size);

        if self.perspective {
            ft_desc = format!("({ft_desc})x2");
        }

        let mut out = builder.create_input("stm", input_shape);

        let pst = if self.psqt_subnet {
            let pst = builder.create_weights("pst", Shape::new(1, input_size));
            saved_format.push(SavedFormat { id: "pst".to_string(), quant: QuantTarget::Float, layout: Layout::Normal });
            Some(operations::matmul(&mut builder, pst, out))
        } else {
            None
        };

        let mut push_saved_format = |layer: usize| {
            let w = format!("l{layer}w");
            let b = format!("l{layer}b");

            if let Some(quants) = &self.quantisations {
                let layout = if self.allow_transpose && layer > 0 && output_buckets {
                    Layout::Transposed
                } else {
                    Layout::Normal
                };

                saved_format.push(SavedFormat { id: w, quant: quants[layer], layout });

                match quants[layer] {
                    QuantTarget::Float => {
                        net_quant = 1;
                        saved_format.push(SavedFormat { id: b, quant: QuantTarget::Float, layout: Layout::Normal });
                    }
                    QuantTarget::I16(q) => {
                        net_quant = net_quant.checked_mul(q).expect("Bias quantisation factor overflowed!");
                        saved_format.push(SavedFormat {
                            id: b,
                            quant: QuantTarget::I16(net_quant),
                            layout: Layout::Normal,
                        });
                    }
                    QuantTarget::I8(q) => {
                        net_quant = net_quant.checked_mul(q).expect("Bias quantisation factor overflowed!");
                        saved_format.push(SavedFormat {
                            id: b,
                            quant: QuantTarget::I8(net_quant),
                            layout: Layout::Normal,
                        });
                    }
                    QuantTarget::I32(_) => unimplemented!("i32 quant is not implemented for TrainerBuilder!"),
                }
            } else {
                saved_format.push(SavedFormat { id: w, quant: QuantTarget::Float, layout: Layout::Normal });
                saved_format.push(SavedFormat { id: b, quant: QuantTarget::Float, layout: Layout::Normal });
            }
        };

        push_saved_format(0);

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

        if let Some(pst) = pst {
            out = operations::add(&mut builder, out, pst);
        }

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

        let factorised_weights = if input_getter.is_factorised() {
            let mut f = vec!["l0w".to_string()];

            if self.psqt_subnet {
                f.push("pst".to_string());
            }

            Some(f)
        } else {
            None
        };

        let mut trainer = Trainer {
            optimiser: O::Optimiser::new(graph, Default::default()),
            input_getter: input_getter.clone(),
            output_getter: self.bucket_getter,
            output_node: out,
            additional_inputs: AdditionalTrainerInputs {
                nstm: self.perspective,
                output_buckets,
                wdl: false,
                dense_inputs: false,
            },
            saved_format: saved_format.clone(),
            factorised_weights,
            sparse_scratch_space: SparseMatrix::default(),
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

        logger::clear_colours();
        println!("{}", logger::ansi("Built Trainer", "34;1"));
        println!("Architecture           : {}", logger::ansi(format!("{ft_desc} -> {output_desc}"), "32;1"));
        println!("Inputs                 : {}", input_getter.description());

        if input_getter.is_factorised() {
            println!("Factoriser             : Will be merged in quantised network for you");
        }

        if output_buckets {
            if self.allow_transpose {
                println!("Output Buckets         : Will be transposed in quantised network for you, output bucketed layers will");
                println!("                       : have weights in form [[[T; layer input size]; layer output size]; buckets]")
            } else {
                println!("Output Buckets         : Will **not** be transposed in quantised network for you, output bucketed layers will");
                println!("                       : have weights in form [[[T; layer output size]; buckets]; layer input size]")
            }
        }

        if let Some(quantisations) = self.quantisations {
            print!("Quantisations          : [");

            for (i, q) in quantisations.iter().enumerate() {
                if i != 0 {
                    print!(", ");
                }

                let q = match *q {
                    QuantTarget::I16(x) => i32::from(x),
                    QuantTarget::I8(x) => i32::from(x),
                    QuantTarget::I32(x) => x,
                    QuantTarget::Float => 1,
                };

                print!("{}", logger::ansi(q.to_string(), "31"));
            }

            println!("]");
        }

        trainer
    }
}
