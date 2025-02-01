use crate::{
    default::{Layout, SavedFormat},
    frontend::NetworkBuilder,
    logger,
    nn::InitSettings,
    optimiser::{self, Optimiser, OptimiserType},
    rng,
    trainer::save::QuantTarget,
    Activation, ExecutionContext, Shape,
};

use super::{
    inputs::SparseInputType,
    outputs::{self, OutputBuckets},
    AdditionalTrainerInputs, Trainer,
};

use bullet_backend::SparseMatrix;

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

    fn push_saved_format(&self, layer: usize, saved_format: &mut Vec<SavedFormat>, net_quant: &mut i16) {
        let w = format!("l{layer}w");
        let b = format!("l{layer}b");

        let layout = if self.allow_transpose && layer > 0 && U::BUCKETS > 1 {
            Layout::Transposed
        } else {
            Layout::Normal
        };

        let (wquant, bquant) = if let Some(quants) = &self.quantisations {
            let bquant = match quants[layer] {
                QuantTarget::Float => {
                    *net_quant = 1;
                    QuantTarget::Float
                }
                QuantTarget::I16(q) => {
                    *net_quant = net_quant.checked_mul(q).expect("Bias quantisation factor overflowed!");
                    QuantTarget::I16(*net_quant)
                }
                QuantTarget::I8(q) => {
                    *net_quant = net_quant.checked_mul(q).expect("Bias quantisation factor overflowed!");
                    QuantTarget::I8(*net_quant)
                }
                QuantTarget::I32(_) => unimplemented!("i32 quant is not implemented for TrainerBuilder!"),
            };

            (quants[layer], bquant)
        } else {
            (QuantTarget::Float, QuantTarget::Float)
        };

        saved_format.push(SavedFormat { id: w, quant: wquant, layout });
        saved_format.push(SavedFormat { id: b, quant: bquant, layout: Layout::Normal });
    }

    pub fn build(self) -> Trainer<O::Optimiser, T, U> {
        let builder = NetworkBuilder::default();

        let output_buckets = U::BUCKETS > 1;

        let input_getter = self.input_getter.clone().expect("Need to set the input features!");
        let input_size = input_getter.num_inputs();
        let input_shape = Shape::new(input_size, 1);

        let mut out = builder.new_input("stm", input_shape);
        let targets = builder.new_input("targets", Shape::new(1, 1));
        let buckets = output_buckets.then(|| builder.new_input("buckets", Shape::new(U::BUCKETS, 1)));
        let l0 = builder.new_affine("l0", input_size, self.ft_out_size);

        let mut still_in_ft = true;
        let mut saved_format = Vec::new();

        if self.ft_out_size % 8 != 0 {
            warning(|| {
                println!("Feature transformer size = {}", self.ft_out_size);
                println!("   is not a multiple of 8.");
                println!("   Why are you doing this?");
                println!("      Please seek help.");
            });
        }

        let mut net_quant = 1i16;
        let mut ft_desc = format!("{} -> {}", input_getter.shorthand(), self.ft_out_size);

        if self.perspective {
            ft_desc = format!("({ft_desc})x2");
        }

        let pst = self.psqt_subnet.then(|| {
            let pst = builder.new_weights("pst", Shape::new(1, input_size), InitSettings::Zeroed);
            saved_format.push(SavedFormat { id: "pst".to_string(), quant: QuantTarget::Float, layout: Layout::Normal });
            pst.matmul(out)
        });

        self.push_saved_format(0, &mut saved_format, &mut net_quant);

        assert!(self.nodes.len() > 1, "Require at least 2 nodes for a working arch!");

        let skip = if self.perspective {
            let (skip, activation) = if let OpType::Activate(act) = self.nodes[0].op {
                (1, act)
            } else {
                warning(|| {
                    println!("Feature transformer is not followed");
                    println!("   by an activation function,");
                    println!("  which is probably erreonous");
                });
                (0, Activation::Identity)
            };

            let ntm = builder.new_input("nstm", input_shape);
            out = l0.forward_sparse_dual_with_activation(out, ntm, activation);
            skip
        } else {
            out = l0.forward(out);
            0
        };

        let mut layer = 1;
        let mut layer_sizes = Vec::new();
        let mut prev_size = self.ft_out_size * if self.perspective { 2 } else { 1 };

        for &NodeType { size, op } in self.nodes.iter().skip(skip) {
            match op {
                OpType::Activate(activation) => out = out.activate(activation),
                OpType::Affine => {
                    still_in_ft = false;
                    let raw_size = size * U::BUCKETS;

                    let l = builder.new_affine(&format!("l{layer}"), prev_size, raw_size);

                    self.push_saved_format(layer, &mut saved_format, &mut net_quant);

                    layer += 1;

                    out = l.forward(out);
                    prev_size = size;

                    layer_sizes.push(size);

                    if let Some(buckets) = buckets {
                        out = out.select(buckets);
                    }
                }
                OpType::PairwiseMul => {
                    if still_in_ft && self.perspective {
                        out = out.pairwise_mul_post_affine_dual();
                    } else {
                        out = out.pairwise_mul();
                    }

                    prev_size /= 2;
                }
            }
        }

        assert!(!still_in_ft);

        if let Some(pst) = pst {
            out = out + pst;
        }

        let output_node = out.node();
        let predicted = out.activate(Activation::Sigmoid);

        match self.loss {
            Loss::None => panic!("No loss function specified!"),
            Loss::SigmoidMSE => predicted.mse(targets),
            Loss::SigmoidMPE(power) => predicted.mpe(targets, power),
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

        let factorised_weights = input_getter.is_factorised().then(|| {
            if self.psqt_subnet {
                vec!["l0w".to_string(), "pst".to_string()]
            } else {
                vec!["l0w".to_string()]
            }
        });

        let sparse_scratch_space = SparseMatrix::zeroed(graph.device(), Shape::new(1, 1), 1);

        let mut trainer = Trainer {
            optimiser: O::Optimiser::new(graph, Default::default()),
            input_getter: input_getter.clone(),
            output_getter: self.bucket_getter,
            output_node,
            additional_inputs: AdditionalTrainerInputs {
                nstm: self.perspective,
                output_buckets,
                wdl: false,
                dense_inputs: false,
            },
            saved_format: saved_format.clone(),
            factorised_weights,
            sparse_scratch_space,
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

        let num_params = trainer.optimiser.graph().get_num_params();
        let fmt = if num_params >= 1_000_000 {
            format!("{:.2}m", num_params as f64 / 1_000_000.0)
        } else {
            format!("{:.2}k", num_params as f64 / 1_000.0)
        };
        println!("Number of Weights      : {fmt}");

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

fn warning(mut f: impl FnMut()) {
    logger::set_colour("31");
    println!("==================================");
    f();
    println!("==================================");
    logger::clear_colours();
}
