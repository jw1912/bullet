use crate::{
    default::{Layout, SavedFormat},
    logger,
    nn::{
        optimiser::{self, OptimiserType},
        GraphCompileArgs, InitSettings, NetworkBuilder,
    },
    trainer::save::QuantTarget,
    Activation, ExecutionContext, Shape,
};

use super::{
    inputs::SparseInputType,
    outputs::{self, OutputBuckets},
    AdditionalTrainerInputs, Trainer,
};

use bullet_core::optimiser::Optimiser;

#[derive(Clone, Copy, Debug)]
pub enum Loss {
    None,
    SigmoidMSE,
    SigmoidMPE(f32),
    SoftmaxCrossEntropy,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OpType {
    Activate(Activation),
    ActivateDual,
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
    ft_init_input_size: Option<usize>,
    output_bucket_ft_biases: bool,
    profile_ft: bool,
    compile_args: GraphCompileArgs,
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
            ft_init_input_size: None,
            output_bucket_ft_biases: false,
            profile_ft: false,
            compile_args: GraphCompileArgs::default(),
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

    pub fn output_bucket_ft_biases(mut self) -> Self {
        assert!(U::BUCKETS > 1);
        self.output_bucket_ft_biases = true;
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

    /// Adds SF-style dual activation
    pub fn add_dual_activation(self) -> Self {
        let size = self.get_last_layer_size() * 2;
        self.add(size, OpType::ActivateDual)
    }

    /// Adds a PSQT subnet directly from inputs to output.
    /// The PSQT weights will be placed **before** all other network weights.
    pub fn psqt_subnet(mut self) -> Self {
        self.psqt_subnet = true;
        self
    }

    pub fn track_ft_profile(mut self) -> Self {
        self.profile_ft = true;
        self
    }

    pub fn with_ft_init_input_size(mut self, size: usize) -> Self {
        assert!(size > 0);
        self.ft_init_input_size = Some(size);
        self
    }

    pub fn set_compile_args(mut self, args: GraphCompileArgs) -> Self {
        self.compile_args = args;
        self
    }

    fn push_saved_format(&self, layer: usize, shape: Shape, saved_format: &mut Vec<SavedFormat>, net_quant: &mut i16) {
        let w = format!("l{layer}w");
        let b = format!("l{layer}b");

        let layout = if self.allow_transpose && layer > 0 && U::BUCKETS > 1 {
            Layout::Transposed(shape)
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
        let mut builder = NetworkBuilder::default();

        let output_buckets = U::BUCKETS > 1;

        let input_getter = self.input_getter.clone().expect("Need to set the input features!");
        let input_size = input_getter.num_inputs();
        let input_shape = Shape::new(input_size, 1);

        let mut out = builder.new_sparse_input("stm", input_shape, input_getter.max_active());
        let buckets = output_buckets.then(|| builder.new_sparse_input("buckets", Shape::new(U::BUCKETS, 1), 1));
        let l0 = builder.new_affine_custom(
            "l0",
            input_size,
            self.ft_out_size,
            if self.output_bucket_ft_biases { U::BUCKETS } else { 1 },
        );

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

        self.push_saved_format(0, l0.weights.shape(), &mut saved_format, &mut net_quant);

        assert!(self.nodes.len() > 1, "Require at least 2 nodes for a working arch!");

        if self.perspective {
            let ntm = builder.new_sparse_input("nstm", input_shape, input_getter.max_active());

            if self.output_bucket_ft_biases {
                out = l0.forward_sparse_dual_with_activation_and_bias_buckets(
                    out,
                    ntm,
                    buckets.unwrap(),
                    Activation::Identity,
                );
            } else {
                out = l0.forward_sparse_dual_with_activation(out, ntm, Activation::Identity);
            }
        } else {
            out = l0.forward(out);
        }

        let profile_node = self.profile_ft.then(|| out.node());

        let mut layer = 1;
        let mut layer_sizes = Vec::new();
        let mut prev_size = self.ft_out_size * if self.perspective { 2 } else { 1 };

        for &NodeType { size, op } in self.nodes.iter() {
            match op {
                OpType::Activate(activation) => out = out.activate(activation),
                OpType::ActivateDual => {
                    out = out.concat(out.activate(Activation::Square)).activate(Activation::CReLU);
                    prev_size = size;
                }
                OpType::Affine => {
                    still_in_ft = false;
                    let raw_size = size * U::BUCKETS;

                    let l = builder.new_affine(&format!("l{layer}"), prev_size, raw_size);

                    self.push_saved_format(layer, l.weights.shape(), &mut saved_format, &mut net_quant);

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

                    still_in_ft = false;

                    prev_size /= 2;
                }
            }
        }

        assert!(!still_in_ft);

        if let Some(pst) = pst {
            out = out + pst;
        }

        let output_node = out.node();
        let output_size = prev_size;
        let targets = builder.new_dense_input("targets", Shape::new(output_size, 1));
        match self.loss {
            Loss::None => panic!("No loss function specified!"),
            Loss::SigmoidMSE => out.activate(Activation::Sigmoid).mse(targets),
            Loss::SigmoidMPE(power) => out.activate(Activation::Sigmoid).mpe(targets, power),
            Loss::SoftmaxCrossEntropy => out.softmax_crossentropy_loss(targets),
        };

        #[allow(clippy::default_constructed_unit_structs)]
        let ctx = ExecutionContext::default();
        builder.set_compile_args(self.compile_args);
        let mut graph = builder.build(ctx);

        if let Some(size) = self.ft_init_input_size {
            let stdev = 1.0 / (size as f32).sqrt();
            graph.get_weights_mut("l0w").seed_random(0.0, stdev, true).unwrap();
        }

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

        if let Some(node) = profile_node {
            graph.profile_node(node, "Profile");
        }

        let trainer = Trainer {
            optimiser: Optimiser::new(graph, Default::default()).unwrap(),
            input_getter: input_getter.clone(),
            output_getter: self.bucket_getter,
            output_node,
            additional_inputs: AdditionalTrainerInputs { wdl: output_size == 3 },
            saved_format: saved_format.clone(),
            factorised_weights,
        };

        logger::clear_colours();
        println!("{}", logger::ansi("Built Trainer", "34;1"));
        println!("Architecture           : {}", logger::ansi(format!("{ft_desc} -> {output_desc}"), "32;1"));
        println!("Inputs                 : {}", input_getter.description());

        let num_params = trainer.optimiser.graph.get_num_params();
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
