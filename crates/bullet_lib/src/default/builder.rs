use crate::{
    default::{Layout, SavedFormat},
    nn::{
        optimiser::{self, OptimiserType},
        InitSettings, NetworkBuilder,
    },
    trainer::{logger, save::QuantTarget},
    Activation, ExecutionContext, Shape,
};

use crate::{
    game::{inputs::SparseInputType, outputs::OutputBuckets},
    value::{
        builder::{Bucket, NoOutputBuckets, OutputBucket},
        loader::B,
    },
};

use super::{AdditionalTrainerInputs, Trainer, Wgt};

use bullet_core::{
    graph::{NodeId, NodeIdTy},
    optimiser::Optimiser,
};

#[derive(Clone, Copy, Debug)]
pub enum Loss {
    None,
    SigmoidMSE,
    SigmoidMPE(f32),
    SoftmaxCrossEntropy,
}

#[derive(Clone, Copy, PartialEq)]
enum OpType {
    Activate(Activation),
    ActivateDual,
    Affine,
    PairwiseMul,
    Scale(f32),
}

struct NodeType {
    size: usize,
    op: OpType,
}

pub struct TrainerBuilder<T: SparseInputType, U, O = optimiser::AdamW> {
    input_getter: Option<T>,
    bucket_getter: U,
    blend_getter: B<T>,
    weight_getter: Option<Wgt<T>>,
    ft_out_size: usize,
    nodes: Vec<NodeType>,
    quantisations: Option<Vec<QuantTarget>>,
    perspective: bool,
    use_win_rate_model: Option<f32>,
    loss: Loss,
    optimiser: O,
    psqt_subnet: bool,
    allow_transpose: bool,
    ft_init_input_size: Option<usize>,
    output_bucket_ft_biases: bool,
    profile_ft: bool,
    quant_round: bool,
}

impl<T: SparseInputType, O: OptimiserType> Default for TrainerBuilder<T, NoOutputBuckets, O> {
    fn default() -> Self {
        Self {
            input_getter: None,
            bucket_getter: NoOutputBuckets,
            blend_getter: |_, wdl| wdl,
            weight_getter: None,
            ft_out_size: 0,
            nodes: Vec::new(),
            quantisations: None,
            perspective: true,
            use_win_rate_model: None,
            loss: Loss::None,
            optimiser: O::default(),
            psqt_subnet: false,
            allow_transpose: true,
            ft_init_input_size: None,
            output_bucket_ft_biases: false,
            profile_ft: false,
            quant_round: false,
        }
    }
}

impl<T: SparseInputType, U, O: OptimiserType> TrainerBuilder<T, U, O> {
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

    /// Round rather than truncate when quantising.
    pub fn round_in_quantisation(mut self) -> Self {
        self.quant_round = true;
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
        self.output_bucket_ft_biases = true;
        self
    }

    pub fn wdl_adjuster(mut self, b: B<T>) -> Self {
        self.blend_getter = b;
        self
    }

    pub fn datapoint_weight_function(mut self, f: Wgt<T>) -> Self {
        assert!(self.weight_getter.is_none(), "Position weight function is already set!");
        self.weight_getter = Some(f);
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

    pub fn use_win_rate_model(mut self, scale: f32) -> Self {
        self.use_win_rate_model = Some(scale);
        self
    }

    /// Applies the given activation function.
    pub fn activate(self, activation: Activation) -> Self {
        let size = self.get_last_layer_size();
        self.add(size, OpType::Activate(activation))
    }

    /// Multiply by `scale`
    pub fn scale(self, scale: f32) -> Self {
        let size = self.get_last_layer_size();
        self.add(size, OpType::Scale(scale))
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
}

impl<T: SparseInputType, O: OptimiserType> TrainerBuilder<T, NoOutputBuckets, O> {
    /// Sets the output buckets.
    pub fn output_buckets<U>(self, bucket_getter: U) -> TrainerBuilder<T, OutputBucket<U>, O>
    where
        U: OutputBuckets<T::RequiredDataType>,
    {
        TrainerBuilder {
            input_getter: self.input_getter,
            bucket_getter: OutputBucket(bucket_getter),
            blend_getter: self.blend_getter,
            weight_getter: self.weight_getter,
            ft_out_size: self.ft_out_size,
            nodes: self.nodes,
            quantisations: self.quantisations,
            perspective: self.perspective,
            use_win_rate_model: self.use_win_rate_model,
            loss: self.loss,
            optimiser: self.optimiser,
            psqt_subnet: self.psqt_subnet,
            allow_transpose: self.allow_transpose,
            ft_init_input_size: self.ft_init_input_size,
            output_bucket_ft_biases: self.output_bucket_ft_biases,
            profile_ft: self.profile_ft,
            quant_round: self.quant_round,
        }
    }
}

impl<T: SparseInputType, U: Bucket, O: OptimiserType> TrainerBuilder<T, U, O>
where
    U::Inner: OutputBuckets<T::RequiredDataType>,
{
    fn push_saved_format(&self, layer: usize, shape: Shape, saved_format: &mut Vec<SavedFormat>, net_quant: &mut i16) {
        let w = format!("l{layer}w");
        let b = format!("l{layer}b");

        let layout = if self.allow_transpose && layer > 0 && U::Inner::BUCKETS > 1 {
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

        let mut wfmt = SavedFormat::new(&w, wquant, layout);
        let mut bfmt = SavedFormat::new(&b, bquant, Layout::Normal);

        if self.quant_round {
            wfmt = wfmt.round();
            bfmt = bfmt.round();
        }

        saved_format.push(wfmt);
        saved_format.push(bfmt);
    }

    pub fn build(self) -> Trainer<O::Optimiser, T, U::Inner> {
        let builder = NetworkBuilder::default();

        let output_buckets = U::Inner::BUCKETS;

        let input_getter = self.input_getter.clone().expect("Need to set the input features!");
        let input_size = input_getter.num_inputs();
        let input_shape = Shape::new(input_size, 1);

        let mut out = builder.new_sparse_input("stm", input_shape, input_getter.max_active());
        let buckets =
            (output_buckets > 1).then(|| builder.new_sparse_input("buckets", Shape::new(output_buckets, 1), 1));
        let l0 = builder.new_affine_custom(
            "l0",
            input_size,
            self.ft_out_size,
            if self.output_bucket_ft_biases { output_buckets } else { 1 },
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
            saved_format.push(SavedFormat::new("pst", QuantTarget::Float, Layout::Normal));
            pst.matmul(out)
        });

        self.push_saved_format(0, l0.weights.node().shape, &mut saved_format, &mut net_quant);

        assert!(self.nodes.len() > 1, "Require at least 2 nodes for a working arch!");

        let apply = |x| {
            if self.output_bucket_ft_biases {
                l0.weights.matmul(x) + l0.bias.matmul(buckets.unwrap())
            } else {
                l0.forward(x)
            }
        };

        out = apply(out);

        if self.perspective {
            let ntm = builder.new_sparse_input("nstm", input_shape, input_getter.max_active());
            out = out.concat(apply(ntm));
        }

        let mut layer = 1;
        let mut layer_sizes = Vec::new();
        let mut prev_size = self.ft_out_size * if self.perspective { 2 } else { 1 };

        for &NodeType { size, op } in self.nodes.iter() {
            match op {
                OpType::Activate(activation) => out = out.activate(activation),
                OpType::ActivateDual => {
                    out = out.concat(out.abs_pow(2.0)).activate(Activation::CReLU);
                    prev_size = size;
                }
                OpType::Affine => {
                    still_in_ft = false;
                    let raw_size = size * output_buckets;

                    let l = builder.new_affine(&format!("l{layer}"), prev_size, raw_size);

                    self.push_saved_format(layer, l.weights.node().shape, &mut saved_format, &mut net_quant);

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
                OpType::Scale(x) => out = x * out,
            }
        }

        assert!(!still_in_ft);

        if let Some(pst) = pst {
            out = out + pst;
        }

        let output_node = out.node();
        let output_size = prev_size;
        let targets = builder.new_dense_input("targets", Shape::new(output_size, 1));

        let raw_loss = if let Some(scale) = self.use_win_rate_model {
            let score = (scale / 340.0) * out;
            let q = score - (270.0 / 340.0);
            let qm = -score - (270.0 / 340.0);
            let wdl = 0.5 * (1.0 + q.sigmoid() - qm.sigmoid());

            match self.loss {
                Loss::SigmoidMSE => wdl.squared_error(targets),
                Loss::SigmoidMPE(power) => wdl.power_error(targets, power),
                _ => panic!("Loss fn incompatible with use_win_rate_model!"),
            }
        } else {
            match self.loss {
                Loss::None => panic!("No loss function specified!"),
                Loss::SigmoidMSE => out.sigmoid().squared_error(targets),
                Loss::SigmoidMPE(power) => out.sigmoid().power_error(targets, power),
                Loss::SoftmaxCrossEntropy => out.softmax_crossentropy_loss(targets),
            }
        };

        if self.weight_getter.is_some() {
            let entry_weights = builder.new_dense_input("entry_weights", Shape::new(1, 1));
            let _ = entry_weights * raw_loss;
        }

        #[allow(clippy::default_constructed_unit_structs)]
        let ctx = ExecutionContext::default();
        let graph = builder.build(ctx);

        if let Some(size) = self.ft_init_input_size {
            let stdev = 1.0 / (size as f32).sqrt();
            let id = NodeId::new(graph.weight_idx("l0w").unwrap(), NodeIdTy::Values);
            graph.get_mut(id).unwrap().seed_random(0.0, stdev, true).unwrap();
        }

        let mut output_desc = format!("{}", layer_sizes[0]);

        for size in layer_sizes.iter().skip(1) {
            output_desc.push_str(&format!(" -> {size}"));
        }

        if output_buckets > 1 {
            if layer_sizes.len() == 1 {
                output_desc = format!("{output_desc}x{output_buckets}");
            } else {
                output_desc = format!("({output_desc})x{output_buckets}");
            }
        }

        let factorised_weights = input_getter.is_factorised().then(|| {
            if self.psqt_subnet {
                vec!["l0w".to_string(), "pst".to_string()]
            } else {
                vec!["l0w".to_string()]
            }
        });

        let trainer = Trainer {
            optimiser: Optimiser::new(graph, Default::default()).unwrap(),
            input_getter: input_getter.clone(),
            output_getter: self.bucket_getter.inner(),
            blend_getter: self.blend_getter,
            weight_getter: self.weight_getter,
            use_win_rate_model: self.use_win_rate_model.is_some(),
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

        println!("Input Layer Layout     : [[T; feature transformer size]; number of inputs]");

        print!("Output Layer Layout    : ");
        if output_buckets > 1 {
            if self.allow_transpose {
                println!("[[[T; layer input size]; layer output size]; output buckets]");
            } else {
                println!("[[[T; layer output size]; output buckets]; layer input size]");
            }
        } else {
            println!("[[T; layer output size]; layer input size]");
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
