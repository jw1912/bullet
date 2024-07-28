use crate::{
    inputs::InputType,
    optimiser::{Optimiser, OptimiserType},
    outputs::OutputBuckets,
    tensor::{self, DeviceBuffer, DeviceHandles, Shape, SparseTensor, Tensor, TensorBatch},
    Activation,
};

use super::{Affine, FeatureTransformer, Node, Operation, QuantiseInfo, Trainer};

enum OpType {
    Activate(Activation),
    Affine,
    PairwiseShrink,
}

struct NodeType {
    size: usize,
    op: OpType,
    in_res_block: bool,
}

pub struct TrainerBuilder<T, U, O> {
    input_getter: T,
    bucket_getter: U,
    ft_out_size: usize,
    nodes: Vec<NodeType>,
    quantisations: Vec<i32>,
    single_perspective: bool,
    in_res_block: bool,
    size: usize,
    optimiser: O,
}

impl<T: InputType, U: OutputBuckets<T::RequiredDataType>, O: OptimiserType> Default for TrainerBuilder<T, U, O> {
    fn default() -> Self {
        Self {
            input_getter: T::default(),
            bucket_getter: U::default(),
            ft_out_size: 0,
            nodes: Vec::new(),
            quantisations: Vec::new(),
            single_perspective: false,
            in_res_block: false,
            size: 0,
            optimiser: O::default(),
        }
    }
}

impl<T: InputType, U: OutputBuckets<T::RequiredDataType>, O: OptimiserType> TrainerBuilder<T, U, O> {
    fn get_last_layer_size(&self) -> usize {
        if let Some(node) = self.nodes.last() {
            node.size
        } else {
            self.ft_out_size * if self.single_perspective { 1 } else { 2 }
        }
    }

    pub fn single_perspective(mut self) -> Self {
        if !self.nodes.is_empty() {
            panic!("You need to set 'single_perspective' before adding any layers!");
        }
        self.single_perspective = true;
        self
    }

    pub fn optimiser(mut self, optimiser: O) -> Self {
        self.optimiser = optimiser;
        self
    }

    pub fn input(mut self, input_getter: T) -> Self {
        self.input_getter = input_getter;
        self
    }

    pub fn output_buckets(mut self, bucket_getter: U) -> Self {
        self.bucket_getter = bucket_getter;
        self
    }

    pub fn quantisations(mut self, quants: &[i32]) -> Self {
        self.quantisations = quants.to_vec();
        self
    }

    pub fn feature_transformer(mut self, size: usize) -> Self {
        assert!(self.nodes.is_empty());
        self.ft_out_size = size;
        self
    }

    fn add(mut self, size: usize, op: OpType) -> Self {
        self.nodes.push(NodeType { size, op, in_res_block: self.in_res_block });

        self
    }

    pub fn add_layer(mut self, size: usize) -> Self {
        self.size += (self.get_last_layer_size() + 1) * size * U::BUCKETS;
        self.add(size, OpType::Affine)
    }

    pub fn activate(self, activation: Activation) -> Self {
        let size = self.get_last_layer_size();
        self.add(size, OpType::Activate(activation))
    }

    pub fn start_residual_block(mut self) -> Self {
        assert!(!self.in_res_block, "Already in residual block!");
        self.in_res_block = true;
        self
    }

    pub fn end_residual_block(mut self) -> Self {
        assert!(self.in_res_block, "Not in residual block!");
        self.in_res_block = false;
        self
    }

    pub fn build(self) -> Trainer<T, U, O::Optimiser> {
        let inp_getter_size = self.input_getter.size();
        let max_active_inputs = self.input_getter.max_active_inputs();

        let buckets = U::BUCKETS;

        let ft_size = (inp_getter_size + 1) * self.ft_out_size;
        let net_size = self.size + ft_size;

        let opt = O::Optimiser::new(net_size);
        let batch_size = 1;
        let mul = if self.single_perspective { 1 } else { 2 };

        unsafe {
            let ftw_shape = Shape::new(self.ft_out_size, inp_getter_size);
            let ftb_shape = Shape::new(1, self.ft_out_size);
            let fto_shape = Shape::new(1, mul * self.ft_out_size);

            let mut ft = FeatureTransformer {
                weights: Tensor::uninit(ftw_shape),
                biases: Tensor::uninit(ftb_shape),
                weights_grad: Tensor::uninit(ftw_shape),
                biases_grad: Tensor::uninit(ftb_shape),
                single_perspective: self.single_perspective,
                outputs: TensorBatch::new(fto_shape, batch_size),
                copy: TensorBatch::new(fto_shape, batch_size),
            };

            let mut offset = 0;
            ft.weights.set_ptr(opt.weights_offset(offset));
            ft.weights_grad.set_ptr(opt.gradients_offset(offset));
            offset += self.ft_out_size * inp_getter_size;

            ft.biases.set_ptr(opt.weights_offset(offset));
            ft.biases_grad.set_ptr(opt.gradients_offset(offset));
            offset += self.ft_out_size;

            let mut nodes = Vec::new();
            let mut inp_size = mul * self.ft_out_size;

            let mut quantiser = Vec::new();
            let mut qi = 0;
            let mut accq = 1;
            if !self.quantisations.is_empty() {
                quantiser.push(QuantiseInfo { val: self.quantisations[qi], start: 0 });
                accq *= self.quantisations[qi];
                qi += 1;
            }

            for NodeType { size, op, in_res_block } in &self.nodes {
                let size = *size;
                let in_res_block = *in_res_block;

                match op {
                    OpType::Affine => {
                        let raw_size = size * buckets;
                        let wsh = Shape::new(inp_size, raw_size);
                        let bsh = Shape::new(1, raw_size);

                        let ones = DeviceBuffer::new(batch_size);
                        ones.load_from_host(&vec![1.0; batch_size]);
                        let mut affine = Affine {
                            weights: Tensor::uninit(wsh),
                            biases: Tensor::uninit(bsh),
                            weights_grad: Tensor::uninit(wsh),
                            biases_grad: Tensor::uninit(bsh),
                            ones,
                        };

                        affine.weights.set_ptr(opt.weights_offset(offset));
                        affine.weights_grad.set_ptr(opt.gradients_offset(offset));

                        if !self.quantisations.is_empty() {
                            quantiser.push(QuantiseInfo { val: self.quantisations[qi], start: offset });
                        }

                        offset += inp_size * raw_size;

                        affine.biases.set_ptr(opt.weights_offset(offset));
                        affine.biases_grad.set_ptr(opt.gradients_offset(offset));

                        if !self.quantisations.is_empty() {
                            accq *= self.quantisations[qi];
                            quantiser.push(QuantiseInfo { val: accq, start: offset });
                            qi += 1;
                        }

                        offset += raw_size;

                        let outputs = TensorBatch::new(bsh, batch_size);
                        nodes.push(Node { outputs, op: Operation::Affine(affine), in_res_block });

                        if buckets > 1 {
                            nodes.push(Node {
                                outputs: TensorBatch::new(Shape::new(1, size), batch_size),
                                op: Operation::Select,
                                in_res_block,
                            });
                        }
                    }
                    OpType::Activate(activation) => {
                        let bsh = Shape::new(1, size);
                        let outputs = TensorBatch::new(bsh, batch_size);
                        nodes.push(Node { outputs, op: Operation::Activate(*activation), in_res_block });
                    }
                    OpType::PairwiseShrink => {
                        assert!(size % 2 == 0, "Can't apply a pairwise shrink layer to an odd number of neurons!");
                        let bsh = Shape::new(1, size / 2);
                        let outputs = TensorBatch::new(bsh, batch_size);
                        nodes.push(Node { outputs, op: Operation::PairwiseShrink, in_res_block });
                    }
                };

                inp_size = size;
            }

            assert_eq!(qi, self.quantisations.len(), "Incorrectly specified number of quantisations!");
            assert_eq!(offset, net_size);

            let inputs = SparseTensor::uninit(batch_size, inp_getter_size, max_active_inputs);

            let results = TensorBatch::new(Shape::new(1, 1), batch_size);
            let error_device = DeviceBuffer::new(1);

            let trainer = Trainer {
                input_getter: self.input_getter,
                bucket_getter: self.bucket_getter,
                handle: DeviceHandles::default(),
                optimiser: opt,
                ft,
                nodes,
                inputs,
                results,
                error_device,
                error: 0.0,
                validation_error: 0.0,
                error_record: Vec::new(),
                validation_record: Vec::new(),
                ft_reg: 0.0,
                used: 0,
                quantiser,
                buckets: tensor::util::calloc(batch_size),
            };

            trainer.randomise_weights(true, true);

            trainer
        }
    }
}
