use crate::{
    ir::NodeId,
    model::{Layout, MType, ModelOperation},
    tensor::{
        DType, DValue, IRTrace, TensorIR,
        operation::{
            CABinary, Power, Unary,
            autograd::{CustomAutogradOp, PassThrough, SoftmaxCrossEntropyLoss},
        },
    },
};

#[derive(Clone, Copy, Debug)]
pub struct PointwiseUnary(pub MType, pub Unary);
impl ModelOperation for PointwiseUnary {
    fn opname(&self) -> String {
        format!("Unary.{:?}", self.1)
    }

    fn inputs(&self) -> Vec<MType> {
        vec![self.0]
    }

    fn output(&self) -> MType {
        self.0
    }

    fn lower(&self, _batch_size: usize, lower: &mut TensorIR, inputs: Vec<NodeId>) -> Result<NodeId, IRTrace> {
        lower.add_unary(inputs[0], self.1)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PointwiseBinary(pub MType, pub CABinary);
impl ModelOperation for PointwiseBinary {
    fn opname(&self) -> String {
        format!("Binary.{:?}", self.1)
    }

    fn inputs(&self) -> Vec<MType> {
        vec![self.0, self.0]
    }

    fn output(&self) -> MType {
        self.0
    }

    fn lower(&self, _batch_size: usize, lower: &mut TensorIR, inputs: Vec<NodeId>) -> Result<NodeId, IRTrace> {
        lower.add_binary(inputs[0], inputs[1], self.1)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ReLU(pub MType);
impl ModelOperation for ReLU {
    fn opname(&self) -> String {
        "ReLU".into()
    }

    fn inputs(&self) -> Vec<MType> {
        vec![self.0]
    }

    fn output(&self) -> MType {
        self.0
    }

    fn lower(&self, batch_size: usize, lower: &mut TensorIR, inputs: Vec<NodeId>) -> Result<NodeId, IRTrace> {
        let ttype = self.0.ttype(batch_size);
        let zero = lower.add_scalar(DValue::zero(ttype.dtype()), ttype.size());
        lower.add_binary(inputs[0], zero, CABinary::Max)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CReLU(pub MType);
impl ModelOperation for CReLU {
    fn opname(&self) -> String {
        "CReLU".into()
    }

    fn inputs(&self) -> Vec<MType> {
        vec![self.0]
    }

    fn output(&self) -> MType {
        self.0
    }

    fn lower(&self, batch_size: usize, lower: &mut TensorIR, inputs: Vec<NodeId>) -> Result<NodeId, IRTrace> {
        let ttype = self.0.ttype(batch_size);
        let zero = lower.add_scalar(DValue::zero(ttype.dtype()), ttype.size());
        let one = lower.add_scalar(DValue::one(ttype.dtype()), ttype.size());
        let relu = lower.add_binary(inputs[0], zero, CABinary::Max)?;
        lower.add_binary(relu, one, CABinary::Min)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SCReLU(pub MType);
impl ModelOperation for SCReLU {
    fn opname(&self) -> String {
        "SCReLU".into()
    }

    fn inputs(&self) -> Vec<MType> {
        vec![self.0]
    }

    fn output(&self) -> MType {
        self.0
    }

    fn lower(&self, batch_size: usize, lower: &mut TensorIR, inputs: Vec<NodeId>) -> Result<NodeId, IRTrace> {
        let crelu = CReLU(self.0).lower(batch_size, lower, inputs)?;
        lower.add_binary(crelu, crelu, CABinary::Mul)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SqrReLU(pub MType);
impl ModelOperation for SqrReLU {
    fn opname(&self) -> String {
        "SqrReLU".into()
    }

    fn inputs(&self) -> Vec<MType> {
        vec![self.0]
    }

    fn output(&self) -> MType {
        self.0
    }

    fn lower(&self, batch_size: usize, lower: &mut TensorIR, inputs: Vec<NodeId>) -> Result<NodeId, IRTrace> {
        let relu = ReLU(self.0).lower(batch_size, lower, inputs)?;
        lower.add_binary(relu, relu, CABinary::Mul)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Sigmoid(pub MType);
impl ModelOperation for Sigmoid {
    fn opname(&self) -> String {
        "Sigmoid".into()
    }

    fn inputs(&self) -> Vec<MType> {
        vec![self.0]
    }

    fn output(&self) -> MType {
        self.0
    }

    fn lower(&self, batch_size: usize, lower: &mut TensorIR, inputs: Vec<NodeId>) -> Result<NodeId, IRTrace> {
        let ttype = self.0.ttype(batch_size);
        let one = lower.add_scalar(DValue::one(ttype.dtype()), ttype.size());
        let neg_one = lower.add_scalar(DValue::neg_one(ttype.dtype()), ttype.size());
        let neg = lower.add_binary(inputs[0], neg_one, CABinary::Mul)?;
        let exp = lower.add_unary(neg, Unary::Exp)?;
        let denom = lower.add_binary(one, exp, CABinary::Add)?;
        lower.add_unary(denom, Unary::Reciprocal)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Reshape(MType, MType);

impl Reshape {
    pub fn new(ty: MType, rows: usize, cols: usize) -> Self {
        let new_ty = MType { rows, cols, ..ty };
        assert_eq!(ty.single_size(), new_ty.single_size(), "Invalid reshape {ty:?} -> {new_ty:?}");
        Self(ty, new_ty)
    }
}

impl ModelOperation for Reshape {
    fn opname(&self) -> String {
        format!("Reshape<{}, {}>", self.1.rows, self.1.cols)
    }

    fn inputs(&self) -> Vec<MType> {
        vec![self.0]
    }

    fn output(&self) -> MType {
        self.1
    }

    fn lower(&self, _batch_size: usize, _lower: &mut TensorIR, inputs: Vec<NodeId>) -> Result<NodeId, IRTrace> {
        Ok(inputs[0])
    }
}

#[derive(Debug, PartialEq)]
pub struct FauxQuantise(pub MType, pub DValue, pub bool);

impl ModelOperation for FauxQuantise {
    fn opname(&self) -> String {
        "FauxQuantise".to_string()
    }

    fn inputs(&self) -> Vec<MType> {
        vec![self.0]
    }

    fn output(&self) -> MType {
        self.0
    }

    fn lower(&self, batch_size: usize, lower: &mut TensorIR, inputs: Vec<NodeId>) -> Result<NodeId, IRTrace> {
        let [input] = inputs[..] else { return Err("Invalid number of inputs!".into()) };
        let op = if self.2 { Unary::Round } else { Unary::Truncate };
        let scalar = lower.add_scalar(self.1, self.0.ttype(batch_size).size());
        let denom = lower.add_unary(scalar, Unary::Reciprocal)?;
        let mul = lower.add_binary(scalar, input, CABinary::Mul)?;
        let int = lower.add_unary(mul, op)?;
        lower.add_binary(int, denom, CABinary::Mul)
    }
}

#[derive(Debug, PartialEq)]
pub struct SoftmaxCrossEntropy(pub MType);

impl ModelOperation for SoftmaxCrossEntropy {
    fn opname(&self) -> String {
        "SoftmaxCrossEntropy".to_string()
    }

    fn inputs(&self) -> Vec<MType> {
        assert_eq!(self.0.layout, Layout::Dense(DType::F32));
        vec![self.0, self.0]
    }

    fn output(&self) -> MType {
        self.0
    }

    fn lower(&self, batch_size: usize, lower: &mut TensorIR, inputs: Vec<NodeId>) -> Result<NodeId, IRTrace> {
        let batch = self.0.cols * if self.0.batch { batch_size } else { 1 };
        let op = SoftmaxCrossEntropyLoss { batch_size: batch.into(), axis_size: self.0.rows };

        lower.add_op(inputs, CustomAutogradOp::new(op)).map(|x| x[0])
    }
}

#[derive(Debug, PartialEq)]
pub struct AbsPower(pub MType);

impl ModelOperation for AbsPower {
    fn opname(&self) -> String {
        "AbsPower".to_string()
    }

    fn inputs(&self) -> Vec<MType> {
        assert_eq!(self.0.layout, Layout::Dense(DType::F32));
        vec![self.0, self.0]
    }

    fn output(&self) -> MType {
        self.0
    }

    fn lower(&self, batch_size: usize, lower: &mut TensorIR, inputs: Vec<NodeId>) -> Result<NodeId, IRTrace> {
        let abs = lower.add_unary(inputs[0], Unary::Abs)?;
        lower.add_op([abs, inputs[1]], Ok::<_, IRTrace>(Power(self.0.ttype(batch_size).size()))).map(|x| x[0])
    }
}

#[derive(Debug, PartialEq)]
pub struct ClipPassThroughGrad(pub MType, pub f32, pub f32);

impl ModelOperation for ClipPassThroughGrad {
    fn opname(&self) -> String {
        "ClipPassThroughGrad".to_string()
    }

    fn inputs(&self) -> Vec<MType> {
        assert_eq!(self.0.layout, Layout::Dense(DType::F32));
        vec![self.0]
    }

    fn output(&self) -> MType {
        self.0
    }

    fn lower(&self, batch_size: usize, lower: &mut TensorIR, inputs: Vec<NodeId>) -> Result<NodeId, IRTrace> {
        let ClipPassThroughGrad(ty, min, max) = *self;
        let op = PassThrough(
            ty.ttype(batch_size),
            Box::new(move |x| {
                let size = x.ty().size();
                let min = x.builder().scalar(min, size);
                let max = x.builder().scalar(max, size);
                x.max(min)?.min(max)
            }),
        );

        lower.add_op(inputs, CustomAutogradOp::new(op)).map(|x| x[0])
    }
}
