use crate::{
    ir::NodeId,
    model::{MType, ModelOperation, ModelIR},
    tensor::{
        DValue, IRTrace, TensorIR,
        operation::{CABinary, Unary},
    },
};

#[derive(Clone, Copy, Debug)]
pub struct PointwiseUnary(pub MType, pub Unary);
impl ModelOperation for PointwiseUnary {
    fn opname(&self) -> String {
        format!("pointwise.unary.{:?}", self.1).to_lowercase()
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

    fn gradient(
        &self,
        _ir: &mut ModelIR,
        _inputs: Vec<NodeId>,
        _output_grad: NodeId,
    ) -> Result<Vec<Option<NodeId>>, IRTrace> {
        unimplemented!()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PointwiseBinary(pub MType, pub CABinary);
impl ModelOperation for PointwiseBinary {
    fn opname(&self) -> String {
        format!("pointwise.binary.{:?}", self.1).to_lowercase()
    }

    fn inputs(&self) -> Vec<MType> {
        vec![self.0]
    }

    fn output(&self) -> MType {
        self.0
    }

    fn lower(&self, _batch_size: usize, lower: &mut TensorIR, inputs: Vec<NodeId>) -> Result<NodeId, IRTrace> {
        lower.add_binary(inputs[0], inputs[1], self.1)
    }

    fn gradient(
        &self,
        _ir: &mut ModelIR,
        _inputs: Vec<NodeId>,
        _output_grad: NodeId,
    ) -> Result<Vec<Option<NodeId>>, IRTrace> {
        unimplemented!()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ReLU(pub MType);
impl ModelOperation for ReLU {
    fn opname(&self) -> String {
        "pointwise.relu".into()
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

    fn gradient(
        &self,
        ir: &mut ModelIR,
        inputs: Vec<NodeId>,
        output_grad: NodeId,
    ) -> Result<Vec<Option<NodeId>>, IRTrace> {
        let output = ir.add_op([inputs[0]], *self)?;
        let derivative = ir.add_op([output], PointwiseUnary(self.0, Unary::IsPositive))?;
        let grad = ir.add_op([derivative, output_grad], PointwiseBinary(self.0, CABinary::Mul))?;
        Ok(vec![Some(grad)])
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CReLU(pub MType);
impl ModelOperation for CReLU {
    fn opname(&self) -> String {
        "pointwise.crelu".into()
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

    fn gradient(
        &self,
        _ir: &mut ModelIR,
        _inputs: Vec<NodeId>,
        _output_grad: NodeId,
    ) -> Result<Vec<Option<NodeId>>, IRTrace> {
        unimplemented!()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SCReLU(pub MType);
impl ModelOperation for SCReLU {
    fn opname(&self) -> String {
        "pointwise.crelu".into()
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

    fn gradient(
        &self,
        _ir: &mut ModelIR,
        _inputs: Vec<NodeId>,
        _output_grad: NodeId,
    ) -> Result<Vec<Option<NodeId>>, IRTrace> {
        unimplemented!()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Sigmoid(pub MType);
impl ModelOperation for Sigmoid {
    fn opname(&self) -> String {
        "pointwise.crelu".into()
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
        let denom = lower.add_binary(one, neg, CABinary::Add)?;
        lower.add_unary(denom, Unary::Reciprocal)
    }

    fn gradient(
        &self,
        _ir: &mut ModelIR,
        _inputs: Vec<NodeId>,
        _output_grad: NodeId,
    ) -> Result<Vec<Option<NodeId>>, IRTrace> {
        unimplemented!()
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
        "pointwise.reshape".into()
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

    fn gradient(
        &self,
        _ir: &mut ModelIR,
        _inputs: Vec<NodeId>,
        output_grad: NodeId,
    ) -> Result<Vec<Option<NodeId>>, IRTrace> {
        Ok(vec![Some(output_grad)])
    }
}

#[derive(Debug, PartialEq)]
pub struct FauxQuantise(pub MType, pub DValue, pub bool);

impl ModelOperation for FauxQuantise {
    fn opname(&self) -> String {
        "faux-quantise".to_string()
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

    fn gradient(
        &self,
        _ir: &mut ModelIR,
        _inputs: Vec<NodeId>,
        output_grad: NodeId,
    ) -> Result<Vec<Option<NodeId>>, IRTrace> {
        Ok(vec![Some(output_grad)])
    }
}
