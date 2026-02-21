use bullet_compiler::{
    ir::Operation,
    tensor::{
        DType, DValue, Size,
        operation::{CABinary, Unary},
    },
};

use super::PType;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemIO {
    pub buf_ty: DType,
    pub p2size: u8,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PointwiseOp {
    Buffer(DType, Size),
    Read(MemIO),
    Write(MemIO),
    AtomicAdd(MemIO),
    Unary { ty: DType, p2size: u8, op: Unary },
    Binary { ty: DType, p2size: u8, op: CABinary },
    Constant { value: DValue, p2size: u8 },
    ThreadId,
    VarSize,
}

impl Operation<PType> for PointwiseOp {
    fn opname(&self) -> String {
        "something".to_string()
    }

    fn inputs(&self) -> Vec<PType> {
        match *self {
            Self::Read(io) => {
                vec![PType::Pointer(io.buf_ty), PType::Variable { ty: DType::I32, p2size: 0 }]
            }
            Self::Write(io) | Self::AtomicAdd(io) => {
                vec![
                    PType::Pointer(io.buf_ty),
                    PType::Variable { ty: DType::I32, p2size: 0 },
                    PType::Variable { ty: io.buf_ty, p2size: io.p2size },
                ]
            }
            Self::Unary { ty, p2size, .. } => vec![PType::Variable { ty, p2size }],
            Self::Binary { ty, p2size, .. } => vec![PType::Variable { ty, p2size }, PType::Variable { ty, p2size }],
            Self::ThreadId => vec![PType::Variable { ty: DType::I32, p2size: 0 }],
            Self::Constant { .. } | Self::Buffer { .. } | Self::VarSize => Vec::new(),
        }
    }

    fn outputs(&self) -> Vec<PType> {
        match *self {
            Self::Buffer(ty, _) => vec![PType::Pointer(ty)],
            Self::Read(io) => vec![PType::Variable { ty: io.buf_ty, p2size: io.p2size }],
            Self::Write(_) | Self::AtomicAdd(_) => Vec::new(),
            Self::Unary { ty, p2size, op } => {
                let ty = match op {
                    Unary::Cast(ty) => ty,
                    Unary::Sgn | Unary::Abs => ty,
                    _ => (ty != DType::I32).then_some(ty).unwrap(),
                };

                vec![PType::Variable { ty, p2size }]
            }
            Self::Binary { ty, p2size, .. } => vec![PType::Variable { ty, p2size }],
            Self::Constant { value, p2size } => vec![PType::Variable { ty: value.dtype(), p2size }],
            Self::ThreadId | Self::VarSize => vec![PType::Variable { ty: DType::I32, p2size: 0 }],
        }
    }
}
