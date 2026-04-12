use std::num::NonZeroU8;

use bullet_compiler::{
    ir::Operation,
    tensor::{
        DType, DValue, Size,
        operation::{CABinary, Unary},
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PType {
    Pointer(DType),
    Variable { ty: DType, p2size: u8 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemIO {
    pub buf_ty: DType,
    pub p2size: u8,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PointwiseOp {
    Buffer(DType, Size),
    Read(MemIO),
    ConditionalRead(MemIO, DValue),
    Write(MemIO),
    #[allow(unused)]
    AtomicAdd(MemIO),
    Unary {
        ty: DType,
        p2size: u8,
        op: Unary,
    },
    Binary {
        ty: DType,
        p2size: u8,
        op: CABinary,
    },
    Power {
        p2size: u8,
    },
    Constant {
        value: DValue,
        p2size: u8,
    },
    ThreadId,
    VarSize,
    Div,
    Rem,
    EvalSize(Size),
    Broadcast(DType, NonZeroU8),
    SpMM {
        nnz: usize,
        rows: usize,
        cols: usize,
        ty: DType,
        p2size: u8,
    },
    SpMMT {
        nnz: usize,
        rows: usize,
        cols: usize,
        ty: DType,
    },
}

impl PointwiseOp {
    pub fn is_unique(&self) -> bool {
        matches!(self, Self::Buffer(_, _) | Self::AtomicAdd(_))
    }
}

impl Operation<PType> for PointwiseOp {
    fn opname(&self) -> String {
        format!("{self:?}").to_lowercase()
    }

    fn inputs(&self) -> Vec<PType> {
        match *self {
            Self::Read(io) => {
                vec![PType::Pointer(io.buf_ty), PType::Variable { ty: DType::I32, p2size: 0 }]
            }
            Self::ConditionalRead(io, _) => {
                vec![
                    PType::Pointer(io.buf_ty),
                    PType::Variable { ty: DType::I32, p2size: 0 },
                    PType::Variable { ty: DType::I32, p2size: 0 },
                ]
            }
            Self::Write(io) | Self::AtomicAdd(io) => {
                vec![
                    PType::Pointer(io.buf_ty),
                    PType::Variable { ty: DType::I32, p2size: 0 },
                    PType::Variable { ty: io.buf_ty, p2size: io.p2size },
                ]
            }
            Self::Unary { ty, p2size, .. } => vec![PType::Variable { ty, p2size }],
            Self::Binary { ty, p2size, .. } => vec![PType::Variable { ty, p2size }; 2],
            Self::Power { p2size } => vec![PType::Variable { ty: DType::F32, p2size }; 2],
            Self::ThreadId => vec![PType::Variable { ty: DType::I32, p2size: 0 }],
            Self::Constant { .. } | Self::Buffer { .. } | Self::VarSize => Vec::new(),
            Self::Div | Self::Rem => vec![PType::Variable { ty: DType::I32, p2size: 0 }; 2],
            Self::EvalSize(_) => vec![PType::Variable { ty: DType::I32, p2size: 0 }],
            Self::Broadcast(ty, _) => vec![PType::Variable { ty, p2size: 0 }],
            Self::SpMM { ty, .. } => {
                vec![PType::Pointer(ty), PType::Pointer(DType::I32), PType::Variable { ty: DType::I32, p2size: 0 }]
            }
            Self::SpMMT { ty, .. } => vec![
                PType::Pointer(ty),
                PType::Pointer(DType::I32),
                PType::Variable { ty: DType::I32, p2size: 0 },
                PType::Variable { ty, p2size: 0 },
            ],
        }
    }

    fn outputs(&self) -> Vec<PType> {
        match *self {
            Self::Buffer(ty, _) => vec![PType::Pointer(ty)],
            Self::Read(io) => vec![PType::Variable { ty: io.buf_ty, p2size: io.p2size }],
            Self::ConditionalRead(io, _) => vec![PType::Variable { ty: io.buf_ty, p2size: io.p2size }],
            Self::Write(_) | Self::AtomicAdd(_) | Self::SpMMT { .. } => Vec::new(),
            Self::Unary { ty, p2size, op } => {
                let ty = match op {
                    Unary::Cast(ty) => ty,
                    Unary::Sgn | Unary::Abs | Unary::IsNonNegative | Unary::IsPositive | Unary::IsZero => ty,
                    _ => (ty != DType::I32).then_some(ty).unwrap(),
                };

                vec![PType::Variable { ty, p2size }]
            }
            Self::Binary { ty, p2size, .. } => vec![PType::Variable { ty, p2size }],
            Self::Power { p2size } => vec![PType::Variable { ty: DType::F32, p2size }],
            Self::Constant { value, p2size } => vec![PType::Variable { ty: value.dtype(), p2size }],
            Self::ThreadId | Self::VarSize => vec![PType::Variable { ty: DType::I32, p2size: 0 }],
            Self::Div | Self::Rem | Self::EvalSize(_) => vec![PType::Variable { ty: DType::I32, p2size: 0 }],
            Self::Broadcast(ty, p2size) => vec![PType::Variable { ty, p2size: p2size.get() }],
            Self::SpMM { ty, p2size, .. } => vec![PType::Variable { ty, p2size }],
        }
    }
}
