use bullet_compiler::tensor::{DType, TType};

use crate::kernel::codegen::Stub;

fn dtype_str(dtype: DType) -> &'static str {
    match dtype {
        DType::I32 => "int",
        DType::F32 => "float",
    }
}

pub struct ComputeStub {
    inputs: Vec<TType>,
    outputs: Vec<TType>,
    source: String,
}

impl ComputeStub {
    pub fn unary(ty: TType, op: &str) -> Self {
        Self { inputs: vec![ty], outputs: vec![ty], source: format!("const {} OUTPUT1 = {op};", dtype_str(ty.dtype())) }
    }

    pub fn binary(ty: TType, op: &str) -> Self {
        Self {
            inputs: vec![ty, ty],
            outputs: vec![ty],
            source: format!("const {} OUTPUT1 = {op};", dtype_str(ty.dtype())),
        }
    }
}

impl From<ComputeStub> for Stub {
    fn from(value: ComputeStub) -> Self {
        let ComputeStub { inputs, outputs, source } = value;

        Stub { terminal: Vec::new(), inputs: inputs.into_iter().map(|x| (x, false)).collect(), outputs, source }
    }
}
