use bullet_compiler::tensor::{DType, Size, TType};

use crate::kernel::codegen::Stub;

pub fn read_stub(ty: TType, size: Size) -> Stub {
    let dtype = match ty.dtype() {
        DType::F32 => "float",
        DType::I32 => "int",
    };

    Stub {
        terminal: Vec::new(),
        inputs: vec![(ty, true), (TType::new(size, DType::I32), false)],
        outputs: vec![TType::new(size, ty.dtype())],
        source: format!("const {dtype} OUTPUT1 = INPUT1[INPUT2];"),
    }
}

pub fn write_stub(ty: TType) -> Stub {
    Stub {
        terminal: vec![0],
        inputs: vec![(ty, true), (ty, false), (TType::new(ty.size(), DType::I32), false)],
        outputs: Vec::new(),
        source: "INPUT1[INPUT3] = INPUT2;".to_string(),
    }
}
