use bullet_compiler::tensor::{DType, Size, TType};

use crate::kernel::codegen::Stub;

pub fn var_size_stub(size: Size) -> Stub {
    Stub {
        terminal: Vec::new(),
        inputs: Vec::new(),
        outputs: vec![TType::new(size, DType::I32)],
        source: String::new(),
    }
}

pub fn thread_idx_stub(size: Size) -> Stub {
    let mut size_str = format!("{}", size.factor());
    for _ in 0..size.var_power() {
        size_str += " * INPUT1";
    }

    let source = format!(
        "\
        const int idx_in_grid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;\n\
        const int OUTPUT1 = idx_in_grid * blockDim.x + threadIdx.x;\n\
        if (OUTPUT1 >= ({size_str})) return;\
    "
    );

    Stub {
        terminal: Vec::new(),
        inputs: vec![(TType::new(size, DType::I32), false)],
        outputs: vec![TType::new(size, DType::I32)],
        source,
    }
}
