use bullet_compiler::tensor::{
    DType, DValue, Size,
    operation::{CABinary, Unary},
};

use crate::pointwise::operations::PointwiseOp;

/// GPU kernel language dialect
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dialect {
    /// CUDA C / HIP C (used by NVIDIA and AMD)
    CudaHip,
    /// Metal Shading Language (used by Apple Silicon)
    Msl,
}

impl Dialect {
    /// Return the dialect matching the currently enabled feature
    pub fn active() -> Self {
        #[cfg(feature = "metal")]
        {
            Self::Msl
        }
        #[cfg(not(feature = "metal"))]
        {
            Self::CudaHip
        }
    }
}

pub fn tystr(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "float",
        DType::I32 => "int",
    }
}

#[allow(dead_code)]
pub fn code_str(op: PointwiseOp, size: Size) -> Option<String> {
    code_str_dialect(op, size, Dialect::active())
}

pub fn code_str_dialect(op: PointwiseOp, size: Size, dialect: Dialect) -> Option<String> {
    let is_msl = dialect == Dialect::Msl;

    // Helper: reinterpret_cast with optional `device` address space qualifier for MSL
    let reinterpret = |ty: &str| {
        if is_msl { format!("reinterpret_cast<device {ty}*>") } else { format!("reinterpret_cast<{ty}*>") }
    };

    // Helper: make_floatN constructor
    let make_vec = |ty: &str, count: u32, vals: &str| {
        if is_msl { format!("{ty}{count}({vals})") } else { format!("make_{ty}{count}({vals})") }
    };

    // Helper: atomic add
    let atomic_add = |ptr_expr: &str, val_expr: &str| {
        if is_msl {
            format!(
                "atomic_fetch_add_explicit((volatile device atomic_float*)({ptr_expr}), {val_expr}, memory_order_relaxed);"
            )
        } else {
            format!("atomicAdd({ptr_expr}, {val_expr});")
        }
    };

    // Helper: powf
    let powf_fn = if is_msl { "pow" } else { "powf" };

    match op {
        PointwiseOp::Buffer { .. } => None,
        PointwiseOp::Div => Some("const int OUT1 = IN1 / IN2;".into()),
        PointwiseOp::Rem => Some("const int OUT1 = IN1 % IN2;".into()),
        PointwiseOp::Read(io) => {
            let ty = tystr(io.buf_ty);
            if io.p2size == 0 {
                Some(format!("const {ty} OUT1 = IN1[IN2];"))
            } else if io.p2size < 3 {
                let sz = 2u32.pow(io.p2size as u32);
                let cast = reinterpret(&format!("{ty}{sz}"));
                Some(format!("const {ty}{sz} OUT1 = {cast}(IN1)[IN2];"))
            } else {
                None
            }
        }
        PointwiseOp::ConditionalRead(io, value) => {
            let ty = tystr(io.buf_ty);
            let vl = match value {
                DValue::F32(x) => format!("{x:E}"),
                DValue::I32(x) => x.to_string(),
            };

            if io.p2size == 0 {
                Some(format!("const {ty} OUT1 = IN3 > 0 ? IN1[IN2] : {vl};"))
            } else if io.p2size < 3 {
                let sz = 2u32.pow(io.p2size as u32);
                let cast = reinterpret(&format!("{ty}{sz}"));
                let vl_vec = if io.p2size == 1 {
                    make_vec(ty, 2, &format!("{vl}, {vl}"))
                } else {
                    make_vec(ty, 4, &format!("{vl}, {vl}, {vl}, {vl}"))
                };
                Some(format!("const {ty}{sz} OUT1 = IN3 > 0 ? {cast}(IN1)[IN2] : {vl_vec};"))
            } else {
                None
            }
        }
        PointwiseOp::Write(io) => {
            let ty = tystr(io.buf_ty);
            if io.p2size == 0 {
                Some("IN1[IN2] = IN3;".to_string())
            } else if io.p2size < 3 {
                let sz = 2u32.pow(io.p2size as u32);
                let cast = reinterpret(&format!("{ty}{sz}"));
                Some(format!("{cast}(IN1)[IN2] = IN3;"))
            } else {
                None
            }
        }
        PointwiseOp::AtomicAdd(io) => {
            let ty = tystr(io.buf_ty);
            if io.p2size == 0 {
                Some(atomic_add("IN1 + IN2", "IN3"))
            } else if io.p2size < 3 {
                let sz = 2u32.pow(io.p2size as u32);
                let cast = reinterpret(&format!("{ty}{sz}"));
                Some(atomic_add(&format!("{cast}(IN1) + IN2"), "IN3"))
            } else {
                None
            }
        }
        PointwiseOp::Constant { value, p2size } => {
            let ty = tystr(value.dtype());
            let vl = match value {
                DValue::F32(x) => format!("{x:E}"),
                DValue::I32(x) => x.to_string(),
            };

            match p2size {
                0 => Some(format!("const {ty} OUT1 = {vl};")),
                1 => Some(format!("const {ty}2 OUT1 = {};", make_vec(ty, 2, &format!("{vl}, {vl}")))),
                2 => Some(format!("const {ty}4 OUT1 = {};", make_vec(ty, 4, &format!("{vl}, {vl}, {vl}, {vl}")))),
                3.. => None,
            }
        }
        PointwiseOp::ThreadId => {
            let size_val = size.get();

            if is_msl {
                Some(format!(
                    "\
                    const int OUT1 = static_cast<int>(metal_tid);\n\
                    if (OUT1 >= ({size_val})) return;\
                "
                ))
            } else {
                Some(format!(
                    "\
                    const int idx_in_grid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;\n\
                    const int OUT1 = idx_in_grid * blockDim.x + threadIdx.x;\n\
                    if (OUT1 >= ({size_val})) return;\
                "
                ))
            }
        }
        PointwiseOp::Broadcast(ty, p2size) => {
            let ty = tystr(ty);
            match p2size.get() {
                1 => Some(format!("const {ty}2 OUT1 = {};", make_vec(ty, 2, "IN1, IN1"))),
                2 => Some(format!("const {ty}4 OUT1 = {};", make_vec(ty, 4, "IN1, IN1, IN1, IN1"))),
                _ => None,
            }
        }
        PointwiseOp::Binary { ty, p2size, op } => {
            let ty = tystr(ty);

            let chars = ['x', 'y', 'z', 'w'];

            let formula = |x, y| match op {
                CABinary::Add => format!("{x} + {y};"),
                CABinary::Mul => format!("{x} * {y};"),
                CABinary::Min => format!("min({x}, {y});"),
                CABinary::Max => format!("max({x}, {y});"),
            };

            match p2size {
                0 => Some(format!("const {ty} OUT1 = {}", formula("IN1".into(), "IN2".into()))),
                3.. => None,
                x => {
                    let sz = 2u32.pow(p2size as u32);
                    let mut s = format!("{ty}{sz} OUT1;");
                    for c in chars.iter().take(2usize.pow(x as u32)) {
                        let ln = formula(format!("IN1.{c}"), format!("IN2.{c}"));
                        s += &format!("OUT1.{c} = {ln}")
                    }
                    Some(s)
                }
            }
        }
        PointwiseOp::Power { p2size } => {
            let chars = ['x', 'y', 'z', 'w'];
            let formula = |x, y| format!("{powf_fn}({x}, {y});");
            match p2size {
                0 => Some(format!("const float OUT1 = {}", formula("IN1".into(), "IN2".into()))),
                3.. => None,
                x => {
                    let sz = 2u32.pow(p2size as u32);
                    let mut s = format!("float{sz} OUT1;");
                    for c in chars.iter().take(2usize.pow(x as u32)) {
                        let ln = formula(format!("IN1.{c}"), format!("IN2.{c}"));
                        s += &format!("OUT1.{c} = {ln}")
                    }
                    Some(s)
                }
            }
        }
        PointwiseOp::Unary { ty, p2size, op } => {
            let ty = tystr(ty);

            let opstr = |x: &str| match op {
                Unary::Sgn => {
                    format!("{x} == {ty}(0) ? {ty}(0) : {x} > {ty}(0) ? {ty}(1) : {ty}(-1)")
                }
                Unary::Reciprocal => format!("1.0F / {x}"),
                Unary::IsPositive => format!("{x} > static_cast<{ty}>(0)"),
                Unary::IsZero => format!("{x} == static_cast<{ty}>(0)"),
                Unary::IsNonNegative => format!("{x} >= static_cast<{ty}>(0)"),
                _ => {
                    let opstr = match op {
                        Unary::Cast(nty) => match nty {
                            DType::F32 => "static_cast<float>",
                            DType::I32 => "static_cast<int>",
                        },
                        Unary::Abs => "abs",
                        Unary::Sin => {
                            if is_msl {
                                "sin"
                            } else {
                                "sinf"
                            }
                        }
                        Unary::Cos => {
                            if is_msl {
                                "cos"
                            } else {
                                "cosf"
                            }
                        }
                        Unary::Tan => {
                            if is_msl {
                                "tan"
                            } else {
                                "tanf"
                            }
                        }
                        Unary::Sinh => {
                            if is_msl {
                                "sinh"
                            } else {
                                "sinhf"
                            }
                        }
                        Unary::Cosh => {
                            if is_msl {
                                "cosh"
                            } else {
                                "coshf"
                            }
                        }
                        Unary::Tanh => {
                            if is_msl {
                                "tanh"
                            } else {
                                "tanhf"
                            }
                        }
                        Unary::Exp => {
                            if is_msl {
                                "exp"
                            } else {
                                "expf"
                            }
                        }
                        Unary::Log => {
                            if is_msl {
                                "log"
                            } else {
                                "logf"
                            }
                        }
                        Unary::Sqrt => {
                            if is_msl {
                                "sqrt"
                            } else {
                                "sqrtf"
                            }
                        }
                        Unary::Round => {
                            if is_msl {
                                "round"
                            } else {
                                "roundf"
                            }
                        }
                        Unary::Truncate => {
                            if is_msl {
                                "trunc"
                            } else {
                                "truncf"
                            }
                        }
                        Unary::Sgn | Unary::Reciprocal | Unary::IsPositive | Unary::IsZero | Unary::IsNonNegative => {
                            unimplemented!()
                        }
                    };

                    format!("{opstr}({x})")
                }
            };

            match p2size {
                0 => Some(format!("const {ty} OUT1 = {};", opstr("IN1"))),
                1 => Some(format!("{ty}2 OUT1;\nOUT1.x = {};\nOUT1.y = {};", opstr("IN1.x"), opstr("IN1.y"),)),
                2 => Some(format!(
                    "{ty}4 OUT1;\nOUT1.x = {};\nOUT1.y = {};\nOUT1.z = {};\nOUT1.w = {};",
                    opstr("IN1.x"),
                    opstr("IN1.y"),
                    opstr("IN1.z"),
                    opstr("IN1.w"),
                )),
                3.. => None,
            }
        }
        PointwiseOp::SpMM { nnz, rows, cols, stride, offset, ty, p2size } => {
            let ty = tystr(ty);
            match p2size {
                0 => Some(format!(
                    "\
                    {ty} OUT1 = 0;
                    int UNIQ1 = IN3 / {rows};
                    int UNIQ2 = {offset} + IN3 % {rows};

                    for (int i = 0; i < {nnz}; i++) {{
                        const int j = IN2[{nnz} * UNIQ1 + i];
                        if (j < 0 || j >= {cols}) break;
                        OUT1 += IN1[j * {stride} + UNIQ2];
                    }}"
                )),
                1 => {
                    assert_eq!(rows % 2, 0);
                    assert_eq!(offset % 2, 0);
                    assert_eq!(stride % 2, 0);
                    let m = rows / 2;
                    let o = offset / 2;
                    let s = stride / 2;
                    let cast = reinterpret(&format!("{ty}2"));
                    let zero_vec = make_vec(ty, 2, "0, 0");
                    Some(format!(
                        "\
                        {ty}2 OUT1 = {zero_vec};
                        int UNIQ1 = IN3 / {m};
                        int UNIQ2 = {o} + IN3 % {m};

                        for (int i = 0; i < {nnz}; i++) {{
                            const int j = IN2[{nnz} * UNIQ1 + i];

                            if (j < 0 || j >= {cols}) break;

                            const {ty}2 a = {cast}(IN1)[j * {s} + UNIQ2];

                            OUT1.x += a.x;
                            OUT1.y += a.y;
                        }}"
                    ))
                }
                2 => {
                    assert_eq!(rows % 4, 0);
                    assert_eq!(offset % 4, 0);
                    assert_eq!(stride % 4, 0);
                    let m = rows / 4;
                    let o = offset / 4;
                    let s = stride / 4;
                    let cast = reinterpret(&format!("{ty}4"));
                    let zero_vec = make_vec(ty, 4, "0, 0, 0, 0");
                    Some(format!(
                        "\
                        {ty}4 OUT1 = {zero_vec};
                        int UNIQ1 = IN3 / {m};
                        int UNIQ2 = {o} + IN3 % {m};

                        for (int i = 0; i < {nnz}; i++) {{
                            const int j = IN2[{nnz} * UNIQ1 + i];

                            if (j < 0 || j >= {cols}) break;

                            const {ty}4 a = {cast}(IN1)[j * {s} + UNIQ2];

                            OUT1.x += a.x;
                            OUT1.y += a.y;
                            OUT1.z += a.z;
                            OUT1.w += a.w;
                        }}"
                    ))
                }
                3.. => None,
            }
        }
        PointwiseOp::SpMMT { nnz, rows, cols, stride, offset, .. } => {
            let atomic = |ptr: &str, val: &str| atomic_add(ptr, val);
            Some(format!(
                "\
                    if (IN4 != 0) {{
                        int UNIQ1 = IN3 / {rows};
                        int UNIQ2 = {offset} + IN3 % {rows};

                        for (int i = 0; i < {nnz}; i++) {{
                            const int j = IN2[{nnz} * UNIQ1 + i];
                            if (j < 0 || j >= {cols}) break;
                            {}
                        }}
                    }}",
                atomic(&format!("IN1 + j * {stride} + UNIQ2"), "IN4")
            ))
        }
    }
}
