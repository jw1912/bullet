use bullet_compiler::tensor::{
    DType, DValue, Size,
    operation::{CABinary, Unary},
};

use crate::pointwise::operations::PointwiseOp;

pub fn tystr(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "float",
        DType::I32 => "int",
    }
}

pub fn code_str(op: PointwiseOp, size: Size) -> Option<String> {
    match op {
        PointwiseOp::Buffer { .. } | PointwiseOp::VarSize => None,
        PointwiseOp::Div => Some("const int OUT1 = IN1 / IN2;".into()),
        PointwiseOp::Rem => Some("const int OUT1 = IN1 % IN2;".into()),
        PointwiseOp::Read(io) => {
            let ty = tystr(io.buf_ty);
            if io.p2size == 0 {
                Some(format!("const {ty} OUT1 = IN1[IN2];"))
            } else if io.p2size < 3 {
                let sz = 2u32.pow(io.p2size as u32);
                Some(format!("const {ty}{sz} OUT1 = reinterpret_cast<{ty}{sz}*>(IN1)[IN2];"))
            } else {
                None
            }
        }
        PointwiseOp::ConditionalRead(io, value) => {
            let ty = tystr(io.buf_ty);
            let vl = match value {
                DValue::F32(x) => x.to_string(),
                DValue::I32(x) => x.to_string(),
            };

            if io.p2size == 0 {
                Some(format!("const {ty} OUT1 = IN3 > 0 ? IN1[IN2] : {vl};"))
            } else if io.p2size < 3 {
                let sz = 2u32.pow(io.p2size as u32);
                let vl = if io.p2size == 2 {
                    format!("make_{ty}2({vl}, {vl})")
                } else {
                    format!("make_{ty}4({vl}, {vl}, {vl}, {vl})")
                };
                Some(format!("const {ty}{sz} OUT1 = IN3 > 0 ? reinterpret_cast<{ty}{sz}*>(IN1)[IN2] : {vl};"))
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
                Some(format!("reinterpret_cast<{ty}{sz}*>(IN1)[IN2] = IN3;"))
            } else {
                None
            }
        }
        PointwiseOp::AtomicAdd(io) => {
            let ty = tystr(io.buf_ty);
            if io.p2size == 0 {
                Some("atomicAdd(IN1 + IN2, IN3);".to_string())
            } else if io.p2size < 3 {
                let sz = 2u32.pow(io.p2size as u32);
                Some(format!("atomicAdd(reinterpret_cast<{ty}{sz}*>(IN1) + IN2, IN3);"))
            } else {
                None
            }
        }
        PointwiseOp::Constant { value, p2size } => {
            let ty = tystr(value.dtype());
            let vl = match value {
                DValue::F32(x) => x.to_string(),
                DValue::I32(x) => x.to_string(),
            };

            match p2size {
                0 => Some(format!("const {ty} OUT1 = {vl};")),
                1 => Some(format!("const {ty}2 OUT1 = make_{ty}2({vl}, {vl});")),
                2 => Some(format!("const {ty}4 OUT1 = make_{ty}4({vl}, {vl}, {vl}, {vl});")),
                3.. => None,
            }
        }
        PointwiseOp::EvalSize(size) => {
            let mut size_str = format!("{}", size.factor());
            for _ in 0..size.var_power() {
                size_str += " * IN1";
            }

            Some(format!("const int OUT1 = {size_str};"))
        }
        PointwiseOp::ThreadId => {
            let mut size_str = format!("{}", size.factor());
            for _ in 0..size.var_power() {
                size_str += " * IN1";
            }

            Some(format!(
                "\
                const int idx_in_grid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;\n\
                const int OUT1 = idx_in_grid * blockDim.x + threadIdx.x;\n\
                if (OUT1 >= ({size_str})) return;\
            "
            ))
        }
        PointwiseOp::Broadcast(ty, p2size) => {
            let ty = tystr(ty);
            match p2size.get() {
                1 => Some(format!("const {ty}2 OUT1 = make_{ty}2(IN1, IN1);")),
                2 => Some(format!("const {ty}4 OUT1 = make_{ty}4(IN1, IN1, IN1, IN1);")),
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
        PointwiseOp::Unary { ty, p2size, op } => {
            let ty = tystr(ty);

            let opstr = |x| match op {
                Unary::Sgn => {
                    format!("{x} == {ty}(0) ? {ty}(0) : x > {ty}(0) ? {ty}(1) : {ty}(-1)")
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
                        Unary::Sin => "sinf",
                        Unary::Cos => "cosf",
                        Unary::Tan => "tanf",
                        Unary::Sinh => "sinhf",
                        Unary::Cosh => "coshf",
                        Unary::Tanh => "tanhf",
                        Unary::Exp => "expf",
                        Unary::Log => "logf",
                        Unary::Sqrt => "sqrtf",
                        Unary::Round => "roundf",
                        Unary::Truncate => "truncf",
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
        PointwiseOp::SpMM { nnz, rows, cols, ty, p2size } => {
            let ty = tystr(ty);
            match p2size {
                0 => Some(format!(
                    "\
                    {ty} OUT1 = 0;
                    int UNIQ1 = IN3 / {rows};
                    int UNIQ2 = IN3 % {rows};

                    for (int i = 0; i < {nnz}; i++) {{
                        const int j = IN2[{nnz} * UNIQ1 + i];
                        if (j < 0 || j >= {cols}) break;
                        OUT1 += IN1[j * {rows} + UNIQ2];
                    }}"
                )),
                1 => {
                    let m = rows / 2;
                    Some(format!(
                        "\
                        {ty}2 OUT1 = make_{ty}2(0, 0, 0, 0);
                        int UNIQ1 = IN3 / {m};
                        int UNIQ2 = IN3 % {m};

                        for (int i = 0; i < {nnz}; i++) {{
                            const int j = IN2[{nnz} * UNIQ1 + i];

                            if (j < 0 || j >= {cols}) break;

                            const {ty}2 a = reinterpret_cast<{ty}2*>(IN1)[j * {m} + UNIQ2];

                            OUT1.x += a.x;
                            OUT1.y += a.y;
                        }}"
                    ))
                }
                2 => {
                    let m = rows / 4;
                    Some(format!(
                        "\
                        {ty}4 OUT1 = make_{ty}4(0, 0, 0, 0);
                        int UNIQ1 = IN3 / {m};
                        int UNIQ2 = IN3 % {m};

                        for (int i = 0; i < {nnz}; i++) {{
                            const int j = IN2[{nnz} * UNIQ1 + i];

                            if (j < 0 || j >= {cols}) break;

                            const {ty}4 a = reinterpret_cast<{ty}4*>(IN1)[j * {m} + UNIQ2];

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
        PointwiseOp::SpMMT { nnz, rows, cols, .. } => Some(format!(
            "\
                if (IN4 != 0) {{
                    int UNIQ1 = IN3 / {rows};
                    int UNIQ2 = IN3 % {rows};

                    for (int i = 0; i < {nnz}; i++) {{
                        const int j = IN2[{nnz} * UNIQ1 + i];
                        if (j < 0 || j >= {cols}) break;
                        atomicAdd(IN1 + j * {rows} + UNIQ2, IN4);
                    }}
                }}"
        )),
    }
}
