use bullet_compiler::tensor::{
    DType, DValue, Size,
    operation::{CABinary, Unary},
};

use crate::{
    pointwise::operations::PointwiseOp,
    runtime::{DeviceProps, Dialect},
};

pub fn tystr(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "float",
        DType::I32 => "int",
    }
}

/// Source literal for `value`, bit casting non-finite floats
/// because `inf`/`NaN` are not valid literals.
fn literal(value: DValue, dialect: Dialect) -> String {
    match value {
        DValue::F32(x) if x.is_finite() => format!("{x:E}"),
        DValue::F32(x) => match dialect {
            Dialect::CudaHip => format!("__int_as_float({:#x})", x.to_bits()),
            Dialect::Msl => format!("as_type<float>({:#x}u)", x.to_bits()),
        },
        DValue::I32(x) => x.to_string(),
    }
}

pub fn code_str(op: PointwiseOp, size: Size, props: &DeviceProps) -> Option<String> {
    let dialect = props.dialect();

    // A whole wave processes one position when its row count is wave-aligned.
    // Tell AMD's compiler that the sparse-index address is uniform so it can
    // use scalar loads instead of repeating the same load in every lane.
    let batch_index = |rows: usize| {
        if props.is_rocm() && props.warp_size().is_some_and(|warp| rows.is_multiple_of(usize::from(warp))) {
            format!("int UNIQ1 = __builtin_amdgcn_readfirstlane(IN3 / {rows});")
        } else {
            format!("int UNIQ1 = IN3 / {rows};")
        }
    };

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
                let cast = dialect.reinterpret_cast(&format!("{ty}{sz}"));
                Some(format!("const {ty}{sz} OUT1 = {cast}(IN1)[IN2];"))
            } else {
                None
            }
        }
        PointwiseOp::ConditionalRead(io, value) => {
            let ty = tystr(io.buf_ty);
            let vl = literal(value, dialect);

            if io.p2size == 0 {
                Some(format!("const {ty} OUT1 = IN3 > 0 ? IN1[IN2] : {vl};"))
            } else if io.p2size < 3 {
                let sz = 2u32.pow(io.p2size as u32);
                let cast = dialect.reinterpret_cast(&format!("{ty}{sz}"));
                let vl_vec = if io.p2size == 1 {
                    dialect.make_vec(ty, 2, &format!("{vl}, {vl}"))
                } else {
                    dialect.make_vec(ty, 4, &format!("{vl}, {vl}, {vl}, {vl}"))
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
                let cast = dialect.reinterpret_cast(&format!("{ty}{sz}"));
                Some(format!("{cast}(IN1)[IN2] = IN3;"))
            } else {
                None
            }
        }
        PointwiseOp::AtomicAdd(io) => {
            let ty = tystr(io.buf_ty);
            if io.p2size == 0 {
                Some(dialect.atomic_add("IN1 + IN2", "IN3"))
            } else if io.p2size < 3 {
                let sz = 2u32.pow(io.p2size as u32);
                let cast = dialect.reinterpret_cast(&format!("{ty}{sz}"));
                Some(dialect.atomic_add(&format!("{cast}(IN1) + IN2"), "IN3"))
            } else {
                None
            }
        }
        PointwiseOp::Constant { value, p2size } => {
            let ty = tystr(value.dtype());
            let vl = literal(value, dialect);

            match p2size {
                0 => Some(format!("const {ty} OUT1 = {vl};")),
                1 => Some(format!("const {ty}2 OUT1 = {};", dialect.make_vec(ty, 2, &format!("{vl}, {vl}")))),
                2 => {
                    Some(format!("const {ty}4 OUT1 = {};", dialect.make_vec(ty, 4, &format!("{vl}, {vl}, {vl}, {vl}"))))
                }
                3.. => None,
            }
        }
        PointwiseOp::ThreadId => {
            let size_val = size.get();
            match dialect {
                Dialect::CudaHip => Some(format!(
                    "\
                    const int idx_in_grid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;\n\
                    const int OUT1 = idx_in_grid * blockDim.x + threadIdx.x;\n\
                    if (OUT1 >= ({size_val})) return;\
                "
                )),
                Dialect::Msl => Some(format!(
                    "\
                    const int OUT1 = static_cast<int>(metal_tid);\n\
                    if (OUT1 >= ({size_val})) return;\
                "
                )),
            }
        }
        PointwiseOp::Broadcast(ty, p2size) => {
            let ty = tystr(ty);
            match p2size.get() {
                1 => Some(format!("const {ty}2 OUT1 = {};", dialect.make_vec(ty, 2, "IN1, IN1"))),
                2 => Some(format!("const {ty}4 OUT1 = {};", dialect.make_vec(ty, 4, "IN1, IN1, IN1, IN1"))),
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
            let formula = |x, y| format!("{}({x}, {y});", dialect.pow());
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
            // the declared type follows the op's output type, which matches the
            // input for everything but a cast; `opstr` still compares against
            // zero in the input type
            let out = tystr(if let Unary::Cast(cast) = op { cast } else { ty });
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
                    let mslidx = usize::from(dialect == Dialect::Msl);
                    let opstr: &str = match op {
                        Unary::Cast(nty) => match nty {
                            DType::F32 => "static_cast<float>",
                            DType::I32 => "static_cast<int>",
                        },
                        Unary::Abs => "abs",
                        Unary::Sin => ["sinf", "sin"][mslidx],
                        Unary::Cos => ["cosf", "cos"][mslidx],
                        Unary::Tan => ["tanf", "tan"][mslidx],
                        Unary::Sinh => ["sinhf", "sinh"][mslidx],
                        Unary::Cosh => ["coshf", "cosh"][mslidx],
                        Unary::Tanh => ["tanhf", "tanh"][mslidx],
                        Unary::Exp => ["expf", "exp"][mslidx],
                        Unary::Log => ["logf", "log"][mslidx],
                        Unary::Sqrt => ["sqrtf", "sqrt"][mslidx],
                        Unary::Round => ["roundf", "round"][mslidx],
                        Unary::Truncate => ["truncf", "trunc"][mslidx],
                        Unary::Identity => "",
                        Unary::Sgn | Unary::Reciprocal | Unary::IsPositive | Unary::IsZero | Unary::IsNonNegative => {
                            unimplemented!()
                        }
                    };

                    format!("{opstr}({x})")
                }
            };

            match p2size {
                0 => Some(format!("const {out} OUT1 = {};", opstr("IN1"))),
                1 => Some(format!("{out}2 OUT1;\nOUT1.x = {};\nOUT1.y = {};", opstr("IN1.x"), opstr("IN1.y"),)),
                2 => Some(format!(
                    "{out}4 OUT1;\nOUT1.x = {};\nOUT1.y = {};\nOUT1.z = {};\nOUT1.w = {};",
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
                    {}
                    int UNIQ2 = {offset} + IN3 % {rows};

                    for (int i = 0; i < {nnz}; i++) {{
                        const int j = IN2[{nnz} * UNIQ1 + i];
                        if (j < 0 || j >= {cols}) break;
                        OUT1 += IN1[j * {stride} + UNIQ2];
                    }}",
                    batch_index(rows)
                )),
                1 => {
                    assert_eq!(rows % 2, 0);
                    assert_eq!(offset % 2, 0);
                    assert_eq!(stride % 2, 0);
                    let m = rows / 2;
                    let o = offset / 2;
                    let s = stride / 2;
                    let cast = dialect.reinterpret_cast(&format!("{ty}2"));
                    let zero_vec = dialect.make_vec(ty, 2, "0, 0");
                    Some(format!(
                        "\
                        {ty}2 OUT1 = {zero_vec};
                        {}
                        int UNIQ2 = {o} + IN3 % {m};

                        for (int i = 0; i < {nnz}; i++) {{
                            const int j = IN2[{nnz} * UNIQ1 + i];

                            if (j < 0 || j >= {cols}) break;

                            const {ty}2 a = {cast}(IN1)[j * {s} + UNIQ2];

                            OUT1.x += a.x;
                            OUT1.y += a.y;
                        }}",
                        batch_index(m)
                    ))
                }
                2 => {
                    assert_eq!(rows % 4, 0);
                    assert_eq!(offset % 4, 0);
                    assert_eq!(stride % 4, 0);
                    let m = rows / 4;
                    let o = offset / 4;
                    let s = stride / 4;
                    let cast = dialect.reinterpret_cast(&format!("{ty}4"));
                    let zero_vec = dialect.make_vec(ty, 4, "0, 0, 0, 0");
                    Some(format!(
                        "\
                        {ty}4 OUT1 = {zero_vec};
                        {}
                        int UNIQ2 = {o} + IN3 % {m};

                        for (int i = 0; i < {nnz}; i++) {{
                            const int j = IN2[{nnz} * UNIQ1 + i];

                            if (j < 0 || j >= {cols}) break;

                            const {ty}4 a = {cast}(IN1)[j * {s} + UNIQ2];

                            OUT1.x += a.x;
                            OUT1.y += a.y;
                            OUT1.z += a.z;
                            OUT1.w += a.w;
                        }}",
                        batch_index(m)
                    ))
                }
                3.. => None,
            }
        }
        PointwiseOp::SpMMT { nnz, rows, cols, stride, offset, .. } => Some(format!(
            "\
                    if (IN4 != 0) {{
                        {}
                        int UNIQ2 = {offset} + IN3 % {rows};

                        for (int i = 0; i < {nnz}; i++) {{
                            const int j = IN2[{nnz} * UNIQ1 + i];
                            if (j < 0 || j >= {cols}) break;
                            {}
                        }}
                    }}",
            batch_index(rows),
            dialect.atomic_add(&format!("IN1 + j * {stride} + UNIQ2"), "IN4")
        )),
    }
}
