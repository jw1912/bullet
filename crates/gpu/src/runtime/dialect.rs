/// GPU kernel language dialect
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dialect {
    /// CUDA C / HIP C (used by NVIDIA and AMD)
    CudaHip,
    /// Metal Shading Language (used by Apple Silicon)
    Msl,
}

impl Dialect {
    pub fn reinterpret_cast(&self, ty: &str) -> String {
        match self {
            Dialect::CudaHip => format!("reinterpret_cast<{ty}*>"),
            Dialect::Msl => format!("reinterpret_cast<device {ty}*>"),
        }
    }

    pub fn make_vec(&self, ty: &str, count: u32, vals: &str) -> String {
        match self {
            Dialect::CudaHip => format!("make_{ty}{count}({vals})"),
            Dialect::Msl => format!("{ty}{count}({vals})"),
        }
    }

    pub fn atomic_add(&self, ptr_expr: &str, val_expr: &str) -> String {
        match self {
            Dialect::CudaHip => format!("atomicAdd({ptr_expr}, {val_expr});"),
            Dialect::Msl => format!(
                "atomic_fetch_add_explicit((volatile device atomic_float*)({ptr_expr}), {val_expr}, memory_order_relaxed);"
            ),
        }
    }

    pub fn pow(&self) -> &'static str {
        match self {
            Dialect::CudaHip => "powf",
            Dialect::Msl => "pow",
        }
    }
}
