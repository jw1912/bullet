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

    pub fn reinterpret_cast(&self, ty: &str) -> String {
        match self {
            Dialect::Msl => format!("reinterpret_cast<device {ty}*>"),
            Dialect::CudaHip => format!("reinterpret_cast<{ty}*>"),
        }
    }

    pub fn make_vec(&self, ty: &str, count: u32, vals: &str) -> String {
        match self {
            Dialect::Msl => format!("{ty}{count}({vals})"),
            Dialect::CudaHip => format!("make_{ty}{count}({vals})"),
        }
    }

    pub fn atomic_add(&self, ptr_expr: &str, val_expr: &str) -> String {
        match self {
            Dialect::Msl => format!(
                "atomic_fetch_add_explicit((volatile device atomic_float*)({ptr_expr}), {val_expr}, memory_order_relaxed);"
            ),
            Dialect::CudaHip => format!("atomicAdd({ptr_expr}, {val_expr});"),
        }
    }

    pub fn pow(&self) -> &'static str {
        match self {
            Dialect::Msl => "pow",
            Dialect::CudaHip => "powf",
        }
    }

    pub fn sin(&self) -> &'static str {
        match self {
            Dialect::Msl => "sin",
            Dialect::CudaHip => "sinf",
        }
    }

    pub fn cos(&self) -> &'static str {
        match self {
            Dialect::Msl => "cos",
            Dialect::CudaHip => "cosf",
        }
    }

    pub fn tan(&self) -> &'static str {
        match self {
            Dialect::Msl => "tan",
            Dialect::CudaHip => "tanf",
        }
    }

    pub fn sinh(&self) -> &'static str {
        match self {
            Dialect::Msl => "sinh",
            Dialect::CudaHip => "sinhf",
        }
    }

    pub fn cosh(&self) -> &'static str {
        match self {
            Dialect::Msl => "cosh",
            Dialect::CudaHip => "coshf",
        }
    }

    pub fn tanh(&self) -> &'static str {
        match self {
            Dialect::Msl => "tanh",
            Dialect::CudaHip => "tanhf",
        }
    }

    pub fn exp(&self) -> &'static str {
        match self {
            Dialect::Msl => "exp",
            Dialect::CudaHip => "expf",
        }
    }

    pub fn log(&self) -> &'static str {
        match self {
            Dialect::Msl => "log",
            Dialect::CudaHip => "logf",
        }
    }

    pub fn sqrt(&self) -> &'static str {
        match self {
            Dialect::Msl => "sqrt",
            Dialect::CudaHip => "sqrtf",
        }
    }

    pub fn round(&self) -> &'static str {
        match self {
            Dialect::Msl => "round",
            Dialect::CudaHip => "roundf",
        }
    }

    pub fn trunc(&self) -> &'static str {
        match self {
            Dialect::Msl => "trunc",
            Dialect::CudaHip => "truncf",
        }
    }
}
