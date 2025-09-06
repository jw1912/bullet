mod bwd;
mod fwd;

use bullet_core::{function, graph::ir::operation::sparse::SparseAffineImpl};

use crate::{CudaDevice, kernel::Kernel};

impl SparseAffineImpl for CudaDevice {
    type Bwd = Kernel;
    type Fwd = Kernel;

    fn bwd(op: function::BackpropSparseAffineActivate<Self>) -> Self::Bwd {
        bwd::kernel(op)
    }

    fn fwd(op: function::SparseAffineActivate<Self>) -> Self::Fwd {
        fwd::kernel(op)
    }
}
