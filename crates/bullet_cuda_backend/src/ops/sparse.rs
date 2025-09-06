mod bwd;
mod fwd;

use acyclib::{device::function, graph::ir::operation::sparse::SparseAffineImpl};

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
