pub mod pointwise;

use std::{collections::HashSet, fmt, rc::Rc};

use bullet_compiler::tensor::{OpType, TType};

use crate::runtime::Dim3;

#[derive(Clone)]
pub struct KernelSrc {
    pub(crate) inputs: Vec<TType>,
    pub(crate) outputs: Vec<TType>,
    pub(crate) source: String,
    pub(crate) requires_var_size_arg: bool,
    pub(crate) arg_order: Vec<(usize, bool)>,
    pub(crate) requires_zero: HashSet<usize>,
    pub(crate) gdim: Rc<dyn Fn(usize) -> Dim3>,
    pub(crate) bdim: Rc<dyn Fn(usize) -> u32>,
    pub(crate) smem: Rc<dyn Fn(usize) -> u32>,
}

impl fmt::Debug for KernelSrc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "gpu.kernel.source")
    }
}

impl KernelSrc {
    /// ### Safety
    ///
    /// I solemnly swear that as long as the passed input and output
    /// tensors to the compiled function have the correct TType and
    /// the variable size is passed correctly, then this kernel will
    /// not invoke UB.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new(
        inputs: Vec<TType>,
        outputs: Vec<TType>,
        source: String,
        requires_var_size_arg: bool,
        arg_order: Vec<(usize, bool)>,
        requires_zero: HashSet<usize>,
        gdim: Rc<dyn Fn(usize) -> Dim3>,
        bdim: Rc<dyn Fn(usize) -> u32>,
        smem: Rc<dyn Fn(usize) -> u32>,
    ) -> Self {
        assert_eq!(arg_order.len(), inputs.len() + outputs.len());
        assert_eq!(
            inputs.len(),
            arg_order.iter().filter_map(|(idx, input)| input.then_some(*idx)).collect::<HashSet<_>>().len()
        );
        assert_eq!(
            outputs.len(),
            arg_order.iter().filter_map(|(idx, input)| (!input).then_some(*idx)).collect::<HashSet<_>>().len()
        );

        Self { inputs, outputs, source, requires_var_size_arg, arg_order, requires_zero, gdim, bdim, smem }
    }
}

impl OpType for KernelSrc {
    fn opname(&self) -> String {
        "gpu.kernel.source".to_string()
    }

    fn inputs(&self) -> Vec<TType> {
        self.inputs.clone()
    }

    fn outputs(&self) -> Vec<TType> {
        self.outputs.clone()
    }
}
