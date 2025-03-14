#[derive(Default)]
pub struct GraphIRCompileArgs {
    pub(super) emit_ir: bool,
}

impl GraphIRCompileArgs {
    pub fn emit_ir(mut self) -> Self {
        self.emit_ir = true;
        self
    }
}
