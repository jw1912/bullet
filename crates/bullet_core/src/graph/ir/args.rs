pub struct GraphIRCompileArgs {
    pub(super) emit_ir: bool,
    pub(super) allow_fusion: bool,
}

impl Default for GraphIRCompileArgs {
    fn default() -> Self {
        Self { emit_ir: false, allow_fusion: true }
    }
}

impl GraphIRCompileArgs {
    pub fn emit_ir(mut self) -> Self {
        self.emit_ir = true;
        self
    }

    pub fn disable_fusion(mut self) -> Self {
        self.allow_fusion = false;
        self
    }
}
