pub struct GraphIRCompileArgs {
    pub(super) emit_ir: bool,
    pub(super) fancy_ir_display: Option<f32>,
    pub(super) allow_fusion: bool,
}

impl Default for GraphIRCompileArgs {
    fn default() -> Self {
        Self { emit_ir: false, fancy_ir_display: None, allow_fusion: true }
    }
}

impl GraphIRCompileArgs {
    pub fn emit_ir(mut self) -> Self {
        self.emit_ir = true;
        self
    }

    pub fn fancy_ir_display(mut self, delay_between_passes: f32) -> Self {
        self.fancy_ir_display = Some(delay_between_passes);
        self
    }

    pub fn disable_fusion(mut self) -> Self {
        self.allow_fusion = false;
        self
    }
}
