pub mod autograd;
pub mod canonicalise;
pub mod eliminate;
pub mod foldrules;
pub mod inline;
pub mod modify;
pub mod ordering;
pub mod rewriterules;

pub use canonicalise::CanonicalisePass;

use crate::tensor::{IRTrace, TensorIR};

pub trait IRTransform: std::fmt::Debug + 'static {
    fn apply(&self, ir: &mut TensorIR) -> Result<(), IRTrace>;
}

#[cfg(test)]
mod tests {
    use crate::tensor::{transform::CanonicalisePass, *};

    #[test]
    fn constant_fold_all() -> Result<(), IRTrace> {
        let builder = IRBuilder::default();

        let inputs = builder.constant(TValue::F32(vec![1.0; 8]));
        let target = builder.constant(TValue::F32(vec![1.0]));

        let weights = builder.constant(TValue::F32(vec![1.0; 8]));
        let bias = builder.constant(TValue::F32(vec![1.0]));

        let prediction = ((weights * inputs)?.reduce_sum([8], 0)? + bias)?;
        let diff = (prediction - target)?;
        let loss = (diff * diff)?;

        let mut program = builder.build([prediction, loss]);

        assert!(program.num_nontrivial_operations()? > 0);

        program.optimise()?;

        assert_eq!(program.num_nontrivial_operations()?, 0);

        program.check_valid()
    }

    #[test]
    fn factorise_exprs() -> Result<(), IRTrace> {
        let builder = IRBuilder::default();

        let a = builder.add_input(1, DType::F32);
        let b = builder.add_input(1, DType::F32);
        let c = builder.add_input(1, DType::F32);
        let d = builder.add_input(1, DType::F32);

        let out = (((a * c)? + (a * d)?)? + ((b * c)? + (b * d)?)?)?;

        let mut program = builder.build([out]);

        assert_eq!(program.num_nontrivial_operations()?, 7, "{program}");

        program.transform(CanonicalisePass::all())?;

        assert_eq!(program.num_nontrivial_operations()?, 3, "{program}");

        program.check_valid()
    }
}
