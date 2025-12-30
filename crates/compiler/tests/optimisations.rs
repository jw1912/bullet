use bullet_compiler::{
    ir::{
        IRTrace,
        graph::IrType,
        operation::{IrCopy, ScalarConstant},
    },
    prelude::*,
};

#[test]
fn constant_fold_all() -> Result<(), IRTrace> {
    let builder = ProgramBuilder::default();

    let inputs = builder.constant(DTypeTensor::F32(vec![1.0; 8]));
    let target = builder.constant(DTypeTensor::F32(vec![1.0]));

    let weights = builder.constant(DTypeTensor::F32(vec![1.0; 8]));
    let bias = builder.constant(DTypeTensor::F32(vec![1.0]));

    let prediction = (weights * inputs).reduce_sum([8], 0) + bias;
    let diff = prediction - target;
    let loss = diff * diff;

    let mut program = builder.build([prediction, loss]);

    assert!(program.num_nontrivial_operations()? > 0);

    program.optimise()?;

    assert_eq!(program.num_nontrivial_operations()?, 0);

    program.check_valid()
}

#[test]
fn constant_fold_scalars() -> Result<(), IRTrace> {
    let builder = ProgramBuilder::default();

    let size = Size::variable();

    let a = builder.add_input(size, DType::F32);
    let b = 1.0 * 1.0 * (a + 0.0) + 0.0;

    let mut program = builder.build([b]);

    assert!(program.num_nontrivial_operations()? > 0);

    program.optimise()?;

    assert_eq!(program.num_nontrivial_operations()?, 1);
    assert_eq!(program.parent_op(b.node())?, Some(&IrCopy(IrType::new(size, DType::F32))));

    program.check_valid()
}

#[test]
fn fold_equiv_arith_subexprs() -> Result<(), IRTrace> {
    for perm in 0..24 {
        let mut perms = [0, 1, 2, 3];
        perms.swap(perm / 6, 3);
        perms.swap((perm % 6) / 2, 2);
        perms.swap(perm % 2, 1);

        for swap in 0..16 {
            let builder = ProgramBuilder::default();

            let a = builder.add_input(1, DType::F32);
            let b = builder.add_input(1, DType::F32);
            let c = builder.add_input(1, DType::F32);
            let d = builder.add_input(1, DType::F32);

            let lhs1 = a * b + a * c + b * c + a * c;
            //let lhs2 = (a + b) * (c + d);

            let muls = [
                if swap & 1 == 0 { a * c } else { c * a },
                if swap & 2 == 0 { a * d } else { d * a },
                if swap & 4 == 0 { b * c } else { c * b },
                if swap & 8 == 0 { b * d } else { d * b },
            ];

            let rhs = muls[perms[0]] + muls[perms[1]] + muls[perms[2]] + muls[perms[3]];

            let zero1 = lhs1 - rhs;
            //let zero2 = lhs2 - rhs;

            let mut program = builder.build([zero1 /*zero2*/]);

            assert!(program.num_nontrivial_operations()? > 0, "{program}");

            program.optimise()?;

            assert_eq!(program.num_nontrivial_operations()?, 0, "{program}");

            let constant = ScalarConstant(0.0.into(), 1.into());
            assert_eq!(program.parent_op(zero1.node())?, Some(&constant), "{program}");
            //assert_eq!(program.parent_op(zero2.node())?, Some(&constant), "{program}");

            program.check_valid()?;
        }
    }

    Ok(())
}
