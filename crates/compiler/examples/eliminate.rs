use bullet_compiler::prelude::*;

fn main() {
    let builder = ProgramBuilder::default();

    let batch = Size::variable();

    let inputs = builder.add_input(batch * 8, DType::F32);
    let target = builder.add_input(batch, DType::F32);

    let weights = builder.constant(DTypeTensor::F32(vec![1.0; 8])).broadcast([8], 0, batch);
    let bias = builder.constant(DTypeTensor::F32(vec![1.0])).broadcast([1], 0, batch);

    let dot = (weights * inputs).reduce_sum([batch, Size::from(8)], 1);

    let d1 = dot + bias - target;
    let d2 = bias - target + dot;
    let loss = (d1 * d2).reduce_sum([batch], 0);
    let zero = d1 - d2;

    let mut program = builder.build([loss, zero]);

    let ops = program.num_nontrivial_operations().unwrap();
    println!("Unoptimised: {ops} operations");
    println!("{}", program.as_highlighted());

    program.optimise().unwrap();

    let ops = program.num_nontrivial_operations().unwrap();
    println!("Optimised: {ops} operations");
    println!("{}", program.as_highlighted());

    let inputs = (inputs.node(), DTypeTensor::F32(vec![1.0; 32]));
    let target = (target.node(), DTypeTensor::F32(vec![1.0; 4]));
    let outputs = program.evaluate([inputs, target]).unwrap();

    println!("loss: {:?} (expect 256.0)", outputs.get(&loss.node()).unwrap());
    println!("zero: {:?} (expect [0.0; 4])", outputs.get(&zero.node()).unwrap());
}
