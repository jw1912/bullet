use bullet_compiler::prelude::*;

fn main() {
    let builder = ProgramBuilder::default();

    let inputs = builder.constant(DTypeTensor::F32(vec![1.0; 8]));
    let target = builder.constant(DTypeTensor::F32(vec![1.0]));

    let weights = builder.constant(DTypeTensor::F32(vec![1.0; 8]));
    let bias = builder.constant(DTypeTensor::F32(vec![1.0]));

    let prediction = (weights * inputs).reduce_sum([8], 0) + bias;
    let diff = prediction - target;
    let loss = diff * diff;

    let program = builder.build([prediction, loss]);

    println!("{program}");

    let outputs = program.evaluate([]).unwrap();

    println!("prediction: {:?}", outputs.get(&prediction.node()).unwrap());
    println!("loss: {:?}", outputs.get(&loss.node()).unwrap());
}
