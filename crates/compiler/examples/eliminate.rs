use bullet_compiler::prelude::*;

fn main() {
    let builder = ProgramBuilder::default();

    let inputs = builder.add_leaf(8, DType::F32);
    let target = builder.add_leaf(1, DType::F32);

    let weights = builder.constant(DTypeTensor::F32(vec![1.0; 8]));
    let bias = builder.constant(DTypeTensor::F32(vec![1.0]));

    let dot = (weights * inputs).reduce_sum([8], 0);

    let [_prediction, loss] = builder.elementwise([dot, bias, target], |[dot, bias, target]| {
        let prediction = dot + bias;
        [prediction, (prediction - target) * (prediction - target)]
    });

    let program = builder.build([loss]);

    println!("{program}");

    let inputs = (inputs.node(), DTypeTensor::F32(vec![1.0; 8]));
    let target = (target.node(), DTypeTensor::F32(vec![1.0; 1]));
    let outputs = program.evaluate([inputs, target]).unwrap();

    println!("loss: {:?}", outputs.get(&loss.node()).unwrap());
}
