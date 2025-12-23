use bullet_compiler::prelude::*;

fn main() {
    let builder = ProgramBuilder::default();

    let batch = Size::variable();

    let inputs = builder.add_leaf(batch * 8, DType::F32);
    let target = builder.add_leaf(batch, DType::F32);

    let a = builder.constant(DTypeTensor::F32(vec![2.0; 8])).broadcast([8], 0, batch);
    let b = builder.constant(DTypeTensor::F32(vec![1.0])).broadcast([1], 0, batch);

    let dot = (a * inputs).reduce_sum(batch + [8], 1);

    let [prediction, loss] = builder.elementwise([dot, b, target], |[dot, b, target]| {
        let prediction = dot + b;
        let diff = prediction - target;
        [prediction, diff * diff]
    });

    let program = builder.build([prediction, loss]);

    println!("{program}");

    let inputs = (inputs.node(), DTypeTensor::F32(vec![1.0; 32]));
    let target = (target.node(), DTypeTensor::F32(vec![1.0; 4]));
    let outputs = program.evaluate([inputs, target]).unwrap();

    println!("prediction: {:?}", outputs.get(&prediction.node()).unwrap());
    println!("loss: {:?}", outputs.get(&loss.node()).unwrap());
}
