use bullet_compiler::{DType, DTypeTensor, ProgramBuilder};

fn main() {
    let builder = ProgramBuilder::default();

    let inputs = builder.add_leaf(8, DType::F32);
    let target = builder.add_leaf(1, DType::F32);

    let a = builder.add_leaf(8, DType::F32);
    let b = builder.add_leaf(1, DType::F32);

    let dot = (a * inputs).reduce_sum([8], 0);

    let [prediction, loss] = builder.elementwise([dot, b, target], |[dot, b, target]| {
        let prediction = dot + b;
        let diff = prediction - target;
        [prediction, diff * diff]
    });

    let program = builder.build([prediction, loss]);

    println!("{program}");

    let outputs = program
        .evaluate([
            (inputs.node(), DTypeTensor::F32(vec![1.0; 8])),
            (target.node(), DTypeTensor::F32(vec![1.0])),
            (a.node(), DTypeTensor::F32(vec![2.0; 8])),
            (b.node(), DTypeTensor::F32(vec![1.0])),
        ])
        .unwrap();

    println!("prediction: {:?}", outputs.get(&prediction.node()).unwrap());
    println!("loss: {:?}", outputs.get(&loss.node()).unwrap());
}
