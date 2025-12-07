use bullet_compiler::{DType, ProgramBuilder, ReduceOp, Size};

fn main() {
    let builder = ProgramBuilder::default();

    let batch = Size::variable();

    let inputs = builder.add_leaf(batch * 8, DType::F32);
    let target = builder.add_leaf(batch, DType::F32);

    let a = builder.add_leaf(8, DType::F32).broadcast([8], batch + [8]);
    let b = builder.add_leaf(1, DType::F32).broadcast([1], batch + [1]);

    let dot = (a * inputs).reduce(batch + [8], batch, ReduceOp::Sum);

    let [prediction, loss] = builder.elementwise([dot, b, target], |[dot, b, target]| {
        let prediction = dot + b;
        let diff = prediction - target;
        [prediction, diff * diff]
    });

    let loss = loss.reduce(batch, [1], ReduceOp::Sum);

    let program = builder.build([prediction, loss]);

    println!("Intermediate Representation:");
    builder.display_ir();
    println!();
    println!("Program Code:");
    println!("{program}");
}
