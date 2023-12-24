use bullet_tensor::{TensorBatch, Shape, Activation};

#[test]
fn test_tensor_lt() {
    let mut xs = [1.0, -1.0, -0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0];

    let x = TensorBatch::new(Shape::new(1, 3), 3);

    x.load_from_cpu(&xs);
    x.activate(Activation::ReLU);
    x.write_to_cpu(&mut xs);

    assert_eq!(xs, [1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0]);
}