use crate::{
    device::Device,
    graph::{
        builder::GraphBuilder,
        operation::{Activation, Operation},
    },
    shape::Shape,
};

pub fn sparse_affine<D: Device>(device: D) {
    let mut builder = GraphBuilder::default();
    let w = builder.create_weights("w", Shape::new(1, 3));
    let b = builder.create_weights("b", Shape::new(1, 1));
    let i = builder.create_input("i", Shape::new(3, 1));
    let out = builder.create_result_of_operation(Operation::Affine(w, i, Some(b)));
    builder.create_result_of_operation(Operation::ReduceAcrossBatch(out));
    let mut graph = builder.build(device);

    graph.get_weights_mut("w").load_dense_from_slice(None, &[-1.0, 4.0, 2.0]);
    graph.get_weights_mut("b").load_dense_from_slice(None, &[1.0]);

    unsafe {
        graph.get_input_mut("i").load_sparse_from_slice(2, Some(2), &[1, -1, 0, 2]);
    }

    let err = graph.forward();
    assert_eq!(err, 7.0);

    let output = graph.get_node(out).get_dense_vals().unwrap();
    assert_eq!(&output, &[5.0, 2.0]);

    graph.backward();

    let mut buf = [0.0; 3];
    graph.get_weights("w").gradients.as_ref().unwrap().write_to_slice(&mut buf);
    assert_eq!(buf, [1.0, 1.0, 1.0]);

    let mut buf = [0.0];
    graph.get_weights("b").gradients.as_ref().unwrap().write_to_slice(&mut buf);
    assert_eq!(buf, [2.0]);
}

pub fn sparse_affine_dual<D: Device>(device: D) {
    let mut builder = GraphBuilder::default();
    let w = builder.create_weights("w", Shape::new(1, 3));
    let b = builder.create_weights("b", Shape::new(1, 1));
    let i1 = builder.create_input("i1", Shape::new(3, 1));
    let i2 = builder.create_input("i2", Shape::new(3, 1));
    let dot = builder.create_input("dot", Shape::new(1, 2));
    let out = builder.create_result_of_operation(Operation::AffineDualActivate(w, i1, i2, b, Activation::Identity));
    let out2 = builder.create_result_of_operation(Operation::Affine(dot, out, None));
    builder.create_result_of_operation(Operation::ReduceAcrossBatch(out2));
    let mut graph = builder.build(device);

    graph.get_weights_mut("w").load_dense_from_slice(None, &[-1.0, 4.0, 2.0]);
    graph.get_weights_mut("b").load_dense_from_slice(None, &[1.0]);
    graph.get_input_mut("dot").load_dense_from_slice(Some(2), &[1.0, 1.0, 1.0, 1.0]);

    unsafe {
        graph.get_input_mut("i1").load_sparse_from_slice(2, Some(2), &[1, -1, 0, 2]);
        graph.get_input_mut("i2").load_sparse_from_slice(2, Some(2), &[2, -1, 1, 0]);
    }

    let err = graph.forward();
    assert_eq!(err, 14.0);

    let output = graph.get_node(out).get_dense_vals().unwrap();
    assert_eq!(&output, &[5.0, 3.0, 2.0, 4.0]);

    graph.backward();

    let mut buf = [0.0; 3];
    graph.get_weights("w").gradients.as_ref().unwrap().write_to_slice(&mut buf);
    assert_eq!(buf, [2.0, 2.0, 2.0]);

    let mut buf = [0.0];
    graph.get_weights("b").gradients.as_ref().unwrap().write_to_slice(&mut buf);
    assert_eq!(buf, [4.0]);
}
