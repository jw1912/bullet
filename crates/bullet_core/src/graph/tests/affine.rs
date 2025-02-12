use crate::{
    device::Device,
    graph::{builder::GraphBuilder, operation::Operation},
    shape::Shape,
};

pub fn matmul<D: Device>(device: D) {
    let mut builder = GraphBuilder::default();
    let w1 = builder.create_weights("w1", Shape::new(1, 3));
    let w2 = builder.create_weights("w2", Shape::new(3, 1));
    let out = builder.create_result_of_operation(Operation::Affine(w1, w2, None));
    builder.create_result_of_operation(Operation::ReduceAcrossBatch(out));
    let mut graph = builder.build(device);

    graph.get_weights_mut("w1").load_dense_from_slice(None, &[-1.0, 4.0, 2.0]);
    graph.get_weights_mut("w2").load_dense_from_slice(Some(2), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);

    let err = graph.forward();
    assert_eq!(err, 26.0);

    let output = graph.get_node(out).get_dense_vals().unwrap();
    assert_eq!(&output, &[13.0, 13.0]);

    graph.backward();

    let mut buf = [0.0; 3];
    graph.get_weights("w1").gradients.as_ref().unwrap().write_to_slice(&mut buf);
    assert_eq!(buf, [2.0, 4.0, 6.0]);

    let mut buf = [0.0; 6];
    graph.get_weights("w2").gradients.as_ref().unwrap().write_to_slice(&mut buf);
    assert_eq!(buf, [-1.0, 4.0, 2.0, -1.0, 4.0, 2.0]);
}
