use crate::{
    device::{Device, OperationError},
    graph::{builder::GraphBuilder, error::GraphError, operation::Operation},
    shape::Shape,
};

pub fn matmul<D: Device>(device: D) -> Result<(), GraphError<D::DeviceError>> {
    let mut builder = GraphBuilder::default();
    let w1 = builder.create_weights("w1", Shape::new(1, 3)).unwrap();
    let w2 = builder.create_weights("w2", Shape::new(3, 1)).unwrap();
    let out = builder.create_result_of_operation(Operation::Matmul(w1, false, w2, false), true)?;
    builder.create_result_of_operation(Operation::ReduceAcrossBatch(out), true)?;
    let mut graph = builder.build(device)?;

    graph.get_weights_mut("w1").load_dense_from_slice(None, &[-1.0, 4.0, 2.0]).unwrap();
    graph.get_weights_mut("w2").load_dense_from_slice(Some(2), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap();

    let err = graph.forward()?;
    assert_eq!(err, 26.0);

    let output = graph.get_node(out).get_dense_vals().unwrap();
    assert_eq!(&output, &[13.0, 13.0]);

    graph.backward()?;

    let mut buf = [0.0; 3];
    graph.get_weights("w1").gradients.as_ref().unwrap().write_to_slice(&mut buf).map_err(OperationError::from)?;
    assert_eq!(buf, [2.0, 4.0, 6.0]);

    let mut buf = [0.0; 6];
    graph.get_weights("w2").gradients.as_ref().unwrap().write_to_slice(&mut buf).map_err(OperationError::from)?;
    assert_eq!(buf, [-1.0, 4.0, 2.0, -1.0, 4.0, 2.0]);

    Ok(())
}

pub fn matmul2<D: Device>(device: D) -> Result<(), GraphError<D::DeviceError>> {
    let mut builder = GraphBuilder::default();
    let w1 = builder.create_weights("w1", Shape::new(2, 2)).unwrap();
    let w2 = builder.create_weights("w2", Shape::new(2, 2)).unwrap();
    let dot = builder.create_dense_input("dot", Shape::new(1, 4)).unwrap();
    let out = builder.create_result_of_operation(Operation::Matmul(w1, false, w2, false), true)?;
    let a = out.reshape(Shape::new(4, 1)).unwrap();
    let err = builder.create_result_of_operation(Operation::Matmul(dot, false, a, false), true)?;
    builder.create_result_of_operation(Operation::ReduceAcrossBatch(err), true)?;
    let mut graph = builder.build(device)?;

    graph.get_weights_mut("w1").load_dense_from_slice(None, &[-1.0, 4.0, 2.0, 1.0]).unwrap();
    graph.get_weights_mut("w2").load_dense_from_slice(Some(2), &[1.0, 2.0, 3.0, 4.0, 1.0, 3.0, 2.0, 4.0]).unwrap();
    graph.get_input_mut("dot").load_from_slice(None, &[1.0; 4]).unwrap();

    let err = graph.forward()?;
    assert_eq!(err, 60.0);

    let output = graph.get_node(out).get_dense_vals().unwrap();
    assert_eq!(&output, &[3.0, 6.0, 5.0, 16.0, 5.0, 7.0, 6.0, 12.0]);

    graph.backward()?;

    let mut buf = [0.0; 4];
    graph.get_weights("w1").gradients.as_ref().unwrap().write_to_slice(&mut buf).map_err(OperationError::from)?;
    assert_eq!(buf, [7.0, 7.0, 13.0, 13.0]);

    let mut buf = [0.0; 8];
    graph.get_weights("w2").gradients.as_ref().unwrap().write_to_slice(&mut buf).map_err(OperationError::from)?;
    assert_eq!(buf, [3.0; 8]);

    Ok(())
}
