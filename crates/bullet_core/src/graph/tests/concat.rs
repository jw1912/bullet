use crate::{
    backend::device::{blas::Shape, Device, OperationError},
    graph::{builder::GraphBuilder, error::GraphError, operation::Operation},
};

pub fn concat<D: Device>(device: D) -> Result<(), GraphError<D::DeviceError>> {
    let mut builder = GraphBuilder::default();
    let w1 = builder.create_weights("w1", Shape::new(3, 1)).unwrap();
    let w2 = builder.create_weights("w2", Shape::new(1, 1)).unwrap();
    let out = builder.create_result_of_operation(Operation::Concat(w1, w2), true)?;
    let dot = builder.create_dense_input("dot", Shape::new(1, 4)).unwrap();
    let out2 = builder.create_result_of_operation(Operation::Matmul(dot, false, out, false), true)?;
    builder.create_result_of_operation(Operation::ReduceAcrossBatch(out2), true)?;
    let mut graph = builder.build(device)?;

    graph
        .get_weights_mut("w1")
        .load_dense_from_slice(Some(3), &[-1.0, 4.0, 2.0, -2.0, 0.0, -3.0, 1.0, 1.0, 1.0])
        .unwrap();
    graph.get_weights_mut("w2").load_dense_from_slice(Some(3), &[1.0, 2.0, 3.0]).unwrap();
    graph.get_input_mut("dot").load_dense_from_slice(None, &[1.0; 4]).unwrap();

    let err = graph.forward()?;

    assert_eq!(err, 9.0);

    graph.backward()?;

    let mut buf = [0.0; 9];
    graph.get_weights("w1").gradients.as_ref().unwrap().write_to_slice(&mut buf).map_err(OperationError::from)?;
    assert_eq!(buf, [1.0; 9]);

    let mut buf = [0.0; 3];
    graph.get_weights("w2").gradients.as_ref().unwrap().write_to_slice(&mut buf).map_err(OperationError::from)?;
    assert_eq!(buf, [1.0; 3]);

    Ok(())
}
