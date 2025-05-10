use crate::{
    backend::device::{Device, OperationError},
    graph::{
        ir::{
            args::GraphIRCompileArgs,
            op::{DiffableFromOutput, GraphIROp, GraphIROpError, GraphIROpErrorType, Reduce},
            shape::Shape,
            GraphIR, GraphIRError,
        },
        GraphError,
    },
};

pub fn sparse_affine<D: Device>(device: D) -> Result<(), GraphError<D::DeviceError>> {
    let mut builder = GraphIR::default();
    let w = builder.add_weights("w", Shape::new(1, 3)).unwrap();
    let b = builder.add_weights("b", Shape::new(1, 1)).unwrap();
    let i = builder.add_sparse_input("i", Shape::new(3, 1), 2).unwrap();
    let v = builder.add_dense_input("v", Shape::new(2, 1)).unwrap();
    let out = builder.add_op(GraphIROp::SparseAffineActivate(w, i, Some(v), Some(b), DiffableFromOutput::Identity))?;
    builder.add_op(GraphIROp::ReduceAcrossBatch(out, Reduce::Sum))?;
    let mut graph = builder.compile(device, GraphIRCompileArgs::default())?;

    graph.get_weights_mut("w").load_dense_from_slice(None, &[-1.0, 4.0, 2.0]).unwrap();
    graph.get_weights_mut("b").load_dense_from_slice(None, &[1.0]).unwrap();
    graph.get_input_mut("v").load_dense_from_slice(Some(2), &[1.0, 2.0, 2.0, 1.0]).unwrap();

    unsafe {
        graph.get_input_mut("i").load_sparse_from_slice(2, Some(2), &[1, -1, 0, 2]).unwrap();
    }

    let err = graph.forward()?;
    assert_eq!(err, 6.0);

    let output = graph.get_node(out.into()).get_dense_vals().unwrap();
    assert_eq!(&output, &[5.0, 1.0]);

    graph.backward()?;

    let mut buf = [0.0; 3];
    graph.get_weights("w").gradients.as_ref().unwrap().write_to_slice(&mut buf).map_err(OperationError::from)?;
    assert_eq!(buf, [2.0, 1.0, 1.0]);

    let mut buf = [0.0];
    graph.get_weights("b").gradients.as_ref().unwrap().write_to_slice(&mut buf).map_err(OperationError::from)?;
    assert_eq!(buf, [2.0]);

    Ok(())
}

pub fn sparse_affine_batched_biases<D: Device>(device: D) -> Result<(), GraphError<D::DeviceError>> {
    let mut builder = GraphIR::default();
    let w = builder.add_weights("w", Shape::new(1, 3)).unwrap();
    let b = builder.add_weights("b", Shape::new(1, 2)).unwrap();
    let i = builder.add_sparse_input("i", Shape::new(3, 1), 2).unwrap();
    let bb = builder.add_sparse_input("bb", Shape::new(2, 1), 1).unwrap();

    let b = builder.add_op(GraphIROp::SparseAffineActivate(b, bb, None, None, DiffableFromOutput::Identity))?;
    let out = builder.add_op(GraphIROp::SparseAffineActivate(w, i, None, Some(b), DiffableFromOutput::Identity))?;
    builder.add_op(GraphIROp::ReduceAcrossBatch(out, Reduce::Sum))?;
    let mut graph = builder.compile(device, GraphIRCompileArgs::default())?;

    graph.get_weights_mut("w").load_dense_from_slice(None, &[-1.0, 4.0, 2.0]).expect("loading weights");
    graph.get_weights_mut("b").load_dense_from_slice(None, &[1.0, 2.0]).expect("loading weights");

    unsafe {
        graph.get_input_mut("i").load_sparse_from_slice(2, Some(2), &[1, -1, 0, 2]).expect("loading inputs");
        graph.get_input_mut("bb").load_sparse_from_slice(1, Some(2), &[1, 1]).expect("loading inputs");
    }

    let err = graph.forward()?;
    assert_eq!(err, 9.0);

    let output = graph.get_node(out.into()).get_dense_vals().expect("loading outputs");
    assert_eq!(&output, &[6.0, 3.0]);

    graph.backward()?;

    let mut buf = [0.0; 3];
    graph.get_weights("w").gradients.as_ref().unwrap().write_to_slice(&mut buf).map_err(OperationError::from)?;
    assert_eq!(buf, [1.0, 1.0, 1.0]);

    let mut buf = [0.0; 2];
    graph.get_weights("b").gradients.as_ref().unwrap().write_to_slice(&mut buf).map_err(OperationError::from)?;
    assert_eq!(buf, [0.0, 2.0]);

    Ok(())
}

pub fn sparse_affine_dual<D: Device>(device: D) -> Result<(), GraphError<D::DeviceError>> {
    let mut builder = GraphIR::default();
    let w = builder.add_weights("w", Shape::new(1, 3)).unwrap();
    let b = builder.add_weights("b", Shape::new(1, 1)).unwrap();
    let i1 = builder.add_sparse_input("i1", Shape::new(3, 1), 2).unwrap();
    let i2 = builder.add_sparse_input("i2", Shape::new(3, 1), 2).unwrap();
    let dot = builder.add_dense_input("dot", Shape::new(1, 2)).unwrap();
    let out = builder.add_op(GraphIROp::SparseAffineDualActivate(w, i1, i2, Some(b), DiffableFromOutput::Identity))?;
    let out2 = builder.add_op(GraphIROp::Matmul(dot, false, out, false))?;
    builder.add_op(GraphIROp::ReduceAcrossBatch(out2, Reduce::Sum))?;
    let mut graph = builder.compile(device, GraphIRCompileArgs::default())?;

    graph.get_weights_mut("w").load_dense_from_slice(None, &[-1.0, 4.0, 2.0]).unwrap();
    graph.get_weights_mut("b").load_dense_from_slice(None, &[1.0]).unwrap();
    graph.get_input_mut("dot").load_dense_from_slice(Some(2), &[1.0, 1.0, 1.0, 1.0]).unwrap();

    unsafe {
        graph.get_input_mut("i1").load_sparse_from_slice(2, Some(2), &[1, -1, 0, 2]).unwrap();
        graph.get_input_mut("i2").load_sparse_from_slice(2, Some(2), &[2, -1, 2, 0]).unwrap();
    }

    let err = graph.forward().unwrap();
    assert_eq!(err, 12.0);

    let output = graph.get_node(out.into()).get_dense_vals().unwrap();
    assert_eq!(&output, &[5.0, 3.0, 2.0, 2.0]);

    graph.backward().unwrap();

    let mut buf = [0.0; 3];
    graph.get_weights("w").gradients.as_ref().unwrap().write_to_slice(&mut buf).map_err(OperationError::from)?;
    assert_eq!(buf, [2.0, 1.0, 3.0]);

    let mut buf = [0.0];
    graph.get_weights("b").gradients.as_ref().unwrap().write_to_slice(&mut buf).map_err(OperationError::from)?;
    assert_eq!(buf, [4.0]);

    Ok(())
}

pub fn sparse_affine_check_not_batched<D: Device>(_device: D) -> Result<(), GraphIRError> {
    let mut builder = GraphIR::default();
    let w = builder.add_dense_input("w", Shape::new(1, 3)).unwrap();
    let b = builder.add_weights("b", Shape::new(1, 1)).unwrap();
    let i1 = builder.add_sparse_input("i1", Shape::new(3, 1), 2).unwrap();
    let i2 = builder.add_sparse_input("i2", Shape::new(3, 1), 2).unwrap();

    let op = GraphIROp::SparseAffineDualActivate(w, i1, i2, Some(b), DiffableFromOutput::Identity);
    let out = builder.add_op(op);

    assert_eq!(out, Err(GraphIRError::Op(GraphIROpError::new(&op, GraphIROpErrorType::BatchedInputNotSupported))));

    Ok(())
}
