use crate::{
    backend::device::{base::Activation, blas::Shape, Device, OperationError},
    graph::{
        ir::{op::GraphIROp, GraphIR},
        GraphError,
    },
};

pub fn relu<D: Device>(device: D) -> Result<(), GraphError<D::DeviceError>> {
    activate(device, Activation::ReLU, [0.0, 0.5, 2.0, 0.0], [0.0, 1.0, 1.0, 0.0])
}

pub fn crelu<D: Device>(device: D) -> Result<(), GraphError<D::DeviceError>> {
    activate(device, Activation::CReLU, [0.0, 0.5, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0])
}

pub fn screlu<D: Device>(device: D) -> Result<(), GraphError<D::DeviceError>> {
    activate(device, Activation::SCReLU, [0.0, 0.25, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0])
}

pub fn sqrrelu<D: Device>(device: D) -> Result<(), GraphError<D::DeviceError>> {
    activate(device, Activation::SqrReLU, [0.0, 0.25, 4.0, 0.0], [0.0, 1.0, 4.0, 0.0])
}

fn activate<D: Device>(
    device: D,
    activation: Activation,
    fwd: [f32; 4],
    bwd: [f32; 4],
) -> Result<(), GraphError<D::DeviceError>> {
    let mut builder = GraphIR::default();
    let w = builder.add_weights("w", Shape::new(1, 1)).unwrap();
    let out = builder.add_op(GraphIROp::Activate(w, activation), true).unwrap();
    builder.add_op(GraphIROp::ReduceAcrossBatch(out), true).unwrap();
    let mut graph = builder.compile(device).unwrap();

    graph.get_weights_mut("w").load_dense_from_slice(Some(4), &[-1.0, 0.5, 2.0, -2.0]).unwrap();

    let err = graph.forward().unwrap();
    assert_eq!(err, fwd.iter().sum());

    let output = graph.get_node(out.into()).get_dense_vals().unwrap();
    assert_eq!(&output, &fwd);

    graph.backward().unwrap();

    let mut buf = [0.0; 4];
    graph.get_weights("w").gradients.as_ref().unwrap().write_to_slice(&mut buf).map_err(OperationError::from)?;
    assert_eq!(buf, bwd);

    Ok(())
}
