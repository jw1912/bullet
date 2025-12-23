use super::*;

use crate::{
    common::{Binary, DType, Size},
    ir::operation::{Constant, IrElementwise},
};

#[test]
fn construct_deconstruct() -> Result<(), IrError> {
    let mut ir = IrGraph::default();

    let x = ir.add_leaf(IrType::new(8, DType::F32));
    let y = ir.add_leaf(IrType::new(8, DType::F32));
    let z = ir.add_leaf(IrType::new(8, DType::F32));

    let w = ir.add_binary(x, y, Binary::Add)?;
    let t = ir.add_binary(z, w, Binary::Mul)?;
    let u = ir.add_binary(t, x, Binary::Add)?;

    assert_eq!(ir.get_node(u)?.ty(), IrType::new(8, DType::F32));

    ir.remove_op(ir.get_parent_op(u)?)?;
    ir.remove_op(ir.get_parent_op(t)?)?;
    ir.remove_op(ir.get_parent_op(w)?)?;
    ir.remove_op(ir.get_parent_op(z)?)?;
    ir.remove_op(ir.get_parent_op(y)?)?;
    ir.remove_op(ir.get_parent_op(x)?)?;

    assert!(ir.ops.is_empty());
    assert!(ir.nodes.is_empty());
    assert!(ir.links.is_empty());

    ir.check_valid()
}

#[test]
fn swap_outputs() -> Result<(), IrError> {
    let mut ir = IrGraph::default();

    let x = ir.add_leaf(IrType::new(8, DType::F32));
    let y = ir.add_leaf(IrType::new(8, DType::F32));
    let z = ir.add_binary(x, y, Binary::Add)?;
    let w = ir.add_binary(z, y, Binary::Add)?;
    let t = ir.add_binary(w, y, Binary::Sub)?;

    ir.register_output(t);

    let new_t = ir.add_binary(x, y, Binary::Add)?;
    ir.swap_outputs(new_t, t)?;
    ir.eliminate_dead_ops()?;

    assert_eq!(ir.num_ops(), 3);
    assert_eq!(ir.num_nodes(), 3);
    assert!(ir.get_node(x).is_ok());
    assert!(ir.get_node(y).is_ok());
    assert!(ir.get_node(t).is_ok());

    ir.check_valid()
}

#[test]
fn replace_op() -> Result<(), IrError> {
    let mut ir = IrGraph::default();

    let x = ir.add_leaf(IrType::new(8, DType::F32));
    let y = ir.add_leaf(IrType::new(8, DType::F32));
    let z = ir.add_binary(x, y, Binary::Add)?;
    let w = ir.add_binary(z, y, Binary::Add)?;
    let t = ir.add_binary(w, y, Binary::Sub)?;

    ir.register_output(t);

    let new_op = IrElementwise::binary(ir.get_node_type(x)?, ir.get_node_type(y)?, Binary::Add)?;
    ir.replace_op(ir.get_parent_op(t)?, [x, y], new_op)?;
    ir.eliminate_dead_ops()?;

    assert_eq!(ir.num_ops(), 3);
    assert_eq!(ir.num_nodes(), 3);
    assert!(ir.get_node(x).is_ok());
    assert!(ir.get_node(y).is_ok());
    assert!(ir.get_node(t).is_ok());

    ir.check_valid()
}

#[test]
fn invalid_addition_size() -> Result<(), IrError> {
    let mut ir = IrGraph::default();

    let x = ir.add_leaf(IrType::new(8, DType::F32));
    let y = ir.add_leaf(IrType::new(16, DType::F32));

    assert!(ir.add_binary(x, y, Binary::Add).is_err());

    Ok(())
}

#[test]
fn invalid_addition_dtype() -> Result<(), IrError> {
    let mut ir = IrGraph::default();

    let x = ir.add_leaf(IrType::new(8, DType::F32));
    let y = ir.add_leaf(IrType::new(8, DType::I32));

    assert!(ir.add_binary(x, y, Binary::Add).is_err());

    Ok(())
}

#[test]
fn invalid_removal() -> Result<(), IrError> {
    let mut ir = IrGraph::default();

    let x = ir.add_leaf(IrType::new(8, DType::F32));
    let y = ir.add_leaf(IrType::new(8, DType::F32));
    let z = ir.add_binary(x, y, Binary::Add)?;

    assert_eq!(ir.get_node(z)?.ty(), IrType::new(8, DType::F32));
    assert!(ir.remove_op(ir.get_parent_op(y)?).is_err());

    Ok(())
}

#[test]
fn invalid_swap_outputs() -> Result<(), IrError> {
    let mut ir = IrGraph::default();

    let x = ir.add_leaf(IrType::new(8, DType::F32));
    let y = ir.add_leaf(IrType::new(8, DType::F32));
    let z = ir.add_binary(x, y, Binary::Add)?;
    let w = ir.add_binary(z, y, Binary::Add)?;
    let t = ir.add_binary(w, y, Binary::Sub)?;

    assert_eq!(ir.swap_outputs(z, t), Err("IrGraph::topo_order_ops: cycle found!".into()));

    Ok(())
}

#[test]
fn evaluate() -> Result<(), IrError> {
    let mut ir = IrGraph::default();
    let size = Size::variable() * 2;

    let x = ir.add_leaf(IrType::new(size, DType::F32));
    let y = ir.add_leaf(IrType::new(size, DType::F32));
    let z = ir.add_leaf(IrType::new(size, DType::F32));

    let w = ir.add_binary(x, y, Binary::Add)?;
    let t = ir.add_binary(z, w, Binary::Mul)?;
    let u = ir.add_binary(t, x, Binary::Add)?;

    let ix = DTypeTensor::F32(vec![1.0, 2.0, 3.0, 4.0]);
    let iy = DTypeTensor::F32(vec![1.0, 1.0, 1.0, 1.0]);
    let iz = DTypeTensor::F32(vec![1.0, 2.0, 3.0, 4.0]);

    ir.register_output(u);

    let outputs = ir.evaluate([(x, ix), (y, iy), (z, iz)])?;

    assert_eq!(outputs, [(u, DTypeTensor::F32(vec![3.0, 8.0, 15.0, 24.0]))].into());

    ir.check_valid()
}

#[test]
fn evaluate_missing_input() -> Result<(), IrError> {
    let mut ir = IrGraph::default();
    let size = Size::variable() * 2;

    let x = ir.add_leaf(IrType::new(size, DType::F32));
    let y = ir.add_leaf(IrType::new(size, DType::F32));
    let z = ir.add_binary(x, y, Binary::Add)?;
    ir.register_output(z);

    let ix = DTypeTensor::F32(vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(ir.evaluate([(x, ix)]), Err("Leaf node not seeded!".into()));

    ir.check_valid()
}

#[test]
fn evaluate_seed_non_leaf() -> Result<(), IrError> {
    let mut ir = IrGraph::default();
    let size = Size::variable() * 2;

    let x = ir.add_leaf(IrType::new(size, DType::F32));
    let y = ir.add_leaf(IrType::new(size, DType::F32));
    let z = ir.add_binary(x, y, Binary::Add)?;
    ir.register_output(z);

    let ix = DTypeTensor::F32(vec![1.0, 2.0, 3.0, 4.0]);
    let inputs = [(x, ix.clone()), (y, ix.clone()), (z, ix)];
    assert_eq!(ir.evaluate(inputs), Err("Seeded non-leaf node!".into()));

    ir.check_valid()
}

#[test]
fn propagate_constants() -> Result<(), IrError> {
    let mut ir = IrGraph::default();

    let x = ir.add_const(DTypeTensor::F32(vec![1.0; 8]));
    let y = ir.add_const(DTypeTensor::F32(vec![1.0; 8]));
    let z = ir.add_binary(x, y, Binary::Add)?;
    let w = ir.add_binary(z, y, Binary::Add)?;
    let t = ir.add_binary(w, y, Binary::Sub)?;

    ir.register_output(t);

    ir.propagate_constants()?;

    println!("{ir}");
    ir.eliminate_dead_ops()?;

    println!("{ir}");

    assert_eq!(ir.num_ops(), 1);
    assert_eq!(ir.num_nodes(), 1);

    for node in [x, y, z, w] {
        assert!(ir.get_node(node).is_err());
    }

    assert!(ir.get_node(t).is_ok());

    let t_op = ir.get_op(ir.get_parent_op(t)?)?;
    assert_eq!(t_op.inputs(), &[]);
    assert_eq!(t_op.outputs(), &[t]);
    assert_eq!(IrOperation::downcast(t_op.op()), Some(&Constant(DTypeTensor::F32(vec![2.0; 8]))));

    ir.check_valid()
}
