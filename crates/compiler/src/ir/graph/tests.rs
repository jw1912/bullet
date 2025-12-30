use crate::core::{DType, Size};

use super::*;

#[test]
fn construct_deconstruct() -> Result<(), IrError> {
    let mut ir = IrGraph::default();

    let x = ir.add_input(IrType::new(8, DType::F32));
    let y = ir.add_input(IrType::new(8, DType::F32));
    let z = ir.add_input(IrType::new(8, DType::F32));

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

    let x = ir.add_input(IrType::new(8, DType::F32));
    let y = ir.add_input(IrType::new(8, DType::F32));
    let z = ir.add_binary(x, y, Binary::Add)?;
    let w = ir.add_binary(z, y, Binary::Add)?;
    let t = ir.add_binary(w, y, Binary::Add)?;
    let new_t = ir.add_binary(x, y, Binary::Add)?;

    let op = ir.get_parent_op(t)?;
    let new_op = ir.get_parent_op(new_t)?;

    ir.register_output(t);
    ir.swap_outputs_unchecked(new_t, t)?;

    assert_eq!(ir.get_parent_op(new_t)?, op);
    assert_eq!(ir.get_parent_op(t)?, new_op);

    ir.check_valid()
}

#[test]
fn replace_input() -> Result<(), IrError> {
    let mut ir = IrGraph::default();

    let x = ir.add_input(IrType::new(8, DType::F32));
    let y = ir.add_input(IrType::new(8, DType::F32));
    let z = ir.add_binary(x, y, Binary::Add)?;
    let w = ir.add_binary(z, y, Binary::Add)?;
    let t = ir.add_binary(w, y, Binary::Add)?;

    ir.register_output(t);
    ir.replace_input_unchecked(x, w)?;

    assert_eq!(ir.get_op(ir.get_parent_op(t)?)?.inputs(), [x, y]);
    assert_eq!(ir.get_node(w)?.children(), 0);

    ir.check_valid()
}

#[test]
fn invalid_addition_size() -> Result<(), IrError> {
    let mut ir = IrGraph::default();

    let x = ir.add_input(IrType::new(8, DType::F32));
    let y = ir.add_input(IrType::new(16, DType::F32));

    assert!(ir.add_binary(x, y, Binary::Add).is_err());

    Ok(())
}

#[test]
fn invalid_addition_dtype() -> Result<(), IrError> {
    let mut ir = IrGraph::default();

    let x = ir.add_input(IrType::new(8, DType::F32));
    let y = ir.add_input(IrType::new(8, DType::I32));

    assert!(ir.add_binary(x, y, Binary::Add).is_err());

    Ok(())
}

#[test]
fn invalid_removal() -> Result<(), IrError> {
    let mut ir = IrGraph::default();

    let x = ir.add_input(IrType::new(8, DType::F32));
    let y = ir.add_input(IrType::new(8, DType::F32));
    let z = ir.add_binary(x, y, Binary::Add)?;

    assert_eq!(ir.get_node(z)?.ty(), IrType::new(8, DType::F32));
    assert!(ir.remove_op(ir.get_parent_op(y)?).is_err());

    Ok(())
}

#[test]
fn invalid_swap_outputs() -> Result<(), IrError> {
    let mut ir = IrGraph::default();

    let x = ir.add_input(IrType::new(8, DType::F32));
    let y = ir.add_input(IrType::new(8, DType::F32));
    let z = ir.add_binary(x, y, Binary::Add)?;
    let w = ir.add_binary(z, y, Binary::Add)?;
    let t = ir.add_binary(w, y, Binary::Add)?;

    ir.swap_outputs_unchecked(z, t)?;

    assert_eq!(ir.check_valid(), Err("Cycle found!".into()));

    Ok(())
}

#[test]
fn evaluate() -> Result<(), IrError> {
    let mut ir = IrGraph::default();
    let size = Size::variable() * 2;

    let x = ir.add_input(IrType::new(size, DType::F32));
    let y = ir.add_input(IrType::new(size, DType::F32));
    let z = ir.add_input(IrType::new(size, DType::F32));

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

    let x = ir.add_input(IrType::new(size, DType::F32));
    let y = ir.add_input(IrType::new(size, DType::F32));
    let z = ir.add_binary(x, y, Binary::Add)?;
    ir.register_output(z);

    let ix = DTypeTensor::F32(vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(ir.evaluate([(x, ix)]), Err("IrInput node not seeded!".into()));

    ir.check_valid()
}

#[test]
fn evaluate_seed_non_leaf() -> Result<(), IrError> {
    let mut ir = IrGraph::default();
    let size = Size::variable() * 2;

    let x = ir.add_input(IrType::new(size, DType::F32));
    let y = ir.add_input(IrType::new(size, DType::F32));
    let z = ir.add_binary(x, y, Binary::Add)?;
    ir.register_output(z);

    let ix = DTypeTensor::F32(vec![1.0, 2.0, 3.0, 4.0]);
    let inputs = [(x, ix.clone()), (y, ix.clone()), (z, ix)];
    assert_eq!(ir.evaluate(inputs), Err("Seeded non-leaf node!".into()));

    ir.check_valid()
}
