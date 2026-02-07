use crate::ir::graph::{DType, Size};

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mock(TType, TType);

impl OpType for Mock {
    fn opname(&self) -> String {
        "binary.mock".to_string()
    }

    fn inputs(&self) -> Vec<TType> {
        vec![self.0, self.1]
    }

    fn outputs(&self) -> Vec<TType> {
        vec![self.0]
    }

    fn equals(&self, _: &Rc<dyn OpType>) -> bool {
        false
    }

    fn evaluate(&self, inputs: Vec<&TValue>, mut outputs: Vec<&mut TValue>) {
        *outputs[0] = inputs[0].clone();
    }
}

fn mock(ir: &mut Graph, lhs: NodeId, rhs: NodeId) -> Result<NodeId, GraphError> {
    let op = Mock(ir.get_node(lhs)?.ty(), ir.get_node(rhs)?.ty());
    if let [output] = ir.add_op([lhs, rhs], op)?[..] {
        Ok(output)
    } else {
        Err("Binary operation had unexpected number of outputs!".into())
    }
}

#[test]
fn construct_deconstruct() -> Result<(), GraphError> {
    let mut ir = Graph::default();

    let x = ir.add_input(TType::new(8, DType::F32));
    let y = ir.add_input(TType::new(8, DType::F32));
    let z = ir.add_input(TType::new(8, DType::F32));

    let w = mock(&mut ir, x, y)?;
    let t = mock(&mut ir, z, w)?;
    let u = mock(&mut ir, t, x)?;

    assert_eq!(ir.get_node(u)?.ty(), TType::new(8, DType::F32));

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
fn swap_outputs() -> Result<(), GraphError> {
    let mut ir = Graph::default();

    let x = ir.add_input(TType::new(8, DType::F32));
    let y = ir.add_input(TType::new(8, DType::F32));
    let z = mock(&mut ir, x, y)?;
    let w = mock(&mut ir, z, y)?;
    let t = mock(&mut ir, w, y)?;
    let new_t = mock(&mut ir, x, y)?;

    let op = ir.get_parent_op(t)?;
    let new_op = ir.get_parent_op(new_t)?;

    ir.register_output(t);
    ir.swap_outputs_unchecked(new_t, t)?;

    assert_eq!(ir.get_parent_op(new_t)?, op);
    assert_eq!(ir.get_parent_op(t)?, new_op);

    ir.check_valid()
}

#[test]
fn replace_input() -> Result<(), GraphError> {
    let mut ir = Graph::default();

    let x = ir.add_input(TType::new(8, DType::F32));
    let y = ir.add_input(TType::new(8, DType::F32));
    let z = mock(&mut ir, x, y)?;
    let w = mock(&mut ir, z, y)?;
    let t = mock(&mut ir, w, y)?;

    ir.register_output(t);
    ir.replace_input_unchecked(x, w)?;

    assert_eq!(ir.get_op(ir.get_parent_op(t)?)?.inputs(), [x, y]);
    assert_eq!(ir.get_node(w)?.children(), 0);

    ir.check_valid()
}

#[test]
fn invalid_removal() -> Result<(), GraphError> {
    let mut ir = Graph::default();

    let x = ir.add_input(TType::new(8, DType::F32));
    let y = ir.add_input(TType::new(8, DType::F32));
    let z = mock(&mut ir, x, y)?;

    assert_eq!(ir.get_node(z)?.ty(), TType::new(8, DType::F32));
    assert!(ir.remove_op(ir.get_parent_op(y)?).is_err());

    Ok(())
}

#[test]
fn invalid_swap_outputs() -> Result<(), GraphError> {
    let mut ir = Graph::default();

    let x = ir.add_input(TType::new(8, DType::F32));
    let y = ir.add_input(TType::new(8, DType::F32));
    let z = mock(&mut ir, x, y)?;
    let w = mock(&mut ir, z, y)?;
    let t = mock(&mut ir, w, y)?;

    ir.swap_outputs_unchecked(z, t)?;

    assert_eq!(ir.check_valid(), Err("Cycle found!".into()));

    Ok(())
}

#[test]
fn evaluate() -> Result<(), GraphError> {
    let mut ir = Graph::default();
    let size = Size::variable() * 2;

    let x = ir.add_input(TType::new(size, DType::F32));
    let y = ir.add_input(TType::new(size, DType::F32));
    let z = mock(&mut ir, x, y)?;

    let ix = TValue::F32(vec![1.0, 2.0, 3.0, 4.0]);
    let iy = TValue::F32(vec![1.0, 1.0, 1.0, 1.0]);
    let ez = TValue::F32(vec![1.0, 2.0, 3.0, 4.0]);

    ir.register_output(z);

    let outputs = ir.evaluate([(x, ix), (y, iy)])?;

    assert_eq!(outputs, [(z, ez)].into());

    ir.check_valid()
}

#[test]
fn evaluate_missing_input() -> Result<(), GraphError> {
    let mut ir = Graph::default();
    let size = Size::variable() * 2;

    let x = ir.add_input(TType::new(size, DType::F32));
    let y = ir.add_input(TType::new(size, DType::F32));
    let z = mock(&mut ir, x, y)?;
    ir.register_output(z);

    let ix = TValue::F32(vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(ir.evaluate([(x, ix)]), Err("Input node not seeded!".into()));

    ir.check_valid()
}

#[test]
fn evaluate_seed_non_leaf() -> Result<(), GraphError> {
    let mut ir = Graph::default();
    let size = Size::variable() * 2;

    let x = ir.add_input(TType::new(size, DType::F32));
    let y = ir.add_input(TType::new(size, DType::F32));
    let z = mock(&mut ir, x, y)?;
    ir.register_output(z);

    let ix = TValue::F32(vec![1.0, 2.0, 3.0, 4.0]);
    let inputs = [(x, ix.clone()), (y, ix.clone()), (z, ix)];
    assert_eq!(ir.evaluate(inputs), Err("Seeded non-leaf node!".into()));

    ir.check_valid()
}
