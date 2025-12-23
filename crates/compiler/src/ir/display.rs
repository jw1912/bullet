use std::fmt;

use crate::ir::{IrError, IrGraph, node::IrNode};

impl fmt::Display for IrGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn map<T>(x: Result<T, IrError>) -> Result<T, fmt::Error> {
            x.map_err(|_| fmt::Error)
        }

        writeln!(f, "start")?;

        for id in map(self.topo_order_ops())? {
            let op = map(self.get_op(id))?;
            let inputs = op.inputs();
            let outputs = op.outputs();

            if outputs.len() > 1 {
                write!(f, "[")?;
            }

            let output_tys =
                map(outputs.iter().map(|x| self.get_node(*x).map(IrNode::ty)).collect::<Result<Vec<_>, _>>())?;

            for (i, (&output, ty)) in outputs.iter().zip(output_tys).enumerate() {
                write!(f, "{output:?} : {ty:?}")?;
                if i != outputs.len() - 1 {
                    write!(f, ", ")?;
                }
            }

            if outputs.len() > 1 {
                write!(f, "]")?;
            }

            write!(f, " = {}(", op.op().opname())?;

            for (i, &input) in inputs.iter().enumerate() {
                write!(f, "{input:?}")?;
                if i != inputs.len() - 1 {
                    write!(f, ", ")?;
                }
            }

            writeln!(f, ")")?;
        }

        write!(f, "return(")?;
        for (i, &output) in self.outputs.iter().enumerate() {
            write!(f, "{output:?}")?;
            if i != self.outputs.len() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, ")")?;

        Ok(())
    }
}

impl IrGraph {
    pub fn as_graphviz(&self) -> Result<String, std::fmt::Error> {
        use std::fmt::Write;

        let mut s = String::new();

        let op_ids = self.topo_order_ops().unwrap();

        let get_label = |id, op| {
            let outputs = self.get_op(op).unwrap().outputs();

            if outputs.len() > 1 {
                let idx = outputs.iter().position(|&x| x == id).unwrap();
                format!(" [label=.{idx}]")
            } else {
                "".to_string()
            }
        };

        writeln!(&mut s, "digraph G {{ node [style=filled,color=lightgrey];")?;

        for op_id in op_ids {
            let op = self.get_op(op_id).unwrap();

            let opname = op.op().opname();
            let inputs = op.inputs();
            let lbl = op_id.inner();

            if inputs.is_empty() {
                writeln!(&mut s, "op{lbl} [label=\"{opname}\", style=filled, color=lightblue];")?;
            } else {
                writeln!(&mut s, "op{lbl} [label=\"{opname}\"];")?;

                for &input in inputs {
                    let parent_op = self.get_parent_op(input).unwrap();
                    let label = get_label(input, parent_op);
                    writeln!(&mut s, "op{} -> op{lbl:?}{label};", parent_op.inner())?;
                }
            }
        }

        writeln!(&mut s, "return [label=\"return\", style=filled, color=green];")?;

        for &output in &self.outputs {
            let parent_op = self.get_parent_op(output).unwrap();
            let label = get_label(output, parent_op);
            writeln!(&mut s, "op{} -> return{label};", parent_op.inner())?;
        }

        write!(&mut s, "}}")?;

        Ok(s)
    }
}
