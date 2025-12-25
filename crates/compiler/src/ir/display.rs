use std::{collections::HashSet, fmt};

use crate::{
    common::Ansi,
    ir::{
        IrError, IrGraph,
        node::{IrNode, IrNodeId},
        operation::{IrOperation, Leaf},
    },
};

impl fmt::Display for IrGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn map<T>(x: Result<T, IrError>) -> Result<T, fmt::Error> {
            x.map_err(|_| fmt::Error)
        }

        write!(f, "irgraph(")?;
        let leaves = self.ops.values().filter(|x| IrOperation::downcast::<Leaf>(x.op()).is_some()).collect::<Vec<_>>();
        let mline = leaves.len() >= 5;

        for (i, leaf) in leaves.iter().enumerate() {
            if mline {
                writeln!(f)?;
                write!(f, "    ")?;
            } else if i != 0 {
                write!(f, ", ")?;
            }

            let node = leaf.outputs()[0];
            let ty = map(self.get_node_type(node))?;

            write!(f, "{node:?}: {ty:?}")?;
        }

        if mline {
            writeln!(f)?;
        }

        writeln!(f, ") {{")?;

        for id in map(self.topo_order_ops())? {
            let op = map(self.get_op(id))?;

            if IrOperation::downcast::<Leaf>(op.op()).is_some() {
                continue;
            }

            let inputs = op.inputs();
            let outputs = op.outputs();

            write!(f, "    ")?;
            if outputs.len() > 1 {
                write!(f, "[")?;
            }

            let output_tys =
                map(outputs.iter().map(|x| self.get_node(*x).map(IrNode::ty)).collect::<Result<Vec<_>, _>>())?;

            for (i, (&output, ty)) in outputs.iter().zip(output_tys).enumerate() {
                if i != 0 {
                    write!(f, ", ")?;
                }

                write!(f, "{output:?}: {ty:?}")?;
            }

            if outputs.len() > 1 {
                write!(f, "]")?;
            }

            write!(f, " = {}(", op.op().opname())?;

            for (i, &input) in inputs.iter().enumerate() {
                if i != 0 {
                    write!(f, ", ")?;
                }

                write!(f, "{input:?}")?;
            }

            writeln!(f, ")")?;
        }

        write!(f, "    return ")?;
        for (i, &output) in self.outputs.iter().enumerate() {
            if i != 0 {
                write!(f, ", ")?;
            }

            write!(f, "{output:?}")?;
        }

        writeln!(f)?;
        write!(f, "}}")?;

        Ok(())
    }
}

impl IrGraph {
    pub fn as_highlighted(&self) -> String {
        let kwd = &Ansi::rgb(183, 120, 221);
        let cnt = &Ansi::rgb(86, 182, 194);
        let inp = &Ansi::rgb(209, 154, 102);
        let brk = &Ansi::rgb(37, 113, 242);
        let typ = &Ansi::rgb(229, 187, 107);

        fn rgb(s: impl fmt::Display, colour: impl fmt::Display) -> String {
            format!("{colour}{s}{}", Ansi::rgb(171, 178, 191))
        }

        let mut s = self.to_string();
        s = s.replace("[", &rgb("[", brk));
        s = s.replace("]", &rgb("]", brk));
        s = s.replace("(", &rgb("(", brk));
        s = s.replace(")", &rgb(")", brk));
        s = s.replace("{", &rgb("{", brk));
        s = s.replace("}", &rgb("}", brk));
        s = format!("{}{s}", Ansi::rgb(171, 178, 191));
        s = s.replace("irgraph", &rgb("irgraph", kwd));
        s = s.replace("return", &rgb("return", kwd));
        s = s.replace("constant", &rgb("constant", kwd));
        s = s.replace(":", &rgb(":", cnt));
        s = s.replace(".", &rgb(".", cnt));
        s = s.replace("=", &rgb("=", cnt));
        s = s.replace("f32", &rgb("f32", typ));
        s = s.replace("i32", &rgb("i32", typ));

        let mut vars = HashSet::new();
        let mut var = 0;
        let mut in_var = false;

        let _ = s.replace(
            |c: char| {
                if in_var {
                    if let Some(digit) = c.to_digit(10) {
                        var = 10 * var + digit;
                    } else {
                        vars.insert(var);
                        var = 0;
                        in_var = false;
                    }
                }

                if c == '%' {
                    in_var = true;
                }

                false
            },
            "",
        );

        let mut vars = vars.into_iter().collect::<Vec<_>>();
        vars.sort();
        vars.reverse();

        for var in vars {
            let id = IrNodeId::new(var as usize);
            let name = if self.is_input(id) {
                format!("%{}{}", Ansi::Clear, rgb(var, inp))
            } else {
                format!("%{}{var}", Ansi::rgb(171, 178, 191))
            };
            s = s.replace(&format!("{id:?}"), &name);
        }

        s = s.replace("%", &rgb("%", cnt));

        s
    }

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
