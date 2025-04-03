use std::{
    collections::HashMap,
    fmt::{self, Write},
    sync::atomic::{AtomicUsize, Ordering},
};

use super::{
    node::AnnotatedNode,
    op::{DiffableFromOutput, UnaryOp},
    shape::Shape,
    GraphIR, GraphIRNode,
};

impl fmt::Display for GraphIR {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let fmt = GraphIRStringFormat::default_colours();
        write!(f, "{}", self.to_formatted_string(&fmt)?)
    }
}

#[derive(Clone, Default)]
pub struct GraphIRStringFormat {
    pub(crate) op_name_fmt: Option<String>,
    pub(crate) op_node_fmt: Option<String>,
    pub(crate) op_fp_fmt: Option<String>,
    pub(crate) op_shape_fmt: Option<String>,
    pub(crate) op_paren_fmt: Option<String>,
    pub(crate) fill_newlines: bool,
}

impl GraphIRStringFormat {
    pub fn default_colours() -> Self {
        Self {
            op_name_fmt: Some("34".to_string()),
            op_node_fmt: Some("36;1".to_string()),
            op_fp_fmt: Some("32".to_string()),
            op_shape_fmt: Some("31".to_string()),
            op_paren_fmt: Some("35".to_string()),
            fill_newlines: true,
        }
    }
}

fn ansi<T: fmt::Display, U: fmt::Display>(x: T, y: U) -> String {
    format!("\x1b[{y}m{x}\x1b[0m")
}

fn maybe_fmt<T: fmt::Display>(x: T, fmt: &Option<String>, count: &AtomicUsize) -> String {
    if let Some(fmt) = fmt {
        count.fetch_add(7 + fmt.len(), Ordering::SeqCst);
        ansi(x, fmt)
    } else {
        x.to_string()
    }
}

impl GraphIR {
    pub fn to_formatted_string(&self, fmt: &GraphIRStringFormat) -> Result<String, fmt::Error> {
        let mut orig_shapes = HashMap::new();
        let mut s = String::new();
        let mut lines = Vec::new();

        let mut max_len = 12;

        for node in self.nodes.iter().flatten() {
            let count = AtomicUsize::new(0);
            let id = maybe_fmt(format!("{:0>2}", node.idx), &fmt.op_node_fmt, &count);
            let name = maybe_fmt(op_name(node), &fmt.op_name_fmt, &count);
            let lp = maybe_fmt("(", &fmt.op_paren_fmt, &count);
            let rp = maybe_fmt(")", &fmt.op_paren_fmt, &count);
            let args = op_args(node, &mut orig_shapes, fmt, &count);

            let line = format!("%{id} = {name}{lp}{args}{rp};");
            let l = line.chars().count() - count.load(Ordering::SeqCst);
            max_len = max_len.max(l);
            lines.push((line, l));
        }

        max_len += max_len % 2;
        let header = "Graph IR";
        let pad = "=".repeat((max_len - 10) / 2);

        writeln!(&mut s, "{pad} {header} {pad}")?;

        for (line, l) in &lines {
            writeln!(&mut s, "{line}{}", " ".repeat(max_len.saturating_sub(*l)))?;
        }

        if fmt.fill_newlines {
            for _ in 0..self.nodes.len() - lines.len() {
                writeln!(&mut s, "{}", " ".repeat(max_len))?;
            }
        }

        writeln!(&mut s, "{}", "=".repeat(max_len))?;

        Ok(s)
    }
}

fn op_args(
    node: &GraphIRNode,
    shapes: &mut HashMap<usize, Shape>,
    fmt: &GraphIRStringFormat,
    count: &AtomicUsize,
) -> String {
    use super::op::GraphIROp::*;

    shapes.insert(node.idx, node.shape);

    let shape = |x: Shape| maybe_fmt(x, &fmt.op_shape_fmt, count);

    let id = |x: &AnnotatedNode| {
        let name = maybe_fmt(x.idx, &fmt.op_node_fmt, count);
        if *shapes.get(&x.idx).unwrap() == x.shape {
            format!("%{name}")
        } else {
            format!("%{name}: {}", shape(x.shape))
        }
    };

    let fp = |fp: &f32| maybe_fmt(fp, &fmt.op_fp_fmt, count);

    match node.parent_operation.as_ref() {
        Some(op) => match op {
            Affine(a, b, c) => format!("{}, {}, {}", id(a), id(b), id(c)),
            Concat(a, b) => format!("{}, {}", id(a), id(b)),
            Gather(input, mask) => format!("{}, {}", id(input), id(mask)),
            LinearCombination(alpha, a, beta, b) => format!("{}, {}, {}, {}", fp(alpha), id(a), fp(beta), id(b)),
            Mask(input, mask) => format!("{}, {}", id(input), id(mask)),
            Matmul(a, ta, b, tb) => format!("{}, {ta}, {}, {tb}", id(a), id(b)),
            PairwiseMul(input, post_concat) => format!("{}, {post_concat}", id(input)),
            PowerError(a, b, pow) => format!("{}, {}, {pow}", id(a), id(b)),
            ReduceAcrossBatch(node) => id(node),
            Select(input, buckets) => format!("{}, {}", id(input), id(buckets)),
            Slice(input, a, b) => format!("{}, {a}, {b}", id(input)),
            SparseAffineActivate(w, i, b, act) => match (b, act) {
                (Some(b), DiffableFromOutput::Identity) => format!("{}, {}, {}", id(w), id(i), id(b)),
                (Some(b), _) => format!("{}, {}, {}, {act:?}", id(w), id(i), id(b)),
                (None, DiffableFromOutput::Identity) => format!("{}, {}", id(w), id(i)),
                (None, _) => format!("{}, {}, {act:?}", id(w), id(i)),
            },
            ToDense(node) => id(node),
            Unary(node, unary) => match unary {
                UnaryOp::DiffableFromOutput(_) => id(node),
                UnaryOp::Add(x) => format!("{}, {}", id(node), fp(x)),
                UnaryOp::Mul(x) => format!("{}, {}", id(node), fp(x)),
                UnaryOp::AbsPow(x) => format!("{}, {}", id(node), fp(x)),
            },
            SparseAffineDualActivate(w, s, n, b, act) => match (b, act) {
                (Some(b), DiffableFromOutput::Identity) => format!("{}, {}, {}, {}", id(w), id(s), id(n), id(b)),
                (Some(b), _) => format!("{}, {}, {}, {}, {act:?}", id(w), id(s), id(n), id(b)),
                (None, DiffableFromOutput::Identity) => format!("{}, {}, {}", id(w), id(s), id(n)),
                (None, _) => format!("{}, {}, {}, {act:?}", id(w), id(s), id(n)),
            },
            MaskedSoftmaxCrossEntropyLoss(mask, input, target) => {
                format!("{}, {}, {}", id(mask), id(input), id(target))
            }
            SoftmaxCrossEntropyLoss(a, b) => format!("{}, {}", id(a), id(b)),
        },
        None => {
            let layout = match node.sparse {
                Some(nnz) => format!("Sparse(f32, {nnz})"),
                None => "Dense(f32)".to_string(),
            };
            format!("{}, {}, {}, {}", shape(node.shape), layout, node.requires_grad, node.batched)
        }
    }
}

fn op_name(node: &GraphIRNode) -> String {
    use super::op::GraphIROp::*;

    match node.parent_operation.as_ref() {
        Some(op) => match op {
            SparseAffineActivate(_, _, b, act) => match (b, act) {
                (Some(_), DiffableFromOutput::Identity) => "SparseAffine",
                (Some(_), _) => "SparseAffineActivate",
                (None, DiffableFromOutput::Identity) => "SparseMatmul",
                (None, _) => "SparseMatmulActivate",
            }
            .to_string(),
            SparseAffineDualActivate(_, _, _, b, act) => match (b, act) {
                (Some(_), DiffableFromOutput::Identity) => "SparseAffineDual",
                (Some(_), _) => "SparseAffineDualActivate",
                (None, DiffableFromOutput::Identity) => "SparseMatmulDual",
                (None, _) => "SparseMatmulDualActivate",
            }
            .to_string(),
            Unary(_, unary) => match unary {
                UnaryOp::DiffableFromOutput(act) => format!("{act:?}"),
                UnaryOp::Add(_) => "Add".to_string(),
                UnaryOp::Mul(_) => "Mul".to_string(),
                UnaryOp::AbsPow(_) => "AbsPow".to_string(),
            },
            _ => {
                let s = format!("{op:?}");
                s.split('(').next().unwrap().to_string()
            }
        },
        None => "Create".to_string(),
    }
}
