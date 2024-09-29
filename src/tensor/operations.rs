mod add;
mod linear;

use diffable::{GraphBuilder, Node};

use super::Tensor;

pub struct Operation;
impl Operation {
    pub fn linear(builder: &mut GraphBuilder<Tensor>, a: Node, b: Node) -> Node {
        builder.create_result_of_operation(linear::linear(), &[a, b])
    }

    pub fn add(builder: &mut GraphBuilder<Tensor>, a: Node, b: Node) -> Node {
        builder.create_result_of_operation(add::add(), &[a, b])
    }
}
