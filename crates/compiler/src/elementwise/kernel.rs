use std::{
    cell::RefCell,
    collections::{HashMap, hash_map},
    sync::atomic::{AtomicBool, Ordering},
};

use crate::{
    DType, Size,
    elementwise::{ElementwiseBuilder, ElementwiseDescription, ElementwiseId, ElementwiseNode},
};

pub struct ElementwiseKernel {
    size: Size,
    num_refs: usize,
    num_muts: usize,
    reads: HashMap<ElementwiseId, (bool, usize)>,
    writes: HashMap<ElementwiseId, usize>,
    desc: ElementwiseDescription,
}

impl ElementwiseKernel {
    pub fn new(
        size: Size,
        reads: HashMap<ElementwiseId, (bool, usize)>,
        writes: HashMap<ElementwiseId, usize>,
        desc: ElementwiseDescription,
    ) -> Self {
        let num_refs = reads.values().filter_map(|x| (!x.0).then_some(x.1)).max().map(|x| x + 1).unwrap_or(0);
        let num_muts = writes.values().max().cloned().map(|x| x + 1).unwrap_or(0);

        Self { size, num_refs, num_muts, reads, writes, desc }
    }

    pub fn size(&self) -> Size {
        self.size
    }

    pub fn num_refs(&self) -> usize {
        self.num_refs
    }

    pub fn num_muts(&self) -> usize {
        self.num_muts
    }

    pub fn reads(&self) -> hash_map::Iter<'_, ElementwiseId, (bool, usize)> {
        self.reads.iter()
    }

    pub fn writes(&self) -> hash_map::Iter<'_, ElementwiseId, usize> {
        self.writes.iter()
    }

    pub fn desc(&self) -> &ElementwiseDescription {
        &self.desc
    }
}

#[derive(Default)]
pub struct ElementwiseKernelBuilder {
    builder: ElementwiseBuilder,
    num_refs: RefCell<usize>,
    num_muts: RefCell<usize>,
    reads: RefCell<HashMap<ElementwiseId, (bool, usize)>>,
    writes: RefCell<HashMap<ElementwiseId, usize>>,
}

impl ElementwiseKernelBuilder {
    pub fn add_input<'a>(&'a self, dtype: DType) -> ElementwiseRef<'a> {
        let mut refs = self.num_refs.borrow_mut();
        let idx = *refs;
        *refs += 1;

        ElementwiseRef { builder: self, idx, dtype }
    }

    pub fn add_input_mut<'a>(&'a self, dtype: DType) -> ElementwiseMut<'a> {
        let mut muts = self.num_muts.borrow_mut();
        let idx = *muts;
        *muts += 1;

        ElementwiseMut { builder: self, idx, dtype, read: AtomicBool::new(false) }
    }

    pub fn build(self, size: Size) -> ElementwiseKernel {
        ElementwiseKernel::new(size, self.reads.into_inner(), self.writes.into_inner(), self.builder.build())
    }
}

pub struct ElementwiseRef<'a> {
    builder: &'a ElementwiseKernelBuilder,
    idx: usize,
    dtype: DType,
}

impl<'a> ElementwiseRef<'a> {
    pub fn read(self) -> ElementwiseNode<'a> {
        let out = self.builder.builder.add_input(self.dtype);

        self.builder.reads.borrow_mut().insert(out.node, (false, self.idx));

        out
    }
}

pub struct ElementwiseMut<'a> {
    builder: &'a ElementwiseKernelBuilder,
    idx: usize,
    dtype: DType,
    read: AtomicBool,
}

impl<'a> ElementwiseMut<'a> {
    pub fn read(&self) -> ElementwiseNode<'a> {
        let read = self.read.fetch_or(true, Ordering::Relaxed);
        assert!(!read, "Already read!");

        let out = self.builder.builder.add_input(self.dtype);

        self.builder.reads.borrow_mut().insert(out.node, (true, self.idx));

        out
    }

    pub fn write(self, node: ElementwiseNode<'a>) {
        self.builder.writes.borrow_mut().insert(node.node, self.idx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adamw() {
        let beta1 = 0.99;
        let beta2 = 0.999;
        let lr = 0.001;

        let builder = ElementwiseKernelBuilder::default();
        let weights = builder.add_input_mut(DType::F32);
        let momentum = builder.add_input_mut(DType::F32);
        let velocity = builder.add_input_mut(DType::F32);
        let gradients = builder.add_input(DType::F32);

        let mut p = weights.read();
        let mut m = momentum.read();
        let mut v = velocity.read();
        let g = gradients.read();

        m = beta1 * m + (1.0 - beta1) * g;
        v = beta2 * v + (1.0 - beta2) * g * g;

        p = 0.99 * p;
        p = p - lr * m / (v.abs_powf(0.5) + 0.00000001);
        p = p.max(-0.99).min(0.99);

        weights.write(p);
        momentum.write(m);
        velocity.write(v);

        let kernel = builder.build(Size::variable());
        let desc = kernel.desc();

        assert_eq!(desc.roots(), 1);
        assert_eq!(desc.leaves(), 4);
    }
}
