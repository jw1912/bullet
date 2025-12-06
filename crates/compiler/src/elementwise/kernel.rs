use std::{
    cell::RefCell,
    collections::{HashMap, hash_map},
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

        ElementwiseMut { builder: self, idx, dtype }
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
}

impl<'a> ElementwiseMut<'a> {
    pub fn read(&self) -> ElementwiseNode<'a> {
        let out = self.builder.builder.add_input(self.dtype);

        self.builder.reads.borrow_mut().insert(out.node, (true, self.idx));

        out
    }

    pub fn write(self, node: ElementwiseNode<'a>) {
        self.builder.writes.borrow_mut().insert(node.node, self.idx);
    }
}
