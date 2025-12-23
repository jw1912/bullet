use std::{
    cell::RefCell,
    collections::{HashMap, HashSet, hash_map},
    sync::atomic::{AtomicBool, Ordering},
};

use crate::{
    common::{DType, DTypeTensor, Size},
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

    pub fn evaluate(&self, reads: &[&DTypeTensor], writes: &mut [&mut DTypeTensor]) -> Option<()> {
        if reads.len() != self.num_refs || writes.len() != self.num_muts {
            return None;
        }

        let read_size = reads.iter().map(|x| x.size()).collect::<HashSet<_>>();
        let write_size = writes.iter().map(|x| x.size()).collect::<HashSet<_>>();

        let sizes = read_size.union(&write_size).collect::<Vec<_>>();

        if sizes.len() != 1 {
            return None;
        }

        let size = *sizes[0];

        for idx in 0..size {
            let inputs = self
                .reads
                .iter()
                .map(|(&id, &(from_write, i))| (id, if from_write { writes[i].read(idx) } else { reads[i].read(idx) }))
                .collect();

            let outputs = self.writes.keys().cloned().collect::<Vec<_>>();

            let values = self.desc.evaluate(inputs, &outputs)?;

            for (out, value) in outputs.iter().zip(values) {
                writes[*self.writes.get(out)?].write(idx, value);
            }
        }

        Some(())
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

    #[test]
    fn evaluate() {
        let builder = ElementwiseKernelBuilder::default();
        let a = builder.add_input(DType::F32);
        let b = builder.add_input_mut(DType::F32);
        let c = builder.add_input_mut(DType::F32);

        let x = a.read();
        let y = b.read();
        let z = x + y;

        c.write(y);
        b.write(z);

        let kernel = builder.build(Size::variable());

        let a = DTypeTensor::F32(vec![1.0; 8]);
        let mut b = DTypeTensor::F32(vec![1.0; 8]);
        let mut c = DTypeTensor::F32(vec![0.0; 8]);

        kernel.evaluate(&[&a], &mut [&mut b, &mut c]).unwrap();

        assert_eq!(b, DTypeTensor::F32(vec![2.0; 8]));
        assert_eq!(c, DTypeTensor::F32(vec![1.0; 8]));
    }

    #[test]
    fn evaluate_wrong_num_args() {
        let builder = ElementwiseKernelBuilder::default();

        let a = builder.add_input_mut(DType::F32);
        let x = a.read();
        let y = 1.0 + x;
        a.write(y);

        let kernel = builder.build(Size::variable());

        let a = DTypeTensor::F32(vec![1.0; 8]);

        assert!(kernel.evaluate(&[&a], &mut []).is_none());
    }

    #[test]
    fn evaluate_wrong_dtype() {
        let builder = ElementwiseKernelBuilder::default();

        let a = builder.add_input_mut(DType::F32);
        let x = a.read();
        let y = 1.0 + x;
        a.write(y);

        let kernel = builder.build(Size::variable());

        let mut a = DTypeTensor::I32(vec![1; 8]);

        assert!(kernel.evaluate(&[], &mut [&mut a]).is_none());
    }
}
