use std::{collections::HashMap, sync::Arc};

use bullet_compiler::frontend::{DType, DValue, IRBuilder, IRTrace};

use crate::runtime::{Device, ReadyToCompileGraph, Stream, TensorInput};

use super::{OptimiserState, OptimiserUpdateValue, utils::Placement};

fn build_clip_op(size: usize, min: f32, max: f32) -> Result<ReadyToCompileGraph, IRTrace> {
    let builder = IRBuilder::default();

    let min = builder.scalar(DValue::F32(min), size);
    let max = builder.scalar(DValue::F32(max), size);
    let w = builder.add_input(size, DType::F32);
    let new_w = w.min(max)?.max(min)?;

    let ir = builder.build([new_w]);

    ReadyToCompileGraph::new(ir, [("w".to_string(), TensorInput::InOut(w.node(), new_w.node()))].into())
}

#[derive(Clone, Debug)]
pub struct WeightClippingParams<T> {
    pub inner: T,
    pub placement: Placement,
    pub min: f32,
    pub max: f32,
}

impl<T: Default> Default for WeightClippingParams<T> {
    fn default() -> Self {
        Self { inner: T::default(), placement: Placement::After, min: -1.98, max: 1.98 }
    }
}

pub struct WeightClipping<D: Device, S: OptimiserState<D>> {
    inner: S,
    placement: Placement,
    op: D::CompiledGraph,
    device: Arc<D>,
    size: usize,
}

impl<D: Device, S: OptimiserState<D>> OptimiserState<D> for WeightClipping<D, S> {
    type Params = WeightClippingParams<S::Params>;

    fn new(device: Arc<D>, size: usize, params: Self::Params) -> Result<Self, D::Error> {
        Ok(Self {
            op: device.compile(build_clip_op(size, params.min, params.max).unwrap())?,
            inner: S::new(device.clone(), size, params.inner.clone())?,
            placement: params.placement,
            device,
            size,
        })
    }

    fn update(
        &mut self,
        stream: &Arc<D::Stream>,
        weights: Arc<D::Buffer>,
        grads: Arc<D::Buffer>,
        gradient_factor: Arc<D::Buffer>,
        learning_rate: Arc<D::Buffer>,
    ) -> Result<OptimiserUpdateValue<D>, D::Error> {
        let mut blocks = Vec::new();

        let args = [("w".to_string(), weights.clone())].into();

        if self.placement == Placement::Before {
            blocks.push(stream.execute_graph(&self.op, &args)?);
        }

        blocks.extend(self.inner.update(stream, weights, grads, gradient_factor, learning_rate)?);

        if self.placement == Placement::After {
            blocks.push(stream.execute_graph(&self.op, &args)?);
        }

        Ok(blocks)
    }

    fn reset(&mut self) -> Result<(), D::Error> {
        self.inner.reset()
    }

    fn set_params(&mut self, params: Self::Params) -> Result<(), D::Error> {
        self.inner.set_params(params.inner)?;
        self.placement = params.placement;

        self.op = self.device.compile(build_clip_op(self.size, params.min, params.max).unwrap())?;

        Ok(())
    }

    fn load_from_checkpoint(map: &mut HashMap<String, &mut Self>, path: &str) -> Result<(), D::Error> {
        let mut map = map.iter_mut().map(|(id, single)| (id.clone(), &mut single.inner)).collect();
        S::load_from_checkpoint(&mut map, path)
    }

    fn write_to_checkpoint(map: &HashMap<String, &Self>, path: &str) -> Result<(), D::Error> {
        let map = map.iter().map(|(id, single)| (id.clone(), &single.inner)).collect();
        S::write_to_checkpoint(&map, path)
    }
}
