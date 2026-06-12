use std::{collections::BTreeMap, fmt::Debug, marker::PhantomData, sync::Arc};

use bullet_gpu::{
    buffer::Buffer,
    runtime::{Device, Gpu, Stream},
};

use crate::optimiser::{OptimiserState, OptimiserUpdateResult};

pub struct WrapOptimiser<O, P> {
    optimiser: O,
    phantom_data: PhantomData<P>,
}

impl<G, O, P> OptimiserState<G> for WrapOptimiser<O, P>
where
    G: Gpu,
    O: OptimiserState<G>,
    P: Clone + Default + Debug + Into<O::Params>,
{
    type Params = P;

    fn new(device: &Arc<Device<G>>, size: usize, params: Self::Params) -> Result<Self, G::Error> {
        Ok(Self { optimiser: O::new(device, size, params.into())?, phantom_data: PhantomData })
    }

    fn update<'a>(
        &'a mut self,
        stream: &Arc<Stream<G>>,
        weights: Arc<Buffer<G>>,
        grads: Arc<Buffer<G>>,
        gradient_factor: Arc<Buffer<G>>,
        learning_rate: Arc<Buffer<G>>,
    ) -> OptimiserUpdateResult<'a, G> {
        self.optimiser.update(stream, weights, grads, gradient_factor, learning_rate)
    }

    fn reset(&mut self) -> Result<(), G::Error> {
        self.optimiser.reset()
    }

    fn set_params(&mut self, params: Self::Params) -> Result<(), G::Error> {
        self.optimiser.set_params(params.into())
    }

    fn load_from_checkpoint(map: &mut BTreeMap<String, &mut Self>, path: &str) -> Result<(), G::Error> {
        let mut map = map.iter_mut().map(|(id, single)| (id.clone(), &mut single.optimiser)).collect();
        O::load_from_checkpoint(&mut map, path)
    }

    fn write_to_checkpoint(map: &BTreeMap<String, &Self>, path: &str) -> Result<(), G::Error> {
        let map = map.iter().map(|(id, single)| (id.clone(), &single.optimiser)).collect();
        O::write_to_checkpoint(&map, path)
    }
}
