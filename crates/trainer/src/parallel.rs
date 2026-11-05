mod comm;
mod data;
mod plan;

pub use data::{DataParallel, StepOutput};

use std::{collections::BTreeMap, sync::Arc};

use bullet_gpu::{
    buffer::SyncOnValue,
    runtime::{Gpu, Stream},
};

use crate::optimiser::OptimiserState;

fn sync_all<G: Gpu, T>(handles: Vec<SyncOnValue<G, T>>) -> Result<(), G::Error> {
    let mut result = Ok(());

    for handle in handles {
        result = result.and(handle.value().map(drop));
    }

    result
}

unsafe fn release_all<G: Gpu, T>(handles: Vec<SyncOnValue<G, T>>) {
    for handle in handles {
        unsafe { handle.release() }
    }
}

fn drain<G: Gpu, T, E>(result: Result<T, E>, streams: &[Arc<Stream<G>>]) -> Result<T, E> {
    if result.is_err() {
        for stream in streams {
            let _ = stream.sync();
        }
    }

    result
}

fn write_state<G: Gpu, S: OptimiserState<G>>(state: &BTreeMap<String, S>, path: &str) -> Result<(), G::Error> {
    let map = state.iter().map(|(id, single)| (id.clone(), single)).collect();
    S::write_to_checkpoint(&map, path)
}

fn load_state<G: Gpu, S: OptimiserState<G>>(state: &mut BTreeMap<String, S>, path: &str) -> Result<(), G::Error> {
    let mut map = state.iter_mut().map(|(id, single)| (id.clone(), single)).collect();
    S::load_from_checkpoint(&mut map, path)
}
