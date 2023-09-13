pub mod bindings;
mod gradient;
pub mod util;

pub use gradient::{calc_gradient, free_everything, malloc_everything};