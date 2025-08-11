#![allow(deprecated)]

pub mod gamerunner;
pub mod testing;

/// Re-exports crates for certain file formats (e.g. Bulletformat)
pub mod formats {
    pub use bulletformat;
    pub use montyformat;
    pub use sfbinpack;
}

pub use crate::{
    game::{inputs, outputs},
    value::loader,
};
