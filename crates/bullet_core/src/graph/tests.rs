mod activate;
mod affine;
mod concat;
mod sparse_affine;

pub use activate::*;
pub use affine::*;
pub use concat::*;
pub use sparse_affine::*;

#[macro_export]
macro_rules! make_tests {
    ($dev:expr $(, $id:ident)+ $(,)?) => {
        $(
            #[test]
            fn $id() {
                tests::$id($dev);
            }
        )+
    };
}

pub use make_tests;
