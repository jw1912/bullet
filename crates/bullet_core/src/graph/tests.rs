mod activate;
mod concat;
mod matmul;
mod sparse_affine;

pub use activate::*;
pub use concat::*;
pub use matmul::*;
pub use sparse_affine::*;

#[macro_export]
macro_rules! make_tests {
    ($dev:expr $(, $id:ident)+ $(,)?) => {
        $(
            #[test]
            fn $id() {
                tests::$id($dev).unwrap()
            }
        )+
    };
}

pub use make_tests;
