mod concat;
mod matmul;
mod sparse_affine;
mod unary;

pub use concat::*;
pub use matmul::*;
pub use sparse_affine::*;
pub use unary::*;

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
