pub struct Single;
impl Single {
    pub const NUM: usize = 1;

    pub fn update_output_bucket(_: &mut usize, _: usize) {}

    pub fn get_bucket(_: usize) -> usize {
        0
    }
}

pub struct MaterialCount<const N: usize>;
impl<const N: usize> MaterialCount<N> {
    pub const NUM: usize = N;

    const DIVISOR: usize = 32 / Self::NUM;

    pub fn update_output_bucket(idx: &mut usize, _: usize) {
        *idx += 1;
    }

    pub fn get_bucket(idx: usize) -> usize {
        (idx - 2) / Self::DIVISOR
    }
}
