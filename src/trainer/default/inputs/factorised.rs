use super::SparseInputType;

pub trait Factorises<T: SparseInputType>: SparseInputType<RequiredDataType = T::RequiredDataType> {
    fn derive_feature(&self, input: &T, feat: usize) -> Option<usize>;
}

#[derive(Clone, Copy, Default)]
pub struct Factorised<A: SparseInputType, B: Factorises<A>> {
    normal: A,
    factoriser: B,
    offset: usize,
}

impl<A: SparseInputType, B: Factorises<A>> Factorised<A, B> {
    pub fn from_parts(normal: A, factoriser: B) -> Self {
        let offset = factoriser.num_inputs();
        Self { normal, factoriser, offset }
    }
}

impl<A: SparseInputType, B: Factorises<A>> SparseInputType for Factorised<A, B> {
    type RequiredDataType = <A as SparseInputType>::RequiredDataType;

    fn num_inputs(&self) -> usize {
        self.normal.num_inputs() + self.factoriser.num_inputs()
    }

    fn max_active(&self) -> usize {
        2 * self.normal.max_active()
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        self.normal.map_features(pos, |stm, ntm| {
            f(self.offset + stm, self.offset + ntm);

            let stm = self.factoriser.derive_feature(&self.normal, stm);
            let ntm = self.factoriser.derive_feature(&self.normal, ntm);

            match (stm, ntm) {
                (Some(stm), Some(ntm)) => {
                    assert!(stm < self.offset && ntm < self.offset, "Factoriser feature exceeded factoriser size!");
                    f(stm, ntm);
                }
                (None, None) => {}
                _ => panic!("One factorised feature existed but the other did not!"),
            }
        });
    }

    fn shorthand(&self) -> String {
        self.normal.shorthand()
    }

    fn description(&self) -> String {
        format!("{}, factorised by {}", self.normal.description(), self.factoriser.description().to_lowercase())
    }

    fn is_factorised(&self) -> bool {
        true
    }

    fn merge_factoriser(&self, unmerged: Vec<f32>) -> Vec<f32> {
        let src_size = self.num_inputs();

        assert_eq!(unmerged.len() % src_size, 0);
        let layer_size = unmerged.len() / src_size;
        let offset = self.factoriser.num_inputs();

        (0..self.normal.num_inputs() * layer_size)
            .map(|elem| {
                let feat = elem / layer_size;
                let idx = elem % layer_size;
                let factoriser = self
                    .factoriser
                    .derive_feature(&self.normal, feat)
                    .map(|feat| unmerged[layer_size * feat + idx])
                    .unwrap_or(0.0);

                unmerged[layer_size * (feat + offset) + idx] + factoriser
            })
            .collect()
    }
}
