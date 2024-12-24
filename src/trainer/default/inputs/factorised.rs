use super::InputType;

pub trait Factorises<T: InputType>: InputType<RequiredDataType = T::RequiredDataType> {
    fn derive_feature(&self, input: &T, feat: usize) -> Option<usize>;
}

#[derive(Clone, Copy, Default)]
pub struct Factorised<A: InputType, B: Factorises<A>> {
    normal: A,
    factoriser: B,
}

impl<A: InputType, B: Factorises<A>> Factorised<A, B> {
    pub fn from_parts(normal: A, factoriser: B) -> Self {
        Self { normal, factoriser }
    }
}

impl<A: InputType, B: Factorises<A>> InputType for Factorised<A, B> {
    type RequiredDataType = <A as InputType>::RequiredDataType;
    type FeatureIter = FactorisedIter<A, B>;

    fn inputs(&self) -> usize {
        self.normal.inputs()
    }

    fn buckets(&self) -> usize {
        self.normal.buckets()
    }

    fn size(&self) -> usize {
        self.normal.size() + self.factoriser.size()
    }

    fn max_active_inputs(&self) -> usize {
        2 * self.normal.max_active_inputs()
    }

    fn feature_iter(&self, pos: &Self::RequiredDataType) -> Self::FeatureIter {
        FactorisedIter {
            iter: self.normal.feature_iter(pos),
            queued: None,
            offset: self.factoriser.size(),
            inputs: *self,
        }
    }

    fn is_factorised(&self) -> bool {
        true
    }

    fn merge_factoriser(&self, unmerged: Vec<f32>) -> Vec<f32> {
        let src_size = self.size();

        assert_eq!(unmerged.len() % src_size, 0);
        let layer_size = unmerged.len() / src_size;
        let offset = self.factoriser.size();

        (0..self.normal.size() * layer_size)
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

    fn description(&self) -> String {
        format!("{} factorised by {}", self.normal.description(), self.factoriser.description().to_lowercase())
    }
}

pub struct FactorisedIter<A: InputType, B: Factorises<A>> {
    iter: A::FeatureIter,
    queued: Option<(usize, usize)>,
    offset: usize,
    inputs: Factorised<A, B>,
}

impl<A: InputType, B: Factorises<A>> FactorisedIter<A, B> {
    fn map_feat(&self, feat: (usize, usize)) -> Option<(usize, usize)> {
        let stm = self.inputs.factoriser.derive_feature(&self.inputs.normal, feat.0);
        let ntm = self.inputs.factoriser.derive_feature(&self.inputs.normal, feat.1);

        match (stm, ntm) {
            (Some(stm), Some(ntm)) => Some((stm, ntm)),
            (None, None) => None,
            _ => panic!("One factorised feature existed but the other did not!"),
        }
    }
}

impl<A: InputType, B: Factorises<A>> Iterator for FactorisedIter<A, B> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(feats) = self.queued {
            self.queued = None;
            Some(feats)
        } else {
            self.iter.next().map(|feat| {
                self.queued = self.map_feat(feat);
                (self.offset + feat.0, self.offset + feat.1)
            })
        }
    }
}
