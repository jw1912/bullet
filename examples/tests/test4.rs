use bullet_lib::{
    game::inputs::{Chess768, SparseInputType},
    value::loader::{DirectSequentialDataLoader, LoadableDataType},
};
use bullet_trainer::{
    model::{ModelDefinition, ModelInputs, ModelInputsMapper, ModelWeights},
    optimiser::{
        Optimiser,
        adam::{AdamW, AdamWParams},
    },
    run::{DefaultDevice, ReadMapLoader, TrainingSchedule, TrainingSteps, train},
};

fn main() {
    let wdl = 0.2;

    let feats = Chess768;
    let num_inputs = feats.num_inputs();
    let nnz = feats.max_active();

    let inputs = ModelInputs::default()
        .add_sparse("stm", (num_inputs, 1), nnz)
        .add_sparse("ntm", (num_inputs, 1), nnz)
        .add_dense("target", (1, 1));

    let mapper = ModelInputsMapper::build(&inputs, move |pnt, _, ((stm, ntm), target)| {
        let mut i = 0;

        feats.map_features(pnt, |sfeat, nfeat| {
            assert!(sfeat.max(nfeat) < num_inputs);
            stm[i] = sfeat as i32;
            ntm[i] = nfeat as i32;
            i += 1;
        });

        assert!(i <= nnz);

        for j in i..nnz {
            stm[j] = -1;
            ntm[j] = -1;
        }

        let score = 1.0 / (1.0 + (-f32::from(pnt.score) / 400.0).exp());
        let result = f32::from(pnt.result() as u8) / 2.0;

        target[0] = wdl * result + (1.0 - wdl) * score;
    });

    let defn = ModelDefinition::build(&inputs, |builder, ((stm, ntm), target)| {
        let l0 = builder.new_affine("l0", 768, 32);
        let l1 = builder.new_affine("l1", 2 * 32, 1);

        let stm_hidden = l0.forward(stm).screlu();
        let ntm_hidden = l0.forward(ntm).screlu();
        let hidden_layer = stm_hidden.concat(ntm_hidden);
        let output = l1.forward(hidden_layer);

        let loss = output.sigmoid().squared_error(target).reduce_sum_batch();

        (Some(loss), vec![("output".to_string(), output)])
    });

    let weights = ModelWeights::new(&defn, 198273612);
    let device = DefaultDevice::new(0).unwrap();
    let mut optimiser = Optimiser::<_, AdamW<_>>::new(defn, weights, device, AdamWParams::default()).unwrap();

    let reader = DirectSequentialDataLoader::new(&["examples/tests/batch.bf"]);
    let loader = ReadMapLoader::new(reader, mapper, 0, 4);

    let schedule = TrainingSchedule {
        steps: TrainingSteps { batch_size: 16_384, batches_per_superbatch: 1, start_superbatch: 1, end_superbatch: 10 },
        log_rate: 128,
        lr_schedule: Box::new(|_, _| 0.001),
    };

    train(&mut optimiser, schedule, loader, |_, _, _, _| {}, |_, _| {}).unwrap();
}
