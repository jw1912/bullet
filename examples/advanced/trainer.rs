use bullet_lib::{
    game::inputs::{get_num_buckets, ChessBucketsMirrored},
    nn::{
        optimiser::{AdamW, AdamWParams},
        ExecutionContext, InitSettings, Shape,
    },
    trainer::save::SavedFormat,
    value::{ValueTrainer, ValueTrainerBuilder},
};

use crate::output_buckets::{CustomOutputBuckets, NUM_OUTPUT_BUCKETS};

pub fn make_trainer(
    input_bucket_layout: [usize; 32],
    hl_size: usize,
) -> ValueTrainer<bullet_core::optimiser::adam::AdamW<ExecutionContext>, ChessBucketsMirrored, CustomOutputBuckets> {
    let num_input_buckets = get_num_buckets(&input_bucket_layout);
    let num_inputs = 768 * num_input_buckets;
    let num_output_buckets = NUM_OUTPUT_BUCKETS;

    assert!(num_input_buckets > 1, "Factoriser is worthless with only one bucket!");
    assert!(num_output_buckets > 1, "1 output bucket does not make sense!");

    let save_format = [
        SavedFormat::id("l0w")
            .add_transform(|builder, _, mut weights| {
                let factoriser = builder.get_weights("l0f").get_dense_vals().unwrap();
                let expanded = factoriser.repeat(weights.len() / factoriser.len());

                for (i, &j) in weights.iter_mut().zip(expanded.iter()) {
                    *i += j;
                }

                weights
            })
            .round()
            .quantise::<i16>(255),
        SavedFormat::id("l0b").round().quantise::<i16>(255),
        SavedFormat::id("l1w").round().quantise::<i8>(64).transpose(),
        SavedFormat::id("l1b"),
        SavedFormat::id("l2w").transpose(),
        SavedFormat::id("l2b"),
        SavedFormat::id("l3w").transpose(),
        SavedFormat::id("l3b"),
    ];

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .inputs(ChessBucketsMirrored::new(input_bucket_layout))
        .output_buckets(CustomOutputBuckets)
        .optimiser(AdamW)
        .save_format(&save_format)
        .build_custom(|builder, (stm, ntm, buckets), targets| {
            // input layer factoriser
            let l0f = builder.new_weights("l0f", Shape::new(hl_size, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(num_input_buckets);

            // input layer weights
            let mut l0 = builder.new_affine("l0", num_inputs, hl_size);
            l0.weights = l0.weights + expanded_factoriser;

            // layerstack weights
            let l1 = builder.new_affine("l1", hl_size, num_output_buckets * 16);
            let l2 = builder.new_affine("l2", 30, num_output_buckets * 32);
            let l3 = builder.new_affine("l3", 32, num_output_buckets);

            // input layer inference
            let stm_subnet = l0.forward(stm).crelu().pairwise_mul();
            let ntm_subnet = l0.forward(ntm).crelu().pairwise_mul();
            let mut out = stm_subnet.concat(ntm_subnet);

            // layerstack inference
            out = l1.forward(out).select(buckets);

            let skip_neuron = out.slice_rows(15, 16);

            out = out.slice_rows(0, 15);
            out = out.concat(out.abs_pow(2.0)).crelu();

            out = l2.forward(out).select(buckets).screlu();
            out = l3.forward(out).select(buckets);

            // network output
            out = out + skip_neuron;

            // squared error loss
            let loss = out.sigmoid().squared_error(targets);

            (out, loss)
        });

    // need to account for factoriser weight magnitudes
    let stricter_clipping = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", stricter_clipping);
    trainer.optimiser.set_params_for_weight("l0f", stricter_clipping);

    trainer
}
