/*
This shows how to define your own input type, note that I plan to rework
this in the near future to make it more flexible.
*/

use bulletformat::ChessBoard;
use bullet::{
    inputs, Activation, LocalSettings, LrScheduler, TrainerBuilder, TrainingSchedule, WdlScheduler,
};

#[derive(Clone, Copy, Debug, Default)]
pub struct ChessBucketsWillow;
impl inputs::InputType for ChessBucketsWillow {
    type RequiredDataType = ChessBoard;

    // The number of inputs per bucket.
    fn inputs(&self) -> usize {
        768
    }

    // The number of buckets.
    fn buckets(&self) -> usize {
        4
    }

    // By default the `ChessBoard` type yields `(u8, u8, u8, u8)`
    // as an iterator, I plan to change it so you can implement
    // a custom iterator instead, if you, say, need to collect the
    // whole board into bitboards before extracting features.
    fn get_feature_indices(
        &self,
        (piece, square, our_ksq, opp_ksq): (u8, u8, u8, u8),
    ) -> (usize, usize) {
        let c = usize::from(piece & 8 > 0);
        let pc = 64 * usize::from(piece & 7);
        let sq = usize::from(square);

        let wks = usize::from(our_ksq & 7 > 3);
        let bks = usize::from(opp_ksq & 7 > 3);

        let wbucket = 2 * wks + bks;
        let bbucket = 2 * bks + wks;

        let wfeat = 768 * wbucket + [0, 384][c] + pc + sq;
        let bfeat = 768 * bbucket + [384, 0][c] + pc + (sq ^ 56);

        (wfeat, bfeat)
    }
}

fn main() {
    let mut trainer = TrainerBuilder::default()
        .set_batch_size(16_384)
        .set_eval_scale(400.0)
        .set_quantisations(&[255, 64])
        .set_input(ChessBucketsWillow)
        .ft(768)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();

    let schedule = TrainingSchedule {
        net_id: "willow".to_string(),
        start_epoch: 1,
        end_epoch: 25,
        wdl_scheduler: WdlScheduler::Constant { value: 0.4 },
        lr_scheduler: LrScheduler::Step {
            start: 0.001,
            gamma: 0.1,
            step: 10,
        },
        save_rate: 1,
    };

    let settings = LocalSettings {
        threads: 4,
        data_file_path: "data/willow.data",
        output_directory: "checkpoints",
    };

    trainer.run(&schedule, &settings);
}