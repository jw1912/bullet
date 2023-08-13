use crate::{arch::{Accumulator, NNUE, NNUEParams, HIDDEN}, position::Position};

const SCALE: i32 = 400;
const QA: i32 = 255;
const QB: i32 = 64;
const QAB: i32 = QA * QB;

pub type QuantisedNNUE = NNUE<i16>;

impl QuantisedNNUE {
    pub fn from_unquantised(nnue: &NNUEParams) -> Box<Self> {
        let mut res = Box::<QuantisedNNUE>::default();

        for (i, &param) in nnue.feature_weights.iter().enumerate() {
            res.feature_weights[i] = (param * f64::from(QA)) as i16;
        }

        for (i, &param) in nnue.feature_bias.iter().enumerate() {
            res.feature_bias[i] = (param * f64::from(QA)) as i16;
        }

        for (i, &param) in nnue.output_weights.iter().enumerate() {
            res.output_weights[i] = (param * f64::from(QB)) as i16;
        }

        res.output_bias = (nnue.output_bias * f64::from(QAB)) as i16;

        res
    }
}

fn activate(x: i16) -> i32 {
    i32::from(x.max(0))
}

pub fn eval(pos: &Position, nnue: &QuantisedNNUE) -> i32 {
    let mut acc = Accumulator::<i16, HIDDEN>::new(nnue.feature_bias);

    for &feature in pos.active.iter().take(pos.num) {
        acc.add_feature(usize::from(feature), nnue);
    }

    let mut sum = 0;
    for (&i, &w) in acc.0.iter().zip(&nnue.output_weights) {
        sum += activate(i) * i32::from(w);
    }

    let flatten = sum / QA;

    (flatten + i32::from(nnue.output_bias)) * SCALE / QAB
}

impl QuantisedNNUE {
    pub fn write_to_bin(&self, output_path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(output_path)?;
        const SIZEOF: usize = std::mem::size_of::<QuantisedNNUE>();
        unsafe {
            file.write_all(
                &std::mem::transmute::<QuantisedNNUE, [u8; SIZEOF]>(self.clone())
            )?;
        }
        Ok(())
    }
}

#[test]
fn test_eval() {
    let nnue = Box::<QuantisedNNUE>::new(
        unsafe {
            std::mem::transmute(*include_bytes!("../maiden-500.bin"))
        }
    );
    const FENS: [&str; 3] = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - ",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -",
    ];
    for fen in FENS {
        let pos = fen.parse::<Position>().unwrap();
        let score = eval(&pos, &nnue);
        println!("eval: {score}");
    }
}