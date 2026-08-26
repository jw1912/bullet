use std::time::{SystemTime, UNIX_EPOCH};

use oorandom::Rand64;

pub fn seeded_rng() -> Rand64 {
    let seed = SystemTime::now().duration_since(UNIX_EPOCH).expect("Guaranteed increasing.").as_micros();

    Rand64::new(seed)
}
