use std::ops::Range;

const COPY_CHUNK: usize = 1 << 22;

#[derive(Clone, Copy)]
pub enum Slot {
    Data,
    Stage(usize),
    Reduced,
}

#[derive(Clone, Copy)]
pub enum Op {
    Copy { tensor: usize, src: (usize, Slot), src_start: usize, dst: (usize, Slot), dst_start: usize, len: usize },
    Reduce { rank: usize, tensor: usize },
}

#[derive(Default)]
pub struct Plan {
    pub rounds: Vec<Vec<Op>>,
}

pub fn shard_ranges(size: usize, ranks: usize) -> Vec<Range<usize>> {
    let chunk = size.div_ceil(ranks).next_multiple_of(4);
    (0..ranks).map(|q| (q * chunk).min(size)..((q + 1) * chunk).min(size)).collect()
}

pub fn split_batch(batch: usize, ranks: usize) -> Vec<Range<usize>> {
    let mut start = 0;

    (0..ranks)
        .map(|rank| {
            let len = batch / ranks + usize::from(rank < batch % ranks);
            start += len;
            start - len..start
        })
        .collect()
}

impl Plan {
    pub fn all_reduce(ranks: usize, sizes: &[usize]) -> Self {
        if ranks == 1 {
            return Self::default();
        }

        let shards = sizes.iter().map(|&size| shard_ranges(size, ranks)).collect::<Vec<_>>();
        let mut rounds = Vec::new();

        let mut scatter = vec![Vec::new(); ranks];
        for (tensor, ranges) in shards.iter().enumerate() {
            for (src, ops) in scatter.iter_mut().enumerate() {
                for (dst, range) in ranges.iter().enumerate().filter(|(dst, _)| *dst != src) {
                    let dst = (dst, Slot::Stage(src));
                    push_chunked(ops, tensor, (src, Slot::Data), range.start, dst, 0, range.len());
                }
            }
        }
        rounds.push(interleave(scatter));

        let mut reduce = Vec::new();
        for (tensor, ranges) in shards.iter().enumerate() {
            for (rank, range) in ranges.iter().enumerate() {
                if !range.is_empty() {
                    reduce.push(Op::Reduce { rank, tensor });
                }
            }
        }
        rounds.push(reduce);

        for k in 1..ranks {
            let mut gather = vec![Vec::new(); ranks];
            for (tensor, ranges) in shards.iter().enumerate() {
                for (src, ops) in gather.iter_mut().enumerate() {
                    let (dst, range) = (((src + k) % ranks, Slot::Data), &ranges[src]);
                    push_chunked(ops, tensor, (src, Slot::Reduced), 0, dst, range.start, range.len());
                }
            }
            rounds.push(interleave(gather));
        }

        Self { rounds }
    }

    pub fn broadcast(ranks: usize, root: usize, sizes: &[usize]) -> Self {
        if ranks == 1 {
            return Self::default();
        }

        let mut ops = Vec::new();
        for (tensor, &size) in sizes.iter().enumerate() {
            for dst in (0..ranks).filter(|&dst| dst != root) {
                push_chunked(&mut ops, tensor, (root, Slot::Data), 0, (dst, Slot::Data), 0, size);
            }
        }

        Self { rounds: vec![ops] }
    }
}

fn push_chunked(
    ops: &mut Vec<Op>,
    tensor: usize,
    src: (usize, Slot),
    src_start: usize,
    dst: (usize, Slot),
    dst_start: usize,
    len: usize,
) {
    for offset in (0..len).step_by(COPY_CHUNK) {
        let len = COPY_CHUNK.min(len - offset);
        ops.push(Op::Copy { tensor, src, src_start: src_start + offset, dst, dst_start: dst_start + offset, len });
    }
}

fn interleave(per_rank: Vec<Vec<Op>>) -> Vec<Op> {
    let longest = per_rank.iter().map(Vec::len).max().unwrap_or(0);
    (0..longest).flat_map(|i| per_rank.iter().filter_map(move |ops| ops.get(i).copied())).collect()
}
