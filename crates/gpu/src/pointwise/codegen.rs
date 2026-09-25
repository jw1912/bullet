//! Golden tests for the kernel code generation path - `generate`, `PointwiseIR::lower`
//! and `write::code_str`.
//!
//! These need no device, so unlike the kernel execution tests they run in CI. Each
//! test renders a kernel for every dialect and compares it against a file under
//! `tests`; run with `BLESS=1` to rewrite those files after an intended
//! change, and read the resulting diff.
//!
//! The rendering covers the launch metadata as well as the source, since a wrong
//! argument order or grid size corrupts a kernel just as effectively as wrong code.

use std::{fmt::Write, fs, path::PathBuf};

use bullet_compiler::tensor::{
    DType, DValue, IRBuilder, Size, TValue, TensorIR,
    operation::{CABinary, Select, SelectPad, SparseMatmul, SparseMatmulBwd, SparseMatmulBwdMulti, SubGraph, Unary},
};

use crate::{
    pointwise::generate::generate,
    runtime::{DeviceProps, Dialect, Dim3},
};

/// A backend to generate for, and the props it reports. Two backends share the
/// `CudaHip` dialect but still differ: only ROCm gets the AMD wave intrinsics, and
/// the wave size decides which reductions can be fused.
#[derive(Clone, Copy)]
struct Backend {
    label: &'static str,
    dialect: Dialect,
    warp_size: Option<u8>,
    is_rocm: bool,
}

const CUDA: Backend = Backend { label: "CUDA", dialect: Dialect::CudaHip, warp_size: Some(32), is_rocm: false };
const ROCM: Backend = Backend { label: "ROCm", dialect: Dialect::CudaHip, warp_size: Some(64), is_rocm: true };
const METAL: Backend = Backend { label: "Metal", dialect: Dialect::Msl, warp_size: Some(32), is_rocm: false };

const BACKENDS: [Backend; 3] = [CUDA, ROCM, METAL];

/// Render a subgraph's kernel, or say why no kernel was produced.
fn render(name: &str, sub: &SubGraph, backend: Backend) -> String {
    let props = DeviceProps::testing(backend.dialect, backend.warp_size, backend.is_rocm);

    let Some((ir, vectorised)) = generate(sub, &props).expect("generate failed") else {
        return "not fusable\n".to_string();
    };

    let src = unsafe { ir.lower(name.to_string(), &props) }.expect("lower failed");

    let mut out = String::new();
    writeln!(&mut out, "vectorised: {vectorised}").unwrap();
    writeln!(&mut out, "inputs:     {:?}", src.inputs).unwrap();
    writeln!(&mut out, "outputs:    {:?}", src.outputs).unwrap();
    writeln!(&mut out, "args:       {:?}", src.arg_order).unwrap();
    writeln!(&mut out, "zeroed:     {:?}", src.requires_zero).unwrap();
    let Dim3 { x, y, z } = src.gdim;
    writeln!(&mut out, "launch:     grid {x}x{y}x{z}, block {}, smem {}", src.bdim, src.smem).unwrap();
    writeln!(&mut out).unwrap();
    out.push_str(&src.source);
    out
}

fn golden(name: &str, sub: &SubGraph) {
    golden_for(name, sub, &BACKENDS);
}

fn golden_for(name: &str, sub: &SubGraph, backends: &[Backend]) {
    let mut actual = String::new();
    for &backend in backends {
        writeln!(&mut actual, "===== {} =====", backend.label).unwrap();
        actual.push_str(&render(name, sub, backend));
        actual.push('\n');
    }

    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests", &format!("{name}.golden")].iter().collect();

    if std::env::var_os("BLESS").is_some() {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, &actual).unwrap();
        return;
    }

    let expected = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("could not read {}: {e}\nrun with BLESS=1 to create it", path.display()));

    assert!(
        expected == actual,
        "generated kernel for `{name}` changed\n\
         --- expected ---\n{expected}\n--- actual ---\n{actual}\n\
         if this is intended, rerun with BLESS=1 and check the diff",
    );
}

/// `SubGraph::new`, with every listed node registered as a graph output.
fn subgraph(ir: TensorIR, inputs: &[bullet_compiler::ir::NodeId], outputs: &[bullet_compiler::ir::NodeId]) -> SubGraph {
    SubGraph::new(ir, inputs.to_vec(), outputs.to_vec()).unwrap()
}

#[test]
fn broadcast_binary() {
    // `a` is broadcast over the inner dimension and `b` over the outer one, so this
    // covers both arms of the broadcast lowering, and passes `x` straight through
    let b = IRBuilder::default();
    let a = b.add_input(8, DType::F32);
    let bb = b.add_input(8, DType::F32);
    let x = b.add_input(4 * 8, DType::F32);
    let y = ((a.broadcast([8], 0, 4).unwrap() * x).unwrap() + bb.broadcast([8, 1], 1, 4).unwrap()).unwrap();

    golden("broadcast_binary", &subgraph(b.build([x, y]), &[a.node(), bb.node(), x.node()], &[x.node(), y.node()]));
}

#[test]
fn power() {
    let b = IRBuilder::default();
    let x = b.add_input(64, DType::F32);
    let y = x.exp().unwrap().pow(x).unwrap();

    golden("power", &subgraph(b.build([y]), &[x.node()], &[y.node()]));
}

/// Every unary op, so that no variant's emitted expression can change unnoticed -
/// several share a code path but differ in the function they call.
#[test]
fn every_unary() {
    let ops = [
        Unary::Sin,
        Unary::Cos,
        Unary::Tan,
        Unary::Sinh,
        Unary::Cosh,
        Unary::Tanh,
        Unary::Exp,
        Unary::Log,
        Unary::Sgn,
        Unary::Abs,
        Unary::Sqrt,
        Unary::Reciprocal,
        Unary::Cast(DType::I32),
        Unary::Cast(DType::F32),
        Unary::IsPositive,
        Unary::IsZero,
        Unary::IsNonNegative,
        Unary::Round,
        Unary::Truncate,
    ];

    let b = IRBuilder::default();
    let x = b.add_input(64, DType::F32);
    let mut outs = vec![x];

    for op in ops {
        let y = x.unary(op).unwrap();
        // casts to int would poison the following float ops, so keep each independent
        outs.push(if y.ty().dtype() == DType::F32 { y } else { y.unary(Unary::Cast(DType::F32)).unwrap() });
    }

    let nodes: Vec<_> = outs.iter().map(|n| n.node()).collect();
    golden("every_unary", &subgraph(b.build(&outs[..]), &[x.node()], &nodes));
}

/// Every commutative-associative binary op, for the same reason.
#[test]
fn every_binary() {
    let b = IRBuilder::default();
    let x = b.add_input(64, DType::F32);
    let y = b.add_input(64, DType::F32);

    let outs: Vec<_> = [CABinary::Add, CABinary::Mul, CABinary::Min, CABinary::Max]
        .into_iter()
        .map(|op| x.binary(y, op).unwrap())
        .collect();

    let nodes: Vec<_> = outs.iter().map(|n| n.node()).collect();
    golden("every_binary", &subgraph(b.build(&outs[..]), &[x.node(), y.node()], &nodes));
}

#[test]
fn scalar_constant() {
    let b = IRBuilder::default();
    let x = b.add_input(64, DType::F32);
    let y = (x + b.scalar(DValue::F32(2.5), 64)).unwrap();

    golden("scalar_constant", &subgraph(b.build([y]), &[x.node()], &[y.node()]));
}

#[test]
fn constant_tensor() {
    let b = IRBuilder::default();
    let x = b.add_input(4, DType::F32);
    let y = (x * b.constant(TValue::F32(vec![1.0, 2.0, 3.0, 4.0]))).unwrap();

    golden("constant_tensor", &subgraph(b.build([y]), &[x.node()], &[y.node()]));
}

#[test]
fn pad() {
    for dim in 0..3 {
        let b = IRBuilder::default();
        let x = b.add_input(8, DType::F32);
        let y = b.add_input(8, DType::F32);
        let cat = (x.pad([2, 2, 2], dim, 0, 2, DValue::F32(0.0)).unwrap()
            + y.pad([2, 2, 2], dim, 2, 0, DValue::F32(0.0)).unwrap())
        .unwrap();

        golden(&format!("pad_dim{dim}"), &subgraph(b.build([cat]), &[x.node(), y.node()], &[cat.node()]));
    }
}

#[test]
fn slice() {
    for dim in 0..3 {
        let b = IRBuilder::default();
        let x = b.add_input(8, DType::F32);
        let y = x.slice([2, 2, 2], dim, 0, 1).unwrap();

        golden(&format!("slice_dim{dim}"), &subgraph(b.build([y]), &[x.node()], &[y.node()]));
    }
}

#[test]
fn select() {
    let b = IRBuilder::default();
    let vals = b.add_input(4 * 12, DType::F32);
    let idx = b.add_input(4, DType::I32);
    let op = Select { dtype: DType::F32, batch: 4.into(), inner: 12.into(), divisor: 3.into() };
    let out = b.add_op([vals, idx], op).unwrap()[0];

    golden("select", &subgraph(b.build([out]), &[vals.node(), idx.node()], &[out.node()]));
}

#[test]
fn select_pad() {
    let b = IRBuilder::default();
    let vals = b.add_input(4 * 4, DType::F32);
    let idx = b.add_input(4, DType::I32);
    let op = SelectPad { dtype: DType::F32, batch: 4.into(), inner: 12.into(), divisor: 3.into() };
    let out = b.add_op([vals, idx], op).unwrap()[0];

    golden("select_pad", &subgraph(b.build([out]), &[vals.node(), idx.node()], &[out.node()]));
}

fn sparse_matmul() -> SparseMatmul {
    SparseMatmul::new(DType::F32, 4usize, 32usize, 8usize, 32usize, 0, 2usize).unwrap()
}

#[test]
fn sparse_matmul_fwd() {
    let b = IRBuilder::default();
    let w = b.add_input(32 * 8, DType::F32);
    let i = b.add_input(4 * 2, DType::I32);
    let out = b.add_op([w, i], sparse_matmul()).unwrap()[0];

    golden("sparse_matmul_fwd", &subgraph(b.build([out]), &[w.node(), i.node()], &[out.node()]));
}

#[test]
fn sparse_matmul_bwd() {
    let b = IRBuilder::default();
    let g = b.add_input(4 * 32, DType::F32);
    let i = b.add_input(4 * 2, DType::I32);
    let op = SparseMatmulBwdMulti::new(SparseMatmulBwd(sparse_matmul()));
    let out = b.add_op([g, i], op).unwrap()[0];

    golden("sparse_matmul_bwd", &subgraph(b.build([out]), &[g.node(), i.node()], &[out.node()]));
}

/// When the row count is a multiple of the wave size, the sparse matmul emits the
/// AMD scalar-load hint - on ROCm only, since the builtin does not exist elsewhere.
/// Both wave sizes reported by real devices must be handled, as must a device that
/// reports none.
#[test]
fn sparse_matmul_wave_aligned() {
    let mm = SparseMatmul::new(DType::F32, 4usize, 256usize, 8usize, 256usize, 0, 2usize).unwrap();

    let b = IRBuilder::default();
    let w = b.add_input(256 * 8, DType::F32);
    let i = b.add_input(4 * 2, DType::I32);
    let out = b.add_op([w, i], mm).unwrap()[0];
    let sub = subgraph(b.build([out]), &[w.node(), i.node()], &[out.node()]);

    let backends = [
        CUDA,
        Backend { label: "ROCm wave32", warp_size: Some(32), ..ROCM },
        Backend { label: "ROCm wave64", ..ROCM },
        Backend { label: "ROCm no wave size", warp_size: None, ..ROCM },
    ];

    golden_for("sparse_matmul_wave_aligned", &sub, &backends);
}

#[test]
fn reduce_sum() {
    // reduces to an atomic add, so the destination needs zeroing before launch
    let b = IRBuilder::default();
    let x = b.add_input(64 * 4, DType::F32);
    let r = x.reduce_sum([4, 64], 0).unwrap();

    golden("reduce_sum", &subgraph(b.build([r]), &[x.node()], &[r.node()]));
}

/// A reduction over an inner dimension that is not warp aligned cannot be fused,
/// and must be reported as such rather than miscompiled.
#[test]
fn reduce_not_warp_aligned() {
    let b = IRBuilder::default();
    let x = b.add_input(12 * 5, DType::F32);
    let r = x.reduce_sum([5, 12], 0).unwrap();

    golden("reduce_not_warp_aligned", &subgraph(b.build([r]), &[x.node()], &[r.node()]));
}

/// A reduction fuses only when its inner dimension is wave aligned, so the same
/// graph is fusable on a wave32 device and not on a wave64 one.
#[test]
fn reduce_warp32_only() {
    let b = IRBuilder::default();
    let x = b.add_input(32 * 4, DType::F32);
    let r = x.reduce_sum([4, 32], 0).unwrap();

    golden("reduce_warp32_only", &subgraph(b.build([r]), &[x.node()], &[r.node()]));
}

/// Generated source must not depend on how many other graphs have been built, or
/// golden tests - and diffing a kernel against a previous run - become useless.
#[test]
fn generation_is_deterministic() {
    let build = || {
        let b = IRBuilder::default();
        let w = b.add_input(32 * 8, DType::F32);
        let i = b.add_input(4 * 2, DType::I32);
        let out = b.add_op([w, i], sparse_matmul()).unwrap()[0];
        let sub = subgraph(b.build([out]), &[w.node(), i.node()], &[out.node()]);
        render("spmm", &sub, CUDA)
    };

    let first = build();
    // allocate unrelated nodes and ops in between, bumping any global counters
    for _ in 0..7 {
        let b = IRBuilder::default();
        let x = b.add_input(8, DType::F32);
        let _ = (x * x).unwrap();
    }

    assert_eq!(first, build(), "kernel source depends on unrelated graph construction");
}

#[test]
fn vector_widths() {
    // `Size` is what drives the vectorisation analysis, so check the widths land
    // where expected rather than silently falling back to scalar
    for (size, vectorised) in [(4usize, true), (8, true), (6, true), (5, false)] {
        let b = IRBuilder::default();
        let x = b.add_input(size, DType::F32);
        let y = (x * x).unwrap();
        let sub = subgraph(b.build([y]), &[x.node()], &[y.node()]);

        let props = DeviceProps::testing(CUDA.dialect, CUDA.warp_size, CUDA.is_rocm);
        let (_, actual) = generate(&sub, &props).unwrap().unwrap();
        assert_eq!(actual, vectorised, "size {size}");
        assert_eq!(Size::from(size).get(), size);
    }
}
