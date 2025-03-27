use bullet_core::backend::cpu::CpuThread;
use bullet_lib::nn::{GraphCompileArgs, InitSettings, NetworkBuilder, Shape};

fn main() {
    let mut builder = NetworkBuilder::default();

    let stm = builder.new_sparse_input("stm", Shape::new(768, 1), 32);
    let ntm = builder.new_sparse_input("ntm", Shape::new(768, 1), 32);
    let weights = builder.new_weights("weights", Shape::new(1, 768), InitSettings::Zeroed);
    let bias = builder.new_weights("bias", Shape::new(1, 1), InitSettings::Zeroed);
    let stm = (weights.matmul(stm) + bias).screlu();
    let ntm = (weights.matmul(ntm) + bias).screlu();
    let out = stm.concat(ntm);
    let outw = builder.new_weights("outw", Shape::new(1, 2), InitSettings::Zeroed);
    let _ = outw.matmul(out);

    let args = GraphCompileArgs::default().emit_ir();

    builder.set_compile_args(args);
    let graph = builder.build(CpuThread);

    println!("{:?}", graph.get_last_device_error());
}
