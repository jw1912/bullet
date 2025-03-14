use bullet_core::backend::cpu::CpuThread;
use bullet_lib::{
    nn::{GraphCompileArgs, InitSettings, NetworkBuilder, Shape},
    Activation,
};

fn main() {
    let mut builder = NetworkBuilder::default();

    let input = builder.new_sparse_input("input", Shape::new(768, 1), 32);
    let weights = builder.new_weights("weights", Shape::new(1, 768), InitSettings::Zeroed);
    let bias = builder.new_weights("bias", Shape::new(1, 1), InitSettings::Zeroed);
    let almost = weights.matmul(input) + bias;
    let _ = almost.activate(Activation::SCReLU);

    let args = GraphCompileArgs::default().emit_ir().allow_fusion();

    builder.set_compile_args(args);
    let graph = builder.build(CpuThread);

    println!("{:?}", graph.get_last_device_error());
}
