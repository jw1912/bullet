use bullet_core::Feat;

use crate::DeviceHandles;

pub unsafe fn sparse_affine_forward(
    handle: DeviceHandles,
    batch_size: usize,
    max_input_size: usize,
    output_size: usize,
    weights: *const f32,
    biases: *const f32,
    inputs: *const Feat,
    outputs: *mut f32,
) {
    let weights = weights as usize;
    let biases = biases as usize;
    let inputs = inputs as usize;
    let outputs = outputs as usize;

    handle.split_workload(batch_size, |_, idx| {
        let weights = weights as *const f32;
        let biases = biases as *const f32;
        let this_inp = (inputs as *const Feat).add(max_input_size * idx);
        let our_out = (outputs as *mut f32).add(2 * output_size * idx);
        let opp_out = our_out.add(output_size);

        for i in 0..output_size {
            *our_out.add(i) = *biases.add(i);
        }

        for i in 0..output_size {
            *opp_out.add(i) = *biases.add(i);
        }

        for i in 0..max_input_size {
            let feat = *this_inp.add(i);
            println!("{idx}: {}, {}", feat.our(), feat.opp());

            if feat.our() == 65_535 {
                break;
            }

            let our_weights = weights.add(output_size * feat.our());
            for j in 0..output_size {
                *our_out.add(j) += *our_weights.add(j);
            }

            let opp_weights = weights.add(output_size * feat.opp());
            for j in 0..output_size {
                *opp_out.add(j) += *opp_weights.add(j);
            }
        }
    });
}

pub unsafe fn sparse_affine_backward(
    handle: DeviceHandles,
    batch_size: usize,
    max_input_size: usize,
    output_size: usize,
    weights_grad: *mut f32,
    biases_grad: *mut f32,
    inputs: *const Feat,
    errors: *const f32,
) {
    for idx in 0..batch_size {
        let this_inp = inputs.add(max_input_size * idx);
        let this_err = errors.add(2 * output_size * idx);
        let our_err = this_err;
        let opp_err = this_err.add(output_size);

        for i in 0..output_size {
            *biases_grad.add(i) += *our_err.add(i);
        }

        for i in 0..output_size {
            *biases_grad.add(i) += *opp_err.add(i);
        }

        for i in 0..max_input_size {
            let feat = *this_inp.add(i);

            if feat.our() == 65_535 {
                break;
            }

            let our_weights = weights_grad.add(output_size * feat.our());
            for j in 0..output_size {
                *our_weights.add(j) += *our_err.add(j);
            }

            let opp_weights = weights_grad.add(output_size * feat.opp());
            for j in 0..output_size {
                *opp_weights.add(j) += *opp_err.add(j);
            }
        }
    }
}
