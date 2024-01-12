use bullet_core::Feat;

use crate::{DeviceHandles, util};

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
    max_active_inputs: usize,
    input_size: usize,
    output_size: usize,
    weights_grad: *mut f32,
    biases_grad: *mut f32,
    inputs: *const Feat,
    errors: *const f32,
) {
    let inputs = inputs as usize;
    let errors = errors as usize;

    let weights_size = input_size * output_size;

    let mut weights_grads = vec![0; handle.threads];
    let mut biases_grads = vec![0; handle.threads];

    for (w, b) in weights_grads.iter_mut().zip(biases_grads.iter_mut()) {
        *w = util::calloc::<f32>(weights_size) as usize;
        *b = util::calloc::<f32>(output_size) as usize;
    }

    handle.split_workload(batch_size, |thread, idx| {
        let inputs = inputs as *const Feat;
        let errors = errors as *const f32;

        let weights = weights_grads[thread] as *mut f32;
        let biases = biases_grads[thread] as *mut f32;

        let this_inp = inputs.add(max_active_inputs * idx);
        let this_err = errors.add(2 * output_size * idx);
        let our_err = this_err;
        let opp_err = this_err.add(output_size);

        for i in 0..output_size {
            *biases.add(i) += *our_err.add(i);
        }

        for i in 0..output_size {
            *biases.add(i) += *opp_err.add(i);
        }

        for i in 0..max_active_inputs {
            let feat = *this_inp.add(i);

            if feat.our() == 65_535 {
                break;
            }

            let our_weights = weights.add(output_size * feat.our());
            for j in 0..output_size {
                *our_weights.add(j) += *our_err.add(j);
            }

            let opp_weights = weights.add(output_size * feat.opp());
            for j in 0..output_size {
                *opp_weights.add(j) += *opp_err.add(j);
            }
        }
    });

    for &w in weights_grads.iter() {
        for i in 0..weights_size {
            *weights_grad.add(i) += *(w as *const f32).add(i);
        }
    }

    for &b in biases_grads.iter() {
        for i in 0..output_size {
            *biases_grad.add(i) += *(b as *const f32).add(i);
        }
    }

    for (&w, &b) in weights_grads.iter().zip(biases_grads.iter()) {
        unsafe {
            util::free(w as *mut f32, weights_size);
            util::free(b as *mut f32, output_size);
        }
    }
}
