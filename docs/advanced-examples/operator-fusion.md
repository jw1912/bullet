# Operator Fusion

From a user's specified graph, a graph intermediate representation (IR) is generated, and optimisation passes
are run on it, the most major of which is automatic operator fusion.
A large part of bullet's speed is due to fusing many of the most time consuming operations in network training.
In particular, for standard dual perspective NNUE networks:
```rust
let stm = builder.new_sparse_input("stm", Shape::new(num_inputs, 1), max_active);
let nstm = builder.new_sparse_input("nstm", Shape::new(num_inputs, 1), max_active);
let l0 = builder.new_affine("l0", num_inputs, hl);
```
We can fuse the first layer calculation into a single operation
```diff
- let stm_subnet = l0.forward(stm).crelu();
- let ntm_subnet = l0.forward(nstm).crelu();
- let l1 = stm_subnet.concat(ntm_subnet);
+ let l1 = l0.forward_sparse_dual_with_activation(stm, nstm, Activation::CReLU);
```

Check out the [fusion](../../examples/extra/fusion.rs) example, compile and run it for yourself with
```
cargo r -r --example fusion
```

Below is the output at the time of writing (may change as more optimisation passes are added).

https://github.com/user-attachments/assets/83285416-e320-4b24-b8d4-62f43b8ab9cf
