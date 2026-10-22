# Conditional RSGM: what was done, how, and why it works

RSGM (`src/hp/models/rsgm.py`) can now learn a conditional density p(pose | c), where c are **keypoints**: the positions of the 22 body joints of SMPL (3D), or their x and y (2D). This is the conditioning of the HuProSO3 paper for inverse kinematics (§4.2), occlusions (§4.3) and 2D → 3D uplifting (§4.4). Branch: `experimental/conditions`.

## Summary

- **Keypoints** (`src/hp/data/conditions.py`): `keypoints(poses, modality)` is the forward kinematics of the paper's skeleton. It gives the same positions as the paper's code, but works on rotation matrices directly, in the dtype and on the device of the poses, and is differentiable.
- **Only the network changes.** The noising process does not touch c, so the target of the loss, the sampler and the likelihood ODE are the same as before. The score network gets c as an extra input.
- **The condition MLP is the paper's:** one hidden layer of 64, output 64, concatenated to the input of the score network.
- **Occlusions are built in from the start:** a hidden joint is a NaN keypoint, and the condition MLP sees a 22-bit mask. With `cond_mask="random"` the loss hides joints at random, as the paper's training for occlusions.
- **The keypoints are computed on the fly.** If the loss gets no condition, it computes the keypoints of the poses (+2.6 ms per step of 512 poses on mps). No change to the trainer or the dataset was needed.
- **Unconditional models are untouched:** the same weights, the same state dict, old runs load.
- **One important caveat (below):** with all keypoints visible, the true p(pose | c) has no density on SO(3)^19. Conditional log-likelihoods are therefore set by smoothing, not intrinsic numbers.

## 1. Keypoints

`keypoints(poses)`: poses [N, 19, 3, 3] (the joints of `AmassJoints4Learning`) → positions [N, 22, 3] in meters.

Forward kinematics along the SMPL tree, with the root at the origin and the root and toe rotations set to the identity (joints 0, 10, 11, not in the 19):

```
G_0 = I,  p_0 = 0
G_i = G_parent(i) · R_i                        (R_i = I for joints 0, 10, 11)
p_i = p_parent(i) + G_parent(i) · (b_i)        (b_i: the bone from the parent in the rest pose)
```

- The bone offsets b_i come from `body_models/smpl_skeleton.npz`, the skeleton of the paper's code (`SmplSkeleton`), read once (`lru_cache`).
- 2D is the x and y of the 3D positions, as in the paper's code (`data/amass.py`: `feat_for_conditioning[..., :2]`): an orthographic view along z, no camera.
- **Why it is right:** `test_keypoints_are_the_forward_kinematics_of_the_papers_code` compares it with `perform_forward_kinematics` (the paper's code) on 200 random poses: equal to 1e-5 m. Differences from the paper's code:
  - no round trip matrix → axis-angle → matrix (which loses precision near 180°);
  - no Python loop over the batch, and it stays on the poses' device;
  - autograd works (`test_keypoints_keep_dtype_and_are_differentiable`).
- **What keypoints cannot see** (`test_wrists_and_head_do_not_move_any_keypoint`): the wrists (LeftHand, RightHand) and the head have no child among the 22 joints, so their rotations move no keypoint. See section 6.

## 2. The model

### What RSGM was

- **Noising.** Every joint diffuses by Brownian motion on SO(3): x_t | x_0 ~ IGSO(3)(x_0, s(t)), s(t) = ∫₀ᵗ β.
- **Score network.** score(x, t) = net([vec(x), t]) / std(t), in Lie algebra coordinates in the frame of x.
- **Loss** (denoising score matching, DSM):

  ```
  L = E_{x_0, t, x_t} ‖ std(t) · ( score(x_t, t) − ∇ log p_t(x_t | x_0) ) ‖²
  ```

  ∇ log p_t(x_t | x_0) is the exact gradient of the IGSO(3) heat kernel.
- **Sampling.** The reverse SDE with geodesic Euler-Maruyama steps, from uniform at t = 1 to t = eps.
- **Log-likelihood.** The probability flow ODE v = −β/2 · score from eps to 1, plus the integral of its exact divergence.

### What changed

The score network gets the condition:

```
f = cond_net([c ⊙ m, m])                 # c: keypoints [22·k] (k = 3 or 2), m: mask [22] (1 = observed)
score(x, t, c) = net([vec(x), t, f]) / std(t)
cond_net: Linear(22·k + 22 → 64) → ReLU → Linear(64 → 64)
```

The loss, the sampler and the ODE are the same, with score(x, t, c) in place of score(x, t):

```
L = E_{(x_0, c), t, x_t} ‖ std(t) · ( score(x_t, t, c) − ∇ log p_t(x_t | x_0) ) ‖²
```

### Why the target does not change

- **The forward kernel does not depend on c.** The noising acts on the rotations only, and c stays fixed (c is a function of x_0, and given x_0 the noise does not care about c):

  ```
  p_t(x_t | c) = ∫ p_t(x_t | x_0) p(x_0 | c) dx_0
  ```

- **The minimizer.** The DSM loss for a fixed c and t is minimized by the conditional expectation of the target:

  ```
  score*(x_t, t, c) = E[ ∇ log p_t(x_t | x_0) | x_t, c ] = ∇ log p_t(x_t | c)
  ```

  This is the standard DSM identity (Vincent 2011), applied to the distribution p(x_0 | c) instead of p(x_0). Differentiating the integral above under the integral sign gives it.
- **One network for all c.** A network that takes c as input can minimize the loss for every c at once. So the minimizer of the averaged loss E_c[...] is ∇ log p_t(x_t | c) for every c.
- **Everything downstream follows.** The reverse SDE with this score samples p(x | c), and the probability flow ODE gives log p(x | c). Both are the unconditional derivations with p(· | c) in place of p(·).

### Why concatenation, and why this MLP

- **It is the paper's architecture.** The flow of the paper computes c = g(c_feat) with an MLP of one hidden layer of 64 and output 64 (Appendix B.1), and feeds c to every layer. Concatenating the same features to the input of the score network is the closest equivalent for an MLP score network. It keeps the comparison with the flow about the model, not about the condition encoder.
- **Stronger variants** (FiLM: per-layer scale and shift from f) are a separate experiment.

## 3. Hidden joints (occlusions)

- **Interface.** A joint with a NaN keypoint (any coordinate) is hidden. `cond_features` sets its keypoint to 0 and its mask bit m_j to 0. The mask is an input of `cond_net`, so a hidden joint is distinguishable from a joint observed at the origin.
  - `test_a_hidden_joint_is_its_mask_bit_whatever_its_keypoint`: the value at a hidden joint has no effect.
- **Why NaN.** The model zoo passes one tensor `cond` [B, cond_dim] through `loss`, `sample` and `log_prob`. A NaN in it carries the mask without a second argument, and it cannot be mistaken for a position.
- **The paper's code** zeroes hidden keypoints and (in some configs) shifts them by −1e5, without a mask input. The explicit mask is cleaner and does the same job.
- **Random masks in training** (`cond_mask="random"`), as `mask_data` with `vary_mask_prob` in the paper's code:

  ```
  p ~ U(0, 1) per pose;   joint j hidden if u_j < p,  u_j ~ U(0, 1)
  ```

  On average half of the joints are hidden, and the share varies from pose to pose (`test_random_masks_hide_half_of_the_joints_on_average`). So the model sees everything from fully observed (p ≈ 0) to fully hidden (p ≈ 1).
- **One model for every occlusion.** Trained with random masks, the model handles any pattern of hidden joints at test time, as the paper's occlusion model `id_0384`. With all joints hidden it is an unconditional prior.
- **Determinism.** In eval mode the masks are drawn from the fixed seed of the validation noise, so the validation loss is the same number every time.

## 4. Keypoints on the fly

- **How.** If `loss(pose)` gets no condition and the model has `condition` set, it computes `keypoints(pose)` under `no_grad`, on the model's device. Then the mask, then the usual loss. The trainer passes no condition for a dataset of poses only, so a conditional model trains with the existing trainer and dataset unchanged.
  - `test_without_a_condition_the_loss_takes_the_keypoints_of_the_poses`: the same loss as with the keypoints given.
- **Cost.** On mps, one step of 512 poses (forward, backward, Adam) takes 8.0 ms conditional against 5.4 ms unconditional: the forward kinematics and the mask. A cache of keypoints was discussed and left out: it would save little on mps.
- **`sample` and `log_prob` need the condition explicitly.** There is no pose to compute it from.

## 5. Interface and compatibility

- **New arguments of RSGM:**
  - `condition`: `None` (unconditional), `"3D"` (cond_dim 66) or `"2D"` (cond_dim 44);
  - `cond_hidden`, `cond_features` (64, 64);
  - `cond_mask`: `"none"` or `"random"`.
- **cond_dim** follows from `condition`; a contradicting cond_dim is refused. Without `condition`, cond_dim > 0 gives a model with a **generic** condition vector (no mask, nothing on the fly). It exists so that RSGM meets the contract of the model zoo for conditional models (`test_model_contract.py`).
- **Unconditional models have no `cond_net`.** Their state dict is unchanged and old runs load as before (checked on `rsgm_c30`).
- **Saving and loading.** The condition settings are build arguments, so `save_model` and `load_model` rebuild a conditional model (`test_a_conditional_model_is_saved_and_loaded`).
- **The likelihood ODE** takes the features f into the `vmap`ed Jacobian: the divergence is taken with respect to x only, with f fixed.
  - `test_the_conditional_divergence_is_the_one_of_finite_differences` checks it against finite differences.

## 6. Caveat: with all keypoints visible, p(pose | c) has no density

- **Counting the degrees of freedom.** Keypoints fix the direction of every bone, but not every rotation:
  - a joint with one child among the 22 joints can twist about the bone to that child without moving any keypoint (the child compensates the twist). 15 such joints: hips, knees, ankles, Spine0, Spine1, neck, collars, shoulders, elbows; 1 free degree each;
  - a joint without a child (head, wrists) is free in all 3 degrees;
  - Spine2 (three children) is fully determined.

  So given c, the poses that fit it form a set of dimension about 15 + 3·3 = **24** in the 57 dimensions of SO(3)^19. The other 33 dimensions are fixed exactly.
- **Consequence for the likelihood.** The true p(pose | c) is concentrated on a 24-dimensional set: it has no density with respect to the uniform measure on SO(3)^19 (its log-density would be +∞ on that set).
  - A model gives a finite number because it is smooth: RSGM gives the density at t = eps, the data smoothed by IGSO(3) with s(eps) ≈ 4·10⁻⁷. For a model that resolved the 33 fixed directions exactly, that smoothing alone gives about ½ log(1 / (2π s(eps))) ≈ 6.5 nats per fixed direction, plus log(8π²)/3 ≈ 1.5 for the normalization of the Haar measure: some 250 nats in all. This is the order of the paper's conditional log-likelihood (217.5, Tab. 7).
  - In practice the number is set by how precisely the network resolves the fixed directions, not by the data.
  - **So conditional log-likelihoods of models with different smoothing (eps, loss_t_min, flow vs diffusion) are not comparable as intrinsic numbers.** MPJPE and MGEO of samples are the reliable metrics for IK. With hidden joints the set grows, but the issue stays for the observed part.
- **Consequence for training and sampling.** The score must be very precise at small t in the 33 fixed directions (it grows like 1/std there). `loss_t_min`, `eps` and the number of sampler steps should matter much more than in the unconditional model, and position errors of a few mm need a precise last part of the reverse SDE. These are the first things to check after training.
- **Note: twists are not free in the data.** The free directions are free given c, but the data still has a distribution along them, which the model has to learn as in the unconditional case.

## Tests

- `tests/test_conditions.py`: the keypoints against the paper's FK, the rest pose, 2D, dtype and autograd, the joints keypoints cannot see, refusals.
- `tests/test_rsgm.py`, section "conditioning on keypoints":
  - no condition weights in an unconditional model;
  - cond_dim and refusals;
  - keypoints on the fly;
  - hidden joints;
  - random masks;
  - `sample` and `log_prob` with a condition;
  - the conditional divergence against finite differences;
  - saving and loading.
- The whole suite passes (262 tests).

## Not done yet

- **Training:** a config or notebook for the conditional RSGM (t23 AdamW settings, `condition: 3D`, all frames).
- **Evaluation:** the IK protocol (MPJPE, MGEO with 1 sample and the mean of 10 samples, min-of-10; Tab. 2 of the paper) and the authors' conditional flows `id_0410`, `id_0384`, `id_0510` as references.
- **Conditional PoseRFM.**
- **Merging with `experimental/igso3_gpu`.** The noise on the device changes the same `loss` of RSGM. The merge is mechanical: the conditional loss computes the condition and the mask, and then the noise is drawn as in either branch.
