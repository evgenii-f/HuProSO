# PoseD-Flow on CUDA: the time goes into the final QR projection, not the flow

An issue found in the Pareto comparison (`explore/evaluation/pareto_posedflow_vs_rsgm.ipynb`) when the models were timed on an RTX 4090 (`runtime_NVIDIA_GeForce_RTX_4090_cuda.json` in the cache of `hp.evaluation.pareto`, 2026-11-11, torch 2.14.1+cu130, median of 5 runs). Nothing is fixed yet; this note says what is wrong and how to fix it.

## Summary

- **Symptom.** On the 4090 every PoseD-Flow variant costs about 400-490 µs per pose at large n, whatever its number of steps: midpoint 10 (20 network calls) and midpoint 100 (200 calls) differ by less than 20%. On MPS the same midpoint 100 costs 171 µs per pose.
- **Cause.** A fixed cost of about 400 µs per pose, linear in n: the projection onto SO(3) at the end of their sampler, `projx` (`hp/models/posedflow.py`, a port of their `SO3.projx`). It is `torch.linalg.qr` of n × 21 matrices 3 × 3, which is very slow on CUDA for a large batch of tiny matrices.
- **Why it matters.** It is their implementation, ported one to one, but it measures a slow path of `torch.linalg.qr` on CUDA rather than the cost of their method: a 3 × 3 projection done once is cheap by nature. With it, PoseD-Flow is off the fronts on CUDA for a reason that has nothing to do with flow matching, while our RSGM is optimized. A reviewer would ask about it.
- **Fix.** The same projection by Gram-Schmidt on the columns. It is the same function (the Q of a QR with a positive diagonal of R), in a few elementwise operations. Keep their QR as an option, `projection="qr"`.
- **Expected effect** (an estimate, to be measured): PoseD-Flow at the cost of its network calls, about 0.46 µs per call per pose on the 4090. Midpoint 10 would be at ~9 µs per pose, next to RSGM 512x4 with the new samplers (6.1 µs), which changes the fast end of the fronts.

## Evidence

Time per pose (µs) on the RTX 4090, `model.sample(n)`:

| model | network calls | n = 1 | 64 | 256 | 1024 | 4096 | 16384 |
|---|---|---|---|---|---|---|---|
| PoseD-Flow, midpoint 10 | 20 | 7,861 | 529 | 433 | 409 | 408 | 407 |
| PoseD-Flow, midpoint 25 | 50 | 18,406 | 720 | 481 | 426 | 423 | 423 |
| PoseD-Flow, Euler 50 | 50 | 17,609 | 710 | 477 | 427 | 421 | 421 |
| PoseD-Flow, midpoint 50 | 100 | 35,703 | 1,030 | 557 | 451 | 445 | 445 |
| PoseD-Flow, midpoint 100 (paper) | 200 | 70,277 | 1,647 | 715 | 503 | 491 | 494 |
| RSGM 512x4, ODE Euler 50 (for scale) | 50 | 21,882 | 355 | 88 | 22 | 6.1 | 4.8 |

- **At n = 1 the time follows the network calls** (about 0.35 ms per call, as RSGM). The flow itself is not slow.
- **At large n the time per pose is flat in n** (about 407 µs from n = 1024 to 16384 for midpoint 10): a cost proportional to the number of poses, with no batching gain. The network calls add only 0.46 µs per call per pose (midpoint 100 minus midpoint 10: 83 µs for 180 calls).
- **On MPS it does not happen:** midpoint 100 costs 171 µs per pose at n = 4096 in total, and on the Mac `projx` of 4096 × 21 matrices takes 8.6 ms (2 µs per pose).

The rest of the sampler (`base_sample`: `roma.rotvec_to_rotmat` of the noise; the ambient Euler or midpoint steps) is elementwise or a network call, so `projx` is the only operation left that can cost ~1.6 s per 4096 poses. **This is inferred, not measured directly on CUDA**: the first step of the fix is to time `projx` alone on the 4090 (see Verification).

## Cause

`hp/models/posedflow.py`, as their `SO3.projx`:

```python
def projx(x):
    q, r = torch.linalg.qr(x)
    s = torch.diagonal(r, dim1=-2, dim2=-1).clone()
    s[s == 0] = 1
    return q * (s / s.abs()).unsqueeze(-2)
```

`sample_21` calls it once, on `x.view(n, 21, 3, 3)`: 86,016 matrices for n = 4096. A batched QR of tiny matrices on CUDA goes through cuSOLVER/MAGMA kernels made for large matrices, with a cost per matrix far above the arithmetic of a 3 × 3 decomposition.

## Fix

1. **A Gram-Schmidt projection** in `hp/models/posedflow.py`, for the columns a, b, c of x:
   e1 = a / |a|; e2 = normalized b − (e1·b) e1; e3 = normalized c − (e1·c) e1 − (e2·c) e2; the result [e1 e2 e3].
   This is the Q of the QR decomposition with the signs of diag(R) made positive, which is what `projx` returns.
2. **An option `projection: "gram_schmidt" | "qr"`** in `PoseDFlowPretrained` and `PoseDFlow`, default `"gram_schmidt"`; `"qr"` reproduces their code exactly.
3. **In `MODELS`** (`hp/evaluation/pareto.py`), `projection="gram_schmidt"` in the `posedflow=dict(...)` of every PoseD-Flow model. The dict is part of the cache identity, so the quality (harness, paper) of PoseD-Flow is computed again rather than reused (see below why), and the old results stay in the cache under the old settings.
4. **A test** that both projections agree on samples of the model (float64 and float32), and that the result is orthogonal.
5. **Re-time** the PoseD-Flow models on the 4090 and on MPS (`--recompute-runtime`), and DPoser full 50 too: its re-run is still ~14% slower than DPoser c10 50 with the same network.

### Verification so far (on the Mac)

On 86,016 matrices (random rotations plus N(0, 0.3²) noise, as the ambient steps leave them off the manifold):

| check | result |
|---|---|
| max \|QR − Gram-Schmidt\|, float64 | 4·10⁻¹³ |
| max \|QR − Gram-Schmidt\|, float32 | 3.5·10⁻⁴ (on ill-conditioned inputs) |
| time on CPU: QR / Gram-Schmidt | 15.1 / 5.8 ms |
| time on MPS: QR / Gram-Schmidt | 8.6 / 0.8 ms |

The float32 difference is why the quality should be recomputed rather than assumed equal. On real samples, which end near SO(3), it should be far smaller.

Still to do: time `projx` alone on the 4090, e.g.

```bash
PYTHONPATH=src python -c "import torch, time; from hp.models.posedflow import projx; x = torch.randn(4096 * 21, 3, 3, device='cuda'); projx(x); torch.cuda.synchronize(); t = time.perf_counter(); projx(x); torch.cuda.synchronize(); print(time.perf_counter() - t, 's')"
```

If it prints ~1.6 s, the cause is confirmed.

## Side note: their projection is onto O(3), not SO(3)

Both QR and Gram-Schmidt return an orthogonal matrix with the sign of the determinant of the input: an input with det < 0 gives a reflection (det = −1), which happened for some of the noisy test matrices above. Their sampler ends near SO(3), so it should not occur in practice, but neither `projx` nor the fix guarantees a rotation. Counting the poses with det < 0 among the samples of each variant (especially the few-step ones: midpoint 10, Euler 20) would settle it.
