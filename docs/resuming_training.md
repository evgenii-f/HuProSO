# Resuming training

A run of `scripts/train_model.py` can be stopped and continued, e.g. when a cluster job has a time limit.

```bash
python scripts/train_model.py model=so3_flow train.time_limit_minutes=1400   # stops cleanly after 1400 minutes
python scripts/train_model.py resume=~/data/experiments/so3_flow/run_0001    # goes on in the same run folder
```

## What is saved

Every checkpoint (`<run>/checkpoint_<step>/`, every `train.checkpoint_every` steps) holds, besides `model.json` and
the weights (`weights.pt`, `weights_ema.pt`), a `train_state.pt`:

| | |
|---|---|
| step, history of the losses | the run goes on from the step, and `history.json` is complete |
| optimizer (Adam moments), scheduler | the learning rate continues where it was |
| EMA counter | the warm-up of the EMA decay |
| random generators (torch, device, numpy, python) | |

A checkpoint is written into a folder that ends in `.tmp` and renamed when it is complete, so a run killed while it
saves leaves no broken checkpoint. The run folder of a finished run holds the final state itself.

**Which checkpoints are kept.** The newest `train.keep_last` (3) and the best `train.keep_best` (3), the others are
deleted. "Best" is the lowest validation loss at the checkpoint (a step that is no multiple of `valid_interval` is
validated for its checkpoint); without a validation set, the mean training loss of the last `valid_interval` steps. The
two sets may overlap (then fewer are kept), and the newest checkpoint is always kept, to resume from. A checkpoint
written when the run is stopped by the time limit or a signal has no validation loss (there is no time for one): it is
kept as the newest, but never ranks as a best. `keep_last=0, keep_best=0` keeps all checkpoints; `keep_last=0` alone
keeps the best and the newest. Each checkpoint holds its loss in `metric.json`;
`hp.training.best_checkpoint(run_dir)` gives the best of the kept ones.

## Resuming

`resume=` takes the run folder (a run that stopped: its newest checkpoint; a finished run: nothing is trained) or a
checkpoint folder. The run goes on in the **same run folder** (no `run_0002`) with the settings of its `config.yaml`;
the settings of the resumed session are written to `config_resume_<step>.yaml`, and TensorBoard events after the
checkpoint are replaced by the new ones.

Only things that do not change the training may be given: `data.rotations` (the same data elsewhere, e.g. on another
machine: the number of training and validation poses must be the same), `data.load_dtype`, `tensorboard`,
`train.device`, `num_workers`, `progress`, `checkpoint_every`, `keep_last`, `keep_best`, `time_limit_minutes`. The data split, the
model and the optimization (`lr`, `batch_size`, `max_iterations`, scheduler, EMA, ...) are refused: the learning rate
schedule depends on them.

**Resuming from an older checkpoint** (not the newest) goes on from it as from any other: the checkpoints after it and
the final files of the run folder belong to the trajectory that is left, and are moved to
`<run>/abandoned_after_<step>/` (not deleted). Resuming a finished run trains nothing; it only rewrites its final
files and `status.json`, in case the run was killed while writing them.

**Exactness.** If the run stopped at the end of a pass over the data, the resumed run is the same as an uninterrupted
one, bit for bit (a test checks it). If it stopped inside a pass, the resumed run starts a new pass: the order of the
poses differs, nothing else. Both are valid training.

## Stopping on time

`train.time_limit_minutes` and the signals `SIGTERM` and `SIGUSR1` (what a scheduler sends before it kills a job, e.g.
`#SBATCH --signal=USR1@300` in SLURM) make the run stop after the current step with a checkpoint. The time limit counts
from the start of the script (loading the data is part of it) and leaves as much time for the last checkpoint as the
previous one took. `status.json`
says whether the run is finished, and the script exits with status **0** when it is, **75** when it stopped early
and can be resumed:

```bash
python scripts/train_model.py resume=$RUN
if [ $? -eq 75 ]; then sbatch job.sh; fi   # resubmit until it is finished
```

A run killed without a warning (`kill -9`, a crash) goes on from its newest checkpoint: at most `checkpoint_every`
steps are lost.
