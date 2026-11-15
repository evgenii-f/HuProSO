#!/bin/zsh
# One line per RSGM run for the architecture ablation (docs/experiments/2026-11-14_score_network_architectures.md):
# the best and last validation loss (history.json), the hybrid of hybrid.py on 500 validation poses (MPVPE mean / APD /
# best of 10 per occlusion) and the hidden-joint error of trajectory.py at steps 100 and 199 (left_leg, arms; deg).
# Run from the root of the repository on the Mac (env amass); run names are relative to ~/data/experiments:
#     zsh explore/evaluation/completion_diagnostics/eval_runs.sh rsgm_posedflow_abl_base_s10 full/rsgm_posedflow_dposer_like_s10
D=explore/evaluation/completion_diagnostics
export PYTHONPATH=$PWD/src HUPROSO_BODY_MODELS=${HUPROSO_BODY_MODELS:-~/data/body_models}
P=${PYTHON:-python}
for name in "$@"; do
  R=~/data/experiments/$name/run_0001
  [ -f $R/status.json ] || { echo "$name: missing"; continue; }
  loss=$($P -c "
import json; v = json.load(open('$R/history.json'))['valid_loss']
print(f'valid best {min(v):.3f} last {v[-1]:.3f}')")
  hyb=$(RS=$R $P $D/hybrid.py 500 rsgm 2>&1 | grep '^rsgm' | sed 's/^rsgm *([0-9]*s): //')
  # both denoisers: column 8 is RSGM's error
  traj=$($P $D/trajectory.py $R 2>&1 | grep -E '^ +(100|199) ' | awk '{print $8}' | paste -sd' ' -)
  echo "$name | $loss | hybrid: $hyb | RSGM err at steps 100,199 (left_leg, arms): $traj"
done
