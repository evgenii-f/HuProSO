# Working in this repository

Pose priors on SO(3)^J and their evaluation: the HuProSO3 flow, RSGM, PoseRFM, DPoser, PoseD-Flow. The library is
`hp` (`src/hp`), command-line tools are in `scripts/`, notebooks in `explore/`, settings in `config/`, tests in
`tests/`. Data, runs and caches live outside the repository, under `~/data` (paths in `config/paths.yaml` or
`~/.config/huproso/paths.yaml`, see `config/paths.example.yaml`).

## Rules

- **Commits, merges and pushes only after the user's explicit yes** for that action: first say which files and which
  message, then wait. Messages start with `feat:`, `fix:`, `docs:`, `test:`, `refactor:` or `chore:` (no scope) and
  have no `Co-Authored-By` line. Git's format: a subject of at most 50 characters (never over 72), a blank line, a
  body wrapped at 72 columns (check: `git log -1 --format=%B | awk 'length > 72'` prints nothing).
- **New plans are discussed before they are carried out.** Small fixes inside an agreed task need no extra round.
- **The servers are run by the user.** For workstation1 / workstation2, give copy-paste commands (one per block; say
  whether a copy overwrites), and ask for a few lines of output (`tail -n 5 <log>`), not whole screens.
- **Validation chooses, the test reports.** Settings, checkpoints and models are chosen on validation poses, with the
  rule written down before the test; a test is run once per row. Say which numbers are validation and which test.
- Checkpoints are loaded with `torch.load(..., weights_only=True)`. Downloaded third-party code is read, not run.
- **Worktrees: only when needed, removed when done.** Work in an existing checkout when no other session uses it; make
  a worktree (`git worktree add ../HuProSO-<topic> <branch>`) only when another session works there, and never switch
  the branch of a worktree another session uses. When the work in a worktree is finished (committed, pushed or
  merged), propose removing it (`git worktree remove`); do not leave uncommitted changes, stashes or half-done
  branches behind in it. While work continues in a worktree or branch, say nothing about it; only when a session ends
  or hands over with worktrees, branches or stashes of its own still open, list them once with the reason.
- Tests: `python -m pytest tests` from the root (a checkout that is not installed: `PYTHONPATH=src`). On the Mac the
  environment is the conda env `amass`; MPS has little memory for D-Flow, run one such job at a time.

## Experiment reports

Results that are not written down are lost between sessions. Reports live in `docs/experiments/` (index and format in
its `README.md`) and are written with the skill `experiment-report` (`.claude/skills/experiment-report/SKILL.md`).

Write or extend a report when a phase of work ends, that is when one of these happens:

- a series of runs that answers a question is complete (an Optuna study, a set of trainings, a test of several rows);
- a result changes a decision: which method, model or setting we use, or what we do next (negative results count);
- before an `experimental/*` branch is merged into `dev` (its report must exist and be current);
- before a session hands over or ends with results that are written nowhere else.

Not for single runs, debugging or numbers of a question that is still open: those go into the report when the phase
ends. A line of work has one report; a later phase adds a dated section to it and updates its summary. Propose the
report (and its commit) to the user at the end of the phase; do not wait to be asked.
