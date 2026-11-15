---
name: experiment-report
description: Write or extend an experiment report in docs/experiments of this repository. Use when a phase of experiments ends (a series of runs answered a question, a result changed a decision, a negative result), before an experimental/* branch is merged into dev, before handing over a session with unrecorded results, or when the user asks to document experiments.
---

# Experiment report

A report answers, for a reader a month later who was not in the session: what question, how it was tested, what came
out (numbers), what we conclude, and what it does not show. Every number must be traceable to a file and a script.

## 1. New report or a new section

- Look at `docs/experiments/README.md`. If the question continues a line that has a report, add a section
  `## YYYY-MM-DD: <phase>` to that report and update its Summary; otherwise start
  `docs/experiments/YYYY-MM-DD_<topic>.md` (the day the experiment started, lowercase topic with underscores).
- If another session owns the line (it started the report, or works on its branch), add a section and leave the rest
  to it; do not rewrite its text.

## 2. Collect the numbers from their sources

- Read every number from where it is stored: the completion cache (`~/data/experiments/completion/cache/<row>/`), the
  Optuna databases (`~/data/experiments/completion/optuna/*.db`, `scripts/tune_completion.py --study <name> --summary`),
  `history.json` and `status.json` of a run, logs. Do not copy numbers from the conversation or from memory without
  checking them.
- For every table state: split (validation or test), number of poses, hypotheses K, units, the model (run folder) and
  the settings. Mark the best value of a column if it helps reading.
- Results on the servers that are not on the Mac: give the user the copy command and wait; do not report numbers you
  cannot read.

## 3. Check the evaluation

- Which numbers are validation and which test. Settings chosen on validation are reported with the rule that chose
  them; a test row is run once. If the test was used for any choice, say so plainly.
- Compare against the right baseline (same poses, same K, same budget of network calls where cost matters).
- Name what the result does not show (one seed, few poses, a single setting, a confound).

## 4. Keep what produced the numbers

- Scripts that produced numbers in the report go into the repository (`scripts/` for tools, `explore/<area>/` for
  one-off analyses), runnable from the root, with a docstring that shows the command. No paths into a scratchpad.
- The report says where runs, caches and databases are.

## 5. Write

Structure (sections can be merged when short):

1. `# Title` — the question or the finding, not the method name alone.
2. `## Summary` — 3-7 bullets: the answer with its key numbers, the decision it leads to.
3. `## Setup` — data, split, models (run folders), settings, scripts, budget, machine where it matters.
4. `## Results` — tables with numbers; one sentence under a table saying what it shows.
5. `## Conclusions` — what we now believe and what we do next; negative results stated as such.
6. `## What it does not show` / open questions.
7. `## Where things are` — runs, caches, databases, scripts.

Style: English; plain sentences; units on every number (mm, cm, nats, minutes); no claims beyond the numbers.

## 6. Index, links, commit

- Add or update the row in the table of `docs/experiments/README.md` (date, link, one line with the result).
- Check the links from the root of the repository:

  ```bash
  python - <<'EOF'
  import re
  from pathlib import Path
  for md in Path("docs").rglob("*.md"):
      for link in re.findall(r"\]\(([^)#\s]+)", md.read_text()):
          if not link.startswith("http") and not (md.parent / link).exists():
              print("broken:", md, link)
  EOF
  ```

- Propose the commit to the user (files and message, e.g. `docs: report on <topic>`) and commit only after a yes.
