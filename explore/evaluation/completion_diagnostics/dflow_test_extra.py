"""D-Flow on the 5000 test poses of PoseD-Flow for the rows not in MODELS of scripts/evaluate_completion.py (the
runs of 2026-11-15 on the servers): Heun EDM inside D-Flow at 100 and 200 network calls, and RSGM 1536x6 trained with
the likelihood weighting. The rows are added to MODELS and evaluate_completion.main() runs, so the cache gets the
usual names (run from the root of the repository; --models picks the rows):
    python explore/evaluation/completion_diagnostics/dflow_test_extra.py --models "RSGM 1536x6 lik: their full" \\
        --device cuda:0 --together --batch 200      # 1536x6 at batch 500 does not fit a 24 GB card
"""
import sys

sys.path.insert(0, "scripts")
import evaluate_completion as ec  # noqa: E402

R = "~/data/experiments/rsgm_posedflow_small_s10/run_0001"
ec.MODELS["RSGM 512x4: their c10, Heun EDM 100 NFE"] = dict(run=R, prior=dict(sampler="heun", schedule="edm", ode_steps=50))
ec.MODELS["RSGM 512x4: their c10, Heun EDM 200 NFE"] = dict(run=R, prior=dict(sampler="heun", schedule="edm", ode_steps=100))
ec.MODELS["RSGM 1536x6 lik: their full"] = dict(run="~/data/experiments/full/rsgm_posedflow_search_lik_s10/run_0001")
ec.MODELS["RSGM 1536x6 lik: their c10"] = dict(run="~/data/experiments/rsgm_posedflow_search_lik_s10/run_0001")
sys.exit(ec.main())
