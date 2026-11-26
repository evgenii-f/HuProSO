"""The harness of generate.py with a mixed score below S_MAX: (1 - LAM) network + LAM partner, the partner an energy
(`energy`, the fit given to generate.py) or another RSGM run (docs/experiments/ace/2026-11-26_network_ideas.md, section
2026-11-26: ideas from ACE for the network). The first three arguments are LAM, S_MAX and the partner; the rest goes to
generate.py. From the root:

    python explore/ace_energy/mix_harness.py 0.35 0.04 energy ~/data/experiments/ace/energy_b_rich_products_jd_r0.001.pt \
        --no-network --samples 1000 --batch 500
    python explore/ace_energy/mix_harness.py 0.585 0.04 ~/data/experiments/factorial/rsgm_posedflow_fact_RD_s10/run_0001 \
        ~/data/experiments/ace/energy_b_rich_products_jd_r0.001.pt --samples 1000 --batch 500
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import complete  # noqa: E402
import generate  # noqa: E402

LAM, S_MAX, PARTNER = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
del sys.argv[1:4]


def mixed_model(fit, reference, device, band):
    """The reference RSGM with the mixed score below S_MAX and the network's own above (band is not used)."""
    if PARTNER == "energy":
        model, _, info = complete.energy_model(fit, reference, device, (0.0, float("inf")))
        partner = model.score
    else:
        model, info = complete.build(dict(run=reference), device), f"mix with {Path(PARTNER).parent.name}"
        partner = complete.build(dict(run=PARTNER), device).score
    network = complete.build(dict(run=reference), device).score

    def score(x, t, f=None):
        if float(model.s(t[0])) >= S_MAX:
            return network(x, t, f)
        return (1 - LAM) * network(x, t, f) + LAM * partner(x, t, f)

    model.score = score
    return model, None, info


generate.energy_model = mixed_model
generate.main()
