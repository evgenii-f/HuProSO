"""The skeleton of the SMPL body model at rest, from the official .pkl into a small .npz.

    python scripts/preprocess/convert_smpl_skeleton.py \
        --pkl ~/data/body_models/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl \
        --out ~/data/body_models/smpl/smpl_neutral_skeleton.npz

The joint positions of SMPL with betas = 0 do not depend on the pose blend shapes or the skinning: they are the
joints of the template, J_regressor @ v_template, moved rigidly along the kinematic tree (smplx returns the same as
joints[:, :24]). So the forward kinematics needs only those 24 rest joints and the parents.

The .pkl of SMPL v1.1.0 is a Python 2 pickle whose arrays are partly chumpy objects (chumpy does not run on
Python 3). It is read with an unpickler that only builds numpy arrays and dictionaries: chumpy.Ch and
scipy.sparse.csc_matrix become plain records of their state, every other class is refused.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np


class _Record:
    """The state of a chumpy.Ch or a scipy csc_matrix, kept as a dictionary."""

    def __setstate__(self, state):
        self.state = state


def _reconstructor(cls, base, state):  # copy_reg._reconstructor, for the classes allowed below only
    if cls is not _Record or base is not object:
        raise pickle.UnpicklingError(f"refused to build {cls}")
    return _Record()


class _SafeUnpickler(pickle.Unpickler):
    ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): np.dtype,
        ("__builtin__", "object"): object,
        ("__builtin__", "set"): set,
        ("copy_reg", "_reconstructor"): _reconstructor,
        ("chumpy.ch", "Ch"): _Record,
        ("scipy.sparse.csc", "csc_matrix"): _Record,
    }

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"refused global {module}.{name}")
        return self.ALLOWED[(module, name)]


def _array(value):
    if isinstance(value, _Record):
        if "x" in value.state:  # chumpy.Ch: the array is its x
            return np.asarray(value.state["x"])
        s = value.state  # csc_matrix: columns compressed
        dense = np.zeros(s["_shape"], dtype=s["data"].dtype)
        for col in range(s["_shape"][1]):
            rows = s["indices"][s["indptr"][col]:s["indptr"][col + 1]]
            dense[rows, col] = s["data"][s["indptr"][col]:s["indptr"][col + 1]]
        return dense
    return np.asarray(value)


def load_smpl_pkl(path) -> dict:
    with open(Path(path).expanduser(), "rb") as f:
        model = _SafeUnpickler(f, encoding="latin1").load()
    return {k: _array(model[k]) for k in ("v_template", "J_regressor", "kintree_table")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    m = load_smpl_pkl(args.pkl)
    joints = m["J_regressor"] @ m["v_template"]  # [24, 3]
    parents = m["kintree_table"][0].astype(np.int64)
    parents[0] = -1
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, joints=joints, parents=parents, source=str(Path(args.pkl).name))
    print(f"{out}: {joints.shape[0]} joints, parents {parents.tolist()}")


if __name__ == "__main__":
    main()
