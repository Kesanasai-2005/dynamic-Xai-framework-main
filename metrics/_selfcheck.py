"""Minimal asserts for notebook-style C/S/F/A. Run: python -m metrics._selfcheck"""

import numpy as np

from metrics import consistency, feature_agreement, fidelity, stability


def main():
    shap_vec = np.array([0.5, -0.4, 0.1, 0.0])
    lime_same = np.array([0.3, -0.2, 0.05, 0.0])
    lime_flip = np.array([-0.3, 0.2, 0.05, 0.0])

    assert consistency.compute(shap_vec, lime_same, top_k=2) == 1.0
    assert consistency.compute(shap_vec, lime_flip, top_k=2) == 0.0
    assert feature_agreement.compute(shap_vec, lime_same, top_k=2) == 1.0
    assert fidelity.compute(0.85) == 0.85
    assert fidelity.compute(1.5) == 1.0
    assert stability.compute(shap_vec, [shap_vec * 0.99]) > 0.99
    print("ok")


if __name__ == "__main__":
    main()
