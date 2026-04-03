from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


Q = 20
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def softplus_inverse(y: float) -> float:
    return math.log(math.expm1(y))


def bin_summary(q: int) -> dict:
    if q == 0:
        sigmoid_lo, sigmoid_hi = -math.inf, logit(0.5 / Q)
        softplus_lo, softplus_hi = -math.inf, softplus_inverse(0.5)
    elif q == Q:
        sigmoid_lo, sigmoid_hi = logit((Q - 0.5) / Q), math.inf
        softplus_lo, softplus_hi = softplus_inverse(Q - 0.5), math.inf
    else:
        sigmoid_lo = logit((q - 0.5) / Q)
        sigmoid_hi = logit((q + 0.5) / Q)
        softplus_lo = softplus_inverse(q - 0.5)
        softplus_hi = softplus_inverse(q + 0.5)
    return {
        "action": q,
        "sigmoid_bin": [sigmoid_lo, sigmoid_hi],
        "softplus_bin": [softplus_lo, softplus_hi],
        "sigmoid_width": None if math.isinf(sigmoid_lo) or math.isinf(sigmoid_hi) else sigmoid_hi - sigmoid_lo,
        "softplus_width": None if math.isinf(softplus_lo) or math.isinf(softplus_hi) else softplus_hi - softplus_lo,
    }


def main():
    z = np.linspace(-8.0, 8.0, 2001)
    sigmoid_quantity = Q * sigmoid(z)
    softplus_quantity = softplus(z)
    sigmoid_action = np.clip(np.round(sigmoid_quantity), 0, Q)
    softplus_action = np.clip(np.round(softplus_quantity), 0, Q)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(z, sigmoid_quantity, label=r"$Q \cdot \sigma(z)$", linewidth=2.0)
    axes[0].plot(z, softplus_quantity, label=r"$\mathrm{softplus}(z)$", linewidth=2.0)
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    axes[0].axhline(5.0, color="black", linewidth=0.8, alpha=0.15, linestyle="--")
    axes[0].axhline(10.0, color="black", linewidth=0.8, alpha=0.15, linestyle="--")
    axes[0].axhline(15.0, color="black", linewidth=0.8, alpha=0.15, linestyle="--")
    axes[0].set_title("Continuous Head Maps")
    axes[0].set_xlabel(r"latent score $z = w^\top x + b$")
    axes[0].set_ylabel("pre-rounded quantity")
    axes[0].legend(frameon=False)

    axes[1].plot(z, sigmoid_action, label="sigmoid direct", linewidth=2.0, drawstyle="steps-mid")
    axes[1].plot(z, softplus_action, label="softplus direct", linewidth=2.0, drawstyle="steps-mid")
    axes[1].axvline(logit(0.5 / Q), color="#1f77b4", linewidth=1.0, alpha=0.35, linestyle="--")
    axes[1].axvline(softplus_inverse(0.5), color="#ff7f0e", linewidth=1.0, alpha=0.35, linestyle="--")
    axes[1].set_title("Rounded Actions")
    axes[1].set_xlabel(r"latent score $z = w^\top x + b$")
    axes[1].set_ylabel("action after rounding / clipping")
    axes[1].legend(frameon=False)

    fig.suptitle("Scalar Direct-Head Geometry for Q = 20")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "scalar_head_shapes_q20.svg", format="svg")
    plt.close(fig)

    payload = {
        "Q": Q,
        "zero_thresholds": {
            "sigmoid_direct_quantity": logit(0.5 / Q),
            "direct_quantity": softplus_inverse(0.5),
        },
        "selected_bin_summaries": [bin_summary(q) for q in (0, 1, 3, 5, 10, 15, 19, 20)],
    }
    (ARTIFACTS / "scalar_head_action_bins_q20.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
