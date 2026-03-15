#!/usr/bin/env python3
"""
Train and export a quaternion-state wrench-input Koopman EDMDc model to .npz.

Exports:
    A, B, centers, gamma, state_dim, input_dim, dt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from Koopman.koopmanEDMDc import KoopmanEDMDc


DATASET_NAME = "koopman_dataset_50Hz_with_wrench.csv"

STATE_COLS_QUAT = ["x", "y", "z", "qw", "qx", "qy", "qz", "u", "v", "w", "p", "q", "r"]
STATE_COLS_EULER = ["x", "y", "z", "phi", "theta", "psi", "u", "v", "w", "p", "q", "r"]
INPUT_COLS = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]


def find_project_root(start: Path) -> Path:
    p = start.resolve()
    for q in [p, *p.parents]:
        if (q / "rosbags").exists():
            return q
    return p


def find_latest_csv(root: Path, name: str) -> Path:
    cands = list(root.rglob(name))
    if not cands:
        raise FileNotFoundError(f"No '{name}' under: {root}")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def euler_to_quat_batch(phi: np.ndarray, theta: np.ndarray, psi: np.ndarray) -> np.ndarray:
    c1 = np.cos(phi * 0.5)
    s1 = np.sin(phi * 0.5)
    c2 = np.cos(theta * 0.5)
    s2 = np.sin(theta * 0.5)
    c3 = np.cos(psi * 0.5)
    s3 = np.sin(psi * 0.5)

    qw = c3 * c2 * c1 + s3 * s2 * s1
    qx = c3 * c2 * s1 - s3 * s2 * c1
    qy = c3 * s2 * c1 + s3 * c2 * s1
    qz = s3 * c2 * c1 - c3 * s2 * s1

    q = np.vstack([qw, qx, qy, qz]).T
    qn = np.linalg.norm(q, axis=1, keepdims=True)
    qn = np.maximum(qn, 1e-12)
    return q / qn


def load_dataset(csv_path: Path):
    df = pd.read_csv(csv_path)

    if "t" not in df.columns:
        raise ValueError("CSV must contain a 't' column.")

    has_quat = all(c in df.columns for c in STATE_COLS_QUAT)
    has_euler = all(c in df.columns for c in STATE_COLS_EULER)

    if not has_quat and not has_euler:
        raise ValueError(
            "Dataset must contain either quaternion-state columns "
            f"{STATE_COLS_QUAT} or Euler-state columns {STATE_COLS_EULER}"
        )

    for c in INPUT_COLS:
        if c not in df.columns:
            df[c] = 0.0

    df = (
        df.sort_values("t")
        .drop_duplicates(subset="t")
        .replace([np.inf, -np.inf], np.nan)
        .copy()
    )

    if has_quat:
        df = df.dropna(subset=STATE_COLS_QUAT)
        q = df[["qw", "qx", "qy", "qz"]].to_numpy(dtype=float)
        qn = np.linalg.norm(q, axis=1, keepdims=True)
        qn = np.maximum(qn, 1e-12)
        q = q / qn
        df.loc[:, ["qw", "qx", "qy", "qz"]] = q
    else:
        df = df.dropna(subset=STATE_COLS_EULER)
        q = euler_to_quat_batch(
            df["phi"].to_numpy(dtype=float),
            df["theta"].to_numpy(dtype=float),
            df["psi"].to_numpy(dtype=float),
        )
        df.loc[:, ["qw", "qx", "qy", "qz"]] = q

    X = df[STATE_COLS_QUAT].to_numpy(dtype=float)
    U = df[INPUT_COLS].to_numpy(dtype=float)
    t = df["t"].to_numpy(dtype=float)
    dt = np.median(np.diff(t)) if len(t) > 1 else 0.05

    print(f"[i] Loaded {len(df)} samples from '{csv_path}' (dt ≈ {dt:.6f} s)")
    return X, U, float(dt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="", help="Path to CSV dataset. If empty, find latest automatically.")
    parser.add_argument("--out", type=str, default="models/koopman_edmdc_wrench_quat_weights.npz", help="Output .npz path")
    parser.add_argument("--n-rbfs", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=3.0)
    parser.add_argument("--ridge", type=float, default=1e-1)
    args = parser.parse_args()

    if args.csv:
        csv_path = Path(args.csv).expanduser().resolve()
    else:
        repo_root = find_project_root(Path(__file__).parent)
        rosbags_dir = repo_root / "rosbags"
        csv_path = find_latest_csv(rosbags_dir, DATASET_NAME)

    X, U, dt = load_dataset(csv_path)
    if len(X) < 3:
        raise RuntimeError("Not enough samples to train.")

    model = KoopmanEDMDc(
        state_dim=13,
        input_dim=6,
        n_rbfs=int(args.n_rbfs),
        gamma=float(args.gamma),
        ridge=float(args.ridge),
    )
    model.fit(X, U)

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out,
        A=np.asarray(model.A_, dtype=float),
        B=np.asarray(model.B_, dtype=float),
        centers=np.asarray(model.centers_, dtype=float),
        gamma=np.array([float(model.gamma)], dtype=float),
        state_dim=np.array([13], dtype=int),
        input_dim=np.array([6], dtype=int),
        dt=np.array([float(dt)], dtype=float),
    )
    print(f"[ok] Saved Koopman weights -> {out}")


if __name__ == "__main__":
    main()
