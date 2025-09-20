import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

GRADES = ["A","B","C","D","E"]

def build_transition_matrix(history):
    # history: list of terms; each term is a list of grade labels for students
    n = len(GRADES)
    counts = np.zeros((n,n), dtype=float)
    for t in range(len(history)-1):
        cur, nxt = history[t], history[t+1]
        for g_cur, g_nxt in zip(cur, nxt):
            i = GRADES.index(g_cur); j = GRADES.index(g_nxt)
            counts[i,j] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        P = np.divide(counts, row_sums, where=row_sums>0)
    # handle rows with no transitions (rare in synthetic)
    for i in range(n):
        if not np.isfinite(P[i]).all():
            P[i] = np.full(n, 1.0/n)
    return P

def forecast(dist, P, steps=1):
    v = dist.copy()
    for _ in range(steps):
        v = v @ P
    return v

def simulate_students(n_students=100, n_terms=3, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.dirichlet(np.ones(len(GRADES)))
    history = []
    cur = rng.choice(GRADES, size=n_students, p=base).tolist()
    history.append(cur)
    # Create a mildly "improving" transition bias
    P = np.eye(len(GRADES))*0.5
    for i in range(len(GRADES)-1):
        P[i, max(0, i-1)] += 0.25  # slight improvement
        P[i, i] += 0.25
    P[-1, -1] = 0.75; P[-1, -2] = 0.25
    for _ in range(n_terms-1):
        nxt = []
        for g in cur:
            i = GRADES.index(g)
            nxt.append(np.random.choice(GRADES, p=P[i]))
        history.append(nxt)
        cur = nxt
    return history

def plot_distribution(v, title, fname):
    plt.figure()
    plt.bar(GRADES, v)
    plt.title(title)
    plt.xlabel("Grade")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.savefig(fname, dpi=180)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", type=int, default=120, help="students to simulate")
    ap.add_argument("--terms", type=int, default=3, help="terms in history")
    ap.add_argument("--steps", type=int, default=2, help="forecast steps")
    args = ap.parse_args()

    hist = simulate_students(args.simulate, args.terms, seed=7)
    P = build_transition_matrix(hist)

    # Current distribution = last observed term
    last = hist[-1]
    counts = np.array([last.count(g) for g in GRADES], dtype=float)
    dist = counts / counts.sum()

    v_future = forecast(dist, P, steps=args.steps)

    print("Transition matrix P:\n", np.round(P, 3))
    print("Current dist:", np.round(dist, 3))
    print(f"Forecast (+{args.steps}):", np.round(v_future, 3))

    plot_distribution(dist, "Current Distribution", "dist_current.png")
    plot_distribution(v_future, f"Forecast (+{args.steps})", "dist_forecast.png")
    print("Saved dist_current.png and dist_forecast.png")
