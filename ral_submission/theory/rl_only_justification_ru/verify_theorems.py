#!/usr/bin/env python3
"""Numerical sanity checks for the theorems in main.tex (RL-only justification).

Every check runs on exactly solvable finite-horizon MDPs with a terminal
success reward, so J, Q, A and the state distributions are computed by dynamic
programming, not estimated. A check reports the worst slack over many random
instances; a theorem's inequality holds when its slack is >= -tol.

These checks catch algebra and constant mistakes in the statements. They do
not certify the neural CDPR/WidowX policy: they exercise the inequalities, not
the assumptions that connect them to the real training run.

    python3 verify_theorems.py            # writes verification_results.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

TOL = 1e-10
RNG = np.random.default_rng(20261104)


# --------------------------------------------------------------------------
# Exact finite-horizon machinery
# --------------------------------------------------------------------------


def random_mdp(S: int, A: int, H: int, goal_frac: float = 0.3):
    P = RNG.dirichlet(np.full(S, 0.3), size=(H, S, A))  # [H,S,A,S']
    goal = np.zeros(S)
    k = max(1, int(round(goal_frac * S)))
    goal[RNG.choice(S, size=k, replace=False)] = 1.0
    d0 = RNG.dirichlet(np.ones(S))
    return P, goal, d0


def random_policy(H: int, S: int, A: int, conc: float = 1.0):
    return RNG.dirichlet(np.full(A, conc), size=(H, S))  # [H,S,A]


def perturb(pi: np.ndarray, scale: float) -> np.ndarray:
    logits = np.log(pi + 1e-300) + scale * RNG.standard_normal(pi.shape)
    logits -= logits.max(axis=-1, keepdims=True)
    out = np.exp(logits)
    return out / out.sum(axis=-1, keepdims=True)


def evaluate(P, goal, pi):
    """Return Q[h,s,a], V[h,s] (V has H+1 rows) for the success objective."""
    H, S, A, _ = P.shape
    V = np.zeros((H + 1, S))
    Q = np.zeros((H, S, A))
    V[H] = goal
    for h in range(H - 1, -1, -1):
        Q[h] = P[h] @ V[h + 1]
        V[h] = (pi[h] * Q[h]).sum(axis=-1)
    return Q, V


def occupancy(P, d0, pi):
    H, S, A, _ = P.shape
    d = np.zeros((H + 1, S))
    d[0] = d0
    for h in range(H):
        d[h + 1] = np.einsum("s,sa,sat->t", d[h], pi[h], P[h])
    return d


def J(P, goal, d0, pi) -> float:
    _, V = evaluate(P, goal, pi)
    return float(d0 @ V[0])


def tv_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(p - q).sum(axis=-1)


def kl_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return (p * (np.log(p + 1e-300) - np.log(q + 1e-300))).sum(axis=-1)


# --------------------------------------------------------------------------
# Lemma 1 (PDL), Lemma 2 (state shift), Theorem 2 (improvement bound),
# Theorem 3 (retention) and its anchor corollary
# --------------------------------------------------------------------------


def check_trust_region(n: int = 4000):
    worst = {
        "pdl_abs_error": 0.0,
        "lemma2_shift_slack": math.inf,
        "thm2_expected_tv_slack": math.inf,
        "thm2_sup_tv_slack": math.inf,
        "thm2_kl_slack": math.inf,
        "thm3_retention_slack": math.inf,
        "anchor_slack": math.inf,
    }
    tightest = {"thm2_expected_tv_ratio": 0.0, "thm3_retention_ratio": 0.0}
    for _ in range(n):
        S = int(RNG.integers(2, 7))
        A = int(RNG.integers(2, 5))
        H = int(RNG.integers(1, 7))
        P, goal, d0 = random_mdp(S, A, H)
        pi = random_policy(H, S, A, conc=float(RNG.choice([0.3, 1.0, 3.0])))
        pj = perturb(pi, float(RNG.choice([0.02, 0.1, 0.4, 1.5])))

        Q, V = evaluate(P, goal, pi)
        Adv = Q - V[:H, :, None]
        d_old = occupancy(P, d0, pi)
        d_new = occupancy(P, d0, pj)
        J_old = float(d0 @ V[0])
        J_new = J(P, goal, d0, pj)

        Abar = (pj * Adv).sum(axis=-1)  # [H,S]: E_{a~pi'} A^pi_h(s,a)
        pdl = float(sum(d_new[h] @ Abar[h] for h in range(H)))
        worst["pdl_abs_error"] = max(worst["pdl_abs_error"], abs((J_new - J_old) - pdl))

        tv = tv_rows(pj, pi)  # [H,S]
        alpha_bar = np.array([d_old[h] @ tv[h] for h in range(H)])
        alpha_inf = float(tv.max())
        eps = float(np.abs(Abar).max())

        # Lemma 2: TV(d_h', d_h) <= sum_{t<h} alpha_bar_t
        for h in range(H + 1):
            lhs = 0.5 * np.abs(d_new[h] - d_old[h]).sum()
            worst["lemma2_shift_slack"] = min(worst["lemma2_shift_slack"], alpha_bar[:h].sum() - lhs)

        L = float(sum(d_old[h] @ Abar[h] for h in range(H)))
        gap = (J_new - J_old) - L
        pen_exp = 2.0 * eps * sum(alpha_bar[:h].sum() for h in range(H))
        pen_sup = H * (H - 1) * alpha_inf ** 2
        kl_max = float(kl_rows(pj, pi).max())
        pen_kl = H * (H - 1) * kl_max / 2.0
        worst["thm2_expected_tv_slack"] = min(worst["thm2_expected_tv_slack"], gap + pen_exp)
        worst["thm2_sup_tv_slack"] = min(worst["thm2_sup_tv_slack"], gap + pen_sup)
        worst["thm2_kl_slack"] = min(worst["thm2_kl_slack"], gap + pen_kl)
        if pen_exp > 1e-9:
            tightest["thm2_expected_tv_ratio"] = max(tightest["thm2_expected_tv_ratio"], -gap / pen_exp)

        # Theorem 3: |J' - J| <= sum_h alpha_bar_h, expectations under the OLD policy
        ret = float(alpha_bar.sum())
        worst["thm3_retention_slack"] = min(worst["thm3_retention_slack"], ret - abs(J_new - J_old))
        if ret > 1e-9:
            tightest["thm3_retention_ratio"] = max(tightest["thm3_retention_ratio"], abs(J_new - J_old) / ret)

        # Anchor corollary: |J(pi) - J(ref)| <= sqrt(H/2 * sum_h E_{d_h^ref} KL(ref || pi))
        D_ref = float(sum(d_old[h] @ kl_rows(pi[h], pj[h]) for h in range(H)))
        worst["anchor_slack"] = min(worst["anchor_slack"], math.sqrt(H * D_ref / 2.0) - abs(J_new - J_old))
    return {"instances": n, "worst": worst, "tightest_observed_ratio": tightest}


# --------------------------------------------------------------------------
# Theorem 4: KL-proximal (mirror-descent) policy iteration
# --------------------------------------------------------------------------


def optimal_policy(P, goal):
    H, S, A, _ = P.shape
    V = np.zeros((H + 1, S))
    V[H] = goal
    pi = np.zeros((H, S, A))
    for h in range(H - 1, -1, -1):
        Qh = P[h] @ V[h + 1]
        best = Qh.argmax(axis=-1)
        pi[h, np.arange(S), best] = 1.0
        V[h] = Qh.max(axis=-1)
    return pi


def check_mirror_descent(n: int = 600):
    worst_slack = math.inf
    worst_ratio = 0.0
    examples = []
    for i in range(n):
        S = int(RNG.integers(2, 7))
        A = int(RNG.integers(2, 5))
        H = int(RNG.integers(2, 8))
        K = int(RNG.choice([5, 20, 80]))
        P, goal, d0 = random_mdp(S, A, H)
        pi0 = random_policy(H, S, A, conc=float(RNG.choice([0.5, 2.0])))
        lam = float(RNG.choice([0.02, 0.1, 0.3]))
        comp = (1 - lam) * optimal_policy(P, goal) + lam / A  # smoothed comparator
        d_comp = occupancy(P, d0, comp)
        D = float(sum(d_comp[h] @ kl_rows(comp[h], pi0[h]) for h in range(H)))
        eta_opt = math.sqrt(8.0 * D / (K * H)) if D > 0 else 1.0
        eta = eta_opt * float(RNG.choice([0.3, 1.0, 3.0]))
        J_comp = J(P, goal, d0, comp)

        pi = pi0.copy()
        regret = 0.0
        for _ in range(K):
            Qk, Vk = evaluate(P, goal, pi)
            regret += J_comp - float(d0 @ Vk[0])
            logits = np.log(pi + 1e-300) + eta * Qk
            logits -= logits.max(axis=-1, keepdims=True)
            pi = np.exp(logits)
            pi /= pi.sum(axis=-1, keepdims=True)
        avg_gap = regret / K
        bound = D / (eta * K) + H * eta / 8.0
        worst_slack = min(worst_slack, bound - avg_gap)
        if bound > 1e-9:
            worst_ratio = max(worst_ratio, avg_gap / bound)
        if i < 3:
            examples.append({"S": S, "A": A, "H": H, "K": K, "D": D, "eta": eta,
                             "avg_gap": avg_gap, "bound": bound,
                             "rate_bound_at_eta_opt": math.sqrt(D * H / (2.0 * K))})
    return {"instances": n, "worst_slack": worst_slack,
            "tightest_observed_gap_over_bound": worst_ratio, "examples": examples}


# --------------------------------------------------------------------------
# Theorem 1: GRPO with binary reward; finite-G coefficient and G -> inf limit
# --------------------------------------------------------------------------


def c_G(p: float, G: int, eps: float = 1e-6) -> float:
    total = 0.0
    for k in range(G + 1):
        q = k / G
        w = math.comb(G, k) * p ** k * (1 - p) ** (G - k)
        d = max(math.sqrt(q * (1 - q)), eps)
        total += w * q * (1 - q) / d
    return total / (p * (1 - p))


def check_grpo(n_mc: int = 400_000):
    rows = []
    worst_rel = 0.0
    for p in (0.03, 0.1, 0.3, 0.5, 0.8):
        for G in (2, 8, 32):
            # Bernoulli "policy" with logit theta: score of an outcome is Y - p,
            # dp/dtheta = p(1-p). One trajectory = one Bernoulli draw.
            Y = (RNG.random((n_mc, G)) < p).astype(float)
            Qm = Y.mean(axis=1, keepdims=True)
            d = np.maximum(np.sqrt(Qm * (1 - Qm)), 1e-6)
            Aadv = (Y - Qm) / d
            Z = Y - p
            mc = float((Aadv * Z).mean(axis=1).mean())
            exact = c_G(p, G) * p * (1 - p)
            se = float((Aadv * Z).mean(axis=1).std() / math.sqrt(n_mc))
            rel = abs(mc - exact) / max(se, 1e-12)
            worst_rel = max(worst_rel, rel)
            rows.append({"p": p, "G": G, "monte_carlo": mc, "formula": exact,
                         "z_score": rel,
                         "population_limit": p * (1 - p) / math.sqrt(p * (1 - p)),
                         "P_mixed": 1 - p ** G - (1 - p) ** G})
    # Advantage magnitude: one success in a binary group of G gives sqrt(G-1).
    max_adv = {G: math.sqrt(G - 1) for G in (4, 8, 16)}
    min_nonzero_std = {G: math.sqrt((1 / G) * (1 - 1 / G)) for G in (4, 8, 16)}
    return {"rows": rows, "worst_z_score": worst_rel,
            "max_abs_binary_advantage": max_adv,
            "min_nonzero_binary_group_std": min_nonzero_std}


# --------------------------------------------------------------------------
# Theorem 6: composition through a handoff distribution
# --------------------------------------------------------------------------


def check_composition(n: int = 20000):
    worst = math.inf
    for _ in range(n):
        m = int(RNG.integers(2, 30))
        rho = RNG.dirichlet(np.full(m, float(RNG.choice([0.2, 1.0, 5.0]))))
        mu = RNG.dirichlet(np.full(m, float(RNG.choice([0.2, 1.0, 5.0]))))
        V2 = RNG.random(m)
        p1 = float(RNG.random())
        lhs = p1 * float(rho @ V2)
        rhs = p1 * (float(mu @ V2) - 0.5 * float(np.abs(rho - mu).sum()))
        worst = min(worst, lhs - rhs)
    return {"instances": n, "worst_slack": worst}


# --------------------------------------------------------------------------
# Section 6 numbers quoted in the text
# --------------------------------------------------------------------------


def quoted_numbers():
    out = {}
    for p in (0.001, 0.01, 0.03, 0.1, 0.3):
        G = 8
        pm = 1 - p ** G - (1 - p) ** G
        out[f"p={p}"] = {"P_mixed_G8": pm,
                         "expected_groups_to_first_success_G8": 1.0 / (1 - (1 - p) ** G)}
    # Combination-lock illustration: m sequential stages, each q under pi_0.
    out["chain"] = {f"m={m},q=0.1": 0.1 ** m for m in (1, 2, 3, 4)}
    # Gaussian D example: per action dim KL(N(mu~, s~^2) || N(mu0, s0^2)).
    s0 = math.exp(-1.2)
    def kl_g(dm, s1, s0_):
        return math.log(s0_ / s1) + (s1 ** 2 + dm ** 2) / (2 * s0_ ** 2) - 0.5
    out["gaussian_kl_per_dim"] = {
        "sigma0": s0,
        "dmu=0.1,s1=s0": kl_g(0.1, s0, s0),
        "dmu=0.5,s1=s0": kl_g(0.5, s0, s0),
        "dmu=0.5,s1=s0/2": kl_g(0.5, s0 / 2, s0),
    }
    # Pinsker on the logged mean KL per update of the 1e-5 continuation.
    out["pinsker_from_mean_kl_0.009"] = math.sqrt(0.009 / 2)
    return out


def main() -> None:
    results = {
        "trust_region_and_retention": check_trust_region(),
        "mirror_descent": check_mirror_descent(),
        "grpo_binary": check_grpo(),
        "composition": check_composition(),
        "quoted_numbers": quoted_numbers(),
    }
    tr = results["trust_region_and_retention"]["worst"]
    passed = {
        "lemma1_pdl": tr["pdl_abs_error"] < 1e-9,
        "lemma2_shift": tr["lemma2_shift_slack"] >= -TOL,
        "theorem2_expected_tv": tr["thm2_expected_tv_slack"] >= -TOL,
        "theorem2_sup_tv": tr["thm2_sup_tv_slack"] >= -TOL,
        "theorem2_kl": tr["thm2_kl_slack"] >= -TOL,
        "theorem3_retention": tr["thm3_retention_slack"] >= -TOL,
        "corollary_anchor": tr["anchor_slack"] >= -TOL,
        "theorem4_mirror_descent": results["mirror_descent"]["worst_slack"] >= -TOL,
        "theorem1_grpo_coefficient": results["grpo_binary"]["worst_z_score"] < 5.0,
        "theorem6_composition": results["composition"]["worst_slack"] >= -TOL,
    }
    passed = {k: bool(v) for k, v in passed.items()}
    results["passed"] = passed
    out = Path(__file__).with_name("verification_results.json")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    for k, v in passed.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
