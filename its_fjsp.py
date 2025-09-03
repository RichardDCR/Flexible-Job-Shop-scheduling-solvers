"""
Integrated Tabu Search (ITS) for Flexible Job Shop Scheduling (FJSSP)

Modes (using file "mfjs10.txt" as an example)

1) One run test: Test one run of algorithm and plot 2 charts
    python its_fjsp.py --instance mfjs10.txt

2) Evaluation/No-Improvement mode with known solution
    python its_fjsp.py --instance mfjs10.txt --runs 100 --evals 1000 --noimp 100 --solution 66

3) Evaluation/No-Improvement mode with unknown solution    
    python its_fjsp.py --instance mfjs10.txt --runs 100 --evals 1000 --noimp 100 --lb 66

4) Time-boxed mode with known solution
     python its_fjsp.py --instance mfjs10.txt --runs 100 --runtime 0.5 --solution 66
     
5) Time-boxed mode with unknown solution     
     python its_fjsp.py --instance mfjs10.txt --runs 100 --runtime 0.5 --lb 66
"""

import argparse
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

# ======================= Instance I/O =======================

@dataclass
class Instance:
    jobs: List[List[List[Tuple[int, int]]]]  # jobs[j][h] = [(machine, proc_time), ...]
    num_jobs: int
    num_machines: int
    total_ops: int

def load_instance(path: str) -> Instance:
    with open(path, "r") as f:
        first = f.readline().strip().split()
        num_jobs, num_machines = int(first[0]), int(first[1])
        jobs = []
        total_ops = 0
        for _ in range(num_jobs):
            parts = list(map(int, f.readline().strip().split()))
            idx = 0
            n_ops = parts[idx]; idx += 1
            job_ops = []
            for _h in range(n_ops):
                k = parts[idx]; idx += 1
                choices = []
                for _ in range(k):
                    m = parts[idx]; t = parts[idx+1]; idx += 2
                    choices.append((m, t))
                job_ops.append(choices)
                total_ops += 1
            jobs.append(job_ops)
    return Instance(jobs=jobs, num_jobs=num_jobs, num_machines=num_machines, total_ops=total_ops)

def build_op_index_maps(inst: Instance):
    op2pair, pair2op = [], {}
    for j, job in enumerate(inst.jobs):
        for h in range(len(job)):
            pair2op[(j, h)] = len(op2pair)
            op2pair.append((j, h))
    return op2pair, pair2op

def ptime_table(inst: Instance):
    """ptime[j][h][m] -> processing time"""
    return [[{m:p for (m,p) in ops} for ops in job] for job in inst.jobs]

# ======================= Plot helpers =======================

def plot_cmax_curve(history: List[float], title: str = "Best Cmax over search"):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not history:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(history, linewidth=2)
    plt.xlabel("Iterations / Evaluations (depending on mode)")
    plt.ylabel("Best Cmax")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_gantt(records: List[Dict[str, float]], num_machines: int, cmax: float,
               save_path: str = None, show: bool = True):
    """Simple Gantt using matplotlib.broken_barh. One row per machine."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except Exception:
        return
    if not records:
        return
    job_ids = sorted({int(r["job"]) for r in records})
    cmap = plt.get_cmap("tab20")
    job_colors = {j: cmap(j % cmap.N) for j in job_ids}

    def label_color(rgba):
        r, g, b, a = rgba
        L = 0.2126*r + 0.7152*g + 0.0722*b
        return "black" if L > 0.55 else "white"

    fig, ax = plt.subplots(figsize=(10, 2 + 0.6 * max(1, num_machines)))
    ax.set_title(f"Gantt (Cmax = {cmax:.1f})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Machines")
    ax.set_yticks([i + 0.4 for i in range(num_machines)])
    ax.set_yticklabels([f"M{i}" for i in range(num_machines)])
    ax.set_ylim(0, num_machines)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    for rec in records:
        j = int(rec["job"]); m = int(rec["machine"])
        s = float(rec["start"]); d = float(rec["finish"] - rec["start"])
        color = job_colors[j]
        ax.broken_barh([(s, d)], (m + 0.05, 0.8),
                       facecolors=color, edgecolors="black", linewidth=0.6)
        ax.text(s + d/2.0, m + 0.45, f"J{j}O{int(rec['op'])}",
                ha="center", va="center", fontsize=8, color=label_color(color))

    ax.set_xlim(0, max(cmax, 1.0))
    patches = [Patch(facecolor=job_colors[j], edgecolor="black", label=f"Job {j}") for j in job_ids]
    if patches:
        ax.legend(handles=patches, ncol=min(4, len(patches)), fontsize=8,
                  loc="upper right", frameon=True)

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

# ======================= Decoder: GT-like guided by per-machine priorities =======================

def decode_with_priorities(inst: Instance,
                           assign: List[int],
                           machine_prios: List[List[int]],
                           ptab: List[List[Dict[int,int]]]) -> Tuple[float, List[Dict[str, float]]]:
    """
    Builds an active schedule using a GT-like loop.
    - Each machine m has a priority list (sequence of op_ids assigned to m).
    - At each step, for every machine consider ONLY its *next* unscheduled op in its priority list
      IF that op is the current next operation for its job (precedence). Among feasible candidates,
      schedule the one that finishes earliest (ties broken by the machine list pointer).
    """
    op2pair, pair2op = build_op_index_maps(inst)
    num_jobs, num_machines, total_ops = inst.num_jobs, inst.num_machines, inst.total_ops

    pr_idx = [0] * num_machines      # pointer inside each machine priority list
    job_next = [0] * num_jobs        # next operation index for each job
    job_ready = [0.0] * num_jobs
    mach_ready = [0.0] * num_machines
    records: List[Dict[str, float]] = []
    scheduled = 0

    while scheduled < total_ops:
        best = None  # candidate tuple: (ECT, tie_rank, m_sel, op_id, j, h, start, p)

        # one candidate per machine (its current list head if job is ready)
        for m in range(num_machines):
            idx = pr_idx[m]
            lst = machine_prios[m]
            if idx >= len(lst):
                continue
            op_id = lst[idx]
            j, h = op2pair[op_id]
            if h != job_next[j]:
                continue  # job not ready for this op yet

            m_ass = assign[op_id]
            p = ptab[j][h].get(m_ass, None)
            if p is None:
                if inst.jobs[j][h]:
                    m_ass = inst.jobs[j][h][0][0]
                    p = inst.jobs[j][h][0][1]
                else:
                    continue
            s = max(job_ready[j], mach_ready[m_ass])
            ect = s + p

            cand = (ect, idx, m_ass, op_id, j, h, s, p)
            if (best is None) or (cand < best):
                best = cand

        # Fallback: if nothing ready, pick next-by-job earliest ECT
        if best is None:
            fallback = None
            for j in range(num_jobs):
                h = job_next[j]
                if h >= len(inst.jobs[j]):
                    continue
                op_id = pair2op[(j, h)]
                m_ass = assign[op_id]
                p = ptab[j][h].get(m_ass, None)
                if p is None and inst.jobs[j][h]:
                    m_ass = inst.jobs[j][h][0][0]; p = inst.jobs[j][h][0][1]
                s = max(job_ready[j], mach_ready[m_ass])
                ect = s + p
                cand = (ect, 10**9, m_ass, op_id, j, h, s, p)
                if (fallback is None) or (cand < fallback):
                    fallback = cand
            best = fallback

        if best is None:
            break

        ect, idx, m_sel, op_id, j_sel, h_sel, s_sel, p_sel = best
        finish = s_sel + p_sel
        job_ready[j_sel] = float(finish)
        mach_ready[m_sel] = float(finish)
        job_next[j_sel] += 1
        scheduled += 1
        records.append({"job": j_sel, "op": h_sel, "machine": m_sel, "start": float(s_sel), "finish": float(finish)})

        # advance pointer on the machine where this op sits
        if op_id in machine_prios[m_sel]:
            if pr_idx[m_sel] < len(machine_prios[m_sel]) and machine_prios[m_sel][pr_idx[m_sel]] == op_id:
                pr_idx[m_sel] += 1
            else:
                try:
                    pos = machine_prios[m_sel].index(op_id)
                    machine_prios[m_sel].pop(pos)
                    if pos < pr_idx[m_sel] and pr_idx[m_sel] > 0:
                        pr_idx[m_sel] -= 1
                except ValueError:
                    pass

    cmax = max(mach_ready) if num_machines > 0 else 0.0
    return cmax, records

# ======================= Initial solution =======================

def initial_solution(inst: Instance):
    """Greedy machine assignment (min p) + random per-machine priority lists."""
    op2pair, pair2op = build_op_index_maps(inst)
    ptab = ptime_table(inst)

    assign = []
    for (j,h) in op2pair:
        best_m, best_p = None, math.inf
        for m, p in inst.jobs[j][h]:
            if p < best_p:
                best_m, best_p = m, p
        assign.append(best_m)

    machine_prios = [[] for _ in range(inst.num_machines)]
    for op_id, (j,h) in enumerate(op2pair):
        m = assign[op_id]
        machine_prios[m].append(op_id)
    for m in range(inst.num_machines):
        random.shuffle(machine_prios[m])

    return assign, machine_prios, ptab

# ======================= Neighborhood & Tabu =======================

def neighbor_reassign(inst: Instance, assign: List[int], machine_prios: List[List[int]]):
    op2pair, _ = build_op_index_maps(inst)
    op_id = random.randrange(len(assign))
    j,h = op2pair[op_id]
    choices = [m for (m,_p) in inst.jobs[j][h]]
    if len(choices) <= 1:
        return None
    old_m = assign[op_id]
    alts = [m for m in choices if m != old_m]
    if not alts:
        return None
    new_m = random.choice(alts)

    a2 = assign[:]
    a2[op_id] = new_m

    mp2 = [lst[:] for lst in machine_prios]
    try:
        mp2[old_m].remove(op_id)
    except ValueError:
        pass
    pos = random.randrange(len(mp2[new_m]) + 1) if mp2[new_m] else 0
    mp2[new_m].insert(pos, op_id)

    attr = ('A', op_id, new_m)
    return a2, mp2, attr

def neighbor_swap(inst: Instance, assign: List[int], machine_prios: List[List[int]]):
    m = random.randrange(inst.num_machines)
    if len(machine_prios[m]) < 2:
        return None
    i, j = sorted(random.sample(range(len(machine_prios[m])), 2))
    mp2 = [lst[:] for lst in machine_prios]
    op_i = mp2[m][i]; op_j = mp2[m][j]
    mp2[m][i], mp2[m][j] = op_j, op_i
    attr = ('S', m, min(op_i, op_j), max(op_i, op_j))
    return assign[:], mp2, attr

def sample_neighbor(inst: Instance, assign: List[int], machine_prios: List[List[int]], p_swap: float = 0.5):
    if random.random() < p_swap:
        return neighbor_swap(inst, assign, machine_prios) or neighbor_reassign(inst, assign, machine_prios)
    else:
        return neighbor_reassign(inst, assign, machine_prios) or neighbor_swap(inst, assign, machine_prios)

# ======================= ITS cores =======================

def its_iterative(inst: Instance,
                  iters: int = 1000,
                  tenure: int = 10,
                  neighbors: int = 60,
                  p_swap: float = 0.5,
                  verbose: bool = True) -> Dict[str, Any]:
    """Default ITS: run fixed iterations, print best Cmax per iteration, keep history and best records."""
    assign, machine_prios, ptab = initial_solution(inst)
    best_c, best_rec = decode_with_priorities(inst, assign, [lst[:] for lst in machine_prios], ptab)
    best_assign, best_prios = assign[:], [lst[:] for lst in machine_prios]
    curr_assign, curr_prios, curr_c = assign[:], [lst[:] for lst in machine_prios], best_c
    history: List[float] = [best_c]

    tabu: Dict[Tuple, int] = {}  # attr -> expire_iter

    for it in range(1, iters + 1):
        cand_a = None; cand_p = None; cand_c = math.inf; cand_attr = None; cand_rec = None

        for _ in range(neighbors):
            nb = sample_neighbor(inst, curr_assign, curr_prios, p_swap=p_swap)
            if nb is None:
                continue
            a2, p2, attr = nb
            is_tabu = (attr in tabu and tabu[attr] > it)

            c2, rec2 = decode_with_priorities(inst, a2, [lst[:] for lst in p2], ptab)

            # aspiration: allow tabu if improves global best
            if is_tabu and c2 + 1e-12 >= best_c:
                continue

            if c2 < cand_c:
                cand_a, cand_p, cand_c, cand_attr, cand_rec = a2, p2, c2, attr, rec2

        if cand_a is not None:
            curr_assign, curr_prios, curr_c = cand_a, cand_p, cand_c
            if cand_attr is not None:
                tabu[cand_attr] = it + tenure
            # global-best update
            if curr_c + 1e-12 < best_c:
                best_c = curr_c
                best_assign, best_prios = curr_assign[:], [lst[:] for lst in curr_prios]
                best_rec = cand_rec

        history.append(best_c)
        if verbose:
            print(f"[iter {it}] best Cmax = {best_c:.2f}")

    return {"best_cmax": best_c, "best_assign": best_assign, "best_prios": best_prios, "history": history, "best_records": best_rec}

def its_eval_budget(inst: Instance,
                    eval_limit: int = 10**9,
                    no_improve_limit: int = 100,
                    tenure: int = 10,
                    neighbors: int = 60,
                    p_swap: float = 0.5,
                    history_out: bool = False) -> Dict[str, Any]:
    """Evaluation-based ITS: stop by eval budget or consecutive non-improving *evaluations*."""
    assign, machine_prios, ptab = initial_solution(inst)
    best_c, best_rec = decode_with_priorities(inst, assign, [lst[:] for lst in machine_prios], ptab)
    best_assign, best_prios = assign[:], [lst[:] for lst in machine_prios]
    curr_assign, curr_prios, curr_c = assign[:], [lst[:] for lst in machine_prios], best_c

    history: List[float] = [best_c] if history_out else []
    tabu: Dict[Tuple, int] = {}

    it = 0
    evals = 0
    no_impr_evals = 0
    stop_reason = None

    while evals < eval_limit and no_impr_evals < no_improve_limit:
        it += 1
        cand_a = None; cand_p = None; cand_c = math.inf; cand_attr = None; cand_rec = None

        for _ in range(neighbors):
            if evals >= eval_limit or no_impr_evals >= no_improve_limit:
                break
            nb = sample_neighbor(inst, curr_assign, curr_prios, p_swap=p_swap)
            if nb is None:
                continue
            a2, p2, attr = nb
            is_tabu = (attr in tabu and tabu[attr] > it)

            c2, rec2 = decode_with_priorities(inst, a2, [lst[:] for lst in p2], ptab)
            evals += 1

            if c2 + 1e-12 < best_c:
                best_c = c2
                best_assign, best_prios = a2[:], [lst[:] for lst in p2]
                best_rec = rec2
                no_impr_evals = 0
            else:
                no_impr_evals += 1

            if history_out:
                history.append(best_c)

            if is_tabu and c2 + 1e-12 >= best_c:
                continue

            if c2 < cand_c:
                cand_a, cand_p, cand_c, cand_attr, cand_rec = a2, p2, c2, attr, rec2

            if evals >= eval_limit or no_impr_evals >= no_improve_limit:
                break

        if cand_a is not None:
            curr_assign, curr_prios, curr_c = cand_a, cand_p, cand_c
            if cand_attr is not None:
                tabu[cand_attr] = it + tenure

        if evals >= eval_limit:
            stop_reason = "budget_exhausted"; break
        if no_impr_evals >= no_improve_limit:
            stop_reason = "no_improvement_limit"; break

    if stop_reason is None:
        stop_reason = "stopped"

    return {
        "best_cmax": best_c,
        "best_assign": best_assign,
        "best_prios": best_prios,
        "best_records": best_rec,
        "evaluations": evals,
        "history": history,
        "stop_reason": stop_reason
    }

def its_timeboxed(inst: Instance,
                  seconds: float = 1.0,
                  tenure: int = 10,
                  neighbors: int = 60,
                  p_swap: float = 0.5) -> Dict[str, Any]:
    """Time-boxed ITS: ignore eval/no-imp caps; track time to best and keep best records."""
    t0 = time.perf_counter()
    assign, machine_prios, ptab = initial_solution(inst)
    best_c, best_rec = decode_with_priorities(inst, assign, [lst[:] for lst in machine_prios], ptab)
    best_assign, best_prios = assign[:], [lst[:] for lst in machine_prios]
    curr_assign, curr_prios, curr_c = assign[:], [lst[:] for lst in machine_prios], best_c

    tabu: Dict[Tuple, int] = {}
    evals = 0
    it = 0
    time_best_ms = 0.0

    def time_left():
        return (time.perf_counter() - t0) < seconds

    while time_left():
        it += 1
        cand_a = None; cand_p = None; cand_c = math.inf; cand_attr = None; cand_rec = None

        for _ in range(neighbors):
            if not time_left():
                break
            nb = sample_neighbor(inst, curr_assign, curr_prios, p_swap=p_swap)
            if nb is None:
                continue
            a2, p2, attr = nb
            is_tabu = (attr in tabu and tabu[attr] > it)

            c2, rec2 = decode_with_priorities(inst, a2, [lst[:] for lst in p2], ptab)
            evals += 1

            if c2 + 1e-12 < best_c:
                best_c = c2
                best_assign, best_prios = a2[:], [lst[:] for lst in p2]
                best_rec = rec2
                time_best_ms = (time.perf_counter() - t0) * 1000.0

            if is_tabu and c2 + 1e-12 >= best_c:
                continue

            if c2 < cand_c:
                cand_a, cand_p, cand_c, cand_attr, cand_rec = a2, p2, c2, attr, rec2

            if not time_left():
                break

        if cand_a is not None:
            curr_assign, curr_prios, curr_c = cand_a, cand_p, cand_c
            if cand_attr is not None:
                tabu[cand_attr] = it + tenure

        if not time_left():
            break

    return {
        "best_cmax": best_c,
        "best_assign": best_assign,
        "best_prios": best_prios,
        "best_records": best_rec,
        "evaluations": evals,
        "time_best_ms": time_best_ms,
        "stop_reason": "time_limit"
    }

# ======================= CLI & Reporting =======================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance", type=str, required=False, default="mfjs10.txt")
    p.add_argument("--runs", type=int, default=1, help="Number of independent runs")
    # Default iteration-based mode
    p.add_argument("--iters", type=int, default=4000, help="Iterations (default mode)")
    # Eval/No-imp mode
    p.add_argument("--evals", type=int, default=None, help="Evaluation budget (per run)")
    p.add_argument("--noimp", type=int, default=4000, help="No-improvement limit (consecutive evaluations)")
    # Runtime mode
    p.add_argument("--runtime", type=float, default=None, help="Seconds per run (time-boxed)")
    # Targets / reporting
    p.add_argument("--solution", type=float, default=None, help="Target Cmax for SR/GAP reports")
    p.add_argument("--lb", type=float, default=None, help="Lower bound for LB summaries")
    # ITS knobs
    p.add_argument("--tenure", type=int, default=10, help="Tabu tenure")
    p.add_argument("--neighbors", type=int, default=60, help="Neighbors sampled per iteration")
    p.add_argument("--swap_prob", type=float, default=0.5, help="Probability of sequencing swap vs reassignment")
    # Plot toggle
    p.add_argument("--no_show", action="store_true", help="Do not display plots for single-run modes")
    return p.parse_args()

def main():
    args = parse_args()
    if not os.path.exists(args.instance):
        print(f"Instance file '{args.instance}' not found.")
        return
    inst = load_instance(args.instance)

    # --------- RUNTIME MODE ---------
    if args.runtime is not None:
        if args.runs <= 1:
            out = its_timeboxed(inst,
                                seconds=float(args.runtime),
                                tenure=args.tenure,
                                neighbors=args.neighbors,
                                p_swap=args.swap_prob)
            print(f"Best Cmax: {out['best_cmax']:.1f}")
            print(f"Evaluations: {out['evaluations']}")
            print(f"Stop reason: {out['stop_reason']}")
            print(f"Time to best: {out['time_best_ms']:.0f} ms")
            if not args.no_show:
                # No evaluation-by-evaluation history here; just Gantt
                if out.get("best_records"):
                    plot_gantt(out["best_records"], inst.num_machines, out["best_cmax"], show=True)
            return

        # batch runtime
        runs = args.runs
        bests: List[float] = []
        evals_list: List[int] = []
        timebest_list: List[float] = []
        hits = 0

        for _ in range(runs):
            out = its_timeboxed(inst,
                                seconds=float(args.runtime),
                                tenure=args.tenure,
                                neighbors=args.neighbors,
                                p_swap=args.swap_prob)
            best = float(out["best_cmax"])
            bests.append(best)
            evals_list.append(int(out["evaluations"]))
            timebest_list.append(float(out.get("time_best_ms", 0.0)))
            if args.solution is not None and int(round(best)) == int(round(args.solution)):
                hits += 1

        avg_cmax = sum(bests) / len(bests) if bests else float('inf')
        avg_evals = sum(evals_list) / len(evals_list) if evals_list else 0.0
        avg_timebest_ms = sum(timebest_list) / len(timebest_list) if timebest_list else 0.0

        gap_str = None
        if args.solution is not None and float(args.solution) != 0.0 and math.isfinite(avg_cmax):
            gap = 100.0 * (avg_cmax - float(args.solution)) / float(args.solution)
            gap_str = f"{gap:.2f}"

        if args.lb is not None:
            best_cmax_all = min(bests) if bests else float('inf')
            median_cmax = statistics.median(bests) if bests else float('inf')
            print("========== Batch Results ==========")
            print(f"Runs: {runs}")
            print(f"average of evaluations per run: {avg_evals:.1f}")
            if math.isfinite(best_cmax_all):
                print(f"best_cmax among all runs: {int(round(best_cmax_all))} (best cmax obtained in all {runs} runs)")
            else:
                print(f"best_cmax among all runs: n/a (best cmax obtained in all {runs} runs)")
            if math.isfinite(median_cmax):
                print(f"median best_cmax:{int(round(median_cmax))} (median cmax of all {runs} runs)")
            else:
                print(f"median best_cmax:n/a (median cmax of all {runs} runs)")
            lb_gap_median = None
            if args.lb != 0 and math.isfinite(median_cmax):
                lb_gap_median = (median_cmax - float(args.lb)) / float(args.lb)
            if lb_gap_median is not None:
                print(f"median LB-GAP: {lb_gap_median:.2f} ((median_cmax-lb)/lb)")
            else:
                print(f"median LB-GAP: n/a ((median_cmax-lb)/lb)")
            print("===================================")
        else:
            sr_pct = (hits / runs) * 100.0 if runs > 0 else 0.0
            sr_display = f"{int(round(sr_pct))}%"
            print("========== Batch Results ==========")
            print(f"Runs: {runs}")
            print(f"runtime: {args.runtime}")
            print(f"Average Evaluations per run: {avg_evals:.1f}")
            if args.solution is not None:
                print(f"Success solution rate: {sr_display}")
            if gap_str is not None:
                print(f"GAP: {gap_str}")
            print(f"Average time best Cmax obtained: {avg_timebest_ms:.0f} ms")
            print("===================================")
        return

    # --------- EVAL/NO-IMP MODE ---------
    if args.evals is not None or args.noimp is not None:
        if args.runs <= 1:
            t0 = time.perf_counter()
            out = its_eval_budget(inst,
                                  eval_limit=int(args.evals) if args.evals is not None else int(1e9),
                                  no_improve_limit=int(args.noimp) if args.noimp is not None else int(1e9),
                                  tenure=args.tenure,
                                  neighbors=args.neighbors,
                                  p_swap=args.swap_prob,
                                  history_out=True)
            t1 = time.perf_counter()
            ms = (t1 - t0) * 1000.0
            print(f"Best Cmax: {out['best_cmax']:.1f}")
            print(f"Evaluations: {out['evaluations']}")
            print(f"Stop reason: {out['stop_reason']}")
            print(f"CPU time: {ms:.2f} ms")
            if not args.no_show:
                if out["history"]:
                    plot_cmax_curve(out["history"], title="Best Cmax over evaluations (ITS)")
                if out.get("best_records"):
                    plot_gantt(out["best_records"], inst.num_machines, out["best_cmax"], show=True)
            return

        # batch eval/noimp
        runs = args.runs
        bests: List[float] = []
        times_ms: List[float] = []
        hits = 0
        noimp_stops = 0
        budget_stops = 0
        success_times_ms: List[float] = []

        eval_cap = int(args.evals) if args.evals is not None else int(1e9)
        noimp_cap = int(args.noimp) if args.noimp is not None else int(1e9)

        for _ in range(runs):
            t0 = time.perf_counter()
            out = its_eval_budget(inst,
                                  eval_limit=eval_cap,
                                  no_improve_limit=noimp_cap,
                                  tenure=args.tenure,
                                  neighbors=args.neighbors,
                                  p_swap=args.swap_prob,
                                  history_out=False)
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0

            best = float(out["best_cmax"])
            bests.append(best)
            times_ms.append(elapsed_ms)

            if args.solution is not None and int(round(best)) == int(round(args.solution)):
                hits += 1
                success_times_ms.append(elapsed_ms)

            if out.get("stop_reason") == "no_improvement_limit":
                if args.solution is None or best > float(args.solution):
                    noimp_stops += 1
            if out.get("stop_reason") == "budget_exhausted":
                budget_stops += 1

        avg_cmax = sum(bests) / len(bests) if bests else float('inf')
        sr_pct = (hits / runs) * 100.0 if runs > 0 else 0.0
        noimp_pct = (noimp_stops / runs) * 100.0 if runs > 0 else 0.0
        budget_pct = (budget_stops / runs) * 100.0 if runs > 0 else 0.0

        if success_times_ms:
            avg_success_ms = sum(success_times_ms) / len(success_times_ms)
            avg_success_str = f"{avg_success_ms:.0f} ms"
        else:
            avg_success_str = "n/a"

        if args.lb is not None:
            best_cmax_all = min(bests) if bests else float('inf')
            median_cmax = statistics.median(bests) if bests else float('inf')
            lb_gap_median = None
            if args.lb != 0 and math.isfinite(median_cmax):
                lb_gap_median = (median_cmax - float(args.lb)) / float(args.lb)

            print("========== Batch Results ==========")
            print(f"Runs: {runs}")
            if math.isfinite(best_cmax_all):
                print(f"best_cmax among all runs: {int(round(best_cmax_all))} (best cmax obtained in all {runs} runs)")
            else:
                print(f"best_cmax among all runs: n/a (best cmax obtained in all {runs} runs)")
            if math.isfinite(median_cmax):
                print(f"median best_cmax:{int(round(median_cmax))} (median cmax of all {runs} runs)")
            else:
                print(f"median best_cmax:n/a (median cmax of all {runs} runs)")
            print()
            print(f"Times no improvement happen: {noimp_pct:.1f}% (times stop by no improvement)")
            if args.evals is not None:
                print(f"Times all evaluations finished: {budget_pct:.0f}% (times finished {int(args.evals)} evals)")
            else:
                print(f"Times all evaluations finished: {budget_pct:.0f}% (times finished budget)")
            print()
            if lb_gap_median is not None:
                print(f"median LB-GAP: {lb_gap_median:.2f} ((median_cmax-lb)/lb)")
            else:
                print(f"median LB-GAP: n/a ((median_cmax-lb)/lb)")
            avg_all_ms = (sum(times_ms) / len(times_ms)) if times_ms else 0.0
            print(f"AVG CPU time per run: {avg_all_ms:.0f} ms ")
            print("===================================")
        else:
            gap_str = None
            if args.solution is not None and float(args.solution) != 0.0 and math.isfinite(avg_cmax):
                gap = 100.0 * (avg_cmax - float(args.solution)) / float(args.solution)
                gap_str = f"{gap:.2f}"

            print("========== Batch Results ==========")
            print(f"Runs: {runs}")
            if args.solution is not None:
                print(f"SR to solution {int(round(args.solution))}: {sr_pct:.1f}%")
            print()
            print(f"Times no improvement happen: {noimp_pct:.1f}%")
            print()
            if gap_str is not None:
                print(f"GAP: {gap_str}")
            print(f"AVG CPU successful run: {avg_success_str}")
            print()
            print("===================================")
        return

    # --------- DEFAULT: iteration-based ---------
    if args.runs <= 1:
        out = its_iterative(inst,
                            iters=args.iters,
                            tenure=args.tenure,
                            neighbors=args.neighbors,
                            p_swap=args.swap_prob,
                            verbose=True)
        print(f"Final best Cmax: {out['best_cmax']:.2f}")
        if not args.no_show:
            if out.get("history"):
                plot_cmax_curve(out["history"], title="Best Cmax over iterations (ITS)")
            if out.get("best_records"):
                plot_gantt(out["best_records"], inst.num_machines, out["best_cmax"], show=True)
    else:
        bests: List[float] = []
        for _ in range(args.runs):
            out = its_iterative(inst,
                                iters=args.iters,
                                tenure=args.tenure,
                                neighbors=args.neighbors,
                                p_swap=args.swap_prob,
                                verbose=False)
            bests.append(float(out["best_cmax"]))
        median_c = statistics.median(bests) if bests else float('inf')
        print("========== Batch Results ==========")
        print(f"Runs: {args.runs}")
        if math.isfinite(min(bests)) and math.isfinite(median_c):
            print(f"best_cmax among all runs: {int(round(min(bests)))}")
            print(f"median best_cmax: {int(round(median_c))}")
        else:
            print("best_cmax among all runs: n/a")
            print("median best_cmax: n/a")
        print("===================================")

if __name__ == "__main__":
    main()
