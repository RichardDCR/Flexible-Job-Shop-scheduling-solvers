"""
Simulated Annealing for Flexible Job Shop Scheduling

Modes (using file "mfjs10.txt" as an example)

1) One run test: Test one run of algorithm and plot 2 charts
    python sa_fjsp.py --instance mfjs10.txt

2) Evaluation/No-Improvement mode with known solution
    python sa_fjsp.py --instance mfjs10.txt --runs 100 --evals 1000 --noimp 100 --solution 66

3) Evaluation/No-Improvement mode with unknown solution    
    python sa_fjsp.py --instance mfjs10.txt --runs 100 --evals 1000 --noimp 100 --lb 66

4) Time-boxed mode with known solution
     python sa_fjsp.py --instance mfjs10.txt --runs 100 --runtime 0.5 --solution 66
     
5) Time-boxed mode with unknown solution     
     python sa_fjsp.py --instance mfjs10.txt --runs 100 --runtime 0.5 --lb 66
"""

import argparse
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

# --------------------------- Data model ---------------------------
@dataclass
class Instance:
    jobs: List[List[List[Tuple[int, int]]]]  # jobs[j][h] = [(m,p), (m,p), ...]
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
    op2pair = []
    pair2op = {}
    for j, job in enumerate(inst.jobs):
        for h in range(len(job)):
            pair2op[(j, h)] = len(op2pair)
            op2pair.append((j, h))
    return op2pair, pair2op

# --------------------------- Schedule evaluation ---------------------------
def evaluate_schedule(machine_assignments: Dict[Tuple[int,int], Tuple[int,int]],
                      op_order: List[Tuple[int,int]],
                      jobs: List[List[List[Tuple[int,int]]]],
                      num_machines: int) -> float:
    """
    Earliest-start decoding. Returns makespan (Cmax).
    - machine_assignments[(j,o)] = (m, dur)
    - op_order is precedence-feasible list of (j,o)
    """
    machine_available = [0] * num_machines
    job_available = [0] * len(jobs)
    op_end = {}

    for j, o in op_order:
        m, dur = machine_assignments[(j, o)]
        start = max(machine_available[m], job_available[j])
        end = start + dur
        op_end[(j, o)] = end
        machine_available[m] = end
        job_available[j] = end

    makespan = max(op_end.values()) if op_end else 0
    return makespan

# --------------------------- Initial solution ---------------------------
def random_solution(jobs):
    """Random machine choices among capable machines + precedence-feasible random sequence."""
    machine_assignments = {}
    for j, ops in enumerate(jobs):
        for o, choices in enumerate(ops):
            machine_assignments[(j, o)] = random.choice(choices)

    # precedence-feasible random OS
    next_op = [0] * len(jobs)
    remaining = sum(len(ops) for ops in jobs)
    op_order = []
    while remaining > 0:
        avail_jobs = [j for j in range(len(jobs)) if next_op[j] < len(jobs[j])]
        j = random.choice(avail_jobs)
        o = next_op[j]
        op_order.append((j, o))
        next_op[j] += 1
        remaining -= 1

    return machine_assignments, op_order

# --------------------------- Validity check for swaps ---------------------------
def is_valid_order(order, jobs):
    idxs = {}
    for i, (j, o) in enumerate(order):
        idxs.setdefault(j, []).append((o, i))
    for j, lst in idxs.items():
        lst.sort()
        positions = [pos for _, pos in lst]
        if positions != sorted(positions):
            return False
    for j in range(len(jobs)):
        if j not in idxs or len(idxs[j]) != len(jobs[j]):
            return False
    return True

# --------------------------- Neighborhoods ---------------------------
def neighbor_assign(curr_assign, curr_order, jobs):
    """Change machine for one random operation to an alternative capable machine."""
    new_assign = curr_assign.copy()
    new_order = curr_order  # unchanged
    keys = list(new_assign.keys())
    random.shuffle(keys)
    for (j, o) in keys:
        choices = jobs[j][o]
        if len(choices) > 1:
            cur = new_assign[(j, o)]
            alts = [c for c in choices if c != cur]
            if alts:
                new_assign[(j, o)] = random.choice(alts)
                return new_assign, new_order
    return curr_assign, curr_order

def neighbor_swap_any(curr_assign, curr_order, jobs):
    """Swap two positions in the order, preserving precedence."""
    new_order = curr_order.copy()
    n = len(new_order)
    for _ in range(300):
        i, j = random.sample(range(n), 2)
        new_order[i], new_order[j] = new_order[j], new_order[i]
        if is_valid_order(new_order, jobs):
            return curr_assign.copy(), new_order
        new_order[i], new_order[j] = new_order[j], new_order[i]
    return curr_assign, curr_order

def neighbor_random_sa(curr_assign, curr_order, jobs, swap_prob=0.5):
    """Mix: with prob swap_prob do a precedence-safe swap, else a machine reassignment."""
    if random.random() < swap_prob:
        return neighbor_swap_any(curr_assign, curr_order, jobs)
    else:
        return neighbor_assign(curr_assign, curr_order, jobs)

# --------------------------- SA cores ---------------------------
def run_sa(jobs, num_machines,
           eval_limit: int = 10**9,        # evaluation budget (each neighbor evaluation counts)
           no_improve_limit: int = 100,    # consecutive non-improving evaluations
           T_start: float = 300.0,
           alpha: float = 0.995,
           swap_prob: float = 0.5,
           history_out: bool = False) -> Dict[str, Any]:
    """
    SA with evaluation-based early stop.
    - Counts evaluations per candidate neighbor decoded.
    - no_improve_limit counts *consecutive evaluations* that don't improve the global best.
    """
    curr_assign, curr_order = random_solution(jobs)
    curr_cost = evaluate_schedule(curr_assign, curr_order, jobs, num_machines)

    best_assign, best_order, best_cost = curr_assign.copy(), curr_order.copy(), curr_cost
    T = T_start
    evals = 0
    no_impr_evals = 0
    stop_reason = None
    history = [best_cost] if history_out else []

    while evals < eval_limit and no_impr_evals < no_improve_limit:
        # propose neighbor
        ns, no = neighbor_random_sa(curr_assign, curr_order, jobs, swap_prob=swap_prob)
        nc = evaluate_schedule(ns, no, jobs, num_machines)
        evals += 1

        # SA acceptance against *current* cost
        delta = nc - curr_cost
        accept = delta < 0 or random.random() < math.exp(-delta / max(T, 1e-12))

        if accept:
            curr_assign, curr_order, curr_cost = ns, no, nc

        # global-best update + no-impr counter
        if nc + 1e-12 < best_cost:
            best_assign, best_order, best_cost = ns.copy(), no.copy(), nc
            no_impr_evals = 0
        else:
            no_impr_evals += 1

        if history_out:
            history.append(best_cost)

        # cool down
        T *= alpha

    if evals >= eval_limit:
        stop_reason = "budget_exhausted"
    elif no_impr_evals >= no_improve_limit:
        stop_reason = "no_improvement_limit"
    else:
        stop_reason = "stopped"

    return {
        "best_assign": best_assign,
        "best_order": best_order,
        "best_cmax": best_cost,
        "evaluations": evals,
        "history": history,
        "stop_reason": stop_reason
    }

def run_sa_timeboxed(jobs, num_machines,
                     seconds: float = 1.0,
                     T_start: float = 300.0,
                     alpha: float = 0.995,
                     swap_prob: float = 0.5) -> Dict[str, Any]:
    """SA with a hard wall-clock limit per run; returns evals and time when best was first achieved."""
    t0 = time.perf_counter()

    curr_assign, curr_order = random_solution(jobs)
    curr_cost = evaluate_schedule(curr_assign, curr_order, jobs, num_machines)
    best_assign, best_order, best_cost = curr_assign.copy(), curr_order.copy(), curr_cost
    time_best_ms = 0.0

    T = T_start
    evals = 0

    def time_left() -> bool:
        return (time.perf_counter() - t0) < seconds

    while time_left():
        ns, no = neighbor_random_sa(curr_assign, curr_order, jobs, swap_prob=swap_prob)
        nc = evaluate_schedule(ns, no, jobs, num_machines)
        evals += 1

        delta = nc - curr_cost
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-12)):
            curr_assign, curr_order, curr_cost = ns, no, nc

        if nc + 1e-12 < best_cost:
            best_assign, best_order, best_cost = ns.copy(), no.copy(), nc
            time_best_ms = (time.perf_counter() - t0) * 1000.0

        T *= alpha

    return {
        "best_assign": best_assign,
        "best_order": best_order,
        "best_cmax": best_cost,
        "evaluations": evals,
        "time_best_ms": time_best_ms,
        "stop_reason": "time_limit"
    }

# --------------------------- Optional plot ---------------------------
def plot_cmax_curve(history: List[float]):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(history, linewidth=2)
    plt.xlabel("Evaluations")
    plt.ylabel("Best Cmax")
    plt.title("Best Cmax over evaluations (SA)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


# --------------------------- Gantt Helpers ---------------------------
def decode_records_for_gantt(machine_assignments: Dict[Tuple[int,int], Tuple[int,int]],
                             op_order: List[Tuple[int,int]],
                             jobs: List[List[List[Tuple[int,int]]]],
                             num_machines: int):
    """
    Earliest-start decoding that also returns per-operation records for plotting.
    Returns (cmax, records). Each record: {"job", "op", "machine", "start", "finish"} with 0-based indices.
    """
    machine_available = [0.0] * num_machines
    job_available = [0.0] * len(jobs)
    records = []

    for j, o in op_order:
        m, dur = machine_assignments[(j, o)]
        start = float(max(machine_available[m], job_available[j]))
        finish = float(start + dur)
        records.append({"job": j, "op": o, "machine": m, "start": start, "finish": finish})
        machine_available[m] = finish
        job_available[j] = finish

    cmax = max(machine_available) if num_machines > 0 else 0.0
    return cmax, records





def plot_gantt(records: List[Dict[str, float]], num_machines: int, cmax: float = None, title: str = None, show: bool = True):
    """
    Publication-style Gantt chart.
    - Machine rows labeled M1..M_k
    - Bars colored per Job with legend (unique colors for up to 20 jobs via 'tab20')
    - Operation labels like J{job}{op:02d} (op index starts at 1)
    Expected record keys: job (0-based), op (0-based), machine (0-based), start, finish.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib import colormaps as mcm
    except Exception:
        return  # Matplotlib not available

    if not records:
        return

    # Prepare figure
    fig_height = 1.0 + 0.6 * max(1, num_machines)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    # Unique color per job (no deprecated get_cmap)
    job_ids = sorted({int(r["job"]) for r in records})
    n_jobs = len(job_ids)
    if n_jobs <= 20:
        cmap = mcm.get_cmap("tab20")
        palette = [cmap(i / 20.0) for i in range(20)]
        colors = {j: palette[i % 20] for i, j in enumerate(job_ids)}
    else:
        # Fallback continuous map for many jobs
        cmap = mcm.get_cmap("hsv")
        colors = {j: cmap(i / max(1, n_jobs - 1)) for i, j in enumerate(job_ids)}

    # Draw bars
    for r in records:
        j = int(r["job"])
        o = int(r["op"])
        m = int(r["machine"])
        start = float(r["start"])
        finish = float(r["finish"])
        width = max(0.0, finish - start)
        y = m + 1  # machine rows start at 1
        color = colors.get(j, None)

        ax.barh(y, width, left=start, height=0.7, align="center",
                edgecolor="black", linewidth=0.7, color=color)

        # Centered label: J{job}{op:02d}
        j_disp = j + 1
        o_disp = o + 1
        ax.text(start + width / 2.0, y, f"J{j_disp}{o_disp:02d}",
                va="center", ha="center", fontsize=8)

    # Axes / ticks
    ax.set_yticks(list(range(1, num_machines + 1)))
    ax.set_yticklabels([f"M{i}" for i in range(1, num_machines + 1)])
    ax.set_ylabel("Machine")
    ax.set_xlabel("Time")

    # Vertical grid only
    ax.xaxis.grid(True, linestyle=":", alpha=0.5)
    ax.yaxis.grid(False)

    # Cmax line
    if cmax is not None:
        ax.axvline(float(cmax), linestyle="--", alpha=0.7)

    # Title
    default_title = f"Gantt (Cmax={int(cmax)})" if cmax is not None else "Gantt"
    ax.set_title(title if title else default_title, pad=8)

    # Legend by job
    handles = [Patch(facecolor=colors[j], edgecolor="black", label=f"Job {j+1}") for j in job_ids]
    if handles:
        ax.legend(handles=handles, loc="upper right", ncol=min(6, len(handles)), frameon=True)

    # Nice limits
    if cmax is not None:
        ax.set_xlim(left=0, right=max(float(cmax) * 1.02, ax.get_xlim()[1]))
    else:
        left, right = ax.get_xlim()
        ax.set_xlim(left=0, right=right)

    fig.tight_layout()
    if show:
        plt.show()
    return ax



# --------------------------- CLI / Main ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance", type=str, required=False, default="mfjs10.txt")
    p.add_argument("--runs", type=int, default=1, help="Number of independent runs")

    # Stopping controls for eval mode
    p.add_argument("--evals", type=int, default=None,
                   help="Maximum number of schedule evaluations before stopping")
    p.add_argument("--noimp", type=int, default=100,
                   help="Stop after this many evaluations without improvement")

    # Performance targets / reporting
    p.add_argument("--solution", type=float, default=None,
                   help="Target Cmax for success-rate and GAP reports")
    p.add_argument("--lb", type=float, default=None,
                   help="Lower bound for LB-GAP reporting")

    # SA knobs
    p.add_argument("--Tstart", type=float, default=300.0, help="Initial temperature")
    p.add_argument("--alpha", type=float, default=0.995, help="Cooling factor per evaluation (0<alpha<1)")
    p.add_argument("--swap_prob", type=float, default=0.5, help="Probability of swap vs assignment neighbor")

    # Time-box mode (seconds). When set, ignores --evals/--noimp.
    p.add_argument("--runtime", type=float, default=None,
                   help="Seconds per run; time-boxed SA (disables --evals and --noimp)")

    p.add_argument("--no_show", action="store_true",
                   help="Do not plot history for single run")
    return p.parse_args()

def main():
    args = parse_args()

    # Load instance
    if not os.path.exists(args.instance):
        print(f"Instance file '{args.instance}' not found.")
        return
    inst = load_instance(args.instance)
    num_jobs, num_machines, jobs = inst.num_jobs, inst.num_machines, inst.jobs

    # ---------- TIME-BOXED MODE ----------
    if args.runtime is not None:
        runs = args.runs
        bests: List[float] = []
        evals_list: List[int] = []
        timebest_list: List[float] = []
        hits = 0

        for _ in range(runs):
            out = run_sa_timeboxed(
                jobs, num_machines,
                seconds=float(args.runtime),
                T_start=args.Tstart,
                alpha=args.alpha,
                swap_prob=args.swap_prob
            )
            best = float(out["best_cmax"])
            bests.append(best)
            evals_list.append(int(out["evaluations"]))
            timebest_list.append(float(out.get("time_best_ms", 0.0)))

            if args.solution is not None and int(round(best)) == int(round(args.solution)):
                hits += 1

        avg_cmax = sum(bests) / len(bests) if bests else float('inf')
        avg_evals = sum(evals_list) / len(evals_list) if evals_list else 0.0
        avg_timebest_ms = sum(timebest_list) / len(timebest_list) if timebest_list else 0.0

        # GAP (only if solution provided and non-zero)
        gap_str = None
        if args.solution is not None and float(args.solution) != 0.0 and math.isfinite(avg_cmax):
            gap = 100.0 * (avg_cmax - float(args.solution)) / float(args.solution)
            gap_str = f"{gap:.2f}"

        # success rate percentage
        sr_pct = (hits / runs) * 100.0 if runs > 0 else 0.0
        sr_display = f"{int(round(sr_pct))}%"

        # Runtime-mode printing: LB-centric if --lb else solution-centric
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
            # LB-GAP median ratio
            lb_gap_median = None
            if args.lb != 0 and math.isfinite(median_cmax):
                lb_gap_median = (median_cmax - float(args.lb)) / float(args.lb)
            if lb_gap_median is not None:
                print(f"median LB-GAP: {lb_gap_median:.2f} ((median_cmax-lb)/lb)")
            else:
                print(f"median LB-GAP: n/a ((median_cmax-lb)/lb)")
            print("===================================")
        else:
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

    # ---------- EVAL/NO-IMPROVEMENT MODE ----------
    else:
        eval_cap = int(args.evals) if args.evals is not None else int(1e9)

        if args.runs <= 1:
            t0 = time.perf_counter()
            out = run_sa(
                jobs, num_machines,
                eval_limit=eval_cap,
                no_improve_limit=args.noimp,
                T_start=args.Tstart,
                alpha=args.alpha,
                swap_prob=args.swap_prob,
                history_out=(not args.no_show)
            )
            t1 = time.perf_counter()
            ms = (t1 - t0) * 1000.0
            print(f"Best Cmax: {out['best_cmax']:.1f}")
            print(f"Evaluations: {out['evaluations']}")
            print(f"Stop reason: {out['stop_reason']}")
            print(f"CPU time: {ms:.2f} ms")
            if (not args.no_show) and out["history"]:
                plot_cmax_curve(out["history"])
            if not args.no_show:
                cmax, recs = decode_records_for_gantt(out["best_assign"], out["best_order"], jobs, num_machines)
                plot_gantt(recs, num_machines, cmax)
        else:
            bests: List[float] = []
            times_ms: List[float] = []
            hits = 0
            noimp_stops = 0
            budget_stops = 0
            success_times_ms: List[float] = []  # time per successful run only

            for _ in range(args.runs):
                t0 = time.perf_counter()
                out = run_sa(
                    jobs, num_machines,
                    eval_limit=eval_cap,
                    no_improve_limit=args.noimp,
                    T_start=args.Tstart,
                    alpha=args.alpha,
                    swap_prob=args.swap_prob,
                    history_out=False
                )
                t1 = time.perf_counter()
                elapsed_ms = (t1 - t0) * 1000.0

                best_cmax = float(out["best_cmax"])
                bests.append(best_cmax)
                times_ms.append(elapsed_ms)

                # success detection
                if args.solution is not None and int(round(best_cmax)) == int(round(args.solution)):
                    hits += 1
                    success_times_ms.append(elapsed_ms)

                # Only count no-improvement stops if final best is WORSE than target solution
                if out.get("stop_reason") == "no_improvement_limit":
                    if args.solution is None:
                        noimp_stops += 1
                    else:
                        if best_cmax > float(args.solution):
                            noimp_stops += 1

                if out.get("stop_reason") == "budget_exhausted":
                    budget_stops += 1

            # Summaries
            avg_cmax = sum(bests) / len(bests) if bests else float('inf')
            sr_pct = (hits / args.runs) * 100.0 if args.runs > 0 else 0.0
            noimp_pct = (noimp_stops / args.runs) * 100.0 if args.runs > 0 else 0.0
            budget_pct = (budget_stops / args.runs) * 100.0 if args.runs > 0 else 0.0

            # Average CPU time for successful runs only
            if success_times_ms:
                avg_success_ms = sum(success_times_ms) / len(success_times_ms)
                avg_success_str = f"{avg_success_ms:.0f} ms"
            else:
                avg_success_str = "n/a"

            if args.lb is not None:
                best_cmax_all = min(bests) if bests else float('inf')
                median_cmax = statistics.median(bests) if bests else float('inf')
                # LB-GAP median ratio
                lb_gap_median = None
                if args.lb != 0 and math.isfinite(median_cmax):
                    lb_gap_median = (median_cmax - float(args.lb)) / float(args.lb)

                print("========== Batch Results ==========")
                print(f"Runs: {args.runs}")
                if math.isfinite(best_cmax_all):
                    print(f"best_cmax among all runs: {int(round(best_cmax_all))} (best cmax obtained in all {args.runs} runs)")
                else:
                    print(f"best_cmax among all runs: n/a (best cmax obtained in all {args.runs} runs)")
                if math.isfinite(median_cmax):
                    print(f"median best_cmax:{int(round(median_cmax))} (median cmax of all {args.runs} runs)")
                else:
                    print(f"median best_cmax:n/a (median cmax of all {args.runs} runs)")
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
                # GAP (solution-based) for non-LB summary
                gap_str = None
                if args.solution is not None and float(args.solution) != 0.0 and math.isfinite(avg_cmax):
                    gap = 100.0 * (avg_cmax - float(args.solution)) / float(args.solution)
                    gap_str = f"{gap:.2f}"

                print("========== Batch Results ==========")
                print(f"Runs: {args.runs}")
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

if __name__ == "__main__":
    main()
