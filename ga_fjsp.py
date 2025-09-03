"""
Genetic Algorithm for Flexible Job Shop Scheduling

Modes (using file "mfjs10.txt" as an example)

1) One run test: Test one run of algorithm and plot 2 charts
    python ga_fjsp.py --instance mfjs10.txt

2) Evaluation/No-Improvement mode with known solution
    python ga_fjsp.py --instance mfjs10.txt --runs 100 --evals 1000 --noimp 100 --solution 66

3) Evaluation/No-Improvement mode with unknown solution    
    python ga_fjsp.py --instance mfjs10.txt --runs 100 --evals 1000 --noimp 100 --lb 66

4) Time-boxed mode with known solution
     python ga_fjsp.py --instance mfjs10.txt --runs 100 --runtime 0.5 --solution 66
     
5) Time-boxed mode with unknown solution     
     python ga_fjsp.py --instance mfjs10.txt --runs 100 --runtime 0.5 --lb 66
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

# --------------------------- Decoder & evaluation ---------------------------
def decode_schedule(inst: Instance, seq: List[int], assign: List[int]) -> Tuple[float, List[Dict[str, float]]]:
    """
    Precedence-safe decoder driven by job-token sequence.
    Returns (Cmax, records) where records carry timing for each op (for potential plots).
    """
    op2pair, pair2op = build_op_index_maps(inst)
    job_next = [0] * inst.num_jobs
    job_ready = [0.0] * inst.num_jobs
    mach_ready = [0.0] * inst.num_machines
    records: List[Dict[str, float]] = []

    def ptime(j, h, m):
        for mm, p in inst.jobs[j][h]:
            if mm == m:
                return p
        # If invalid machine, default to first capable (repair)
        return inst.jobs[j][h][0][1]

    # Schedule next operation whenever a job token appears
    for job_id in seq:
        j = job_id
        if job_next[j] >= len(inst.jobs[j]):
            continue
        h = job_next[j]
        flat = pair2op[(j, h)]
        m = assign[flat]
        # repair if needed
        if not any(mm == m for (mm, _p) in inst.jobs[j][h]):
            m = inst.jobs[j][h][0][0]

        p = ptime(j, h, m)
        start = max(job_ready[j], mach_ready[m])
        finish = start + p

        job_ready[j] = float(finish)
        mach_ready[m] = float(finish)
        job_next[j] += 1

        records.append({"job": j, "op": h, "machine": m, "start": float(start), "finish": float(finish)})

    # In case any ops remain for a job (rare with well-formed seq), finish in precedence order
    for j in range(inst.num_jobs):
        while job_next[j] < len(inst.jobs[j]):
            h = job_next[j]
            flat = pair2op[(j, h)]
            m = assign[flat]
            if not any(mm == m for (mm, _p) in inst.jobs[j][h]):
                m = inst.jobs[j][h][0][0]
            p = ptime(j, h, m)
            start = max(job_ready[j], mach_ready[m])
            finish = start + p
            job_ready[j] = float(finish)
            mach_ready[m] = float(finish)
            job_next[j] += 1
            records.append({"job": j, "op": h, "machine": m, "start": float(start), "finish": float(finish)})

    cmax = max(mach_ready) if inst.num_machines > 0 else 0.0
    return cmax, records

# --------------------------- GA operators ---------------------------
def initial_population(inst: Instance, pop_size: int, greedy_ratio: float = 0.2) -> List[Tuple[List[int], List[int]]]:
    op2pair, pair2op = build_op_index_maps(inst)

    # Greedy assignment: min processing time per (j,h)
    greedy_assign = []
    for (j, h) in op2pair:
        m_best, p_best = None, math.inf
        for m, p in inst.jobs[j][h]:
            if p < p_best:
                m_best, p_best = m, p
        greedy_assign.append(m_best)

    # Sequence template: expand jobs by their op counts
    seq_template = []
    for j, job in enumerate(inst.jobs):
        seq_template.extend([j] * len(job))

    pop = []
    n_greedy = max(1, int(pop_size * greedy_ratio))
    for _ in range(n_greedy):
        seq = seq_template[:]
        random.shuffle(seq)
        pop.append((seq, greedy_assign[:]))

    while len(pop) < pop_size:
        assign = []
        for (j, h) in op2pair:
            m = random.choice(inst.jobs[j][h])[0]
            assign.append(m)
        seq = seq_template[:]
        random.shuffle(seq)
        pop.append((seq, assign))

    return pop

def tournament_select(pop, fitness, k: int = 3):
    idxs = random.sample(range(len(pop)), k)
    best = min(idxs, key=lambda i: fitness[i])  # minimize Cmax
    s, a = pop[best]
    return s[:], a[:]

def pox_crossover(seq1: List[int], seq2: List[int], num_jobs: int):
    jobs = list(range(num_jobs))
    random.shuffle(jobs)
    cut = len(jobs) // 2
    JA = set(jobs[:cut]); JB = set(jobs[cut:])

    def make_child(keep_parent, fill_parent, keep_set):
        child = [None] * len(keep_parent)
        for i, j in enumerate(keep_parent):
            if j in keep_set:
                child[i] = j
        fill_iter = (j for j in fill_parent if j not in keep_set)
        for i in range(len(child)):
            if child[i] is None:
                child[i] = next(fill_iter)
        return child

    c1 = make_child(seq1, seq2, JA)
    c2 = make_child(seq2, seq1, JB)
    return c1, c2

def uniform_assignment_crossover(assign1: List[int], assign2: List[int]):
    c1, c2 = assign1[:], assign2[:]
    for i in range(len(assign1)):
        if random.random() < 0.5:
            c1[i], c2[i] = c2[i], c1[i]
    return c1, c2

def mutate_sequence_swap(seq: List[int], pmut: float):
    if random.random() < pmut and len(seq) >= 2:
        i, j = random.sample(range(len(seq)), 2)
        seq[i], seq[j] = seq[j], seq[i]

def mutate_assignment(inst: Instance, assign: List[int], pmut: float):
    op2pair, _ = build_op_index_maps(inst)
    for idx in range(len(assign)):
        if random.random() < pmut:
            j, h = op2pair[idx]
            choices = [m for (m, _p) in inst.jobs[j][h]]
            if len(choices) > 1:
                current = assign[idx]
                alt = random.choice([m for m in choices if m != current])
                assign[idx] = alt

# --------------------------- Optional plot ---------------------------
def plot_cmax_curve(history: List[float], save_path=None, show=True):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return  # silently skip if matplotlib not available
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history, linewidth=2)
    ax.set_title("Best Cmax over evaluations")
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Best Cmax")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Cmax curve saved to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

# --------------------------- Gantt plot ---------------------------
def plot_gantt(records: List[Dict[str, float]], num_machines: int, cmax: float = None, title: str = None, show: bool = True):
    """
    Publication-style Gantt chart.
    - Machine rows labeled M1..M_k
    - Bars colored per Job with legend (unique colors for up to 20 jobs via 'tab20')
    - Operation labels like J{job}{op:02d} (job/op shown as 1-based)
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

    # Figure size scales with number of machines
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

        # Centered label: J{job}{op:02d} (1-based display)
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

    # Title (default includes Cmax if provided)
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



# --------------------------- GA (eval/no-imp) ---------------------------
def run_ga(inst: Instance,
           pop_size: int = 60,
           eval_limit: int = 5000,          # evaluation budget
           no_improve_limit: int = 100,     # consecutive non-improving evaluations
           tournament_k: int = 3,
           p_crossover: float = 0.9,
           p_mut_seq: float = 0.10,
           p_mut_assign: float = 0.05,
           elitism: int = 2,
           show_plots: bool = True,
           verbose: bool = True) -> Dict[str, Any]:
    """
    GA for FJSSP with evaluation-based early stop.
    Representation:
      - seq: job-token list (length = total_ops), precedence-feasible by construction.
      - assign: flat machine list indexed by (j,h) via pair2op (machine ids).
    Fitness = makespan from decode_schedule(inst, seq, assign).
    """
    # --- helpers ---
    history: List[float] = [] if show_plots else None
    evals = 0
    no_impr_evals = 0
    stop_reason = None

    def eval_individual(seq: List[int], assign: List[int]) -> float:
        nonlocal evals, best_so_far, best_seq, best_assign, no_impr_evals
        cmax, _records = decode_schedule(inst, seq, assign)
        evals += 1
        if cmax + 1e-12 < best_so_far:
            best_so_far = cmax
            best_seq = seq[:]
            best_assign = assign[:]
            no_impr_evals = 0
        else:
            no_impr_evals += 1
        if history is not None:
            history.append(best_so_far)
        return cmax

    # --- init ---
    population = initial_population(inst, pop_size, greedy_ratio=0.2)

    fitness: List[float] = []
    best_so_far = float("inf")
    best_seq: List[int] = []
    best_assign: List[int] = []

    # evaluate initial population
    for (seq, assign) in population:
        if evals >= eval_limit or no_impr_evals >= no_improve_limit:
            stop_reason = "budget_exhausted" if evals >= eval_limit else "no_improvement_limit"
            break
        fitness.append(eval_individual(seq, assign))

    if stop_reason is not None:
        if show_plots and history:
            plot_cmax_curve(history, show=True)
        return {
            "best_cmax": best_so_far,
            "best_seq": best_seq,
            "best_assign": best_assign,
            "evaluations": evals,
            "history": history if history is not None else [],
            "stop_reason": stop_reason,
        }

    # --- GA loop ---
    gen = 0
    while evals < eval_limit and no_impr_evals < no_improve_limit:
        gen += 1

        # elitism
        elite_idx = sorted(range(len(population)), key=lambda i: fitness[i])[:max(0, min(elitism, pop_size))]
        new_population: List[Tuple[List[int], List[int]]] = [
            (population[i][0][:], population[i][1][:]) for i in elite_idx
        ]

        # offspring
        while len(new_population) < pop_size:
            p1_seq, p1_assign = tournament_select(population, fitness, k=tournament_k)
            p2_seq, p2_assign = tournament_select(population, fitness, k=tournament_k)

            if random.random() < p_crossover:
                c1_seq, c2_seq = pox_crossover(p1_seq, p2_seq, inst.num_jobs)
                c1_assign, c2_assign = uniform_assignment_crossover(p1_assign, p2_assign)
            else:
                c1_seq, c1_assign = p1_seq[:], p1_assign[:]
                c2_seq, c2_assign = p2_seq[:], p2_assign[:]

            mutate_sequence_swap(c1_seq, p_mut_seq)
            mutate_sequence_swap(c2_seq, p_mut_seq)
            mutate_assignment(inst, c1_assign, p_mut_assign)
            mutate_assignment(inst, c2_assign, p_mut_assign)

            new_population.append((c1_seq, c1_assign))
            if len(new_population) < pop_size:
                new_population.append((c2_seq, c2_assign))

        # evaluate new population
        population = new_population
        fitness = []
        for (seq, assign) in population:
            if evals >= eval_limit or no_impr_evals >= no_improve_limit:
                stop_reason = "budget_exhausted" if evals >= eval_limit else "no_improvement_limit"
                break
            fitness.append(eval_individual(seq, assign))

        if stop_reason is not None:
            break

        if verbose and gen % 10 == 0:
            print(f"[gen {gen}] best={best_so_far:.2f} evals={evals} noimp={no_impr_evals}")

    if stop_reason is None:
        stop_reason = "budget_exhausted" if evals >= eval_limit else "no_improvement_limit"

    if show_plots and history:
        plot_cmax_curve(history, show=True)

    return {
        "best_cmax": best_so_far,
        "best_seq": best_seq,
        "best_assign": best_assign,
        "evaluations": evals,
        "history": history if history is not None else [],
        "stop_reason": stop_reason,
    }

# --------------------------- GA (time-boxed) ---------------------------
def run_ga_timeboxed(inst: Instance,
                     pop_size: int = 60,
                     seconds: float = 1.0,
                     tournament_k: int = 3,
                     p_crossover: float = 0.9,
                     p_mut_seq: float = 0.10,
                     p_mut_assign: float = 0.05,
                     elitism: int = 2) -> Dict[str, Any]:
    """
    GA for FJSSP with a hard wall-clock limit per run.
    - Ignores no-improvement and evaluation budgets.
    - Returns time (ms) when the FINAL best Cmax was first achieved.
    """
    t0 = time.perf_counter()

    # --- helpers ---
    evals = 0
    best_so_far = float("inf")
    best_seq: List[int] = []
    best_assign: List[int] = []
    time_best_ms = 0.0  # time at which current global best was first achieved (ms since t0)

    def time_left() -> bool:
        return (time.perf_counter() - t0) < seconds

    def eval_individual(seq: List[int], assign: List[int]) -> float:
        nonlocal evals, best_so_far, best_seq, best_assign, time_best_ms
        if not time_left():
            return float("inf")
        cmax, _ = decode_schedule(inst, seq, assign)
        evals += 1
        if cmax + 1e-12 < best_so_far:
            best_so_far = cmax
            best_seq = seq[:]
            best_assign = assign[:]
            time_best_ms = (time.perf_counter() - t0) * 1000.0
        return cmax

    # --- init population ---
    population = initial_population(inst, pop_size, greedy_ratio=0.2)

    # evaluate initial population (may partially evaluate due to time)
    fitness = []
    for (seq, assign) in population:
        if not time_left():
            break
        fitness.append(eval_individual(seq, assign))

    # main loop until time is up
    gen = 0
    while time_left():
        gen += 1

        # elitism
        if fitness:
            elite_idx = sorted(range(len(population)), key=lambda i: fitness[i])[:max(0, min(elitism, len(population)))]
            new_population: List[Tuple[List[int], List[int]]] = [(population[i][0][:], population[i][1][:]) for i in elite_idx]
        else:
            new_population = []

        # fill with children
        while len(new_population) < pop_size and time_left():
            base_pop = population if population else initial_population(inst, pop_size, greedy_ratio=0.2)
            base_fit = fitness if fitness else [eval_individual(s, a) for (s, a) in base_pop]

            p1_seq, p1_assign = tournament_select(base_pop, base_fit, k=tournament_k)
            p2_seq, p2_assign = tournament_select(base_pop, base_fit, k=tournament_k)

            if random.random() < p_crossover:
                c1_seq, c2_seq = pox_crossover(p1_seq, p2_seq, inst.num_jobs)
                c1_assign, c2_assign = uniform_assignment_crossover(p1_assign, p2_assign)
            else:
                c1_seq, c1_assign = p1_seq[:], p1_assign[:]
                c2_seq, c2_assign = p2_seq[:], p2_assign[:]

            mutate_sequence_swap(c1_seq, p_mut_seq)
            mutate_sequence_swap(c2_seq, p_mut_seq)
            mutate_assignment(inst, c1_assign, p_mut_assign)
            mutate_assignment(inst, c2_assign, p_mut_assign)

            new_population.append((c1_seq, c1_assign))
            if len(new_population) < pop_size:
                new_population.append((c2_seq, c2_assign))

        # evaluate new population
        population = new_population
        fitness = []
        for (seq, assign) in population:
            if not time_left():
                break
            fitness.append(eval_individual(seq, assign))

    return {
        "best_cmax": best_so_far,
        "best_seq": best_seq,
        "best_assign": best_assign,
        "evaluations": evals,
        "time_best_ms": time_best_ms,
        "stop_reason": "time_limit",
    }

# --------------------------- CLI / Main ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance", type=str, required=False, default="mfjs10.txt")
    p.add_argument("--pop", type=int, default=60)
    p.add_argument("--evals", type=int, default=None,
                   help="Maximum number of schedule evaluations before stopping")
    p.add_argument("--noimp", type=int, default=100,
                   help="Stop after this many evaluations without improvement")
    p.add_argument("--runs", type=int, default=1, help="Number of independent runs")
    p.add_argument("--solution", type=float, default=None,
                   help="Target Cmax to count exact hits / compute GAP")
    p.add_argument("--lb", type=float, default=None,
                   help="Lower bound for LB-GAP reporting in eval/no-imp mode")
    p.add_argument("--no_show", action="store_true",
                   help="Do not show plots (still saved as PNG for single run)")
    # Time-box mode (seconds). When set, ignores --evals/--noimp.
    p.add_argument("--runtime", type=float, default=None,
                   help="Seconds per run; time-boxed GA (disables --evals and --noimp)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.instance):
        print(f"Instance file '{args.instance}' not found. Put it in the same folder or pass --instance path.")
    else:
        inst = load_instance(args.instance)

        # ---------- TIME-BOXED MODE ----------
        if args.runtime is not None:
            runs = args.runs
            bests: List[float] = []
            evals_list: List[int] = []
            timebest_list: List[float] = []
            hits = 0

            for _ in range(runs):
                out = run_ga_timeboxed(
                    inst,
                    pop_size=args.pop,
                    seconds=float(args.runtime),
                    tournament_k=3,
                    p_crossover=0.9,
                    p_mut_seq=0.10,
                    p_mut_assign=0.05,
                    elitism=2
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

            print("========== Batch Results ==========")
            print(f"Runs: {runs}")
            if args.lb is not None:
                # LB-centric summary for runtime mode
                best_cmax_all = min(bests) if bests else float('inf')
                median_cmax = statistics.median(bests) if bests else float('inf')
                if evals_list:
                    print(f"average of evaluations per run: { (sum(evals_list)/len(evals_list)):.1f}")
                else:
                    print(f"average of evaluations per run: n/a")
                if best_cmax_all != float('inf'):
                    print(f"best_cmax among all runs: {int(round(best_cmax_all))} (best cmax obtained in all {runs} runs)")
                else:
                    print(f"best_cmax among all runs: n/a (best cmax obtained in all {runs} runs)")
                if median_cmax != float('inf'):
                    print(f"median best_cmax:{int(round(median_cmax))} (median cmax of all {runs} runs)")
                else:
                    print(f"median best_cmax:n/a (median cmax of all {runs} runs)")
                # LB-GAP median ratio
                lb_gap_median = None
                if args.lb is not None and args.lb != 0 and median_cmax != float('inf'):
                    lb_gap_median = (median_cmax - float(args.lb)) / float(args.lb)
                if lb_gap_median is not None:
                    print(f"median LB-GAP: {lb_gap_median:.2f} ((median_cmax-lb)/lb)")
                else:
                    print(f"median LB-GAP: n/a ((median_cmax-lb)/lb)")
            else:
                # Original runtime summary
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
                out = run_ga(inst,
                             pop_size=args.pop,
                             eval_limit=eval_cap,
                             no_improve_limit=args.noimp,
                             tournament_k=3,
                             p_crossover=0.9,
                             p_mut_seq=0.10,
                             p_mut_assign=0.05,
                             elitism=2,
                             show_plots=(not args.no_show),
                             verbose=True)
                print(f"Best Cmax: {out['best_cmax']:.1f}")

                if not args.no_show:
                    cmax_best, recs = decode_schedule(inst, out["best_seq"], out["best_assign"])
                    plot_gantt(recs, inst.num_machines, cmax_best)
            else:
                bests: List[float] = []
                times_ms: List[float] = []
                hits = 0
                noimp_stops = 0
                budget_stops = 0
                success_times_ms: List[float] = []  # time per successful run only

                for _ in range(args.runs):
                    t0 = time.perf_counter()
                    out = run_ga(
                        inst,
                        pop_size=args.pop,
                        eval_limit=eval_cap,
                        no_improve_limit=args.noimp,
                        tournament_k=3,
                        p_crossover=0.9,
                        p_mut_seq=0.10,
                        p_mut_assign=0.05,
                        elitism=2,
                        show_plots=False,
                        verbose=False
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
                            if best_cmax > float(args.solution):  # strictly greater
                                noimp_stops += 1

                    if out.get("stop_reason") == "budget_exhausted":
                        budget_stops += 1

                # Summaries common pieces
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

                # If LB mode, print LB-centric summary
                if args.lb is not None:
                    best_cmax_all = min(bests) if bests else float('inf')
                    median_cmax = statistics.median(bests) if bests else float('inf')

                    # LB-GAP median ratio (not percent): (median_cmax - lb)/lb
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
