"""
REINFORCEMENT LEARNING GENETIC ALGORITHM (RLGA) for Flexible Job Shop Scheduling (FJSSP)

Modes (using file "mfjs10.txt" as an example)

1) One run test: Test one run of algorithm and plot 2 charts
    python rlga_fjsp.py --instance mfjs10.txt

2) Evaluation/No-Improvement mode with known solution
    python rlga_fjsp.py --instance mfjs10.txt --runs 100 --evals 1000 --noimp 100 --solution 66

3) Evaluation/No-Improvement mode with unknown solution    
    python rlga_fjsp.py --instance mfjs10.txt --runs 100 --evals 1000 --noimp 100 --lb 66

4) Time-boxed mode with known solution
     python rlga_fjsp.py --instance mfjs10.txt --runs 100 --runtime 0.5 --solution 66
     
5) Time-boxed mode with unknown solution     
     python rlga_fjsp.py --instance mfjs10.txt --runs 100 --runtime 0.5 --lb 66
"""
import random
import math
import matplotlib.pyplot as plt
from collections import defaultdict, namedtuple
import numpy as np
import argparse
import os
import time
import statistics
from matplotlib.patches import Patch

# -----------------------------
# FJSP instance parsing
# -----------------------------

def parse_fjsp_instance(path):
    with open(path, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    header = lines[0].split()
    n_jobs, n_machines = int(header[0]), int(header[1])
    jobs_ops = []
    total_ops = 0
    for j in range(n_jobs):
        tokens = list(map(int, lines[1 + j].split()))
        tptr = 0
        num_ops = tokens[tptr]; tptr += 1
        job = []
        for _ in range(num_ops):
            k = tokens[tptr]; tptr += 1
            alt = {}
            for _ in range(k):
                m = tokens[tptr]; p = tokens[tptr+1]
                tptr += 2
                alt[m] = p  # machines in file are 0-based
            job.append(alt)
        jobs_ops.append(job)
        total_ops += num_ops
    op_index_map = {}
    idx = 0
    for j, job in enumerate(jobs_ops):
        for o, _ in enumerate(job):
            op_index_map[(j, o)] = idx
            idx += 1
    return jobs_ops, n_jobs, n_machines, total_ops, op_index_map

# -----------------------------
# Chromosome representation helpers
# -----------------------------

def build_helpers(jobs_ops, op_index_map):
    n_jobs = len(jobs_ops)
    job_num_ops = [len(job) for job in jobs_ops]
    job_occurrences_template = []
    for j, k in enumerate(job_num_ops):
        job_occurrences_template.extend([j]*k)
    allowed_machines = {op_index_map[(j,o)]: sorted(list(alt.keys()))
                        for j, job in enumerate(jobs_ops) for o, alt in enumerate(job)}
    proc_time = {(op_index_map[(j,o)], m): jobs_ops[j][o][m]
                 for j, job in enumerate(jobs_ops) for o in range(len(job)) for m in jobs_ops[j][o]}
    return job_num_ops, job_occurrences_template, allowed_machines, proc_time

# -----------------------------
# Decoding / Scheduling
# -----------------------------

OpInfo = namedtuple("OpInfo", ["job","op","machine","start","end"])

def decode_schedule(OS, MA, n_jobs, n_mach, op_index_map, proc_time):
    job_next_op = [0]*n_jobs
    job_ready = [0]*n_jobs
    mach_ready = [0]*n_mach
    schedule_by_machine = defaultdict(list)
    
    for job in OS:
        o = job_next_op[job]
        flat = op_index_map[(job, o)]
        m = MA[flat]
        p = proc_time.get((flat, m), None)
        if p is None:
            raise RuntimeError("Invalid machine assignment encountered.")
        start = max(job_ready[job], mach_ready[m])
        end = start + p
        schedule_by_machine[m].append(OpInfo(job, o, m, start, end))
        job_ready[job] = end
        mach_ready[m] = end
        job_next_op[job] += 1
    cmax = 0
    for m in schedule_by_machine:
        for op in schedule_by_machine[m]:
            if op.end > cmax:
                cmax = op.end
    return cmax, schedule_by_machine

# -----------------------------
# Fitness tracking & evaluation counter
# -----------------------------

class EvalCounter:
    def __init__(self):
        self.count = 0

EVAL_COUNTER = EvalCounter()

def evaluate(OS, MA, n_jobs, n_mach, op_index_map, proc_time):
    EVAL_COUNTER.count += 1
    cmax, sched = decode_schedule(OS, MA, n_jobs, n_mach, op_index_map, proc_time)
    return -cmax, cmax, sched

# -----------------------------
# Genetic operators
# -----------------------------

def init_individual(job_occurrences_template, n_ops, allowed_machines):
    OS = job_occurrences_template.copy()
    random.shuffle(OS)
    MA = [None]*n_ops
    for flat in range(n_ops):
        MA[flat] = random.choice(allowed_machines[flat])
    return OS, MA

def pox_crossover(os1, os2, n_jobs):
    jobs = list(range(n_jobs))
    subset = set(random.sample(jobs, k=max(1, len(jobs)//2)))
    child = [None]*len(os1)
    for i, j in enumerate(os1):
        if j in subset:
            child[i] = j
    fill_values = [j for j in os2 if j not in subset]
    it = iter(fill_values)
    for i in range(len(os1)):
        if child[i] is None:
            child[i] = next(it)
    return child

def uniform_crossover_ma(ma1, ma2, n_ops, allowed_machines):
    child = []
    for flat in range(n_ops):
        choice = ma1[flat] if random.random() < 0.5 else ma2[flat]
        if choice not in allowed_machines[flat]:
            choice = random.choice(allowed_machines[flat])
        child.append(choice)
    return child

def mutate_os_swap(os, pm):
    child = os.copy()
    if random.random() < pm:
        i, j = random.sample(range(len(os)), 2)
        child[i], child[j] = child[j], child[i]
    return child

def mutate_ma_random(ma, pm, allowed_machines):
    child = ma.copy()
    per_gene = min(1.0, pm)
    for flat in range(len(ma)):
        if random.random() < per_gene:
            child[flat] = random.choice(allowed_machines[flat])
    return child

def tournament_select(pop, fit, k=2):
    best = None
    for _ in range(k):
        idx = random.randrange(len(pop))
        if best is None or fit[idx] > fit[best]:
            best = idx
    return best

# -----------------------------
# RL: Self-learning controller for Pc and Pm
# -----------------------------

class RLController:
    def __init__(self, n_states=20, n_actions=10, alpha=0.75, gamma=0.2, epsilon=0.85):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions), dtype=float)
        self.iteration = 0  # Nti
    
    def select_action(self, state):
        self.iteration += 1
        if random.random() <= self.epsilon:
            return int(np.argmax(self.Q[state, :]))
        return random.randrange(self.n_actions)
    
    def update(self, s, a, r, s_next, a_next=None):
        nts = self.n_states
        nta = self.n_actions
        use_sarsa = (self.iteration < (nts * nta) // 2)
        q_sa = self.Q[s, a]
        if use_sarsa and a_next is not None:
            target = r + self.gamma * self.Q[s_next, a_next]
        else:
            target = r + self.gamma * np.max(self.Q[s_next, :])
        self.Q[s, a] = (1 - self.alpha) * q_sa + self.alpha * target

def compute_state(avg_fit_hist, div_hist, best_fit_hist, w1=0.35, w2=0.35, w3=0.30):
    f_first = avg_fit_hist[0]
    d_first = div_hist[0] if div_hist[0] != 0 else 1.0
    m_first = best_fit_hist[0]
    f_star = avg_fit_hist[-1] / (f_first if f_first != 0 else 1.0)
    d_star = (div_hist[-1] / (d_first if d_first != 0 else 1.0)) if d_first != 0 else 0.0
    m_star = best_fit_hist[-1] / (m_first if m_first != 0 else 1.0)
    S = w1*f_star + w2*d_star + w3*m_star
    state = int(min(19, max(0, math.floor(S / 0.05))))
    return state

def sample_pc_from_action(a):
    low = 0.40 + 0.05*a
    high = min(0.90, low + 0.05)
    return random.uniform(low, high)

def sample_pm_from_action(a):
    low = 0.01 + 0.02*a
    high = min(0.21, low + 0.02)
    return random.uniform(low, high)

# -----------------------------
# SLGA main loop (single run)
# -----------------------------

def run_SLGA(jobs_ops, n_jobs, n_mach, n_ops, op_index_map, pop_size=None, generations=None,
             elite_rate=0.02, tournament_k=2, seed=None):
    # Fast defaults
    if pop_size is None:
        pop_size = max(30, 3 * n_jobs * n_mach)
    if generations is None:
        generations = max(60, 3 * n_jobs * n_mach)
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    
    job_num_ops, job_occurrences_template, allowed_machines, proc_time = build_helpers(jobs_ops, op_index_map)
    
    population = [init_individual(job_occurrences_template, n_ops, allowed_machines) for _ in range(pop_size)]
    fitness, cmax_vals, schedules = [], [], []
    for ind in population:
        fit, cmax, sched = evaluate(ind[0], ind[1], n_jobs, n_mach, op_index_map, proc_time)
        fitness.append(fit); cmax_vals.append(cmax); schedules.append(sched)
    
    avg_fit_hist = [float(np.mean(fitness))]
    div_hist = [float(np.std(fitness))]
    best_fit_hist = [float(np.max(fitness))]
    
    rl_pc = RLController()
    rl_pm = RLController()
    
    eval_trace = [(EVAL_COUNTER.count, min(cmax_vals))]
    elite_count = max(1, int(elite_rate * pop_size))
    
    for _ in range(generations):
        s = compute_state(avg_fit_hist, div_hist, best_fit_hist)
        a_pc = rl_pc.select_action(s)
        a_pm = rl_pm.select_action(s)
        Pc = sample_pc_from_action(a_pc)
        Pm = sample_pm_from_action(a_pm)
        
        ranked = sorted(range(population.__len__()), key=lambda i: fitness[i], reverse=True)
        new_population = [population[i] for i in ranked[:elite_count]]
        new_fitness = [fitness[i] for i in ranked[:elite_count]]
        new_cmax = [cmax_vals[i] for i in ranked[:elite_count]]
        new_schedules = [schedules[i] for i in ranked[:elite_count]]
        
        while len(new_population) < pop_size:
            i1 = tournament_select(population, fitness, k=tournament_k)
            i2 = tournament_select(population, fitness, k=tournament_k)
            p1_OS, p1_MA = population[i1]
            p2_OS, p2_MA = population[i2]
            
            if random.random() < Pc:
                c_OS = pox_crossover(p1_OS, p2_OS, n_jobs)
                c_MA = uniform_crossover_ma(p1_MA, p2_MA, n_ops, allowed_machines)
            else:
                c_OS = p1_OS.copy()
                c_MA = p1_MA.copy()
            
            c_OS = mutate_os_swap(c_OS, Pm)
            c_MA = mutate_ma_random(c_MA, Pm/2.0, allowed_machines)
            
            fit, cmax, sched = evaluate(c_OS, c_MA, n_jobs, n_mach, op_index_map, proc_time)
            new_population.append((c_OS, c_MA))
            new_fitness.append(fit)
            new_cmax.append(cmax)
            new_schedules.append(sched)
        
        population, fitness, cmax_vals, schedules = new_population, new_fitness, new_cmax, new_schedules
        
        avg_fit_hist.append(float(np.mean(fitness)))
        div_hist.append(float(np.std(fitness)))
        best_fit_hist.append(float(np.max(fitness)))
        
        rc = 0.0
        rm = 0.0
        if len(best_fit_hist) >= 2:
            prev_best_fit = best_fit_hist[-2]
            curr_best_fit = best_fit_hist[-1]
            rc = (curr_best_fit - prev_best_fit) / (abs(prev_best_fit) if prev_best_fit != 0 else 1.0)
        if len(avg_fit_hist) >= 2:
            prev_avg = avg_fit_hist[-2]
            curr_avg = avg_fit_hist[-1]
            rm = (curr_avg - prev_avg) / (abs(prev_avg) if prev_avg != 0 else 1.0)
        
        s_next = compute_state(avg_fit_hist, div_hist, best_fit_hist)
        a_pc_next = rl_pc.select_action(s_next)
        a_pm_next = rl_pm.select_action(s_next)
        rl_pc.update(s, a_pc, rc, s_next, a_pc_next)
        rl_pm.update(s, a_pm, rm, s_next, a_pm_next)
        
        eval_trace.append((EVAL_COUNTER.count, min(cmax_vals)))
    
    best_idx = int(np.argmin(cmax_vals))
    best_ind = population[best_idx]
    best_sched = schedules[best_idx]
    best_cmax = cmax_vals[best_idx]
    best_OS, best_MA = best_ind
    
    results = {
        "best_cmax": best_cmax,
        "best_OS": best_OS,
        "best_MA": best_MA,
        "best_schedule": best_sched,
        "eval_trace": eval_trace,
    }
    return results

# -----------------------------
# Plotting utilities (single-run, shows plots)
# -----------------------------

def make_job_colors(n_jobs):
    # Use modern colormaps API for stable, distinct colors
    from matplotlib import colormaps as mcm
    if n_jobs <= 20:
        cmap = mcm.get_cmap("tab20")
        palette = [cmap(i / 20.0) for i in range(20)]
        return [palette[i % 20] for i in range(n_jobs)]
    else:
        cmap = mcm.get_cmap("hsv")
        return [cmap(i / max(1, n_jobs - 1)) for i in range(n_jobs)]

def plot_gantt(schedule_by_machine, n_jobs, title="Gantt"):
    colors = make_job_colors(n_jobs)
    machines = sorted(schedule_by_machine.keys())
    # Compute figure height based on number of machine rows
    fig_height = 1.0 + 0.6 * max(1, len(machines))
    fig, ax = plt.subplots(figsize=(12, fig_height))

    # Compute Cmax from schedule
    cmax = 0.0
    for m in machines:
        for op in schedule_by_machine[m]:
            if op.end > cmax:
                cmax = op.end

    # Draw bars machine by machine
    yticks = []
    yticklabels = []
    legend_jobs = set()

    for row_idx, m in enumerate(machines, start=1):
        ops = sorted(schedule_by_machine[m], key=lambda x: x.start)
        for op in ops:
            color = colors[op.job % n_jobs]
            left = float(op.start)
            width = float(op.end - op.start)
            ax.barh(row_idx, width, left=left, height=0.7, align="center",
                    edgecolor="black", linewidth=0.7, color=color)
            # Label like J{job}{op:02d} with 1-based job/op
            j_disp = int(op.job) + 1
            o_disp = int(op.op) + 1
            ax.text(left + width/2.0, row_idx, f"J{j_disp}{o_disp:02d}", va="center", ha="center", fontsize=8)
            legend_jobs.add(op.job)
        yticks.append(row_idx)
        yticklabels.append(f"M{m+1}")

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylabel("Machine")
    ax.set_xlabel("Time")

    # Vertical grid and Cmax line
    ax.xaxis.grid(True, linestyle=":", alpha=0.5)
    ax.yaxis.grid(False)
    if cmax > 0:
        ax.axvline(cmax, linestyle="--", alpha=0.7)

    # Title
    ax.set_title(title if title else "Gantt", pad=8)

    # Legend by job (sorted)
    handles = [Patch(facecolor=colors[j], edgecolor="black", label=f"Job {j+1}") for j in sorted(legend_jobs)]
    if handles:
        ax.legend(handles=handles, loc="upper right", ncol=min(6, len(handles)), frameon=True)

    fig.tight_layout()
    return fig, ax

def plot_cmax_over_evals(eval_trace, title="Best Cmax over evaluations"):
    xs = [x for x, _ in eval_trace]
    ys = [y for _, y in eval_trace]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xs, ys)
    ax.set_xlabel("Fitness evaluations")
    ax.set_ylabel("Best Cmax so far")
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    return fig, ax

# -----------------------------
# Batch mode (time-budgeted)
# -----------------------------

def run_batch_time(inst_path, args):
    # Time-batch mode: require runs and runtime; solution is optional
    if args.runs is None or args.runtime is None:
        raise SystemExit("Time-batch mode requires --runs and --runtime.")
    runs = int(args.runs)
    runtime_s = float(args.runtime)
    target = float(args.solution) if args.solution is not None else None

    success_count = 0
    bests = []
    evals_per_run = []
    times_best_ms = []
    base_seed = args.seed

    for r in range(runs):
        seed_r = (base_seed + r) if base_seed is not None else None
        if seed_r is not None:
            random.seed(seed_r); np.random.seed(seed_r)

        jobs_ops, N_JOBS, N_MACH, N_OPS, OP_INDEX = parse_fjsp_instance(inst_path)
        job_num_ops, job_occurrences_template, allowed_machines, proc_time = build_helpers(jobs_ops, OP_INDEX)
        pop = args.pop or max(30, 3 * N_JOBS * N_MACH)

        start_eval = EVAL_COUNTER.count
        population = [init_individual(job_occurrences_template, N_OPS, allowed_machines) for _ in range(pop)]
        fitness, cmax_vals, schedules = [], [], []

        best_c = float("inf")
        best_time_ms = 0.0
        t0 = time.perf_counter()

        # Evaluate initial population
        for ind in population:
            fit, cmax, sched = evaluate(ind[0], ind[1], N_JOBS, N_MACH, OP_INDEX, proc_time)
            fitness.append(fit); cmax_vals.append(cmax); schedules.append(sched)
            now_ms = (time.perf_counter() - t0) * 1000.0
            if cmax < best_c:
                best_c = cmax
                best_time_ms = now_ms
            if (time.perf_counter() - t0) >= runtime_s:
                break

        # Evolve while time remains
        if (time.perf_counter() - t0) < runtime_s:
            avg_fit_hist = [float(np.mean(fitness))]
            div_hist = [float(np.std(fitness))]
            best_fit_hist = [float(np.max(fitness))]
            rl_pc = RLController()
            rl_pm = RLController()
            elite_count = max(1, int(args.elite * pop))

            while (time.perf_counter() - t0) < runtime_s:
                s = compute_state(avg_fit_hist, div_hist, best_fit_hist)
                a_pc = rl_pc.select_action(s)
                a_pm = rl_pm.select_action(s)
                Pc = sample_pc_from_action(a_pc)
                Pm = sample_pm_from_action(a_pm)

                ranked = sorted(range(len(population)), key=lambda i: fitness[i], reverse=True)
                new_population = [population[i] for i in ranked[:elite_count]]
                new_fitness = [fitness[i] for i in ranked[:elite_count]]
                new_cmax = [cmax_vals[i] for i in ranked[:elite_count]]
                new_schedules = [schedules[i] for i in ranked[:elite_count]]

                while len(new_population) < pop and (time.perf_counter() - t0) < runtime_s:
                    i1 = tournament_select(population, fitness, k=args.tk)
                    i2 = tournament_select(population, fitness, k=args.tk)
                    p1_OS, p1_MA = population[i1]
                    p2_OS, p2_MA = population[i2]

                    if random.random() < Pc:
                        c_OS = pox_crossover(p1_OS, p2_OS, N_JOBS)
                        c_MA = uniform_crossover_ma(p1_MA, p2_MA, N_OPS, allowed_machines)
                    else:
                        c_OS = p1_OS.copy()
                        c_MA = p1_MA.copy()

                    c_OS = mutate_os_swap(c_OS, Pm)
                    c_MA = mutate_ma_random(c_MA, Pm/2.0, allowed_machines)

                    fit, cmax, sched = evaluate(c_OS, c_MA, N_JOBS, N_MACH, OP_INDEX, proc_time)
                    new_population.append((c_OS, c_MA))
                    new_fitness.append(fit)
                    new_cmax.append(cmax)
                    new_schedules.append(sched)

                    now_ms = (time.perf_counter() - t0) * 1000.0
                    if cmax < best_c:
                        best_c = cmax
                        best_time_ms = now_ms

                population, fitness, cmax_vals, schedules = new_population, new_fitness, new_cmax, new_schedules
                if (time.perf_counter() - t0) >= runtime_s:
                    break

                avg_fit_hist.append(float(np.mean(fitness)))
                div_hist.append(float(np.std(fitness)))
                best_fit_hist.append(float(np.max(fitness)))
                rc = 0.0
                rm = 0.0
                if len(best_fit_hist) >= 2:
                    prev_best_fit = best_fit_hist[-2]
                    curr_best_fit = best_fit_hist[-1]
                    rc = (curr_best_fit - prev_best_fit) / (abs(prev_best_fit) if prev_best_fit != 0 else 1.0)
                if len(avg_fit_hist) >= 2:
                    prev_avg = avg_fit_hist[-2]
                    curr_avg = avg_fit_hist[-1]
                    rm = (curr_avg - prev_avg) / (abs(prev_avg) if prev_avg != 0 else 1.0)
                s_next = compute_state(avg_fit_hist, div_hist, best_fit_hist)
                a_pc_next = rl_pc.select_action(s_next)
                a_pm_next = rl_pm.select_action(s_next)
                rl_pc.update(s, a_pc, rc, s_next, a_pc_next)
                rl_pm.update(s, a_pm, rm, s_next, a_pm_next)

        # End of a single run
        evals = EVAL_COUNTER.count - start_eval
        evals_per_run.append(evals)
        bests.append(best_c)
        times_best_ms.append(best_time_ms)
        if (target is not None) and (best_c <= target):
            success_count += 1

    # Summary
    avg_best = float(np.mean(bests)) if bests else float('nan')
    gap = ((avg_best - target) / target) if (bests and (target is not None)) else float('nan')
    avg_evals = float(np.mean(evals_per_run)) if evals_per_run else 0.0
    success_rate = 100.0 * success_count / runs if runs > 0 else 0.0
    avg_time_best = float(np.mean(times_best_ms)) if times_best_ms else 0.0

    print("========== Batch Results ==========")
    if args.lb is not None:
        lb = float(args.lb)
        best_overall = int(min(bests)) if bests else None
        median_best = statistics.median(bests) if bests else float("nan")
        median_gap = (median_best - lb) / lb if bests else float("nan")
        print(f"Runs: {runs}")
        print(f"average of evaluations per run: {avg_evals:.1f}")
        print(f"best_cmax among all runs: {best_overall} (best cmax obtained in all {runs} runs)")
        print(f"median best_cmax:{int(round(median_best))} (median cmax of all {runs} runs)")
        print(f"median LB-GAP: {median_gap:.2f} ((median_cmax-lb)/lb)")
    else:
        print(f"Runs: {runs} ")
        print(f"runtime: {runtime_s}")
        print(f"Average Evaluations per run: {avg_evals:.1f}")
        if target is not None:
            success_rate = 100.0 * success_count / runs if runs > 0 else 0.0
            gap = 100*(avg_best - target) / target if bests else float("nan")
            print(f"Success solution rate: {success_rate:.0f}%")
            print(f"GAP: {gap:.2f}")
        print(f"Average time best Cmax obtained: {int(round(avg_time_best))} ms")
print("==============================")

# -----------------------------
# Main
# -----------------------------


# -----------------------------
# Batch mode (eval-budgeted, LB-based summary)
# -----------------------------

def run_batch_evals_lb(inst_path, args):
    runs = int(args.runs)
    max_evals = int(args.evals)
    noimp_limit = int(args.noimp)
    lb = float(args.lb)

    # counters and collectors
    stop_noimp = 0
    stop_maxevals = 0
    bests = []
    cpu_ms = []

    base_seed = args.seed

    for r in range(runs):
        seed_r = (base_seed + r) if base_seed is not None else None
        if seed_r is not None:
            random.seed(seed_r); np.random.seed(seed_r)

        jobs_ops, N_JOBS, N_MACH, N_OPS, OP_INDEX = parse_fjsp_instance(inst_path)
        job_num_ops, job_occurrences_template, allowed_machines, proc_time = build_helpers(jobs_ops, OP_INDEX)

        # population constrained by eval budget
        pop = args.pop or max(30, 3 * N_JOBS * N_MACH)
        pop = min(pop, max_evals)

        start_eval = EVAL_COUNTER.count
        population = [init_individual(job_occurrences_template, N_OPS, allowed_machines) for _ in range(pop)]
        fitness, cmax_vals, schedules = [], [], []

        best_c = float('inf')
        last_improve_eval = 0
        stop_reason = None
        t0 = time.perf_counter()

        # Evaluate initial population
        for ind in population:
            fit, cmax, sched = evaluate(ind[0], ind[1], N_JOBS, N_MACH, OP_INDEX, proc_time)
            fitness.append(fit); cmax_vals.append(cmax); schedules.append(sched)
            if cmax < best_c:
                best_c = cmax
                last_improve_eval = 0
            else:
                last_improve_eval += 1
            if last_improve_eval >= noimp_limit:
                stop_reason = "noimp"; break
            if (EVAL_COUNTER.count - start_eval) >= max_evals:
                stop_reason = "maxevals"; break

        # If not stopped, evolve
        if stop_reason is None:
            avg_fit_hist = [float(np.mean(fitness))]
            div_hist = [float(np.std(fitness))]
            best_fit_hist = [float(np.max(fitness))]
            rl_pc = RLController(); rl_pm = RLController()
            elite_count = max(1, int(args.elite * pop))

            while True:
                s = compute_state(avg_fit_hist, div_hist, best_fit_hist)
                a_pc = rl_pc.select_action(s)
                a_pm = rl_pm.select_action(s)
                Pc = sample_pc_from_action(a_pc)
                Pm = sample_pm_from_action(a_pm)

                ranked = sorted(range(len(population)), key=lambda i: fitness[i], reverse=True)
                new_population = [population[i] for i in ranked[:elite_count]]
                new_fitness = [fitness[i] for i in ranked[:elite_count]]
                new_cmax = [cmax_vals[i] for i in ranked[:elite_count]]
                new_schedules = [schedules[i] for i in ranked[:elite_count]]

                while len(new_population) < pop:
                    i1 = tournament_select(population, fitness, k=args.tk)
                    i2 = tournament_select(population, fitness, k=args.tk)
                    p1_OS, p1_MA = population[i1]
                    p2_OS, p2_MA = population[i2]

                    if random.random() < Pc:
                        c_OS = pox_crossover(p1_OS, p2_OS, N_JOBS)
                        c_MA = uniform_crossover_ma(p1_MA, p2_MA, N_OPS, allowed_machines)
                    else:
                        c_OS = p1_OS.copy()
                        c_MA = p1_MA.copy()

                    c_OS = mutate_os_swap(c_OS, Pm)
                    c_MA = mutate_ma_random(c_MA, Pm/2.0, allowed_machines)

                    fit, cmax, sched = evaluate(c_OS, c_MA, N_JOBS, N_MACH, OP_INDEX, proc_time)
                    new_population.append((c_OS, c_MA))
                    new_fitness.append(fit)
                    new_cmax.append(cmax)
                    new_schedules.append(sched)

                    if cmax < best_c:
                        best_c = cmax
                        last_improve_eval = 0
                    else:
                        last_improve_eval += 1

                    if last_improve_eval >= noimp_limit:
                        stop_reason = "noimp"; break
                    if (EVAL_COUNTER.count - start_eval) >= max_evals:
                        stop_reason = "maxevals"; break

                population, fitness, cmax_vals, schedules = new_population, new_fitness, new_cmax, new_schedules

                if stop_reason is None:
                    avg_fit_hist.append(float(np.mean(fitness)))
                    div_hist.append(float(np.std(fitness)))
                    best_fit_hist.append(float(np.max(fitness)))
                    rc = 0.0; rm = 0.0
                    if len(best_fit_hist) >= 2:
                        prev_best_fit = best_fit_hist[-2]; curr_best_fit = best_fit_hist[-1]
                        rc = (curr_best_fit - prev_best_fit) / (abs(prev_best_fit) if prev_best_fit != 0 else 1.0)
                    if len(avg_fit_hist) >= 2:
                        prev_avg = avg_fit_hist[-2]; curr_avg = avg_fit_hist[-1]
                        rm = (curr_avg - prev_avg) / (abs(prev_avg) if prev_avg != 0 else 1.0)
                    s_next = compute_state(avg_fit_hist, div_hist, best_fit_hist)
                    a_pc_next = rl_pc.select_action(s_next); a_pm_next = rl_pm.select_action(s_next)
                    rl_pc.update(s, a_pc, rc, s_next, a_pc_next); rl_pm.update(s, a_pm, rm, s_next, a_pm_next)
                else:
                    break

        t1 = time.perf_counter()
        cpu_ms.append((t1 - t0) * 1000.0)

        if stop_reason == "noimp":
            stop_noimp += 1
        elif stop_reason == "maxevals":
            stop_maxevals += 1
        else:
            # Fallback (shouldn't happen): count as maxevals if budget reached, else noimp
            if (EVAL_COUNTER.count - start_eval) >= max_evals:
                stop_maxevals += 1
            else:
                stop_noimp += 1

        bests.append(best_c)

    # Stats
    best_overall = int(min(bests)) if bests else None
    median_best = statistics.median(bests) if bests else float('nan')
    median_gap = (median_best - lb) / lb if bests else float('nan')
    noimp_rate = 100.0 * stop_noimp / runs if runs > 0 else 0.0
    maxeval_rate = 100.0 * stop_maxevals / runs if runs > 0 else 0.0
    avg_cpu = float(np.mean(cpu_ms)) if cpu_ms else 0.0

    # helper for pretty % (avoid trailing .0)
    def fmt_pct(x):
        return f"{x:.1f}%" if abs(x - round(x)) >= 0.05 else f"{int(round(x))}%"

    print("========== Batch Results ==========")
    print(f"Runs: {runs}")
    print(f"best_cmax among all runs: {best_overall}")
    print(f"median best_cmax: {int(round(median_best))}")
    print(f"Times no improvement happen: {fmt_pct(noimp_rate)} (times stop by no improvement)")
    print(f"Times all evaluations finished: {fmt_pct(maxeval_rate)} (times finished {max_evals} evals)")
    print(f"median LB-GAP: {median_gap:.2f} ((median_cmax-lb)/lb)")
    print(f"AVG CPU time per run: {int(round(avg_cpu))} ms ")
    print("===================================")
def main():
    parser = argparse.ArgumentParser(description="SLGA for FJSP (fast defaults, plots in single-run, batch modes available).")
    parser.add_argument("--instance", type=str, default="mfjs10.txt",
                        help='Path to FJSP instance file (default: "mfjs10.txt")')
    parser.add_argument("--pop", type=int, default=50, help="Population size (default: fast auto: 3*m*n, min 30)")
    parser.add_argument("--gens", type=int, default=60, help="Generations (default: fast auto: 3*m*n, min 60)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: None)")
    parser.add_argument("--elite", type=float, default=0.02, help="Elite rate (default: 0.02)")
    parser.add_argument("--tk", type=int, default=3, help="Tournament k (default: 3)")
    # batch mode (eval/noimp) - already implemented earlier, optional use
    parser.add_argument("--runs", type=int, default=None, help="Batch runs (enables batch mode if set)")
    parser.add_argument("--evals", type=int, default=None, help="Max evaluations per run (batch mode by eval cap)")
    parser.add_argument("--noimp", type=int, default=None, help="Stop if this many evals pass with no improvement (batch eval mode)")
    parser.add_argument("--solution", type=float, default=None, help="Target Cmax for success in batch modes")
    parser.add_argument("--runtime", type=float, default=None, help="Per-run wall time in seconds (time-batch mode)")
    parser.add_argument("--lb", type=float, default=None, help="Lower bound Cmax for LB-based batch summaries")
    args = parser.parse_args()

    inst_path = args.instance
    if not os.path.exists(inst_path):
        here = os.path.dirname(os.path.abspath(__file__))
        alt = os.path.join(here, inst_path)
        if os.path.exists(alt):
            inst_path = alt
        else:
            raise FileNotFoundError(f"Instance not found: {args.instance}")

    # Time-budgeted batch mode
    if args.runs is not None and args.runtime is not None:
        run_batch_time(inst_path, args)
        return

    # LB eval-budgeted batch mode
    if args.runs is not None and args.evals is not None and args.noimp is not None and args.lb is not None:
        run_batch_evals_lb(inst_path, args)
        return

    # Evaluation-budget batch mode (from earlier request)
    if args.runs is not None:
        if args.evals is None or args.noimp is None or args.solution is None:
            raise SystemExit("Batch mode requires --runs, --evals, --noimp, and --solution.")
        runs = int(args.runs); max_evals = int(args.evals); noimp_limit = int(args.noimp); target = float(args.solution)
        success_count = 0; noimp_stops = 0; bests = []; cpu_times_success = []
        base_seed = args.seed
        for r in range(runs):
            seed_r = (base_seed + r) if base_seed is not None else None
            if seed_r is not None:
                random.seed(seed_r); np.random.seed(seed_r)
            jobs_ops, N_JOBS, N_MACH, N_OPS, OP_INDEX = parse_fjsp_instance(inst_path)
            job_num_ops, job_occurrences_template, allowed_machines, proc_time = build_helpers(jobs_ops, OP_INDEX)
            pop = args.pop or max(30, 3 * N_JOBS * N_MACH)
            pop = min(pop, max_evals)
            start_eval = EVAL_COUNTER.count
            population = [init_individual(job_occurrences_template, N_OPS, allowed_machines) for _ in range(pop)]
            fitness, cmax_vals, schedules = [], [], []
            best_c = float('inf'); last_improve_eval = 0; stop_reason = None; t0 = time.perf_counter()
            for ind in population:
                fit, cmax, sched = evaluate(ind[0], ind[1], N_JOBS, N_MACH, OP_INDEX, proc_time)
                fitness.append(fit); cmax_vals.append(cmax); schedules.append(sched)
                if cmax < best_c: best_c = cmax; last_improve_eval = 0
                else: last_improve_eval += 1
                if (target is not None) and (best_c <= target): stop_reason = "success"; break
                if last_improve_eval >= noimp_limit: stop_reason = "noimp"; break
                if (EVAL_COUNTER.count - start_eval) >= max_evals: stop_reason = "maxevals"; break
            if stop_reason is None:
                avg_fit_hist = [float(np.mean(fitness))]; div_hist = [float(np.std(fitness))]; best_fit_hist = [float(np.max(fitness))]
                rl_pc = RLController(); rl_pm = RLController(); elite_count = max(1, int(args.elite * pop))
                while True:
                    s = compute_state(avg_fit_hist, div_hist, best_fit_hist)
                    a_pc = rl_pc.select_action(s); a_pm = rl_pm.select_action(s)
                    Pc = sample_pc_from_action(a_pc); Pm = sample_pm_from_action(a_pm)
                    ranked = sorted(range(len(population)), key=lambda i: fitness[i], reverse=True)
                    new_population = [population[i] for i in ranked[:elite_count]]
                    new_fitness = [fitness[i] for i in ranked[:elite_count]]
                    new_cmax = [cmax_vals[i] for i in ranked[:elite_count]]
                    new_schedules = [schedules[i] for i in ranked[:elite_count]]
                    while len(new_population) < pop:
                        i1 = tournament_select(population, fitness, k=args.tk)
                        i2 = tournament_select(population, fitness, k=args.tk)
                        p1_OS, p1_MA = population[i1]; p2_OS, p2_MA = population[i2]
                        if random.random() < Pc:
                            c_OS = pox_crossover(p1_OS, p2_OS, N_JOBS); c_MA = uniform_crossover_ma(p1_MA, p2_MA, N_OPS, allowed_machines)
                        else:
                            c_OS = p1_OS.copy(); c_MA = p1_MA.copy()
                        c_OS = mutate_os_swap(c_OS, Pm); c_MA = mutate_ma_random(c_MA, Pm/2.0, allowed_machines)
                        fit, cmax, sched = evaluate(c_OS, c_MA, N_JOBS, N_MACH, OP_INDEX, proc_time)
                        new_population.append((c_OS, c_MA))
                        new_fitness.append(fit); new_cmax.append(cmax); new_schedules.append(sched)
                        if cmax < best_c: best_c = cmax; last_improve_eval = 0
                        else: last_improve_eval += 1
                        if (target is not None) and (best_c <= target): stop_reason = "success"; break
                        if last_improve_eval >= noimp_limit: stop_reason = "noimp"; break
                        if (EVAL_COUNTER.count - start_eval) >= max_evals: stop_reason = "maxevals"; break
                    population, fitness, cmax_vals, schedules = new_population, new_fitness, new_cmax, new_schedules
                    if stop_reason is None:
                        avg_fit_hist.append(float(np.mean(fitness))); div_hist.append(float(np.std(fitness))); best_fit_hist.append(float(np.max(fitness)))
                        rc = 0.0; rm = 0.0
                        if len(best_fit_hist) >= 2: prev_best_fit = best_fit_hist[-2]; curr_best_fit = best_fit_hist[-1]; rc = (curr_best_fit - prev_best_fit)/ (abs(prev_best_fit) if prev_best_fit != 0 else 1.0)
                        if len(avg_fit_hist) >= 2: prev_avg = avg_fit_hist[-2]; curr_avg = avg_fit_hist[-1]; rm = (curr_avg - prev_avg) / (abs(prev_avg) if prev_avg != 0 else 1.0)
                        s_next = compute_state(avg_fit_hist, div_hist, best_fit_hist); a_pc_next = rl_pc.select_action(s_next); a_pm_next = rl_pm.select_action(s_next)
                        rl_pc.update(s, a_pc, rc, s_next, a_pc_next); rl_pm.update(s, a_pm, rm, s_next, a_pm_next)
                    if stop_reason is not None: break
            t1 = time.perf_counter(); duration_ms = (t1 - t0) * 1000.0
            if stop_reason == "success": success_count += 1; cpu_times_success.append(duration_ms)
            elif stop_reason == "noimp": noimp_stops = locals().get('noimp_stops',0)+1
            bests.append(best_c)
        avg_best = float(np.mean(bests)) if bests else float('nan')
        gap = (avg_best - target) / target if bests else float('nan')
        success_rate = 100.0 * success_count / runs if runs > 0 else 0.0
        print("========== Batch Results ==========")
        print(f"Runs: {runs}")
        print(f"Success Rate to solution {int(target)}: {success_rate:.1f}%")
        print(f"GAP: {gap:.2f}")
        print("===================================")
        return

    # Single-run: show plots
    jobs_ops, N_JOBS, N_MACH, N_OPS, OP_INDEX = parse_fjsp_instance(inst_path)
    results = run_SLGA(jobs_ops, N_JOBS, N_MACH, N_OPS, OP_INDEX,
                       pop_size=args.pop, generations=args.gens,
                       elite_rate=args.elite, tournament_k=args.tk, seed=args.seed)
    best_sched = results["best_schedule"]
    best_cmax = results["best_cmax"]
    eval_trace = results["eval_trace"]
    print(f"Best Cmax: {best_cmax}")
    plot_cmax_over_evals(eval_trace)
    plot_gantt(best_sched, N_JOBS, title=f"Gantt (Cmax={best_cmax})")
    plt.show()

if __name__ == "__main__":
    main()
