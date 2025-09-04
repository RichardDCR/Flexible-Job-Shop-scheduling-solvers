# Flexible-Job-Shop-scheduling-solvers
A repository with four scripts that have solvers for FJSSP. Simulated Annealing, Genetic Algorithm, Tabular Search and RL+GA using Q-learning.

## Algorith Script
- [sa_fjsp.py](sa_fjsp.py): (Integrated) Simulated annealing script. Structure based on [1]
- [ga_fjsp.py](ga_fjsp.py): Genetic algorithm script. Structure based on [2]
- [its_fjsp.py](its_fjsp.py): (Integrated) Tabular Search script. Structure based on [1]
- [rlga_fjsp.py](rlga_fjsp.py): Reinforcement Learning Genetic Algorithm script. Structure based on [2]

## How to use it - commands
Each algorith has 5 workmodes.

(file "mfjs10.txt" as an instance file and "rlga_fjsp.py" as algorithm file, can be replaced by other)

1) One run test: Test one run of the algorithm and plot cmax chart and gant plot

    `python rlga_fjsp.py --instance mfjs10.txt`

3) Evaluation/No-Improvement mode with known solution
   
    `python rlga_fjsp.py --instance mfjs10.txt --runs 100 --evals 1000 --noimp 100 --solution 66`

5) Evaluation/No-Improvement mode with unknown solution
    
    `python rlga_fjsp.py --instance mfjs10.txt --runs 100 --evals 1000 --noimp 100 --lb 66`

7) Time-boxed mode with known solution
   
     `python rlga_fjsp.py --instance mfjs10.txt --runs 100 --runtime 0.5 --solution 66`
     
9) Time-boxed mode with unknown solution
        
     `python rlga_fjsp.py --instance mfjs10.txt --runs 100 --runtime 0.5 --lb 66`
## Dataset
Dataset to be used in all scripts can be found in [SchedulingLab/fjsp-instances](https://github.com/SchedulingLab/fjsp-instances)

The format for FJSP is:
- First line: `<number of jobs> <number of machines>`
- Then one line per job: `<number of operations>` and then, for each operation, `<number of machines for this operation>` and for each machine, a pair `<machine> <processing time>`.
- Machine index starts at 0.

## References
1. P. Fattahi, M. S. Mehrabad, and F. Jolai. [Mathematical Modeling and Heuristic Approaches to Flexible Job Shop Scheduling Problems](https://doi.org/10.1007/s10845-007-0026-8). Journal of Intelligent Manufacturing, 18(3):331–342, 2007.
2. R. Chen, B. Yang, S. Li, and S. Wang, [A self-learning genetic algorithm based on reinforcement learning for flexible job-shop scheduling problem](https://doi.org/10.1016/j.cie.2020.106778). Comput Ind Eng, vol. 149, p. 106778, 2020.
