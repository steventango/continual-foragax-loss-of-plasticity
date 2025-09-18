#!/bin/bash
clear
parallel --eta --ungroup taskset -c {1} python3 run_ppo.py -c cfg/ant/cbp.yml -s {1} ::: $(seq 0 19)
parallel --eta --ungroup taskset -c {1} python3 run_ppo.py -c cfg/ant/l2.yml -s {1} ::: $(seq 0 19)
parallel --eta --ungroup taskset -c {1} python3 run_ppo.py -c cfg/ant/ns.yml -s {1} ::: $(seq 0 19)
parallel --eta --ungroup taskset -c {1} python3 run_ppo.py -c cfg/ant/redo.yml -s {1} ::: $(seq 0 19)
parallel --eta --ungroup taskset -c {1} python3 run_ppo.py -c cfg/ant/std.yml -s {1} ::: $(seq 0 19)
