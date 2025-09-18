#!/bin/bash
clear

parallel --eta --ungroup taskset -c {1} python3 run_ppo.py -c cfg/foragax/cbp.yml -s {1} ::: $(seq 20 24) &
parallel --eta --ungroup taskset -c {1} python3 run_ppo.py -c cfg/foragax/std.yml -s {1} ::: $(seq 20 24) &
