#!/bin/bash
clear

parallel -j 15 --eta --ungroup taskset -c {1} python3 run_ppo.py -c cfg/foragax/ForagaxTwoBiomeSmall-v2-15/PPO_CB/{1}.yml -s {1} ::: $(seq 0 29)
parallel -j 15 --eta --ungroup taskset -c {1} python3 run_ppo.py -c cfg/foragax/ForagaxTwoBiomeSmall-v2-15/PPO/{1}.yml -s {1} ::: $(seq 0 29)
