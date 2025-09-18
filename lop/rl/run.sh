#!/bin/bash
clear

parallel --eta --ungroup taskset -c {1} python3 run_ppo.py -c cfg/foragax/ForagaxTwoBiomeSmall-v2-15/cbp/{1}.yml -s {1} ::: $(seq 0 29) &
wait
parallel --eta --ungroup taskset -c {1} python3 run_ppo.py -c cfg/foragax/ForagaxTwoBiomeSmall-v2-15/std/{1}.yml -s {1} ::: $(seq 0 29) &
wait
