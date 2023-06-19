#!/bin/sh

source env/bin/activate

python3 temperature_tuning.py tournesol_scores_2023-05-04.csv temp_tuning_n_test=5.csv

shutdown

exit
