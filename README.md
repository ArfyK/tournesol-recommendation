**List of Files**
  - `l_tuning.py`
  - `temperature_tuning.py`

## `l_tuning.py`
This script compares to value for the `l` parameter used in the objective function `F` defined in `recommendation.py`:
  - the default l = 1/10;
  - l = m * 1/10 where m is the mean of all criterias means. At first this value was supposed to ensure that the two terms in `F` are homogeneous to a score. But as I'm writing these lines I realise that I should have used sqrt(m) ! As we will see below, the results are nonetheless pretty good so I keep it that way for now. Fine tuning the parameter should be done later.  

These two parameters are also compared with three `random` algorithms:
  - `random` which uniformly samples a bundles of videos without any prior selection;
  - `r_50` (resp. `r_75`) which uniformly samples the bundle from the videos with a tournesol score above the median (resp. the third quartile);
  - `r_agg_50` (resp. `r_agg_75`) which uniformly samples the bundle from the videos with an "aggregated score" above the median (resp. the third quartile). The "aggregated score" is actually the objective function applied on single video. It's defined in `recommendation.py`.

# How to use the script
First set the tests parameters in the script:
  - the number of tests `n_tests`;
  - the size of the subdatasets that will be sampled for each test `size`. 

Then to run the script using the dataset tournesol_scores_2023-05-04.csv type:
`python3 l_tuning.py tournesol_scores_2023-05-04.csv`

This will create two files:
  - `l_tuning_n_tests=<n>_size=<size>.csv` containing the results;
  - `l_tuning.png' plotting the distribution of the maximum of each criteria. 

# Results analysis
On `l_tuning.png` we can observe that:
  - except for the "engaging" criteria, l=1/10*m performs slightly better or similarly to l=1/10. Regarding the number of channel featured its median is 1 channel higher;
  - `r_50` (resp. `r_75`) and `r_agg_50` (resp. `r_agg_75`) perform similarly. They out perform `random`; 
  - regarding the criteria, the random algorithms are outperformed by the greedy ones. They are better in terms of number of channels.

Those results led me to use l = 1/10*m for the tuning of the temperature.

## `temperature_tuning.py`
This script performs tests several temperature parameters used in the `random_greedy` algorithm from `recommendation.py`. 
The different `random_greedy` algorithms are also compared with two `random` algorithms:
  - the first uniformly samples videos having a tournesol score above the third quartile;
  - the second uniformly samples videos having an "aggregated" score above the third quartile.  

# How to use the script
First set the tests parameters in the script:
  - the number of tests `n_tests`;
  - the list of temperature values to be tested `temp_list`.

Then to run the script using the dataset tournesol_scores_2023-05-04.csv type:
`python3 temperature_tuning.py tournesol_scores_2023-05-04.csv`

This will create three files: 
  - `temp_tuning_n_test=<n>_t=<temp_list>.csv` containing the results 
  - `temperature_criteria_comparison_t=<temp_list>_n_tests=<n>.png` plotting and comparing the distribution of the maximum of each criteria and the number of channels featured in selections for each algorithm. 
  - `temperature_coverage_tournesolscore_t=<temp_list>n_tests=<n>.png` plotting the comparison of the coverage of the top 200 tournesol scores.
  - `temperature_coverage_aggregatedscore_t=<temp_list>_n_tests=<n>.png` plotting the comparison of the coverage of the top 200 aggregated scores. The aggregated score of a video is a score that aggregates its tournesol score and the score of secondary criteria in a way resembling the objective function. 

# Results analysis
First recall that the probablity distribution is p(x) = exp(T*x).

**T in [0.01, 0.1, 1, 10, 100]**
On temperature_criteria_comparison_t\=0.01_0.1_1.0_10.0_100.0_n_tests\=100.0.png we can observe that: 
  - as expected the higher temperature T=100 outperforms the rest, followed by T=10. The three last seem to have quite similar performances;
  - the two `random` algorithms have similar performances and seem to have slightly better performances than the `random_greedy` ones, expect for T=100.

On temperature_coverage_tournesolscore_t=0.01_0.1_1.0_10.0_100.0n_tests=100.0.png (the other coverage plot using the aggregated scores is similar) we can observe that: 
  - the two `random` algorithms seem to well cover the top 200 with frequencies between 1% and 4%;
  - the two `random` algorithms outperform the `random_greedy` ones except for T=100;
  - the higher temperature covers well the top 100 with frequencies between 2 and 10%. Some videos have significantly higher frequencies (hence the use of a log scale): 20% for a few videos, about 70% for one video and close to 100% for two videos (one of those two videos is also always chosen by T=10).

Those quite high frequencies led me to investigate the temperatures between 10 and 100.

**T in [20, 40, 60, 80]**
On temperature_criteria_comparison_t=20.0_40.0_60.0_80.0_n_tests=100.0.png we can observe that: 
  - the performances are approximately sorted according to the temperature;
  - on some criterias the temperatures above 20 have a signficantly lower dispersion despite having better scores;
  - T=20 performs slightly better than the two `random` algorithms.

On temperature_coverage_tournesolscore_t=20.0_40.0_60.0_80.0n_tests=100.0.png (the other coverage plot using the aggregated scores is similar) we can observe that: 
  - only the top 100 is well covered by the `random_greedy` algorithms;
  - the three same videos as before are chosen too often;
  - T=60 and T=80 have a slightly more cover the top 200 than T=100;
  - except for a few videos, T=40 and the `random` algorithms perform similarly.

In conclusion, because of the issue of the three over-chosen videos, it is unclear to me what temperature should be chosen between 60, 80 and 100 (or maybe even a higher one). 
