**Table of contents**
  - `recommendation.py`
  - `sample_tuning.py`
  - `temperature_tuning.py`
  - Next steps

## `recommendation.py`
This file contains several recommendation algorithms:
  - `deterministic_greedy` which greedily optimizes the objective function F;
  - `random_greedy` which, broadly speaking, samples the videos one by one using the probability distribution exp(F(S)/T) where T is a temperature parameter and F(S) is the objective value of the current bundle. 
  - `random` uniformly samples the bundle. It can sample from the whole dataset or only from the set of videos scoring above some given quantile. Two types of scoring are available : the tournesol score (default) et the "aggregated score". The aggregated score of a video is a score that aggregates its tournesol score and the score of secondary criteria in a way resembling the objective function. 
  - `deterministic_random_sample` which applies the greedy algorithm on a random sample of videos with a tournesol score above some specified quantile.

When run as a script it compares the `deterministic_greedy`, `random_greedy` and `random` algorithms using the parameters
identified with `l_tuning.py` and `temp_tuning.py`, see below.

# How to use the script
First set the tests parameters in the script:
  - the number of tests `n_tests`;
  - the size of the subdatasets that will be sampled for each test `size`; 
  - the temperature `T`.

Then to run the script using the dataset tournesol_scores_2023-05-04.csv type:
`python3 recommendation.py tournesol_scores_2023-05-04.csv` 

This will create two files : 
  - `algo_comparison_n_test=<n>_size=<size>.csv` containing the results;
  - `algorithms_comparison.png` plotting the distribution of the maximum of each criteria. Those maximum are normalized according to (x - min)/(max - min). 

# Result analysis
On `algorithms_comparison.png` we can observe that:
  - `deterministic_greedy` and `random_greedy` outperform the `random` algorithm;
  - regarding the criterias the two greedy algorithms have similar performances except for the following criterias: 'importance', 'largely_recommended', 'diversity_inclusion', 'engaging' where `deterministic_greedy' performs better;
  - the two random algorithms seem to feature slightly more channels than the deterministic one. 
  - the 'entertaining_relaxing' criteria features **negative maximums** which should not be possible given the normalization we used.

## `sample_tuning.py`
This script tests several combinations of quantile and sample size used in the `deterministic_random_sample` algorithm and compare them with the `random` algorithm used with q=0.75 (r_75).

# How to use the script
First set the tests parameters in the script:
  - the number of tests `n_tests`;
  - the quantile `quantile`;
  - the list of sample sizes `size_list`

Then to run the script using the dataset tournesol_scores_2023-05-04.csv type:
`python3 sample_tuning.py tournesol_scores_2023-05-04.csv`

This will create three files: 
  - `sample_size_n_test=<n>_size=<size_list>_q=<quantile>.csv` containing the results; 
  - `sample_size_criteria_comparison_q=<quantile>_size=<size_list>_n_tests=<n>.png` plotting and comparing the distribution of the maximum of each criteria and the number of channels featured in selections for each algorithm; 
  - `sample_size_coverage_size=<size_list>_q=<quantile>_n_tests=<n>.png` plotting the comparison of the coverage of the top 200 tournesol scores for each algorithm.

# Results analysis
We tested several sample size for both q=0.5 and q=0.75. The algorithms' performances were mainly assessed according to their coverage of the top 200, more precisely:
  1) no video should appear in more than 20% of the bundles;
  2) all videos from the top 100 should appear in at least 1% of the bundles.

For comparison `deterministic_greedy` obtains an objective value of about 709 on the complete dataset.

**q = 0.75 and sample size in [40, 65, 90, 115, 140]**
On sample_size_criteria_comparison_q\=0.75_size\=40_65_90_115_140_n_tests\=500.png we observe that: 
  - The objective value is increasing with the sample size. It remains above r_75 for all sizes;
  - all sample sizes have better performances than r_75 with respect to the following criterias: diversity_inclusion, importance, engaging, largely_recommended, pedagogy, better_habits, reliability;
  - the performances are approximately the same with respect to the following criterias: layman_friendly, entertaining_relaxing, backfiring_risk;
  - The performances are increasing with the sample size for each criteria except entertaining_relaxing where it's decreasing;
  - the number of channels featured in the bundle in decreasing with the sample size. r_75 features more channels.

On sample_size_coverage_size\=40_65_90_115_140_q\=0.75n_tests\=500.png we observe that:
  - r_75 uniformly covers the top 200 with a frequency of 1-2%;
  - the size 40 also uniformly covers the top 200 with a higher frequency near 5%;
  - the size should be less than 115 in order to satisfy the two criteria above;

According to those results a sample size of 90 could be a good trade-off between performance and coverage. 

**q = 0.5 and sample size in [40, 65, 90, 115, 140]**
On sample_size_criteria_comparison_q\=0.5_size\=40_65_90_115_140_n_tests\=500.png we observe that:
  - the performance is increasing with n for the objective value and every criteria except layman_friendly, backfiref_risk and entertaining_relaxing where it's roughly constant;
  - r_75 has lower performances on every criteria except on layman_friendly, backfiref_risk and entertaining_relaxing where the performances are similar;
  - in terms of number of channels r_75 is slightly better. 

On sample_size_coverage_size\=40_65_90_115_140_q\=0.5n_tests\=500.png we can observe that every parameters meets the two criteria above.

These results lead me to test higher values for n.

**q = 0.5 and sample size in [150, 190, 230, 280, 320]**
On sample_size_criteria_comparison_q\=0.5_size\=150_190_230_280_320_n_tests\=500.png we can observe that:
  - the objective value is still increasing with n;
  - the performances are similar albeit increasing with n;
  - the comparison with r_75 is qualitatively the same as before.

On sample_size_coverage_size\=150_190_230_280_320_q\=0.5n_tests\=500.png we can observe that:
  - n should be less than 300 to meet criteria 2) above;
  - the videos ranked between 100 and 200 are barely covered with n above 200.

In conclusion if we are only interested in a good coverage of the top 100 we could use n=230.

## `temperature_tuning.py`
This script tests several temperature parameters used in the `random_greedy` algorithm from `recommendation.py`. 
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
  - `temp_tuning_n_test=<n>_t=<temp_list>.csv` containing the results; 
  - `temperature_criteria_comparison_t=<temp_list>_n_tests=<n>.png` plotting and comparing the distribution of the maximum of each criteria and the number of channels featured in selections for each algorithm; 
  - `temperature_coverage_tournesolscore_t=<temp_list>n_tests=<n>.png` plotting the comparison of the coverage of the top 200 tournesol scores for each algorithm;
  - `temperature_coverage_aggregatedscore_t=<temp_list>_n_tests=<n>.png` plotting the comparison of the coverage of the top 200 aggregated scores. The aggregated score of a video is a score that aggregates its tournesol score and the score of secondary criteria in a way resembling the objective function. 

# Results analysis
First recall that the probablity distribution is p(x) = exp(T*x).

**T in [0.01, 0.1, 1, 10, 100]**
On temperature_criteria_comparison_t\=0.01_0.1_1.0_10.0_100.0_n_tests\=100.0.png we can observe that: 
  - as expected the lower temperature T=0.01 outperforms the rest, followed by T=0.1. The other three seem to have quite similar performances;

  - the two `random` algorithms have similar performances and seem to have slightly better performances than the `random_greedy` ones, expect for T=0.01.
  - the criterias 'diversity_inclusion', 'layman_friendly', 'backfiring_risks', 'entertaining_relaxing' and 'reliability' feature **negative maximums** which should not be possible given the normalization we used.

On temperature_coverage_tournesolscore_t=0.01_0.1_1.0_10.0_100.0n_tests=100.0.png (the other coverage plot using the aggregated scores is similar) we can observe that: 
  - the two `random` algorithms seem to well cover the top 200 with frequencies between 1% and 4%;
  - the two `random` algorithms outperform the `random_greedy` ones except for T=0.01;
  - the higher temperature covers well the top 100 with frequencies between 2 and 10%. Some videos have significantly higher frequencies (hence the use of a log scale): 20% for a few videos, about 70% for one video and close to 100% for two videos (one of those two videos is also always chosen by T=0.1).

Those quite high frequencies led me to investigate the temperatures between 0.01 and 0.1.

**T in [1/20, 1/40, 1/60, 1/80]**
On temperature_criteria_comparison_t=20.0_40.0_60.0_80.0_n_tests=100.0.png we can observe that: 
  - the performances are approximately sorted according to the temperature;
  - on some criterias the temperatures below 1/20 have a signficantly lower dispersion despite having better scores;
  - T=1/20 performs slightly better than the two `random` algorithms.
  - the criteria 'entertaining_relaxing' features **negative maximums** which should not be possible given the normalization we used.

On temperature_coverage_tournesolscore_t=20.0_40.0_60.0_80.0n_tests=100.0.png (the other coverage plot using the aggregated scores is similar) we can observe that: 
  - only the top 100 is well covered by the `random_greedy` algorithms;
  - the three same videos as before are chosen too often;
  - T=1/60 and T=1/80 have a slightly more cover the top 200 than T=1/100;
  - except for a few videos, T=1/40 and the `random` algorithms perform similarly.

In conclusion, because of the issue of the three over-chosen videos, it is unclear to me what temperature should be chosen between 1/60, 1/80 and 1/100 (or maybe even a higher one). 

## Next steps
  - Investigate the presence of negative maximums;
  - Add a term about the number of channels in the objective function;
  - Add a term for the recency of videos in the objective function. 
  - Investigate the videos that are chosen too frequently. 
  - Maybe change the way we introduce the randomness: we could count how many times each video is recommended and add a term in the scoring function that gives more chance to be selected to videos that have been selected less times

