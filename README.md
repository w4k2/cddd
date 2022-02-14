# Centroid Distance Drift Detector

### *Before use, prepare real data streams! (make prepare_data)*

This repository supplements the results and provides source code implementations prepared for the research article entitled "Concept drift detector based on centroid distance analysis". Most of the results are in raw form - confusion matrices and metrics scores. If one is interested in a much deeper analysis, use the `analyze_results_main.py` script to generate all the plots. Below is a set of the most relevant results.

## Supervised CDDD

![Podpis](./results/ranking_plots/ranks_gradual_CDDD.png)
*Synthetic data streams - gradual drift*

![](./results/ranking_plots/ranks_incremental_CDDD.png)
*Synthetic data streams - incremental drift*

![](./results/ranking_plots/ranks_sudden_CDDD.png)
*Synthetic data streams - sudden drift*

![](./results/ranking_plots/ranks_recurring_CDDD.png)
*Synthetic data streams - reccuring drift*

![](./results/ranking_plots/ranks_real_CDDD.png)
*Real data streams*

## Unsupervised CDDD

![Podpis](./results/ranking_plots/ranks_gradual_CDDD-V2.png)
*Synthetic data streams - gradual drift*

![](./results/ranking_plots/ranks_incremental_CDDD-V2.png)
*Synthetic data streams - incremental drift*

![](./results/ranking_plots/ranks_sudden_CDDD-V2.png)
*Synthetic data streams - sudden drift*

![](./results/ranking_plots/ranks_recurring_CDDD-V2.png)
*Synthetic data streams - reccuring drift*

![](./results/ranking_plots/ranks_real_CDDD-V2.png)
*Synthetic data streams - reccuring drift*
