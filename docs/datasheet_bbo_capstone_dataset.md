## Collection Process

The dataset was collected through an iterative weekly optimisation process rather than through a single batch collection stage. In each round, the existing observations were used to fit a surrogate model, generate candidate points, score those candidates under several acquisition and heuristic criteria, and select one portal-ready submission per function. Once new evaluations became available, they were appended to the relevant per-function CSV files and the process repeated.

The collection process is therefore adaptive. The dataset grows in response to earlier observations and earlier modelling decisions.

At a high level, query generation proceeds as follows:

1. The existing observation history for one function is loaded.
2. A Gaussian process surrogate is fit, with optional output transformation and optional input warping.
3. A trust region is reconstructed from the historical sequence.
4. A mixed candidate pool is generated, including global Sobol points, trust-region Sobol points, and local samples around strong previously observed regions.
5. Candidates are scored using several criteria, including expected improvement, log expected improvement, probability of improvement, upper confidence bound sweeps, Thompson-style scores, and exploration-oriented scores combining novelty and uncertainty.
6. Duplicates are removed after six-decimal floor rounding, including points that would collide with already observed locations.
7. A final recommendation is selected using a portfolio-style rule, in some cases incorporating multi-seed consensus.
8. One new point per function is submitted, and the resulting evaluations are appended when they become available.

The collection strategy is therefore neither purely deterministic nor purely random. Some parts are fixed and rule-based, such as loading, rounding, and deduplication. Others are stochastic, such as Sobol candidate generation and seed-dependent exploration.

The dated folders in `reports/` suggest that the process unfolded across multiple snapshots, including `2026-03-19-plus`, `2026-03-20`, `2026-04-02`, `2026-04-02-audited`, and `2026-04-02-fixed-signflip`. These folders make the sequential nature of the collection visible. However, the repository does not include a single explicit manifest mapping every observation to a clearly labelled round number, so the chronology can only be reconstructed partially from the available files.

No ethics review, consent process, or withdrawal mechanism is documented, which is appropriate given that the dataset does not involve human participants. The more relevant issue is methodological: because later points were selected using the optimiser, the dataset is policy-dependent rather than a random sample of each search space.