## High-Level Overview

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

No ethics review, consent process, or withdrawal mechanism is documented, which is appropriate given that the dataset does not involve human participants. The more relevant issue is methodological: because later points were selected using the optimiser, the dataset is policy-dependent rather than a random sample of each search space.

### Data Breakdown

## Nature Of The Data

1. **Describe the structure of the initial dataset.**

   Each CSV has columns `function`, `index`, `x0` to `x7`, and `y0`. Unused dimensions are blank. 

2. **How does the dataset evolve as you add new queries weekly?**

   The data evolve sequentially. Each week, the optimiser reads the current history, proposes one new point per function, and the returned `y0` values are appended to the relevant CSV files.

3. **Does the function include noise or randomness?**

   The repository does not include repeated evaluations at identical points, so there is no direct empirical estimate of observation noise. The modelling pipeline treats the objectives as expensive black boxes and uses GP uncertainty to guide decisions.

4. **Based on observations, does the function appear unimodal, multimodal, noisy, or smooth?**

   The functions are heterogeneous. `function_5` shows a strong boundary-driven high-value region, while `function_3` and `function_6` have negative outputs and relatively small best-so-far improvements. The progress plots and GP diagnostics suggest a mixture of smooth structure, local optima, and regions where exploration remains valuable.

## Your Optimisation Strategy

1. **Which optimisation method(s) did you use?**

   I used Bayesian optimisation with Gaussian process surrogate models. The pipeline combines global Sobol candidates, trust-region candidates, elite-region candidates, and a portfolio selector using acquisition and heuristic scores.

2. **Why did you choose this method?**

   The functions are expensive, continuous, and low-to-moderate dimensional. Bayesian optimisation is suitable because it can use small datasets efficiently and quantify uncertainty.

3. **How did you balance exploration and exploitation?**

   Candidate scoring includes expected improvement, probability of improvement, upper confidence bound, Thompson-style scores, posterior uncertainty, and novelty. The final selector can choose from exploitation-oriented and exploration-oriented strategies. Trust-region sampling focuses locally, while global Sobol sampling preserves broader coverage.

4. **Did your strategy change over the weeks? Why?**

   Yes. The approach evolved from broader global search toward a portfolio method with trust-region reconstruction, elite-region sampling and stagnation detection. This was introduced because several functions began showing local structure and potential stagnation.

## Data Handling And Preprocessing

1. **Did you rescale or normalise inputs? Why or why not?**

   Inputs are already represented in the unit box, approximately `[0, 1]`, so no additional input rescaling is needed before modelling.

2. **Did you train any surrogate models?**

   Yes. The pipeline trains one Gaussian process surrogate per function.

3. **If yes, what preprocessing did the surrogate require?**

   The preferred setup uses output power transforms such as Box-Cox, sign-flipped Box-Cox, or Yeo-Johnson when helpful. It also supports input warping. The default kernel is an additive linear plus Matern 1.5 kernel with ARD.

4. **Did you handle outliers or unusual data points?**

   The pipeline does not delete high or low outputs as outliers. Instead, it uses transformations and robust fallback model configurations. Candidate deduplication is handled after six-decimal floor rounding to match portal precision.

## Weekly Iteration And Learning

1. **How did new data points change your understanding of the function landscape?**

   New data clarified which functions had strong local basins and which still benefited from broader exploration.

2. **Did you encounter local optima? How did you detect them?**

   Yes. Stagnation flags and repeated trust-region recommendations suggested local optima or local plateaus.

3. **Which queried inputs were most informative and why?**

   Boundary and near-boundary points were especially informative for `function_5`, whose best value occurs at the upper boundary in all four dimensions. Local trust-region points were informative for functions where strong regions had already been found.

4. **If you restarted, what would you do differently?**

   A stricter round-by-round manifest, including the exact command, seed, selected strategy, and returned value for every submission.

## Performance And Results

1. **What is the best output value you achieved?**

   | Function | Best `y0` | Best input vector |
   | --- | ---: | --- |
   | `function_1` | `0.184524` | `[0.443519, 0.407772]` |
   | `function_2` | `0.651454` | `[0.716439, 0.857645]` |
   | `function_3` | `-0.003253` | `[0.391058, 0.458474, 0.448532]` |
   | `function_4` | `0.701305` | `[0.418031, 0.410481, 0.406548, 0.416485]` |
   | `function_5` | `8662.405001` | `[0.999999, 0.999999, 0.999999, 0.999999]` |
   | `function_6` | `-0.154283` | `[0.441579, 0.390526, 0.604394, 0.786880, 0.171566]` |
   | `function_7` | `3.223213` | `[0.218691, 0.218479, 0.440618, 0.271435, 0.335377, 0.659655]` |
   | `function_8` | `9.958745` | `[0.107466, 0.115199, 0.129783, 0.155887, 0.772725, 0.572681, 0.124179, 0.123077]` |

2. **How confident are you that this is near the global maximum? Why?**

   Confidence varies by function. It is higher where progress has plateaued after local exploration and lower where the search space is higher-dimensional or improvements have been more recent. Because the data are small and adaptively collected, these results should be treated as strong observed values rather than proven global optima.

3. **Did your results align with expectations?**

   No. The lower-dimensional functions turned out to have complexity I didn't expect, and required more reliance on surrogate uncertainty and trust-region behaviour. Higher-dimensional functions proved easier to interpret that I thought.

## Ethical, Practical And General Considerations

1. **How does this black-box optimisation task relate to real-world applications?**

   It mirrors settings where evaluations are expensive and the underlying mechanism is unknown, such as laboratory experiments.

2. **What limitations arise from the synthetic nature of the function?**

   The hidden functions do not include real-world constraints, measurement costs, or safety issues.

3. **Would your strategy scale to more serious or more expensive problems? Why or why not?**

   The strategy would scale conceptually to expensive continuous optimisation, but it would need cost-aware acquisition and careful validation before use in a serious domain.

4. **What risks or pitfalls should a future user be aware of?**

   Key risks include allowing visualising data to warp the way you think about the landscape. Matplotlib visualisations can only from the data that they have and new points can drastically change the shape of the landscape. 
