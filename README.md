# olfactory circuit selection
Source codes associated with the preprint: 
```
Developmental and evolutionary constraints on olfactory circuit selection
Naoki Hiratani and Peter E Latham
Gatsby Computational Neuroscience Unit, University College London
doi: https://doi.org/10.1101/2020.12.22.423799
```

Questions on the manuscript and the codes should be addressed to Naoki Hiratani (N.Hiratani@gmail.com).

Directory “mle” corresponds to the results under the maximum likelihood estimation depicted in Figures 3 and 4. 
* mle_theory.py and mle_simul.py generate the theoretical and simulation results depicted in Figure 3 and Figure 4A-C.
* mle_plot_error-h_curve.py plots the error curves in Figure 3. 
* logis_theory.py generates the theoretical lines in Figure 4DE.

Directory “sgd” corresponds to the results under stochastic gradient descent learning depicted in Figures 5 and 6.
* sgd_theory.py and sgd_simul.py generate the theoretical and simulation results depicted in Figure 5B-E.
* dual_simul.py generates the simulation results under a dual pathway model depicted in Figure 6.
