## GACEM: Generalized Auto-regressive Cross Entropy Method for Multi-Modal Black Box Constraint Satisfaction Problem

This repository contains the code for paper (...) ICML 2020.

# Dependancies


# Reproduce ICML Experiments
To re-produce results `icml_command.sh` contains all the commands that were used to train and plot the results.
Just find which command you want to run and un-comment the relative parts.
For example if you want to run all the algorithms for function "Stybnlinski-2D" and get the plots uncomment the following lines:

```
# ---------- styblinski 2D
# CPU compatible
./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_2d_cem_dim_study.yaml -ns 5
./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_2d_cem_sg_fixed_sigma_dim_study.yaml -ns 5
./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_2d_cempp_gauss_dim_study.yaml -ns 5
./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_2d_cempp_kde_dim_study.yaml -ns 5
./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_2d_gacem_off_fixed_sigma_dim_study.yaml -ns 5
./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_2d_gacem_onp_fixed_sigma_dim_study.yaml -ns 5

# plot figures
./run.sh scripts/plot_experiments.py \
data/styblinski_2d_cem_gauss_optim_on_co0p4_ns25 \
data/styblinski_2d_cem_gauss_optim_on_co0p4_ns25_sig10 \
data/styblinski_2d_cem_gauss_optim_off_co0p4_ns25 \
data/styblinski_2d_cem_kde_optim_off_co0p4_ns25 \
data/styblinski_2d_gacem_ns5_lay3x100_e10_b10_nr40_csp_off_fixed_sigma \
data/styblinski_2d_gacem_ns25_lay3x100_e10_b10_nr40_csp_onp_fixed_sigma \
--save-path experiments_results/styblinski_2d \
-l cem_adaptive_variance cem_fixed_variance cem++sg cem++kde gacem_off gacem_onp \
-g 10
```

The following algorithms are implemented:

* Continuous domain adaptive sampling with Autoregressive policies
* Cross Entropy Method (Single Gaussian (SG), KDE (Kernel Desnsity Estimation), with adaptive co-variance, with fixed variance)

CEM has different variants:
Optimization vs. Constraint satisfaction, with on-policy or off-policy

# Architecture of the code




# Running custom experiments
to run:

```
cd autoreg_ckt
./run.sh scripts/run_alg.py --help
./run.sh scripts/run_alg.py spec_files/cem/styblinski.yaml
./run.sh scripts/run_alg.py spec_files/autoregressive/styblinski.yaml
```

to load:
```
./run.sh scripts/run_alg.py data/path_to_spec_yaml.yaml --load
```

to resume with minor changes (like increased number of iterations):
update the yaml file's parameters in the data folder and run:
```
./run.sh scripts/run_alg.py data/path_to_spec_yaml.yaml --load
```

to download:
```
./run.sh scripts/download_data.py --help
```

to plot tsne or pca:
```
./run.sh scripts/compare_solutions.py --help
```

to animate 2D 

# stored data
Results are stored in data folder according to the name, suffix, prefix specified in the yaml and 
the date and time of running the algorithm.

