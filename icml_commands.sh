#!/usr/bin/env bash
# uncomment the sections you want to run

# ---------- levy 2D (runs on CPU in reasonable time)
#./run.sh scripts/run_experiments.py spec_files/icml_paper/levy_dim/levy_2d_cem_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/levy_dim/levy_2d_cempp_gauss_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/levy_dim/levy_2d_cempp_kde_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/levy_dim/levy_2d_gacem_dim_study.yaml -ns 5

# plot figures
#./run.sh scripts/plot_experiments.py \
#data/levy_2d_cem_gauss_optim_on_co0p4_ns25 \
#data/levy_2d_cem_gauss_optim_off_co0p4_ns25 \
#data/levy_2d_cem_kde_optim_off_co0p4_ns25 \
#data/levy_2d_gacem_co0p4_ns5_lay3x20_e5_b10_nr20 \
#--save-path experiments_results/levy_2d \
#-l cem cempp_sg cempp_kde gacem \
#-g 0.4

# ---------- levy 5D
# CPU compatible
#./run.sh scripts/run_experiments.py spec_files/icml_paper/levy_dim/levy_5d_cem_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/levy_dim/levy_5d_cempp_gauss_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/levy_dim/levy_5d_cempp_kde_dim_study.yaml -ns 5
# GPU only (otherwise will take forever)
#./run.sh scripts/run_experiments.py spec_files/icml_paper/levy_dim/levy_5d_gacem_dim_study.yaml -ns 5

# plot figures
#./run.sh scripts/plot_experiments.py \
#data/levy_5d_cem_gauss_optim_on_co0p4_ns25 \
#data/levy_5d_cem_gauss_optim_off_co0p4_ns25 \
#data/levy_5d_cem_kde_optim_off_co0p4_ns25 \
#data/levy_5d_gacem_co0p4_ns5_lay3x20_e5_b10_nr20 \
#--save-path experiments_results/levy_5d \
#-l cem cempp_sg cempp_kde gacem \
#-g 0.4

# ---------- levy 20D
# CPU compatible
#./run.sh scripts/run_experiments.py spec_files/icml_paper/levy_dim/levy_20d_cem_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/levy_dim/levy_20d_cempp_gauss_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/levy_dim/levy_20d_cempp_kde_dim_study.yaml -ns 5
# GPU only (otherwise will take forever)
#./run.sh scripts/run_experiments.py spec_files/icml_paper/levy_dim/levy_20d_gacem_dim_study.yaml -ns 5

# plot figures
#./run.sh scripts/plot_experiments.py \
#data/levy_20d_cem_gauss_optim_on_co0p4_ns25 \
#data/levy_20d_cem_gauss_optim_off_co0p4_ns25 \
#data/levy_20d_cem_kde_optim_off_co0p4_ns25 \
#data/levy_20d_gacem_co0p4_ns5_lay3x200_e5_b10_nr20 \
#--save-path experiments_results/levy_20d \
#-l cem cempp_sg cempp_kde gacem \
#-g 0.4

# ---------- styblinski 2D
# CPU compatible
#./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_2d_cem_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_2d_cempp_gauss_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_2d_cempp_kde_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_2d_gacem_dim_study.yaml -ns 5

# plot figures
#./run.sh scripts/plot_experiments.py \
#data/styblinski_2d_cem_gauss_optim_on_co0p4_ns25 \
#data/styblinski_2d_cem_gauss_optim_off_co0p4_ns25 \
#data/styblinski_2d_cem_kde_optim_off_co0p4_ns25 \
#data/styblinski_2d_gacem_co0p4_ns5_lay3x20_e5_b10_nr20 \
#--save-path experiments_results/styblinski_2d \
#-l cem cempp_sg cempp_kde gacem \
#-g 20

# ---------- styblinski 5D
# CPU compatible
#./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_5d_cem_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_5d_cempp_gauss_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_5d_cempp_kde_dim_study.yaml -ns 5
## GPU only (otherwise will take forever)
#./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_5d_gacem_dim_study.yaml -ns 5

# plot figures
#./run.sh scripts/plot_experiments.py \
#data/styblinski_5d_cem_gauss_optim_on_co0p4_ns25 \
#data/styblinski_5d_cem_gauss_optim_off_co0p4_ns25 \
#data/styblinski_5d_cem_kde_optim_off_co0p4_ns25 \
#data/styblinski_5d_gacem_co0p4_ns5_lay3x20_e5_b10_nr20 \
#--save-path experiments_results/styblinski_5d \
#-l cem cempp_sg cempp_kde gacem \
#-g 20

# ---------- styblinski 20D
# CPU compatible
#./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_20d_cem_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_20d_cempp_gauss_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_20d_cempp_kde_dim_study.yaml -ns 5
# GPU only (otherwise will take forever)
#./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_20d_gacem_dim_study.yaml -ns 5

# plot figures
#./run.sh scripts/plot_experiments.py \
#data/styblinski_20d_cem_gauss_optim_on_co0p4_ns25 \
#data/styblinski_20d_cem_gauss_optim_off_co0p4_ns25 \
#data/styblinski_20d_cem_kde_optim_off_co0p4_ns25 \
#data/styblinski_20d_gacem_co0p4_ns5_lay3x200_e5_b10_nr20 \
#--save-path experiments_results/styblinski_20d \
#-l cem cempp_sg cempp_kde gacem \
#-g 20


# ---------- ackley 2D (runs on CPU in reasonable time)
#./run.sh scripts/run_experiments.py spec_files/icml_paper/ackley_dim/ackley_2d_cem_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/ackley_dim/ackley_2d_cempp_gauss_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/ackley_dim/ackley_2d_cempp_kde_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/ackley_dim/ackley_2d_gacem_dim_study.yaml -ns 5

# plot figures
#./run.sh scripts/plot_experiments.py \
#data/ackley_2d_cem_gauss_optim_on_co0p4_ns25 \
#data/ackley_2d_cem_gauss_optim_off_co0p4_ns25 \
#data/ackley_2d_cem_kde_optim_off_co0p4_ns25 \
#data/ackley_2d_gacem_co0p4_ns5_lay3x20_e5_b10_nr20 \
#--save-path experiments_results/ackley_2d \
#-l cem cempp_sg cempp_kde gacem \
#-g 3.5

# ---------- ackley 5D
# CPU compatible
#./run.sh scripts/run_experiments.py spec_files/icml_paper/ackley_dim/ackley_5d_cem_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/ackley_dim/ackley_5d_cempp_gauss_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/ackley_dim/ackley_5d_cempp_kde_dim_study.yaml -ns 5
# GPU only (otherwise will take forever)
#./run.sh scripts/run_experiments.py spec_files/icml_paper/ackley_dim/ackley_5d_gacem_dim_study.yaml -ns 5

# plot figures
#./run.sh scripts/plot_experiments.py \
#data/ackley_5d_cem_gauss_optim_on_co0p4_ns25 \
#data/ackley_5d_cem_gauss_optim_off_co0p4_ns25 \
#data/ackley_5d_cem_kde_optim_off_co0p4_ns25 \
#data/ackley_5d_gacem_co0p4_ns5_lay3x20_e5_b10_nr20 \
#--save-path experiments_results/ackley_5d \
#-l cem cempp_sg cempp_kde gacem \
#-g 3.5

# ---------- ackley 20D
# CPU compatible
#./run.sh scripts/run_experiments.py spec_files/icml_paper/ackley_dim/ackley_20d_cem_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/ackley_dim/ackley_20d_cempp_gauss_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/ackley_dim/ackley_20d_cempp_kde_dim_study.yaml -ns 5
# GPU only (otherwise will take forever)
#./run.sh scripts/run_experiments.py spec_files/icml_paper/ackley_dim/ackley_20d_gacem_dim_study.yaml -ns 5

# plot figures
#./run.sh scripts/plot_experiments.py \
#data/ackley_20d_cem_gauss_optim_on_co0p4_ns25 \
#data/ackley_20d_cem_gauss_optim_off_co0p4_ns25 \
#data/ackley_20d_cem_kde_optim_off_co0p4_ns25 \
#data/ackley_20d_gacem_co0p4_ns5_lay3x200_e5_b10_nr20 \
#--save-path experiments_results/ackley_20d \
#-l cem cempp_sg cempp_kde gacem \
#-g 3.5


# ---------- Plot clusters
## ackley
#./run.sh scripts/compare_solutions.py spec_files/icml_paper/cluster_plots/ackley_2d.yaml
#./run.sh scripts/compare_solutions.py spec_files/icml_paper/cluster_plots/ackley_5d.yaml
#./run.sh scripts/compare_solutions.py spec_files/icml_paper/cluster_plots/ackley_20d.yaml
#
## styblinski
#./run.sh scripts/compare_solutions.py spec_files/icml_paper/cluster_plots/styblinski_2d.yaml
#./run.sh scripts/compare_solutions.py spec_files/icml_paper/cluster_plots/styblinski_5d.yaml
#./run.sh scripts/compare_solutions.py spec_files/icml_paper/cluster_plots/styblinski_20d.yaml
#
## levy
#./run.sh scripts/compare_solutions.py spec_files/icml_paper/cluster_plots/levy_2d.yaml
#./run.sh scripts/compare_solutions.py spec_files/icml_paper/cluster_plots/levy_5d.yaml
#./run.sh scripts/compare_solutions.py spec_files/icml_paper/cluster_plots/levy_20d.yaml
