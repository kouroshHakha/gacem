#!/usr/bin/env bash
# uncomment the sections you want to run

# ---------- levy 2D (runs on CPU in reasonable time)
#./run.sh scripts/run_experiments.py spec_files/icml_paper/levy_dim/levy_2d_cem_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/levy_dim/levy_2d_cem_sg_fixed_sigma_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/levy_dim/levy_2d_cempp_kde_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/levy_dim/levy_2d_gacem_onp_fixed_sigma_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/levy_dim/levy_2d_gacem_off_fixed_sigma_dim_study.yaml -ns 5

# plot figures
#./run.sh scripts/plot_experiments.py \
#data/levy_2d_cem_gauss_optim_on_co0p4_ns25 \
#data/levy_2d_cem_gauss_optim_on_co0p4_ns25_sig10 \
#data/levy_2d_cem_gauss_optim_off_co0p4_ns25 \
#data/levy_2d_cem_kde_optim_off_co0p4_ns25 \
#data/levy_2d_gacem_ns5_lay3x100_e10_b10_nr40_csp_off_fixed_sigma \
#data/levy_2d_gacem_ns25_lay3x100_e10_b10_nr40_csp_onp_fixed_sigma \
#--save-path experiments_results/levy_2d \
#-l cem_adaptive_variance cem_fixed_variance cem++sg cem++kde gacem_off gacem_onp \
#-g 0.4

# ---------- levy 5D
# CPU compatible
#./run.sh scripts/run_experiments.py spec_files/icml_paper/levy_dim/levy_5d_cem_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/levy_dim/levy_5d_cem_sg_fixed_sigma_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/levy_dim/levy_5d_cempp_gauss_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/levy_dim/levy_5d_cempp_kde_dim_study.yaml -ns 5
# GPU only (otherwise will take forever)
#./run.sh scripts/run_experiments.py spec_files/icml_paper/levy_dim/levy_5d_gacem_onp_fixed_sigma_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/levy_dim/levy_5d_gacem_off_fixed_sigma_dim_study.yaml -ns 5

# plot figures
#./run.sh scripts/plot_experiments.py \
#data/levy_5d_cem_gauss_optim_on_co0p4_ns25 \
#data/levy_5d_cem_gauss_optim_on_co0p4_ns25_sig10 \
#data/levy_5d_cem_gauss_optim_off_co0p4_ns25 \
#data/levy_5d_cem_kde_optim_off_co0p4_ns25 \
#data/levy_5d_gacem_ns5_lay3x100_e10_b10_nr40_csp_off_fixed_sigma \
#data/levy_5d_gacem_ns25_lay3x100_e10_b10_nr40_csp_onp_fixed_sigma \
#--save-path experiments_results/levy_5d \
#-l cem_adaptive_variance cem_fixed_variance cem++sg cem++kde gacem_off gacem_onp \
#-g 0.4

# ---------- levy 20D
# CPU compatible
#./run.sh scripts/run_experiments.py spec_files/icml_paper/levy_dim/levy_20d_cem_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/levy_dim/levy_20d_cem_sg_fixed_sigma_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/levy_dim/levy_20d_cempp_gauss_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/levy_dim/levy_20d_cempp_kde_dim_study.yaml -ns 5
# GPU only (otherwise will take forever)
#./run.sh scripts/run_experiments.py spec_files/icml_paper/levy_dim/levy_20d_gacem_off_fixed_sigma_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/levy_dim/levy_20d_gacem_onp_fixed_sigma_dim_study.yaml -ns 5

# plot figures
#./run.sh scripts/plot_experiments.py \
#data/levy_20d_cem_gauss_optim_on_co0p4_ns25 \
#data/levy_20d_cem_gauss_optim_on_co0p4_ns25_sig10 \
#data/levy_20d_cem_gauss_optim_off_co0p4_ns25 \
#data/levy_20d_cem_kde_optim_off_co0p4_ns25 \
#data/levy_20d_cem_gauss_optim_on_co0p4_ns25_sig10 \
#data/levy_20d_gacem_ns5_lay3x100_e10_b10_nr40_csp_off_fixed_sigma \
#data/levy_20d_gacem_ns25_lay3x100_e10_b10_nr40_csp_onp_fixed_sigma \
#--save-path experiments_results/levy_20d \
#-l cem_adaptive_variance cem_fixed_variance cem++sg cem++kde gacem_off gacem_onp \
#-g 8

# ---------- styblinski 2D
# CPU compatible
#./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_2d_cem_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_2d_cem_sg_fixed_sigma_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_2d_cempp_gauss_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_2d_cempp_kde_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_2d_gacem_off_fixed_sigma_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_2d_gacem_onp_fixed_sigma_dim_study.yaml -ns 5

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

# ---------- styblinski 5D
# CPU compatible
#./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_5d_cem_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_5d_cem_sg_fixed_sigma_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_5d_cempp_gauss_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_5d_cempp_kde_dim_study.yaml -ns 5
## GPU only (otherwise will take forever)
#./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_5d_gacem_off_fixed_sigma_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_5d_gacem_onp_fixed_sigma_dim_study.yaml -ns 5

# plot figures
#./run.sh scripts/plot_experiments.py \
#data/styblinski_5d_cem_gauss_optim_on_co0p4_ns25 \
#data/styblinski_5d_cem_gauss_optim_on_co0p4_ns25_sig10 \
#data/styblinski_5d_cem_gauss_optim_off_co0p4_ns25 \
#data/styblinski_5d_cem_kde_optim_off_co0p4_ns25 \
#data/styblinski_5d_gacem_ns25_lay3x100_e10_b10_nr40_csp_onp_fixed_sigma \
#data/styblinski_5d_gacem_ns5_lay3x100_e10_b10_nr40_csp_off_fixed_sigma \
#--save-path experiments_results/styblinski_5d \
#-l cem_adaptive_variance cem_fixed_variance cem++sg cem++kde gacem_off gacem_onp \
#-g 5

# ---------- styblinski 20D
# CPU compatible
#./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_20d_cem_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_20d_cem_sg_fixed_sigma_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_20d_cempp_gauss_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_20d_cempp_kde_dim_study.yaml -ns 5
# GPU only (otherwise will take forever)
#./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_20d_gacem_off_fixed_sigma_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/styblinski_dim/styblinski_20d_gacem_onp_fixed_sigma_dim_study.yaml -ns 5

# plot figures
#./run.sh scripts/plot_experiments.py \
#data/styblinski_20d_cem_gauss_optim_on_co0p4_ns25 \
#data/styblinski_20d_cem_gauss_optim_on_co0p4_ns25_sig10 \
#data/styblinski_20d_cem_gauss_optim_off_co0p4_ns25 \
#data/styblinski_20d_cem_kde_optim_off_co0p4_ns25 \
#data/styblinski_20d_gacem_ns5_lay3x100_e10_b10_nr40_csp_off_fixed_sigma \
#data/styblinski_20d_gacem_ns25_lay3x100_e10_b10_nr40_csp_onp_fixed_sigma \
#--save-path experiments_results/styblinski_20d \
#-l cem_adaptive_variance cem_fixed_variance cem++sg cem++kde gacem_off gacem_onp \
#-g 10


# ---------- ackley 2D (runs on CPU in reasonable time)
#./run.sh scripts/run_experiments.py spec_files/icml_paper/ackley_dim/ackley_2d_cem_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/ackley_dim/ackley_2d_cem_sg_fixed_sigma_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/ackley_dim/ackley_2d_cempp_gauss_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/ackley_dim/ackley_2d_cempp_kde_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/ackley_dim/ackley_2d_gacem_off_fixed_sigma_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/ackley_dim/ackley_2d_gacem_onp_fixed_sigma_dim_study.yaml -ns 5

## plot figures
#./run.sh scripts/plot_experiments.py \
#data/ackley_2d_cem_gauss_optim_on_co0p4_ns25 \
#data/ackley_2d_cem_gauss_optim_on_co0p4_ns25_sig10 \
#data/ackley_2d_cem_gauss_optim_off_co0p4_ns25 \
#data/ackley_2d_cem_kde_optim_off_co0p4_ns25 \
#data/ackley_2d_gacem_ns5_lay3x100_e10_b10_nr40_csp_off_fixed_sigma \
#data/ackley_2d_gacem_ns25_lay3x100_e10_b10_nr40_csp_onp_fixed_sigma \
#--save-path experiments_results/ackley_2d \
#-l cem_adaptive_variance cem_fixed_variance cem++sg cem++kde gacem_off gacem_onp \
#-g 3.5

# ---------- ackley 5D
# CPU compatible
#./run.sh scripts/run_experiments.py spec_files/icml_paper/ackley_dim/ackley_5d_cem_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/ackley_dim/ackley_5d_cem_sg_fixed_sigma_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/ackley_dim/ackley_5d_cempp_gauss_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/ackley_dim/ackley_5d_cempp_kde_dim_study.yaml -ns 5
# GPU only (otherwise will take forever)
#./run.sh scripts/run_experiments.py spec_files/icml_paper/ackley_dim/ackley_5d_gacem_off_fixed_sigma_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/ackley_dim/ackley_5d_gacem_onp_fixed_sigma_dim_study.yaml -ns 5

# plot figures
#./run.sh scripts/plot_experiments.py \
#data/ackley_5d_cem_gauss_optim_on_co0p4_ns25 \
#data/ackley_5d_cem_gauss_optim_on_co0p4_ns25_sig10 \
#data/ackley_5d_cem_gauss_optim_off_co0p4_ns25 \
#data/ackley_5d_cem_kde_optim_off_co0p4_ns25 \
#data/ackley_5d_gacem_ns5_lay3x100_e10_b10_nr40_csp_off_fixed_sigma \
#data/ackley_5d_gacem_ns25_lay3x100_e10_b10_nr40_csp_onp_fixed_sigma \
#--save-path experiments_results/ackley_5d \
#-l cem_adaptive_variance cem_fixed_variance cem++sg cem++kde gacem_off gacem_onp \
#-g 1.5

# ---------- ackley 20D
# CPU compatible
#./run.sh scripts/run_experiments.py spec_files/icml_paper/ackley_dim/ackley_20d_cem_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/ackley_dim/ackley_20d_cem_sg_fixed_sigma_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/ackley_dim/ackley_20d_cempp_gauss_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/ackley_dim/ackley_20d_cempp_kde_dim_study.yaml -ns 5
# GPU only (otherwise will take forever)
#./run.sh scripts/run_experiments.py spec_files/icml_paper/ackley_dim/ackley_20d_gacem_off_fixed_sigma_dim_study.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/ackley_dim/ackley_20d_gacem_onp_fixed_sigma_dim_study.yaml -ns 5

# plot figures
#./run.sh scripts/plot_experiments.py \
#data/ackley_20d_cem_gauss_optim_on_co0p4_ns25 \
#data/ackley_20d_cem_gauss_optim_on_co0p4_ns25_sig10 \
#data/ackley_20d_cem_gauss_optim_off_co0p4_ns25 \
#data/ackley_20d_cem_kde_optim_off_co0p4_ns25 \
#data/ackley_20d_gacem_ns5_lay3x100_e10_b10_nr40_csp_off_fixed_sigma \
#data/ackley_20d_gacem_ns25_lay3x100_e10_b10_nr40_csp_onp_fixed_sigma \
#--save-path experiments_results/ackley_20d \
#-l cem_adaptive_variance cem_fixed_variance cem++sg cem++kde gacem_off gacem_onp \
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

# ----------- Plot mode discovery
# synt 2D
#./run.sh scripts/run_experiments.py spec_files/icml_paper/synt_mode_discovery/synt_2d_cem_fixed_sigma.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/synt_mode_discovery/synt_2d_gacem_off.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/synt_mode_discovery/synt_2d_gacem_off_fixed_sigma.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/synt_mode_discovery/synt_2d_gacem_onp.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/synt_mode_discovery/synt_2d_gacem_onp_fixed_sigma.yaml -ns 5

# plot figures
#./run.sh scripts/plot_experiments.py \
#data/synt_2d_cem_gauss_optim_on_co0p4_ns25_sig10 \
#data/synt_2d_gacem_ns25_lay2x100_e10_b10_nr20_csp_off_fixed_sigma \
#data/synt_2d_gacem_ns25_lay2x100_e10_b10_nr20_csp_onp_fixed_sigma \
#--save-path experiments_results/synt_2d \
#-l cem gacem_off gacem_onp \
#-g 2

# plot clusters
#./run.sh scripts/compare_solutions.py spec_files/icml_paper/synt_mode_discovery/clusters_2d.yaml

# synt 5D
#./run.sh scripts/run_experiments.py spec_files/icml_paper/synt_mode_discovery/synt_5d_cem_fixed_sigma.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/synt_mode_discovery/synt_5d_gacem_off.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/synt_mode_discovery/synt_5d_gacem_off_fixed_sigma.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/synt_mode_discovery/synt_5d_gacem_onp.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/synt_mode_discovery/synt_5d_gacem_onp_fixed_sigma.yaml -ns 5

# plot figures
#./run.sh scripts/plot_experiments.py \
#data/synt_5d_cem_gauss_optim_on_co0p4_ns25_sig10 \
#data/synt_5d_gacem_ns25_lay2x100_e10_b10_nr20_csp_off_fixed_sigma \
#data/synt_5d_gacem_ns25_lay2x100_e10_b10_nr20_csp_onp_fixed_sigma \
#--save-path experiments_results/synt_5d \
#-l cem gacem_off gacem_onp \
#-g 2
#data/synt_5d_gacem_ns25_lay2x100_e10_b10_nr20_csp_off \
#data/synt_5d_gacem_ns25_lay2x100_e10_b10_nr20_csp_onp \

# plot clusters
#./run.sh scripts/compare_solutions.py spec_files/icml_paper/synt_mode_discovery/clusters_5d.yaml

# synt 3D
#./run.sh scripts/run_experiments.py spec_files/icml_paper/synt_mode_discovery/synt_3d_cem_fixed_sigma.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/synt_mode_discovery/synt_3d_gacem_off.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/synt_mode_discovery/synt_3d_gacem_off_fixed_sigma.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/synt_mode_discovery/synt_3d_gacem_onp.yaml -ns 5
#./run.sh scripts/run_experiments.py spec_files/icml_paper/synt_mode_discovery/synt_3d_gacem_onp_fixed_sigma.yaml -ns 5

# plot figures
#./run.sh scripts/plot_experiments.py \
#data/synt_3d_cem_gauss_optim_on_co0p4_ns25_sig10 \
#data/synt_3d_gacem_ns25_lay2x100_e10_b10_nr20_csp_off_fixed_sigma \
#data/synt_3d_gacem_ns25_lay2x100_e10_b10_nr20_csp_onp_fixed_sigma \
#--save-path experiments_results/synt_3d \
#-l cem_fixed gacem_off gacem_onp \
#-g 2

# plot clusters
#./run.sh scripts/compare_solutions.py spec_files/icml_paper/synt_mode_discovery/clusters_3d.yaml
