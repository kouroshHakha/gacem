#!/usr/bin/env bash


#python plt_cost.py -ckpt \
#       data/search_fig_20191119155016/checkpoint.tar \
#       data/search_fig_20191119162020/checkpoint.tar \
#       data/search_fig_20191119162621/checkpoint.tar \
#       data/search_fig_20191119162642/checkpoint.tar \
#       data/search_fig_20191119162714/checkpoint.tar \
#       -l 2d 3d 4d 5d 10d \
#       -f dimension_effect
#
#python plt_cost.py -ckpt \
#       data/search_fig_20191120163005/checkpoint.tar \
#       data/search_fig_20191120173838/checkpoint.tar \
#       data/search_fig_20191120175028/checkpoint.tar \
#       -l 5D_beta1_goal4 5D_beta0p2_goal4 5D_beta0p2_goal3 \
#       -f 5D_beta_goal_effect

#
#python plt_cost.py -ckpt \
#       data/search_fig_20191121102409/checkpoint.tar \
#       data/search_fig_20191121104010/checkpoint.tar \
#       data/search_fig_20191121104032/checkpoint.tar \
#       -l 2D_cutoff_0p2 2D_cutoff_0p4 2D_cutoff_0p5 \
#       -f percentage_cutoff


python plt_cost.py -ckpt \
       data/search_fig_20191122171659_ackley_20d_nr_mix_1_b1/checkpoint.tar \
       data/search_fig_20191122201644_ackley_20d_nr_mix_5_b1/checkpoint.tar \
       data/search_fig_20191122201820_ackley_20d_nr_mix_5_b10/checkpoint.tar \
       data/search_fig_20191122201903_ackley_20d_nr_mix_1_b10/checkpoint.tar \
       -l nr_mix_1_b1 nr_mix_5_b1 nr_mix_5_b10 nr_mix_1_b10 \
       -f beta_nr_mix_effect


