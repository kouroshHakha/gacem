#!/usr/bin/env bash


python plt_cost.py -ckpt \
       data/search_fig_20191119155016/checkpoint.tar \
       data/search_fig_20191119162020/checkpoint.tar \
       data/search_fig_20191119162621/checkpoint.tar \
       data/search_fig_20191119162642/checkpoint.tar \
       data/search_fig_20191119162714/checkpoint.tar \
       -l 2d 3d 4d 5d 10d -f dimension_effect

python plt_cost.py -ckpt \
       data/search_fig_20191120163005/checkpoint.tar \
       data/search_fig_20191120173838/checkpoint.tar \
       data/search_fig_20191120175028/checkpoint.tar \
       -l 5D_beta1_goal4 5D_beta0p2_goal4 5D_beta0p2_goal3 \
       -f 5D_beta_goal_effect

