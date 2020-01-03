
The following algorithms are implemented:

* Continuous domain adaptive sampling with Autoregressive policies
* Cross Entropy Method (Single Gaussian (SG), KDE (Kernel Desnsity Estimation))

CEM has different variants:
Optimization vs. Constraint satisfaction
With buffer filter or without buffer filter

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

