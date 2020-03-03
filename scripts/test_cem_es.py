import cma
import numpy as np

from bb_eval_engine.util.importlib import import_bb_env
from utils.pdb import register_pdb_hook
from bb_eval_engine.data.design import Design
register_pdb_hook()

bb_env = import_bb_env('bb_envs/src/benchmark_functions/envs/synt_20d_env1.yaml')


vecs = list(bb_env.params_vec.values())
dim = len(vecs)
params_min = np.array([0] * dim)
params_max = np.array([len(x) - 1 for x in vecs])

start = (params_min + params_max) / 2
std_init = np.max((params_max - params_min) / 2 / 3)
opts = dict(seed=10, bounds=[params_min, params_max], CMA_active=False,
            popsize=100,
            CMA_mu=30,
            termination_callback=lambda *args: False,)
es = cma.CMAEvolutionStrategy(start, std_init, opts)

def ff(x): # one instance shape = D
    lo = np.zeros(x.shape) + params_min
    hi = np.zeros(x.shape) + params_max
    samples = np.clip(x, lo, hi)
    samples = np.floor(samples).astype('int')
    xnew = bb_env.evaluate([Design(samples)])[0]
    return xnew

niter = 0
while niter < 100:
    samples = es.ask()
    sample_dsns = [ff(x) for x in samples]
    values = np.array([x['val'] for x in sample_dsns])

    indx = np.argsort(values)
    samples_np = np.array(samples)[indx][:es.sp.weights.mu]
    values_np = np.array(values)[indx][:es.sp.weights.mu]

    ndata = len(samples_np)
    if ndata < es.popsize:
        nextra_rows = es.popsize - ndata
        extra_rows = np.zeros((nextra_rows, samples_np.shape[-1]))
        data = np.concatenate([samples_np, extra_rows], axis=0)


    es.tell(data, np.arange(len(data)), copy=True)
    # es.tell(samples_np, values_np, copy=True)
    es.logger.add()
    es.disp()
    niter += 1

es.logger.plot()
print(sample_dsns[-1], sample_dsns[-1].specs)
cma.s.figshow()
breakpoint()
