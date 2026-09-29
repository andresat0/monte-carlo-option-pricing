"""Regenerate three experiments and machine-readable results without network data."""
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pricing import black_scholes_call, plain_call, antithetic_call, merton_call_mc, merton_call_benchmark

OUT = Path(__file__).resolve().parent / 'results'
BASE = (100.0, 100.0, 0.05, 0.2, 1.0)
SEED = 20260928
REPLICATIONS = 200
BUDGETS = [1000, 4000, 16000, 64000]


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold',
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    benchmark = black_scholes_call(*BASE)
    seed_stream = np.random.SeedSequence(SEED)
    def rng():
        return np.random.default_rng(seed_stream.spawn(1)[0])
    records = []
    for n in BUDGETS:
        for name, method in [('Plain MC', plain_call), ('Antithetic MC', antithetic_call)]:
            estimates = [method(*BASE,n,rng()) for _ in range(REPLICATIONS)]
            prices = np.array([x.price for x in estimates])
            errors = prices-benchmark
            coverage = np.mean([lo <= benchmark <= hi for lo, hi in [x.confidence_interval for x in estimates]])
            records.append({'method':name,'payoff_evaluations':n,'replications':REPLICATIONS,
                            'rmse':float(np.sqrt(np.mean(errors**2))),
                            'empirical_variance':float(prices.var(ddof=1)),
                            'mean_reported_se':float(np.mean([x.standard_error for x in estimates])),
                            'interval_coverage':float(coverage)})
    fig, ax = plt.subplots(figsize=(7.6,4.8))
    for name, color in [('Plain MC','#2563eb'),('Antithetic MC','#0f766e')]:
        subset = [r for r in records if r['method']==name]
        ax.loglog(BUDGETS,[r['rmse'] for r in subset],'o-',label=name,color=color)
    first_rmse = records[0]['rmse']
    ax.loglog(BUDGETS,first_rmse*np.sqrt(BUDGETS[0]/np.array(BUDGETS)),
              '--',color='#94a3b8',label='N^-1/2 reference (scaled)')
    ax.set(title='Sampling error falls with simulation budget',xlabel='Payoff evaluations per estimate',ylabel='RMSE across 200 independent replications')
    ax.legend(frameon=False)
    ax.grid(alpha=.15,which='both')
    fig.tight_layout(); fig.savefig(OUT/'convergence.png',dpi=170); plt.close(fig)

    ratios = [records[i]['empirical_variance']/records[i+1]['empirical_variance'] for i in range(0,len(records),2)]
    fig, ax = plt.subplots(figsize=(7.6,4.8))
    bars = ax.bar([f'{n:,}' for n in BUDGETS],ratios,color='#0f766e',width=.55)
    ax.axhline(1,color='#94a3b8',linestyle='--')
    ax.bar_label(bars,fmt='%.2fx',padding=4)
    ax.set_ylim(0,max(ratios)*1.2)
    ax.set(title='Antithetic pairs reduce estimator variance',xlabel='Equal payoff-evaluation budget',ylabel='Empirical variance: plain / antithetic')
    fig.tight_layout(); fig.savefig(OUT/'variance_reduction.png',dpi=170); plt.close(fig)

    sensitivity = []
    fig, axes = plt.subplots(1,2,figsize=(10.2,4.4))
    for ax, sweep, values in [(axes[0],'intensity',[0,.25,.5,.75,1]),
                              (axes[1],'jump_std',[0,.1,.2,.3,.4])]:
        rows = []
        for value in values:
            intensity = value if sweep=='intensity' else .5
            jump_std = value if sweep=='jump_std' else .2
            estimate = merton_call_mc(*BASE,200_000,intensity,-.05,jump_std,rng())
            exact = merton_call_benchmark(*BASE,intensity,-.05,jump_std)
            row = {'sweep':sweep,'intensity':intensity,'jump_mean':-.05,'jump_std':jump_std,
                   'benchmark':exact,**asdict(estimate)}
            rows.append(row); sensitivity.append(row)
        ax.plot(values,[r['benchmark'] for r in rows],color='#1e3a8a',label='Poisson-mixture benchmark')
        ax.errorbar(values,[r['price'] for r in rows],yerr=[1.96*r['standard_error'] for r in rows],
                    fmt='o',capsize=4,color='#0f766e',label='MC with pointwise 95% interval')
        ax.set(xlabel='Jump intensity (per year)' if sweep=='intensity' else 'Log-jump standard deviation',
               ylabel='European-call value',title='Jump frequency' if sweep=='intensity' else 'Jump dispersion')
        ax.grid(alpha=.15)
    handles,labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=9)
    fig.suptitle('Jump assumptions change the model price',fontweight='bold')
    fig.tight_layout(rect=(0,.08,1,.93)); fig.savefig(OUT/'jump_sensitivity.png',dpi=170); plt.close(fig)
    examples = {name:asdict(method(*BASE,200_000,rng())) for name,method in [('plain',plain_call),('antithetic',antithetic_call)]}
    data = {'seed':SEED,'base':dict(zip(['spot','strike','rate','volatility','maturity'],BASE)),
            'black_scholes_benchmark':benchmark,'replicated_experiments':records,
            'variance_ratios':ratios,'jump_sensitivity':sensitivity,'example_estimates':examples,
            'environment':{'python':platform.python_version(),**{name:version(name) for name in ['numpy','scipy','matplotlib']}}}
    (OUT/'results.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({'benchmark':benchmark,'variance_ratios':ratios,'examples':examples},indent=2))


if __name__=='__main__':
    main()
