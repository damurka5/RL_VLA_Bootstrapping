"""Rebuild manuscript figures from retained evidence, without model execution."""
from pathlib import Path
import csv
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ral-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
rows = list(csv.DictReader((ROOT/'evidence/latent_control_40m_validation.csv').open()))
assert len(rows) == 81
x = [int(r['step'])/1e6 for r in rows]
y = [float(r['validation/success_rate'])*100 for r in rows]
fig, ax = plt.subplots(figsize=(3.5, 2.05), layout='constrained')
ax.plot(x, y, color='#17678a', linewidth=1, marker='.', markersize=2.8)
ax.scatter([x[0], x[-1]], [y[0], y[-1]], color='#17678a', s=18, zorder=3)
ax.annotate(f'{y[0]:.2f}%', (x[0], y[0]), xytext=(4, -13), textcoords='offset points', fontsize=7)
ax.annotate(f'{y[-1]:.2f}%', (x[-1], y[-1]), xytext=(-38, -16), textcoords='offset points', fontsize=7)
ax.set(xlabel='Cumulative selected actions (millions)', ylabel='Strict success (%)',
       xlim=(19.6, 40.8), ylim=(48, 68), xticks=[20,25,30,35,40], yticks=[50,55,60,65])
ax.grid(axis='y', alpha=.2)
fig.savefig(ROOT/'figures/cdpr_training.pdf')
fig.savefig(ROOT/'figures/cdpr_training.png', dpi=220)
plt.close(fig)
d = json.loads((ROOT/'evidence/latent_control_40m_comparison_20261019.json').read_text())
fig, axes = plt.subplots(1,2,figsize=(3.5,2.05), layout='constrained')
colors=['#8b98a5','#17678a']
for j,(key,label) in enumerate([('control20m','20M'),('final40m','40M')]):
 vals = [100*d['checkpoints'][key][k] for k in ['grasped','lifted','strict']]
 axes[0].bar([i+(j-.5)*.36 for i in range(3)],vals,width=.36,color=colors[j],label=label)
 q=d['comparisons']['final40m']['strict_by_target_y_quartile']
 vals=[100*q[f'Q{i}']['baseline' if j==0 else 'candidate'] for i in range(1,5)]
 axes[1].plot(range(1,5),vals,color=colors[j],marker='o',markersize=3,label=label)
axes[0].set(xticks=range(3),xticklabels=['Grasp','Lift','Strict'],ylabel='Attempts (%)',ylim=(0,100))
axes[0].tick_params(axis='x',labelsize=7)
axes[1].set(xticks=range(1,5),xticklabels=['Q1','Q2','Q3','Q4'],ylabel='Strict success (%)',ylim=(0,100))
axes[0].set_title('Task milestones',fontsize=8)
axes[1].set_title('Target-y quartile',fontsize=8)
axes[1].legend(frameon=False,fontsize=7,loc='upper right')
for ax in axes: ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.savefig(ROOT/'figures/cdpr_diagnostics.pdf')
fig.savefig(ROOT/'figures/cdpr_diagnostics.png',dpi=220)
plt.close(fig)
print('Generated two evidence-backed figures.')
