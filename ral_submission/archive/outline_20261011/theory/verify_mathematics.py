"""Independent finite-model checks; no robot training or theorem certification.
Run from any directory: python3 /path/to/ral_submission/theory/verify_mathematics.py
Uses only Python's standard library. Writes verification.json beside this file.
"""
from itertools import product
from math import comb, erf, exp, pi, sqrt
from pathlib import Path
import json


def normal_pdf(x):
    return exp(-0.5 * x * x) / sqrt(2 * pi)


def normal_cdf(x):
    return (1 + erf(x / sqrt(2))) / 2


def mixed(p, g):
    return 1 - p**g - (1-p)**g


def group_coefficient(p, g, eps=1e-6):
    total = 0.0
    for k in range(g+1):
        q = k/g
        total += (comb(g,k)*p**k*(1-p)**(g-k)
                  *q*(1-q)/max(sqrt(q*(1-q)),eps))
    return total/(p*(1-p))


def integrate(f, lo, hi, n=12000):
    assert n % 2 == 0
    step = (hi-lo)/n
    return step/3*(f(lo)+f(hi)+sum((4 if i%2 else 2)*f(lo+i*step)
                                 for i in range(1,n)))


checks = {}
max_error = 0.0
for g in [2,4,8]:
    for p in [.01,.03,.1,.3,.5,.8,.99]:
        expected_normalized = expected_centered = expected_loo = 0.0
        for ys in product([0,1], repeat=g):
            q = sum(ys)/g
            prob = p**sum(ys)*(1-p)**(g-sum(ys))
            z = [y-p for y in ys]  # Bernoulli-logit score
            sd = max(sqrt(q*(1-q)),1e-6)
            expected_normalized += prob*sum((y-q)/sd*s for y,s in zip(ys,z))/g
            expected_centered += prob*sum((y-q)*s for y,s in zip(ys,z))/g
            expected_loo += prob*sum((y-(sum(ys)-y)/(g-1))*s for y,s in zip(ys,z))/g
        grad = p*(1-p)
        errors = [abs(expected_normalized-group_coefficient(p,g)*grad),
                  abs(expected_centered-(g-1)/g*grad),abs(expected_loo-grad)]
        max_error = max(max_error,*errors)
assert max_error < 1e-12
checks['binary_group_enumeration'] = {'passed':True,'max_absolute_error':max_error,
                                    'group_sizes':[2,4,8],'probabilities_per_size':7}
checks['mixed_group_examples'] = {'G':8,'p_0.03':mixed(.03,8),'p_0.30':mixed(.30,8)}

# Each outcome is a complete sequential trajectory: failure at gates 1/2/3,
# or full success. Enumerate every possible group, not just random samples.
returns = [(0,0,0),(1,0,0),(1,1,0),(1,1,1)]
visited = [(True,False,False),(True,True,False),(True,True,True),(True,True,True)]
groups_checked = 0
minimum_gain = float('inf')
for g in [2,3,4]:
    for outcomes in product(range(4),repeat=g):
        ms = [returns[i] for i in outcomes]
        advantages = []
        for k in range(3):
            q = sum(m[k] for m in ms)/g
            sd = max(sqrt(q*(1-q)),1e-6)
            advantages.append([(m[k]-q)/sd for m in ms])
        counts = [sum(visited[i][k] for i in outcomes) for k in range(3)]
        represented = sum(n>0 for n in counts)
        for qs in [(.2,.4,.7),(.8,.3,.1),(.9,.9,.9)]:
            direction = [0.,0.,0.]
            for i,outcome in enumerate(outcomes):
                for k in range(3):
                    if not visited[outcome][k]:
                        continue
                    credit = advantages[k][i]
                    if k<2:
                        bonus = advantages[2][i]
                        if ms[i][k] and bonus<0:
                            bonus = 0.
                        credit += bonus
                    contribution = credit*(ms[i][k]-qs[k])
                    assert contribution >= -1e-12
                    direction[k] += contribution/(represented*counts[k])
            before = qs[0]*qs[1]*qs[2]
            for eta in [.01,1.,10.]:
                new = [q/(q+(1-q)*exp(-eta*d)) for q,d in zip(qs,direction)]
                gain = new[0]*new[1]*new[2]-before
                minimum_gain = min(minimum_gain,gain)
                assert gain >= -1e-12
        groups_checked += 1
checks['protected_sequential_gate_model'] = {'passed':True,'enumerated_groups':groups_checked,
    'parameter_vectors':3,'step_sizes':[.01,1.,10.],'minimum_success_gain':minimum_gain}

# Correct latent scoring with clipped physical actions, single decision.
# Outcome Y=1[clip(U,-1,1)>.5]. Its expected derivative is known analytically.
clipping_results=[]
for mu,sigma in [(-.4,.7),(.2,.9),(.8,.6)]:
    density=lambda u: normal_pdf((u-mu)/sigma)/sigma
    exact=normal_pdf((.5-mu)/sigma)/sigma
    latent=integrate(lambda u:density(u)*(u-mu)/sigma**2,.5,mu+12*sigma)
    clipped=(integrate(lambda u:density(u)*(u-mu)/sigma**2,.5,1.)
             +(1-mu)*(1-normal_cdf((1-mu)/sigma))/sigma**2)
    assert abs(latent-exact)<1e-8
    clipping_results.append({'mean':mu,'sigma':sigma,'exact_gradient':exact,
                             'latent_score_integral':latent,'clipped_sample_gaussian_score':clipped})
checks['clipped_action_score']={'passed':True,'cases':clipping_results}

# Two-decision process with B shared over both decisions:
# U1=theta+B+noise; h2=1[clip(U1)>0]; U2=theta+.4*h2+B+noise.
# Y=1[U1>0 and U2>0]. Compare the correct conditional score with the exact
# one-step mixture marginal scored independently at each time.
a=.7; sigma=.65; shift=.4

def success(theta):
    return sum(.5*normal_cdf((theta+b)/sigma)*normal_cdf((theta+shift+b)/sigma)
               for b in [-a,a])

def marginal_score(u,theta,offset):
    means=[theta+offset-a,theta+offset+a]
    ds=[normal_pdf((u-m)/sigma)/sigma for m in means]
    return sum(d*(u-m)/sigma**2 for d,m in zip(ds,means))/sum(ds)

persistent=[]
for theta in [-.3,.1,.5]:
    exact=conditional=independent=0.
    for b in [-a,a]:
        m1=theta+b; m2=theta+shift+b
        p1=normal_cdf(m1/sigma); p2=normal_cdf(m2/sigma)
        exact+=.5*(normal_pdf(m1/sigma)/sigma*p2+p1*normal_pdf(m2/sigma)/sigma)
        s1=integrate(lambda u:normal_pdf((u-m1)/sigma)/sigma*(u-m1)/sigma**2,0,9)
        s2=integrate(lambda u:normal_pdf((u-m2)/sigma)/sigma*(u-m2)/sigma**2,0,9)
        conditional+=.5*(s1*p2+p1*s2)
        i1=integrate(lambda u:normal_pdf((u-m1)/sigma)/sigma*marginal_score(u,theta,0),0,9)
        i2=integrate(lambda u:normal_pdf((u-m2)/sigma)/sigma*marginal_score(u,theta,shift),0,9)
        independent+=.5*(i1*p2+p1*i2)
    fd=(success(theta+1e-5)-success(theta-1e-5))/2e-5
    assert abs(exact-conditional)<1e-8 and abs(exact-fd)<1e-8
    persistent.append({'theta':theta,'exact_gradient':exact,'finite_difference':fd,
                       'conditional_score_integral':conditional,'independent_exact_marginal_score':independent})
assert max(abs(v['independent_exact_marginal_score']-v['exact_gradient']) for v in persistent)>.01
checks['persistent_offset_score']={'passed':True,'cases':persistent}

# Shared scalar parameters can reverse arithmetic-mean ascent under scene
# normalization, even though each scene's individual coefficient is positive.
gradients=[-.8,1.]; ps=[.01,.5]
grad_j=sum(gradients)/2
d=sum(group_coefficient(p,8)*g for p,g in zip(ps,gradients))/2
assert grad_j*d<0
checks['shared_parameter_counterexample']={'passed':True,'scene_probabilities':ps,
    'scene_success_derivatives':gradients,'mean_success_derivative':grad_j,
    'expected_normalized_direction':d,'directional_alignment':grad_j*d}
checks['prefix_tradeoff']={'passed':True,'before':.8*.8*.5,'after':.8*.5*.7}
assert checks['prefix_tradeoff']['after']<checks['prefix_tradeoff']['before']

out={'scope':'Finite-model algebra checks, not CDPR policy verification',
     'checks':checks,'all_passed':True}
Path(__file__).with_name('verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
