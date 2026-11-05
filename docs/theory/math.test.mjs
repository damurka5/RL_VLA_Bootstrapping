import assert from 'node:assert/strict';
import { mixProbability, groupProbabilities, populationObjective, finiteCoefficient,
  finiteObjective, composition, compensationFloor, trustBound, evidence } from './math.mjs';
const close = (a, b, tolerance = 1e-10) => assert.ok(Math.abs(a - b) < tolerance, `${a} != ${b}`);
// Enumerate all binary groups and compute E[mean(A_i * score_i)] directly
// for the Bernoulli policy theta=p. Its dp/dtheta=1, so the result is c_G(p).
function enumeratedGradient(p, G, epsilon) {
  let total = 0;
  for (let bits = 0; bits < 2 ** G; bits++) {
    const ys = Array.from({ length: G }, (_, i) => (bits >> i) & 1);
    const n = ys.reduce((a, b) => a + b, 0), mean = n / G;
    const std = Math.max(Math.sqrt(mean * (1 - mean)), epsilon);
    const g = ys.reduce((sum, y) => sum + (y - mean) / std * (y ? 1 / p : -1 / (1 - p)), 0) / G;
    total += p ** n * (1 - p) ** (G - n) * g;
  }
  return total;
}
for (const G of [2, 4, 8]) {
  for (const p of [0.001, 0.03, 0.16, 0.25, 0.5, 0.9, 0.999]) {
    close(groupProbabilities(p, G).reduce((a, b) => a + b, 0), 1);
    for (const eps of [1e-6, 0.6]) close(finiteCoefficient(p, G, eps), enumeratedGradient(p, G, eps));
    close(finiteCoefficient(p, G), finiteCoefficient(1 - p, G));
    const h = 1e-6;
    close((finiteObjective(p + h, G) - finiteObjective(p - h, G)) / (2 * h), finiteCoefficient(p, G), 1e-7);
  }
  close(finiteCoefficient(0, G), Math.sqrt(G - 1));
  close(finiteCoefficient(1, G), Math.sqrt(G - 1));
}
close(mixProbability(0), 0); close(mixProbability(1), 0);
close(mixProbability(0.5), 254 / 256);
close(finiteCoefficient(0.37, 2), 1);
// The two empirical points reproduce the exact strict-success denominators.
for (const point of [evidence.before, evidence.after]) {
  close(composition(point.lift / evidence.episodes, point.strict / point.lift), point.strict / evidence.episodes);
}
const baseline = evidence.before.strict / evidence.episodes;
const finish = evidence.after.strict / evidence.after.lift;
assert.ok(composition(0.81, finish) > baseline);
assert.ok(composition(0.70, finish) < baseline);
close(composition(compensationFloor(baseline, finish), finish), baseline);
assert.equal(compensationFloor(baseline, 0), Infinity);
close(composition(0, finish), 0);
close(trustBound(baseline, 0.02, 128, 0.009).penalty, 73.152);
close(trustBound(baseline, 0.02, 128, 0).lower, baseline + 0.02);
// A higher transformed mean need not imply a higher ordinary mean.
assert.ok(populationObjective(0.02) + populationObjective(0.485) > populationObjective(0.01) + populationObjective(0.5));
assert.ok((0.02 + 0.485) / 2 < (0.01 + 0.5) / 2);
console.log('Theory math checks passed: exact group enumeration, limits, derivatives, evidence counts, compensation, KL penalty.');
