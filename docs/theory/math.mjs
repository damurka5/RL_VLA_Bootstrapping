// Pure functions shared by the teaching plots and numerical checks.
export function mixProbability(p, groupSize = 8) {
  return Math.max(0, 1 - p ** groupSize - (1 - p) ** groupSize);
}
export function choose(n, k) {
  let value = 1;
  for (let i = 1; i <= k; i++) value *= (n - i + 1) / i;
  return value;
}
export function groupProbabilities(p, groupSize = 8) {
  return Array.from({ length: groupSize + 1 }, (_, n) =>
    choose(groupSize, n) * p ** n * (1 - p) ** (groupSize - n));
}
export function populationWeight(p, epsilon = 1e-6) {
  return 1 / Math.max(Math.sqrt(p * (1 - p)), epsilon);
}
export function populationObjective(p) {
  return 2 * Math.asin(Math.sqrt(p));
}
// Algebraically cancel p(1-p) before evaluating. This also gives the
// continuous endpoint limits; an endpoint group still has zero advantage.
export function finiteCoefficient(p, groupSize = 8, epsilon = 1e-6) {
  let value = 0;
  for (let n = 1; n < groupSize; n++) {
    const v = n / groupSize;
    const variance = v * (1 - v);
    value += choose(groupSize, n) * p ** (n - 1) * (1 - p) ** (groupSize - n - 1)
      * variance / Math.max(Math.sqrt(variance), epsilon);
  }
  return value;
}
export function finiteObjective(p, groupSize = 8) {
  const segments = 256;
  const step = p / segments;
  let sum = finiteCoefficient(0, groupSize) + finiteCoefficient(p, groupSize);
  for (let i = 1; i < segments; i++) sum += (i % 2 ? 4 : 2) * finiteCoefficient(i * step, groupSize);
  return sum * step / 3;
}
export function composition(lift, finish) { return lift * finish; }
export function compensationFloor(baseline, finish) {
  return baseline === 0 ? 0 : finish === 0 ? Infinity : baseline / finish;
}
export function trustBound(success, localGain, horizon, maxKL) {
  const penalty = horizon * (horizon - 1) * maxKL / 2;
  return { penalty, raw: success + localGain - penalty,
    lower: Math.max(0, success + localGain - penalty) };
}
export const evidence = Object.freeze({
  // Exact counts from docs/artifacts/latent_control_60m_comparison_20261104/review_summary.json.
  episodes: 2048,
  before: Object.freeze({ label: '40M', grasp: 1832, lift: 1764, strict: 1245 }),
  after: Object.freeze({ label: '60M', grasp: 1887, lift: 1832, strict: 1416 }),
  // Launcher preflight and YAML, not inferred from the two evaluation points.
  horizon: 128, groupSize: 8, actionsPerDecision: 4, clipLow: 0.20, clipHigh: 0.28,
  bowlTolerance: 0.057, plateTolerance: 0.091, liftHeight: 0.05,
});
