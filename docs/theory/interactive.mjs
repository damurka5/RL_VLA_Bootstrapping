import { mixProbability, groupProbabilities, populationWeight, populationObjective,
  finiteCoefficient, composition, compensationFloor, trustBound, evidence } from './math.mjs';

const $ = id => document.getElementById(id);
const number = (n, digits = 3) => Number.isFinite(n) ? n.toFixed(digits).replace('.', ',') : 'нет конечного значения';
const percent = n => number(100 * n, 2) + '%';
const signed = n => (n >= 0 ? '+' : '') + number(100 * n, 2) + ' п.п.';
const text = (id, value) => { $(id).textContent = value; };
const NS = 'http://www.w3.org/2000/svg';
function element(parent, tag, attrs = {}, content) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (content !== undefined) node.textContent = content;
  parent.appendChild(node);
  return node;
}
function plot(id, { xmin = 0, xmax = 1, ymin = 0, ymax = 1, xticks = [0, .25, .5, .75, 1],
  yticks = [0, .25, .5, .75, 1], xformat = v => number(v, 2), yformat = v => number(v, 2), xlabel = 'Вероятность', ylabel = '', height = 330 } = {}) {
  const svg = $(id);
  svg.replaceChildren();
  const L = 60, R = 20, T = 40, B = 52, W = 620, H = height;
  const x = v => L + (v - xmin) / (xmax - xmin) * (W - L - R);
  const y = v => H - B - (v - ymin) / (ymax - ymin) * (H - T - B);
  for (const v of yticks) {
    element(svg, 'line', { x1: L, x2: W - R, y1: y(v), y2: y(v), class: 'grid' });
    element(svg, 'text', { x: L - 9, y: y(v) + 4, 'text-anchor': 'end' }, yformat(v));
  }
  for (const v of xticks) element(svg, 'text', { x: x(v), y: H - B + 23, 'text-anchor': 'middle' }, xformat(v));
  element(svg, 'text', { x: L, y: 16 }, ylabel);
  element(svg, 'text', { x: (L + W - R) / 2, y: H - 7, 'text-anchor': 'middle' }, xlabel);
  function line(fn, cls = 'curve', lo = xmin, hi = xmax) {
    const points = Array.from({ length: 241 }, (_, i) => {
      const v = lo + (hi - lo) * i / 240;
      return x(v) + ',' + y(Math.max(ymin, Math.min(ymax, fn(v))));
    }).join(' ');
    element(svg, 'polyline', { points, class: cls });
  }
  function dot(px, py, label = '', dx = 8, dy = -10, fill = 'var(--accent)') {
    element(svg, 'circle', { cx: x(px), cy: y(py), r: 5, class: 'mark', style: 'fill:' + fill });
    if (label) element(svg, 'text', { x: x(px) + dx, y: y(py) + dy, style: 'fill:var(--text)' }, label);
  }
  return { svg, x, y, line, dot, L, R, T, B, W, H };
}
function rangeValue(id, formatter = percent) {
  const value = Number($(id).value);
  text(id + '-value', formatter(value));
  return value;
}
function on(id, callback, event = 'input') { $(id).addEventListener(event, callback); }

function geometry() {
  const centimeters = rangeValue('distance', v => number(v, 1) + ' см');
  const tolerance = Number($('target-geometry').value) * 100;
  const ok = centimeters <= tolerance;
  text('geometry-result', `d = ${number(centimeters, 1)} см ${ok ? '≤' : '>'} ${number(tolerance, 1)} см. Геометрическое условие ${ok ? 'выполнено' : 'не выполнено'}.`);
  $('geometry-result').className = 'lab-result ' + (ok ? 'positive' : 'negative');
  const p = plot('geometry-plot', { xmax: 20, ymin: -.08, ymax: 1.15, xticks: [0, 5, 10, 15, 20], yticks: [0, 1],
    xlabel: 'Расстояние d, см', ylabel: 'Только условие d ≤ R_tol', xformat: v => String(v), yformat: v => v ? 'Да' : 'Нет' });
  element(p.svg, 'polyline', { points: [[0, 1], [tolerance, 1], [tolerance, 0], [20, 0]].map(([x,y]) => p.x(x)+','+p.y(y)).join(' '), class: 'curve' });
  element(p.svg, 'line', { x1: p.x(tolerance), x2: p.x(tolerance), y1: p.y(0), y2: p.y(1.1), class: 'secondary' });
  element(p.svg, 'text', { x: p.x(tolerance) + 7, y: p.y(1.08) }, 'R_tol = ' + number(tolerance, 1) + ' см');
  p.dot(centimeters, ok ? 1 : 0);
}
on('distance', geometry); on('target-geometry', geometry, 'change'); geometry();

function objective() {
  const value = rangeValue('objective-p');
  const f = populationObjective(value), weight = populationWeight(value);
  text('objective-readout', `F(p) = ${number(f)}; наклон F′(p) = ${number(weight)}. Шкала F идёт от 0 до π.`);
  const p = plot('objective-plot', { ymax: Math.PI, yticks: [0, Math.PI/4, Math.PI/2, 3*Math.PI/4, Math.PI],
    yformat: v => number(v, 2), xlabel: 'Вероятность p одного контекста', ylabel: 'F(p), от 0 до π' });
  p.line(populationObjective);
  const span = Math.min(.08, value, 1 - value);
  p.line(v => f + weight * (v - value), 'secondary', value - span, value + span);
  p.dot(value, f);
}
on('objective-p', objective); objective();

function groups() {
  const value = rangeValue('group-p');
  const probs = groupProbabilities(value);
  text('group-readout', `P смешанной группы = ${percent(mixProbability(value))}. Все неудачи: ${percent(probs[0])}. Все успехи: ${percent(probs[8])}.`);
  const p = plot('group-plot', { xmin: -.6, xmax: 8.6, xticks: [0,1,2,3,4,5,6,7,8],
    xlabel: 'n успешных попыток из G = 8', ylabel: 'Вероятность группы', xformat: v => String(v), yformat: v => number(100*v, 0)+'%' });
  probs.forEach((prob, n) => {
    element(p.svg, 'rect', { x: p.x(n) - 20, y: p.y(prob), width: 40, height: p.y(0) - p.y(prob),
      fill: n === 0 || n === 8 ? 'var(--muted)' : 'var(--accent)', opacity: .8 });
    if (prob > .01) element(p.svg, 'text', { x: p.x(n), y: p.y(prob)-7, 'text-anchor': 'middle' }, number(100*prob, 1)+'%');
  });
}
on('group-p', groups); groups();

function weights() {
  const value = rangeValue('weight-p');
  const finite = finiteCoefficient(value), population = populationWeight(value);
  text('weight-readout', `c₈(p) = ${number(finite)}. Истинная нормировка: ${number(population)}. Смешанных групп: ${percent(mixProbability(value))}.`);
  const p = plot('weight-plot', { xmin: .001, xmax: .999, ymin: 0, ymax: Math.log10(40),
    yticks: [1,2,5,10,20,40].map(Math.log10), yformat: v => number(10**v, 0), ylabel: 'Множитель градиента, лог. шкала', xlabel: 'Вероятность p' });
  p.line(v => Math.log10(populationWeight(v)), 'secondary');
  p.line(v => Math.log10(finiteCoefficient(v)));
  p.dot(value, Math.log10(finite));
  p.dot(value, Math.log10(population), '', 8, -10, 'var(--muted)');
}
on('weight-p', weights); weights();

const baseline = evidence.before.strict / evidence.episodes;
const state = { lift: evidence.after.lift / evidence.episodes,
  finish: evidence.after.strict / evidence.after.lift, mode: 'Измеренные частоты 60M' };
// Keep exact rational preset values, not values rounded to a slider step.
for (const id of ['lift-q','finish-q','signal-lift','signal-finish']) $(id).step = 'any';
function drawComposition() {
  text('composition-mode', state.mode);
  for (const [id,key] of [['lift-q','lift'],['finish-q','finish'],['signal-lift','lift'],['signal-finish','finish']]) {
    $(id).value = state[key]; text(id + '-value', percent(state[key]));
  }
  const success = composition(state.lift, state.finish), diff = success - baseline;
  const floor = compensationFloor(baseline, state.finish);
  const feasible = floor <= 1;
  const threshold = feasible ? `При этом q завершения нужен q подъёма ≥ ${percent(floor)}.` : 'При этом q завершения компенсировать потерю невозможно даже при 100% подъёмов.';
  text('composition-readout', `J = ${percent(success)}; к 40M: ${signed(diff)} ${threshold}`);
  $('composition-readout').className = 'lab-result ' + (diff >= -1e-12 ? 'positive' : 'negative');
  const p = plot('composition-plot', { xlabel: 'q подъёма внутри полной инструкции', ylabel: 'J полной инструкции', yformat: v => number(100*v,0)+'%' });
  p.line(x => x * state.finish);
  p.line(() => baseline, 'secondary');
  if (feasible) element(p.svg, 'line', { x1: p.x(floor), x2: p.x(floor), y1: p.y(0), y2: p.y(baseline), class: 'secondary' });
  p.dot(evidence.before.lift / evidence.episodes, baseline, '40M', -48, 19, 'var(--muted)');
  p.dot(evidence.after.lift / evidence.episodes, evidence.after.strict / evidence.episodes, '60M', 8, -12, 'var(--good)');
  const x = p.x(state.lift), y = p.y(success);
  element(p.svg, 'polygon', { points: `${x},${y-8} ${x+8},${y} ${x},${y+8} ${x-8},${y}`, fill: 'var(--accent)', stroke: 'var(--surface)', 'stroke-width': 2 });
  drawSignal(); drawBound();
}
function drawSignal() {
  const value = composition(state.lift, state.finish);
  const mix = mixProbability(value);
  text('signal-readout', `p одной попытки = ${percent(value)}. Из 100 групп в среднем ${number(100*mix,1)} содержат и успех, и неудачу.${value===0||value===1 ? ' Все группы одинаковы: сигнал этой награды нулевой.' : ' Это возможность сравнения, не величина улучшения.'}`);
  const p = plot('signal-plot', { xlabel: 'p: успех одной попытки', ylabel: 'P_mix: смешанных групп из 8', xformat: v => number(100*v,0)+'%', yformat: v => number(100*v,0)+'%' });
  p.line(v => mixProbability(v)); p.dot(value, mix, 'сценарий', value>.8 ? -82:8, mix>.85?20:-10);
}
function drawBound() {
  const success = composition(state.lift, state.finish);
  const gain = rangeValue('local-gain', v => number(v,1)+' п.п.') / 100;
  const logarithm = rangeValue('max-kl', v => 'δ = ' + (10**v).toExponential(2).replace('.',','));
  const delta = 10 ** logarithm, H = Number($('bound-horizon').value);
  const result = trustBound(success, gain, H, delta);
  text('bound-readout', `Нижняя граница: ${percent(result.lower)}. Текущий J: ${percent(success)}. ${result.raw <= 0 ? 'Осталась лишь граница ноль.' : result.lower > success ? 'При этих условиях рост установлен.' : 'Эта оценка не устанавливает рост.'}`);
  text('bound-calculation', `Расчёт: ${percent(success)} + ${number(100*gain,1)} п.п. − ${number(100*result.penalty,3)} п.п. = ${percent(result.raw)} до ограничения снизу нулём.`);
  const p = plot('bound-plot', { height:300, xmin:-7,xmax:-2,xticks:[-7,-6,-5,-4,-3,-2],xlabel:'δ: максимальная KL, логарифмическая шкала',
    ylabel:'Условная нижняя граница J_new', ymax:1.1, xformat:v=>'10'+({'-7':'⁻⁷','-6':'⁻⁶','-5':'⁻⁵','-4':'⁻⁴','-3':'⁻³','-2':'⁻²'}[v]), yformat:v=>number(100*v,0)+'%' });
  p.line(v=>trustBound(success,gain,H,10**v).lower);
  p.line(()=>success,'secondary');
  const critical = 2*gain/(H*(H-1));
  if (critical >= 1e-7 && critical <= 1e-2) {
    const x = p.x(Math.log10(critical));
    element(p.svg,'line',{x1:x,x2:x,y1:p.y(0),y2:p.y(1.05),class:'secondary'});
    element(p.svg,'text',{x:x+6,y:p.y(1.02)},'порог роста');
  }
  p.dot(logarithm, result.lower);
}
for (const [id,key] of [['lift-q','lift'],['finish-q','finish'],['signal-lift','lift'],['signal-finish','finish']]) {
  on(id,()=>{ state[key]=Number($(id).value); state.mode='Сценарий: изменён вручную'; drawComposition(); });
}
document.querySelectorAll('[data-preset]').forEach(button=>button.addEventListener('click',()=>{
  const preset=button.dataset.preset;
  if (preset==='before'||preset==='after') {
    const point=evidence[preset];
    state.lift=point.lift/evidence.episodes; state.finish=point.strict/point.lift;
    state.mode='Измеренные частоты '+point.label;
  } else {
    state.lift=preset==='compensated'?.81:preset==='failure'?.70:0;
    state.finish=evidence.after.strict/evidence.after.lift;
    state.mode='Сценарий: '+(preset==='compensated'?'просадка компенсирована':preset==='failure'?'просадка не компенсирована':'обязательный этап перестал работать');
  }
  drawComposition();
}));
on('local-gain',drawBound); on('max-kl',drawBound); on('bound-horizon',drawBound,'change');
drawComposition();
