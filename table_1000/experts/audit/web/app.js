// Expert iteration site: lines → steps → per-attempt search trees, from the /api/ routes of site.py.
// Media is rendered only on an explicit click (or by the server when auto-render is on).
const $ = id => document.getElementById(id), NS = 'http://www.w3.org/2000/svg';
const colors = {a2: '#cf941f'};
const terminals = {a1: '规划', a3: '物理', planning: '规划', rollback: '回滚'};
const results = {success: 'ok', b: 'b', error: 'error', pending: 'pending'};
const CATEGORIES = ['a1', 'a2', 'a3', 'b'];
const failureText = counts => CATEGORIES.map(c => `${c} ${(counts || {})[c] || 0}`).join(' · ');
const searchNames = {solver_calls: '求解调用', candidate_rejections: '候选淘汰', decision_exhaustions: '决策用尽', expert_rejections: '专家接住的规划拒绝'};
const DX = 200, DY = 84;
let line = null, summary = null, step = null, chosen = null, chosenItem = null, poll = null;
const pad = n => String(n).padStart(2, '0');
const short = batch => batch.replace(/^\d{8}-\d{6}-/, '');

async function api(path, body) {
  const response = await fetch(path, body ? {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)} : {});
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || response.statusText);
  return value;
}
function el(tag, attrs = {}, parent, text) {
  const e = tag.startsWith('svg:') ? document.createElementNS(NS, tag.slice(4)) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  if (parent) parent.append(e);
  return e;
}
function activate(element, fn, label) {
  element.setAttribute('tabindex', '0'); element.setAttribute('role', 'button'); element.setAttribute('aria-label', label);
  element.onclick = e => { e.stopPropagation(); fn(); };
  element.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fn(); } };
}

// Lines and steps.
async function loadLines() {
  const lines = await api('/api/lines');
  $('line').replaceChildren(...lines.map(l => {
    const o = el('option', {value: l.id}, undefined, `${l.scene} · ${l.robot}/${l.line} · ${l.batches} 批次${l.browsable ? '' : '（#114 前记录，无搜索树）'}`);
    if (!l.browsable) { o.disabled = true; o.className = 'deprecated'; }
    return o;
  }));
  if (!lines.length) $('empty').textContent = '没有开发批次（runs/expert/<场景>/<机器人>/<编号>/）。';
  return lines;
}
function stepLabel(s) {
  const mark = s.media_total && s.media === s.media_total ? '✓ ' : '　';
  const counts = `${s.selected_nodes}/${s.tree_nodes}`;
  return s.deprecated ? `${mark}（已废弃）${s.name} · ${counts}` : `${mark}s${pad(s.step)} ${s.name} · ${counts}`;
}
async function loadLine(keep) {
  line = $('line').value;
  summary = await api(`/api/line?id=${encodeURIComponent(line)}`);
  const build = summary.build;
  $('build').hidden = !build;
  if (build) $('build').textContent = build.state === 'error' ? `建立索引失败：\n${build.error}` :
    `正在建立索引${build.total ? `（${build.done}/${build.total} 批次）` : ''}…${summary.steps.length ? '下面是上次的索引。' : ''}`;
  $('build').className = build && build.state === 'error' ? 'error' : 'building';
  $('line-failures').textContent = (summary.steps.length ? `失败（按失败计）：${failureText(summary.failures)}` : '') +
    (summary.before_facts && summary.before_facts.length ? `\n另有 ${summary.before_facts.length} 个 #114 前批次（无搜索树），未列入。` : '');
  $('scope').textContent = summary.scene ? `${summary.scene} · ${summary.robot} · ${summary.expert} · layout ${summary.layout} · seed ${summary.seed}` : '';
  const previous = keep && step ? step.key : null;
  $('step').replaceChildren(...summary.steps.map(s => {
    const o = el('option', {value: s.key, title: s.name}, undefined, stepLabel(s));
    if (s.deprecated) o.className = 'deprecated';
    return o;
  }));
  const wanted = previous || decodeURIComponent(location.hash.slice(1)).split('|')[1];
  if (summary.steps.some(s => s.key === wanted)) $('step').value = wanted;
  if (summary.steps.length) await loadStep(keep);
  else { $('forest').replaceChildren(); clearDetail(); }
  watch();
}
function stepInfo() {
  const box = $('step-info'), s = summary.steps.find(v => v.key === step.key); box.replaceChildren();
  if (!s.deprecated) { el('p', {}, box, `s${pad(s.step)} · ${s.name}`); el('p', {class: 'hint'}, box, `失败：${failureText(s.failures)}`); return; }
  const r = s.replacement || {candidates: []}, by = new Map(summary.steps.filter(v => !v.deprecated).map(v => [v.name, v]));
  const names = r.candidates.map(n => by.has(n) ? `s${pad(by.get(n).step)} ${n}` : n);
  el('div', {class: 'deprecated-note'}, box, `已废弃（不编号）。原编号 ${(s.source_steps || []).map(n => 's' + pad(n)).join('、')}。` +
    (names.length ? `可能的替代者：${names.join('、')}${r.certain ? '' : '（推断不确定）'}。` : '未能推断替代者。'));
}
async function loadStep(keep) {
  const opened = keep ? new Set([...document.querySelectorAll('.tree.open')].map(c => c.dataset.tree)) : new Set();
  step = await api(`/api/step?line=${encodeURIComponent(line)}&key=${encodeURIComponent($('step').value)}`);
  $('step').title = step.name; stepInfo();
  const forest = $('forest'); forest.replaceChildren();
  const nodes = step.trees.reduce((n, t) => n + t.nodes.length, 0);
  $('forest-title').textContent = `${step.trees.length} 棵树 · ${nodes} 个节点`;
  step.trees.forEach((tree, i) => forest.append(treeCard(tree, i, opened.has(tree.id))));
  $('empty').hidden = !!step.trees.length; $('empty').textContent = step.trees.length ? '' : '此 step 没有物理状态记录。';
  history.replaceState(null, '', `#${encodeURIComponent(line)}|${encodeURIComponent(step.key)}`);
  if (keep && chosenItem) {
    const tree = step.trees.find(t => t.id === chosenItem.tree), list = tree && [...tree.nodes, ...tree.edges];
    const item = list && list.find(i => i.id === chosenItem.id);
    const element = item && document.querySelector(`[data-tree="${CSS.escape(tree.id)}"] [data-id="${CSS.escape(item.id)}"]`);
    if (item && element) return choose(tree, item, chosenItem.node, element);
  }
  clearDetail();
}

// Trees.
function treeCard(tree, i, open) {
  const card = el('div', {class: 'tree' + (tree.selected ? ' selected' : ''), 'data-tree': tree.id});
  const head = el('div', {class: 'head'}, card);
  el('b', {}, head, `树 ${i + 1}`);
  el('span', {class: 'tag ' + (results[tree.result] || '')}, head, tree.result_label);
  if (tree.selected) el('span', {class: 'tag chain'}, head, '进入完整流程');
  for (const c of CATEGORIES) if (tree.failure_counts[c]) el('span', {class: 'tag' + (c === 'a2' ? ' recover' : '')}, head, `${c}×${tree.failure_counts[c]}`);
  if (!tree.edges.length) el('span', {class: 'tag idle', title: '物体已在前面的 step 处理，或判据直接满足'}, head, `本步无动作（${tree.frames} 帧）`);
  el('span', {class: 'meta'}, head, `${short(tree.batch)}${tree.repeat ? ` · repeat ${tree.repeat}` : ''} · 专家 ${tree.commit || '?'} · ${tree.started ? tree.started.replace('T', ' ') : '时间未知'} · ${tree.nodes.length} 节点`);
  if (tree.calls || tree.run_command) {
    const cmd = el('div', {class: 'cmd'}, card);
    if (tree.calls) {
      const top = tree.calls.children.find(c => c.main.actions);
      const d = el('details', {class: 'calls'}, cmd);
      el('summary', {}, d, `调用记录 · ${top ? top.main.actions : tree.calls.name}`);
      d.append(callTree(tree, card, tree.calls, null));
    }
    if (tree.run_command) el('div', {class: 'run', title: tree.run_command}, cmd, `运行：${tree.run_command} · commit ${tree.commit || '?'}`);
  }
  const body = el('div', {class: 'body'}, card);
  const toggle = () => { card.classList.toggle('open'); if (card.classList.contains('open') && !body.firstChild) drawTree(tree, body); };
  head.onclick = toggle;
  if (open) toggle();
  return card;
}
// The step's recorded calls: step → expert command → subtask → skills. Control frames (retry) are
// transparent; a callback argument shows the value the call below it received.
function shown(value) {
  if (value === null || typeof value !== 'object') return String(value);
  if (value.$ === 'callback') return `回调 ${value.name.replace(/\.<locals>/g, '')}`;
  if (value.$ && value.value && typeof value.value === 'object' && !Array.isArray(value.value)) return `${value.$}(${value.value.obj ?? '…'})`;
  const text = JSON.stringify(value.$ ? value.value : value);
  return text.length > 90 ? text.slice(0, 90) + '…' : text;
}
function visible(children) {
  return children.flatMap(c => c.control ? visible(c.children).map(([g, via]) => [g, via || `经 ${c.name}${c.arguments.attempts ? ' ≤' + c.arguments.attempts : ''}`]) : [[c, null]]);
}
function callTree(tree, card, frame, via) {
  const box = el('details', {class: 'call ' + frame.kind}), line = el('summary', {}, box);
  if (frame.kind !== 'subtask') box.open = true;
  const at = el('a', {href: '#', class: 'frame', title: '跳到对应节点'}, line, `${frame.enter}–${frame.exit}`);
  at.onclick = e => { e.preventDefault(); e.stopPropagation(); jump(tree, card, frame); };
  el('b', {}, line, ` ${frame.name}`);
  const main = Object.entries(frame.main).filter(([k]) => k !== 'actions')
    .map(([k, v]) => `${k}=${shown(v)}${k in frame.resolved ? ` → ${shown(frame.resolved[k])}` : ''}`);
  if (main.length) el('span', {class: 'args'}, line, ' ' + main.join(' · '));
  if (via) el('span', {class: 'tag'}, line, via);
  if (frame.branch !== 'main') el('span', {class: 'args'}, line, ` · 分支 ${frame.branch}`);
  if (Object.keys(frame.arguments || {}).length) {
    const more = el('details', {class: 'more'}, box); el('summary', {}, more, '全部参数');
    el('pre', {}, more, JSON.stringify(frame.arguments, null, 1));
  }
  for (const [child, through] of visible(frame.children)) box.append(callTree(tree, card, child, through));
  return box;
}
function jump(tree, card, frame) {
  if (!card.classList.contains('open')) card.querySelector('.head').click();
  const id = `${tree.id}/${frame.branch}@${frame.enter}`;
  const node = tree.nodes.find(n => n.members.includes(id)) ||
    tree.nodes.filter(n => n.branch === frame.branch && n.at <= frame.enter).sort((a, b) => b.at - a.at)[0];
  const element = node && card.querySelector(`[data-id="${CSS.escape(node.id)}"]`);
  if (element) { choose(tree, node, true, element); element.scrollIntoView({block: 'center'}); }
}
function layout(tree) {
  const kids = {};
  for (const e of tree.edges) (kids[e.source] ??= []).push(e);
  const chain = new Set(tree.chain);
  for (const list of Object.values(kids)) list.sort((a, b) => (chain.has(b.id) - chain.has(a.id)) || (a.start - b.start));
  const pos = {}; let column = 0;
  (function place(id, depth) {
    const out = kids[id] || [];
    if (!out.length) { pos[id] = [column++, depth]; return; }
    for (const e of out) place(e.target, depth + 1);
    const xs = out.map(e => pos[e.target][0]);
    pos[id] = [(Math.min(...xs) + Math.max(...xs)) / 2, depth];
  })(tree.root, 0);
  return pos;
}
function drawTree(tree, body) {
  const pos = layout(tree), chain = new Set(tree.chain);
  const chainNodes = new Set([tree.root, ...tree.edges.filter(e => chain.has(e.id)).map(e => e.target)]);
  const xs = Object.values(pos).map(p => p[0]), ys = Object.values(pos).map(p => p[1]);
  const w = (Math.max(...xs) + 1) * DX + 120, h = (Math.max(...ys) + 1) * DY + 40;
  const svg = el('svg:svg', {width: w, height: h, viewBox: `0 0 ${w} ${h}`}, body);
  const at = id => [60 + pos[id][0] * DX + DX / 2, 30 + pos[id][1] * DY];
  for (const edge of tree.edges) {
    const [x1, y1] = at(edge.source), [x2, y2] = at(edge.target), on = tree.selected && chain.has(edge.id);
    const g = el('svg:g', {class: 'edge', 'data-id': edge.id}, svg);
    const d = `M${x1},${y1 + 13} C${x1},${(y1 + y2) / 2} ${x2},${(y1 + y2) / 2} ${x2},${y2 - 13}`;
    if (on) el('svg:path', {d, fill: 'none', stroke: 'var(--chain)', 'stroke-width': 9, 'stroke-opacity': .45}, g);
    el('svg:path', {class: 'line', d, fill: 'none', stroke: colors[edge.category] || '#9baab7', 'stroke-width': 3}, g);
    el('svg:path', {class: 'hit', d, fill: 'none', stroke: 'transparent', 'stroke-width': 16}, g);
    const label = edge.name.length > 18 ? edge.name.slice(0, 17) + '…' : edge.name;
    el('svg:text', {x: x2 + 8, y: y2 - DY / 2 + (x1 === x2 ? 4 : 18)}, g, `${label}${edge.category === 'a2' ? ' · ' + edge.category : ''}`);
    el('svg:title', {}, g, `${edge.name} · ${edge.category} · ${edge.start}→${edge.end}`);
    activate(g, () => choose(tree, edge, false, g), `${edge.category}: ${edge.name}`);
  }
  for (const node of tree.nodes) {
    const [x, y] = at(node.id), on = tree.selected && chainNodes.has(node.id);
    const g = el('svg:g', {class: 'node', 'data-id': node.id}, svg);
    el('svg:circle', {class: 'hit', cx: x, cy: y, r: 20, fill: 'transparent'}, g);
    if (node.terminal) el('svg:rect', {x: x - 12, y: y - 12, width: 24, height: 24, fill: '#f1f4f6', stroke: '#7d8b96', 'stroke-width': 2}, g);
    else el('svg:circle', {cx: x, cy: y, r: 12, fill: on ? '#fff1d6' : '#fff', stroke: on ? 'var(--chain)' : '#7d9fb3', 'stroke-width': on ? 3 : 2}, g);
    el('svg:text', {x: x - 18, y: y + 4, 'text-anchor': 'end'}, g, `${node.at}`);
    let row = y + 4;
    const searched = Object.values(node.search || {}).reduce((n, l) => Math.max(n, l.count), 0);
    if (searched) { el('svg:text', {x: x + 17, y: row, class: 'search'}, g, `搜索×${searched}`); row += 13; }
    const still = node.retries.filter(r => !r.moved).length;
    if (still) { el('svg:text', {x: x + 17, y: row, class: 'retry'}, g, `重试×${still}`); row += 13; }
    if (node.terminal) el('svg:text', {x: x + 17, y: row, class: 'term'}, g, terminals[node.terminal]);
    activate(g, () => choose(tree, node, true, g), `控制帧 ${node.at}`);
  }
}

// Details of a node or an edge.
function clearDetail() {
  if (chosen) chosen.classList.remove('chosen');
  chosen = null; chosenItem = null;
  $('clip').pause(); $('clip').removeAttribute('src'); $('clip').hidden = true;
  $('still').removeAttribute('src'); $('still-link').hidden = true; $('missing').hidden = true;
  $('kind').textContent = '选择节点或边'; $('title').textContent = ''; $('metrics').replaceChildren();
  for (const id of ['timing', 'search', 'retries', 'checks']) $(id).hidden = true;
  $('detail').textContent = '';
}
function table(rows) {
  const t = el('table');
  for (const [k, v, mono] of rows) { const tr = el('tr', {}, t); el('th', {}, tr, k); el('td', mono ? {class: 'mono'} : {}, tr, v); }
  return t;
}
function showMedia(item, isNode) {
  const src = `/media/${line}/${item.media.split('/').pop()}`;
  if (item.media_ready) {
    if (isNode) { $('still').src = src; $('still-link').href = src; $('still-link').hidden = false; }
    else { $('clip').hidden = false; $('clip').src = src; $('clip').load(); $('clip').play().catch(() => {}); }
    return;
  }
  $('missing').hidden = false;
  const job = activeJobs.find(j => j.media.includes(item.media));
  $('missing-text').textContent = job ? `渲染中（作业 ${job.id}：${jobState(job)}）` : '媒体尚未生成';
  $('render-item').hidden = !!job;
  $('render-item').onclick = () => submit({scope: 'item', media: [item.media]});
}
function choose(tree, item, isNode, element) {
  clearDetail();
  chosen = element; chosen.classList.add('chosen'); chosenItem = {tree: tree.id, id: item.id, node: isNode};
  const where = ['树', `${short(tree.batch)}${tree.repeat ? ' · repeat ' + tree.repeat : ''}`, true];
  if (isNode) {
    $('kind').textContent = item.terminal ? `被弃分支末端 · ${terminals[item.terminal]}` : '状态节点';
    $('title').textContent = `控制帧 ${item.at}`;
    const still = item.retries.filter(r => !r.moved), moved = item.retries.filter(r => r.moved);
    $('metrics').append(table([['搜索开销', Object.keys(item.search || {}).length ? '见下方分层' : '无记录'],
      ['无运动重试', `${still.length} 次` + (still.length ? `（${still.map(r => terminals[r.trigger]).join('、')} 触发）` : '')],
      ...(moved.length ? [['回滚到此', `${moved.length} 次（被弃分支见树）`]] : []), ['分支', item.branch, true], where]));
    const layers = Object.entries(item.search || {});
    if (layers.length) {
      $('search').hidden = false; $('search').open = true;
      $('search-detail').replaceChildren(table(layers.map(([layer, row]) => [searchNames[layer] || layer,
        `${row.count} 次 · CPU ${row.cpu_s.toFixed(2)} s · wall ${row.wall_s.toFixed(2)} s`])),
        el('p', {class: 'hint'}, undefined, '各层耗时含嵌套（候选淘汰含其中的求解），不可跨层相加。'),
        ...layers.map(([layer, row]) => el('pre', {}, undefined, `${searchNames[layer] || layer}\n` +
          Object.entries(row.reasons).sort((a, b) => b[1] - a[1]).map(([r, n]) => `${n} × ${r}`).join('\n'))));
    }
    if (item.retries.length) { $('retries').hidden = false; $('retry-title').textContent = `搜索开销 · 回滚恢复到此 ${item.retries.length} 次`; $('retry-detail').textContent = JSON.stringify(item.retries, null, 1); }
    if (item.checks.length) { $('checks').hidden = false; $('measurements').textContent = JSON.stringify(item.checks, null, 1); }
  } else {
    $('kind').textContent = item.category; $('title').textContent = item.name;
    $('metrics').append(table([['控制帧', `${item.start} → ${item.end}`],
      ['时长', `${((item.end - item.start) / 25).toFixed(2)} s（${item.end - item.start} 控制帧，25 Hz）`], ['类别', item.category], where]));
    if (item.cpu_s != null) {
      $('timing').hidden = false; $('timing').open = false;
      $('timing-detail').textContent = JSON.stringify({CPU_s: item.cpu_s, wall_s: item.wall_s, ...item.timing_scope,
        note: '所属调用包含规划与执行；其下多条边共享该值，不代表本段耗时。'}, null, 1);
    }
    const checks = [...(item.checks || []), ...(item.failures || [])];
    if (checks.length) { $('checks').hidden = false; $('measurements').textContent = JSON.stringify(checks, null, 1); }
  }
  if (item.media_unavailable) {
    $('missing').hidden = false; $('missing-text').textContent = item.media_unavailable; $('render-item').hidden = true;
  } else if (item.media) showMedia(item, isNode);
  $('detail').textContent = JSON.stringify({tree: {batch: tree.batch, commit: tree.commit, started: tree.started, result: tree.result,
    source_step: tree.source_step, parent_checkpoint: tree.parent_checkpoint, prefix_note: tree.prefix_note,
    cpu_s: tree.cpu_s, wall_s: tree.wall_s}, ...item}, null, 1);
}

// Render jobs, settings and refreshing.
let activeJobs = [];
const jobState = j => ({queued: '排队', preparing: '准备状态', rendering: `渲染 ${j.done}/${j.stills + j.clips}`, publishing: '发布',
  done: '完成', failed: '失败'})[j.state];
async function submit(body) {
  try {
    const {job} = await api('/api/render', {line, ...body});
    if (!job) alert('没有缺的媒体（或已在渲染中）。');
    await refreshJobs(); if (chosenItem) await loadStep(true);
  } catch (error) { alert(`提交失败：${error.message}`); }
}
async function refreshJobs() {
  const jobs = await api(`/api/jobs?line=${encodeURIComponent(line)}`);
  const finished = activeJobs.filter(a => jobs.some(j => j.id === a.id && ['done', 'failed'].includes(j.state)));
  activeJobs = jobs.filter(j => !['done', 'failed'].includes(j.state));
  $('jobs').replaceChildren(...jobs.slice(-6).reverse().map(j => el('div', {}, undefined,
    `作业 ${j.id} · ${j.scope === 'line' ? '整条线' : j.scope === 'step' ? `step ${j.step}` : '单项'} · ${j.stills} 图 ${j.clips} 视频 · ${jobState(j)}${j.error ? ' · ' + j.error.slice(-120) : ''}`)));
  if (jobs.length) $('jobs').prepend(el('label', {}, undefined, '渲染作业'));
  return finished.length;
}
function watch() {
  clearInterval(poll);
  poll = setInterval(async () => {
    const finished = await refreshJobs();
    if (summary.build || finished) await loadLine(true);
    else if (!activeJobs.length) clearInterval(poll);
  }, 5000);
}
$('render-step').onclick = () => submit({scope: 'step', step: step.key}).then(watch);
$('render-line').onclick = () => {
  const missing = summary.steps.reduce((n, s) => n + s.media_total - s.media, 0);
  if (confirm(`提交整条线缺的媒体（约 ${missing} 项，GPU 作业经 gq）？`)) submit({scope: 'line'}).then(watch);
};
$('open-settings').onclick = async () => { $('auto-render').checked = (await api('/api/settings')).auto_render; $('settings').showModal(); };
$('auto-render').onchange = () => api('/api/settings', {auto_render: $('auto-render').checked});
$('close-settings').onclick = () => $('settings').close();
$('expand').onclick = () => { for (const c of document.querySelectorAll('.tree:not(.open) .head')) c.click(); };
$('collapse').onclick = () => { for (const c of document.querySelectorAll('.tree.open .head')) c.click(); };
$('line').onchange = () => { history.replaceState(null, '', `#${encodeURIComponent($('line').value)}`); loadLine(false); };
$('step').onchange = () => loadStep(false);

const lines = await loadLines();
const requested = decodeURIComponent(location.hash.slice(1)).split('|')[0];
// The URL's line, else the viewable line whose newest batch started last.
const viewable = lines.filter(l => l.browsable).sort((a, b) => (b.latest || '').localeCompare(a.latest || ''));
if (lines.some(l => l.id === requested && l.browsable)) $('line').value = requested;
else if (viewable.length) $('line').value = viewable[0].id;
if ($('line').value) { await loadLine(false); await refreshJobs(); watch(); }
