"""Teleoperate the cases of a scene's development batches, one step at a time, in a browser.

``uv run python scripts/experts/teleop.py 000002 --gpu 0 --port 8765``
then open ``http://<server>:8765/`` (for example through ``ssh -L 8765:localhost:8765 <server>``).

The page lists the cases of ``--runs`` (``runs/expert/<scene>/*``, see
``table_1000/experts/teleop.py``), lowest success rate over the ``--repeats`` runs first. Loading
one resets the simulation to its start (the checkpoint before the failed step); the operator then
does the step by hand, watching an outside view (the scene's first, which they can turn, pan and
zoom) and the robot's RGB cameras: drag a view to move the TCP target, buttons and keys for the
rest (listed on the page).
"Check" shows the step's check on the current state and goes on; "Done" and "Not done" end the
step as the operator judges, the check's verdict then is kept alongside. Every attempt is a batch
``<output>/<scene>/<date>-<time>-<commit>-step<k>-seed<s>/`` (``<output>`` = ``runs/teleop``) like
``run_expert.py --dev --steps k --from <batch>`` writes: the trajectory (``trajectories.hdf5`` when
done, else ``trajectories_failed.hdf5``), ``episodes/<seed>.json`` with ``teleop`` (the case, the
checks asked for, the verdict, the check at the end), the step's checkpoint when done, and
``summary.json``.
"""

import argparse
import dataclasses
import json
import os
import queue
from pathlib import Path
import shlex
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DISPLAY_WIDTH = 640    # px, the widest robot camera rendered

PAGE = '''<!doctype html><html><head><meta charset="utf-8"><title>Case teleop</title>
<style>
body{margin:0;background:#111;color:#ddd;font:13px sans-serif;display:flex;height:100vh}
#cases{width:520px;flex:none;overflow:auto;border-right:1px solid #333}
#cases table{border-collapse:collapse;width:100%}#cases td,#cases th{padding:3px 6px;text-align:left;border-bottom:1px solid #222}
#cases tr.case{cursor:pointer}#cases tr.case:hover{background:#223}#cases tr.on{background:#335}
#main{flex:1;overflow:auto;padding:6px}
#v{display:flex;flex-wrap:wrap;gap:6px}#v img{border:1px solid #333;cursor:crosshair;user-select:none;height:auto}
#v img.outside{width:100%;max-width:1200px}#v img.robot{width:calc(33% - 6px);max-width:400px}
button{margin:3px 4px 3px 0;padding:6px 12px;background:#2a2a3a;color:#ddd;border:1px solid #555;cursor:pointer}
button.on{background:#447;border-color:#88c}button.closing{background:#634;border-color:#c86}
#s{white-space:pre;font-family:monospace;padding:4px 0}#help{color:#888;white-space:pre;font-family:monospace;padding-top:6px}
</style></head><body>
<div id="cases"><table><thead><tr><th>rate</th><th>step</th><th>seed</th><th>start from</th><th>failure</th><th>run</th></tr></thead>
<tbody id="rows"></tbody></table></div>
<div id="main">
<div><button onclick="cmd('check')">Check</button><button onclick="cmd('done')">Done</button>
<button onclick="cmd('not done')">Not done</button><button onclick="reload()">Restart case</button>
<button onclick="loadCases()">Refresh cases</button>
<label>failure start: <input id="back" type="number" value="0" min="0" step="0.5" style="width:4em"> s before the failure</label></div>
<div id="arms"></div>
<div id="s">pick a case on the left</div>
<div id="v">__IMAGES__</div>
<div id="help">left drag: move the TCP target (red dot) in that image's plane    Shift+wheel: TCP target toward / away from that camera
outside view: right drag turn, middle drag or Shift+right drag pan, wheel zoom
keys (click the page first): Space gripper  Tab arm  Z/X C/V B/N TCP roll/pitch/yaw  I/K J/L U/O TCP along base x/y/z
W/S A/D base  R/F lift  arrows head  P pause  hold Shift: faster</div>
</div>
<script>
let current=null,outside=[];
const post=(p,b)=>fetch(p,{method:'POST',body:JSON.stringify(b)});
function cmd(c,extra){post('/command',Object.assign({command:c},extra||{}))}
let currentStart='step';
function load(c,start){current=c;currentStart=start;
 cmd('load',{case:c,start,back_s:parseFloat(document.getElementById('back').value)||0});render()}
function reload(){if(current)load(current,currentStart)}
let cases=[];
async function loadCases(){cases=await (await fetch('/cases')).json();render()}
function render(){const rows=document.getElementById('rows');rows.innerHTML='';
 for(const c of cases){const tr=document.createElement('tr');tr.className='case';
  if(current&&c.batch==current.batch&&c.seed==current.seed&&c.step==current.step)tr.className+=' on';
  const rate=c.rate==null?'-':`${c.successes}/${c.runs}`;
  tr.innerHTML=`<td>${rate}</td><td>${c.step}</td><td>${c.seed}</td><td></td><td>${c.failure||''}</td><td>${c.run}</td>`;
  const td=tr.children[3];
  const b=(label,start,ok)=>{const e=document.createElement('button');e.textContent=label;e.disabled=!ok;
   e.onclick=ev=>{ev.stopPropagation();load(c,start)};td.appendChild(e)};
  b('step','step',true);b('failure','failure',!!c.trajectory);
  tr.title=c.step_name;rows.appendChild(tr)}}
const held=new Set();
function key(k,d){post('/key',{key:k,down:d})}
addEventListener('keydown',e=>{if(e.repeat)return;e.preventDefault();held.add(e.code);key(e.code,true)});
addEventListener('keyup',e=>{e.preventDefault();held.delete(e.code);key(e.code,false)});
addEventListener('blur',()=>{for(const k of held)key(k,false);held.clear()});
// Mouse moves are summed and sent every 40 ms.
const pending={};
function add(path,body){const k=path+'|'+body.view+'|'+(body.kind||'');const p=pending[k]||(pending[k]={path,body:Object.assign({},body,{dx:0,dy:0})});
 p.body.dx+=body.dx;p.body.dy+=body.dy}
setInterval(()=>{for(const k in pending){const p=pending[k];delete pending[k];if(p.body.dx||p.body.dy)post(p.path,p.body)}},40);
function setup(){for(const img of document.querySelectorAll('#v img')){
 const view=img.dataset.view,isOutside=outside.includes(view);img.className=isOutside?'outside':'robot';
 let last=null,mode=null;img.draggable=false;
 const scale=()=>img.naturalWidth/img.clientWidth;
 img.addEventListener('contextmenu',e=>e.preventDefault());
 img.addEventListener('mousedown',e=>{last=[e.clientX,e.clientY];e.preventDefault();
  mode=!isOutside||e.button==0?'tcp':(e.button==1||e.shiftKey?'pan':'orbit')});
 addEventListener('mouseup',()=>{last=null;mode=null});
 addEventListener('mousemove',e=>{if(!last||!e.buttons)return;const s=scale(),dx=(e.clientX-last[0])*s,dy=(e.clientY-last[1])*s;
  last=[e.clientX,e.clientY];if(mode=='tcp')add('/drag',{view,dx,dy});else add('/view',{view,kind:mode,dx,dy})});
 img.addEventListener('wheel',e=>{e.preventDefault();const n=Math.sign(-e.deltaY);
  if(e.shiftKey||!isOutside)post('/drag',{view,dx:0,dy:0,wheel:n});else post('/view',{view,kind:'zoom',dx:0,dy:n})},{passive:false});
}}
let armsHtml='';
function arms(s){const el=document.getElementById('arms');
 const html=!s.arms?'':s.arms.map(a=>{const g=s.grippers[a];
  return `<button class="${a==s.arm?'on':''}" onclick="cmd('arm',{arm:'${a}'})">control ${a}</button>`+
   `<button class="${g.closing?'closing':''}" onclick="cmd('gripper',{arm:'${a}'})">${a} gripper: ${g.closing?'closed':'open'} (${(g.gap*1000).toFixed(0)} mm) - toggle</button>`}).join(' ');
 if(html!=armsHtml){armsHtml=html;el.innerHTML=html}}
setInterval(async()=>{try{const s=await (await fetch('/state')).json();const el=document.getElementById('s');
 let t=s.message||'';if(s.arm)t+=`\\ncontrolling ${s.arm}${s.paused?'  PAUSED':''}  t ${s.time_s.toFixed(1)} s  tcp ${s.tcp}`;
 if(s.unreachable&&s.unreachable.length)t+=`  IK fails: ${s.unreachable}`;
 if(s.check)t+=`\\ncheck at ${(s.check.at/25).toFixed(1)} s: ${s.check.passed?'PASSED':'FAILED: '+s.check.reason}`;
 el.textContent=t;arms(s)}catch(e){}},300);
(async()=>{outside=await (await fetch('/views')).json();setup()})();
loadCases();
</script></body></html>'''


def save_attempt(output, report, case, log, metadata):
    """Save an operated step, keeping the checkpoint's source batch through cleanup."""
    from table_1000.common.batch import load_record
    from table_1000.experts.generate import merge_checkpoints, summarize
    source_batch = Path(case['start']).parent.parent if log['start']['kind'] == 'step' else Path(case['batch'])
    report['teleop'] = {'case': case, **log}
    (output / 'episodes' / f"{report['seed']}.json").write_text(json.dumps(report, indent=1, default=float))
    sources = list((output / 'checkpoints').glob('*.hdf5'))
    merge_checkpoints(sources, output, {key: metadata[key] for key in ('scene', 'robot', 'physics', 'commit')})
    for checkpoint in sources:
        checkpoint.unlink()
    summary = {**metadata, 'from': str(source_batch), **summarize([report], [report['seed']]),
               'teleop': {'case': case, 'start': log['start'], 'verdict': log.get('verdict'),
                         'final_check': log.get('final_check')},
               'load': load_record([report]), 'videos': {}, 'fails': {}}
    (output / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=float) + '\n')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('scene')
    parser.add_argument('--physics', default='isaac-cpu')
    parser.add_argument('--runs', type=Path, default=ROOT / 'runs' / 'expert', help='where the cases come from')
    parser.add_argument('--batches', type=Path, nargs='+', default=[], help='more batches whose cases are listed')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs' / 'teleop', help='where the attempts go')
    parser.add_argument('--views', nargs='+', help="views and robot RGB cameras shown (default: the scene's views "
                                                    "and the robot's RGB cameras)")
    parser.add_argument('--assets', type=Path, default=ROOT / 'assets')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--render-every', type=int, default=3, help='control steps per streamed frame')
    args = parser.parse_args(argv)
    from table_1000.sim.isaaclab.app import start_app
    from table_1000.sim.isaaclab.render import RT2_ARGS
    app = start_app(gpu=args.gpu, workers=4, extra_args=RT2_ARGS)
    try:
        run(args, argv)
    finally:
        app.close()


def run(args, argv):
    import numpy as np
    from table_1000.common.batch import commit
    from table_1000.experts import teleop
    from table_1000.experts.generate import Generator, read_checkpoints
    from table_1000.robots import load_spec
    from table_1000.robots.teleop import Server
    from table_1000.scenes import load
    from table_1000.sim import CONTROL_HZ
    from table_1000.sim.isaaclab import recording, render
    from scripts.scenes.render import camera_sets
    from scripts.scenes.run_expert import run_name
    render.configure()
    scene = load(args.scene, 'scene', ROOT)
    spec = load_spec(Path(scene.ROBOT['asset']).name, ROOT)
    names = args.views or [next(iter(scene.VIEWS)), *(name for name, camera in spec.cameras.items()
                                                      if camera.sensor.kind == 'rgb')]
    cameras, views = camera_sets(scene, spec, names)
    # Robot cameras at most DISPLAY_WIDTH wide: the page shows them small, and rendering sets the pace.
    cameras = {name: camera if camera.sensor.resolution[0] <= DISPLAY_WIDTH else dataclasses.replace(
        camera, sensor=camera.sensor.resized(DISPLAY_WIDTH, round(camera.sensor.resolution[1] * DISPLAY_WIDTH
                                                                  / camera.sensor.resolution[0])))
               for name, camera in cameras.items()}
    generator = Generator(args.scene, ROOT / 'outputs' / 'teleop' / args.scene, args.assets, ROOT, args.physics,
                          cameras=cameras, views=views)
    env, expert = generator.env, generator.expert
    operator = teleop.Operator()
    state = {'message': 'pick a case on the left'}

    def json_reply(value):
        return json.dumps(value, default=float).encode(), 'application/json'

    def on_drag(event):
        operator.drag(event['view'], event.get('dx', 0.), event.get('dy', 0.), event.get('wheel', 0.))
        return b'ok', 'text/plain'

    def on_view(event):
        operator.move_view(event['view'], event['kind'], event.get('dx', 0.), event.get('dy', 0.))
        return b'ok', 'text/plain'

    def on_command(event):
        operator.command(event.pop('command'), **event)
        return b'ok', 'text/plain'

    server = Server(operator.keys, names, args.port, page=PAGE,
                    gets={'/cases': lambda: json_reply(teleop.cases(args.runs, args.scene, args.batches)),
                          '/views': lambda: json_reply(list(orbits)),
                          '/state': lambda: json_reply({**state, **operator.status})},
                    posts={'/drag': on_drag, '/view': on_view, '/command': on_command})
    orbits = {name: teleop.Orbit(view) for name, view in views.items()}
    drag_cameras = {name: (lambda ctx, orbit=orbit: orbit.camera()) for name, orbit in orbits.items()}
    for name in cameras:
        def robot_camera(ctx, name=name):
            param = ctx.env.obs['sensor_param'][name]
            pose = param['cam2world'][0]
            return pose[:3, :3], pose[:3, 3], float(param['intrinsic'][0][0, 0])
        drag_cameras[name] = robot_camera
    counter = {'shown': 0}

    def move_views():
        """Apply the operator's moves of the outside views; whether any moved."""
        import torch
        moved = set()
        for name, kind, dx, dy in teleop.Operator.drain(operator.views):
            if name in orbits:
                orbits[name].move(kind, dx, dy)
                moved.add(name)
        origin = env.env.scene.env_origins[:1]
        for name in moved:
            view = orbits[name].view
            env.env.scene[name].set_world_poses_from_view(
                torch.tensor([view['position']], dtype=torch.float32, device=env.device) + origin,
                torch.tensor([view['target']], dtype=torch.float32, device=env.device) + origin)
        return bool(moved)

    def publish():
        server.publish({name: outputs['rgb'][0] for name, outputs in env.render(names).items()})

    def show(ctx, status, targets, arm):
        operator.status = status
        counter['shown'] += 1
        if move_views() or not counter['shown'] % args.render_every:
            env.set_markers([(ctx.base_transform() @ targets[arm])[:3, 3]], .012)
            publish()

    # Kit takes over SIGINT and SIGTERM (also while the environment is built): from here Ctrl+C or
    # kill ends the session at once (an attempt under way is lost).
    def stop(signum, frame):
        print(f'stopped by signal {signum}', flush=True)
        os._exit(130)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    publish()
    print(f'teleop on http://<host>:{args.port}/', flush=True)
    checkout = commit(ROOT)
    command = None
    while True:
        if command is None:
            try:
                command = operator.commands.get(timeout=.1)
            except queue.Empty:
                if move_views():        # between cases the outside view still moves
                    publish()
                continue
        if command['command'] != 'load':
            command = None         # check / done outside a case
            continue
        case, kind, back_s = command['case'], command.get('start', 'step'), float(command.get('back_s') or 0.)
        command = None
        step, seed = int(case['step']), int(case['seed'])
        path = Path(case['start'])
        found = read_checkpoints(path)[1] if path.is_file() else {}
        if kind == 'failure':
            if not case.get('trajectory'):
                state['message'] = f'seed {seed}: the failed run of this case was not recorded, start from the step'
                continue
            # The state back_s before the failure, not before the failed step began.
            index = max(int(case['step_at']), int(case['failure_at']) - 1 - round(back_s * CONTROL_HZ))
            report = json.loads((Path(case['batch']) / 'episodes' / f'{seed}.json').read_text())
            if report.get('call_trace') and not back_s:
                from table_1000.experts.trace import snapshot_state
                initial, snapshot = snapshot_state(case['batch'], report)
                index = snapshot['frame']['enter'] - 1
            else:
                initial = teleop.trajectory_state(case['trajectory'], seed, index)
            attempt, start = (found[seed][0], found[seed][2]) if seed in found else \
                (int(case.get('attempt') or 0), teleop.trajectory_start(case['trajectory'], seed))
            source, label = f'{case["trajectory"]}#demo_{seed}/states/{index}', f'step{step}-seed{seed}-failure'
            begins = {'kind': 'failure', 'index': index, 'before_failure_s': (int(case['failure_at']) - 1 - index) / CONTROL_HZ}
        else:
            if seed not in found:
                state['message'] = f'seed {seed} has no checkpoint in {path}'
                continue
            attempt, initial, start = found[seed]
            start = None if step == 1 else start
            source, label, begins = str(path), f'step{step}-seed{seed}', {'kind': 'step'}
        name = run_name(checkout, label)
        output = args.output / args.scene / name
        output.mkdir(parents=True)
        for sub in ('episodes', 'checkpoints'):
            (output / sub).mkdir()
        generator.output, generator.steps, generator.source = output, (step, step), source
        recording.redirect(env, output)
        log = {'start': begins}
        generator.expert = teleop.HumanExpert(
            expert, step, lambda ctx, entry: teleop.operate(ctx, entry, operator, show, drag_cameras, log))
        where = 'its start' if kind == 'step' else f'{begins["before_failure_s"]:.1f} s before the failure'
        state['message'] = (f'case: step {step} ({case["step_name"]}), seed {seed}, from {case["run"]}, at {where}\n'
                            f'recording to {output}')
        operator.status = {}
        report = generator.run(seed, attempt, initial, start)
        command = log.pop('next', None)
        save_attempt(output, report, case, log,
                     {'run': name, 'command': shlex.join(sys.argv if argv is None else ['teleop.py', *argv]),
                      'scene': args.scene, 'robot': spec.name, 'physics': args.physics, 'commit': checkout,
                      'steps_run': [step, step]})
        final = log.get('final_check') or {}
        state['message'] = (f'saved {output.name}: {report["status"]} (operator: {log.get("verdict")}; check '
                            f'{"passed" if final.get("passed") else "failed: " + str(final.get("reason"))})'
                            + ('' if report['status'] != 'error' else '\n' + report['failure']['traceback'][-600:]))
        operator.status = {}


if __name__ == '__main__':
    main()
