"""Keyboard teleoperation in a browser, for checking a robot by hand.

The simulator runs headless in the main thread at 25 Hz; a small HTTP server
in a thread serves a page that shows the cameras (an MJPEG stream) and sends
key presses back. Every action is appended to ``actions.jsonl`` so a session
can be replayed (:func:`replay`).

Keys (held down; Shift for faster):

=============  ===========================================
W / S          base forward / back
A / D          base turn left / right
R / F          torso lift up / down
arrows         head yaw / pitch
Tab            switch the controlled arm
I / K          TCP +X / -X (forward / back, base frame)
J / L          TCP +Y / -Y
U / O          TCP up / down
Z / X, C / V,  TCP roll, pitch, yaw (+ / -)
B / N
Space          toggle the gripper
Backspace      reset the scene
=============  ===========================================
"""

import io
import json
from pathlib import Path
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

LINEAR, ANGULAR, ROTATION = .05, .6, .5          # TCP m/s, base / head rad/s, TCP rad/s
BASE_SPEED, LIFT_SPEED = .3, .1                  # m/s
FAST = 2.

PAGE = '''<!doctype html><html><head><meta charset="utf-8"><title>Teleop</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px sans-serif}#v{display:flex;flex-wrap:wrap;gap:6px;padding:6px}
img{max-width:100%;height:auto;border:1px solid #333}#s{padding:6px 10px;white-space:pre;font-family:monospace}</style></head>
<body><div id="v">__IMAGES__</div><div id="s">click the page, then use the keys</div>
<script>
const held=new Set();
function send(k,d){fetch('/key',{method:'POST',body:JSON.stringify({key:k,down:d})})}
addEventListener('keydown',e=>{if(e.repeat)return;e.preventDefault();held.add(e.code);send(e.code,true)});
addEventListener('keyup',e=>{e.preventDefault();held.delete(e.code);send(e.code,false)});
addEventListener('blur',()=>{for(const k of held)send(k,false);held.clear()});
setInterval(async()=>{try{const r=await fetch('/status');document.getElementById('s').textContent=await r.text()}catch(e){}},300);
</script></body></html>'''


class Keys:
    """Key state shared between the HTTP thread and the simulation loop."""

    def __init__(self):
        self.held, self.pressed, self.lock = set(), queue.Queue(), threading.Lock()

    def event(self, key, down):
        with self.lock:
            if down:
                if key not in self.held:
                    self.pressed.put(key)
                self.held.add(key)
            else:
                self.held.discard(key)

    def axis(self, positive, negative):
        with self.lock:
            return float(positive in self.held) - float(negative in self.held)

    def fast(self):
        with self.lock:
            return FAST if {'ShiftLeft', 'ShiftRight'} & self.held else 1.

    def presses(self):
        while not self.pressed.empty():
            yield self.pressed.get()


def action_from_keys(keys, arm, gap, dt, yaw=0.):
    """One control step's action dict from the held keys. TCP keys move along the base's axes;
    the arm's ``ee_delta`` action takes deltas in the articulation's root frame, which is the
    world for a mobile base (its virtual joints start at the origin), so ``yaw`` is the base's
    heading there and 0 for a fixed robot (its root is the base)."""
    k = keys.fast()
    action = {'base': [[BASE_SPEED * k * keys.axis('KeyW', 'KeyS'), ANGULAR * k * keys.axis('KeyA', 'KeyD')]]}
    lift = LIFT_SPEED * k * keys.axis('KeyR', 'KeyF') * dt
    if lift:
        action['lift'] = [[lift]]
    head = [ANGULAR * k * keys.axis('ArrowLeft', 'ArrowRight') * dt, ANGULAR * k * keys.axis('ArrowDown', 'ArrowUp') * dt]  # yaw, pitch
    if any(head):
        action['head'] = [head]
    move = np.array([keys.axis('KeyI', 'KeyK'), keys.axis('KeyJ', 'KeyL'), keys.axis('KeyU', 'KeyO')]) * LINEAR * k * dt
    turn = np.array([keys.axis('KeyZ', 'KeyX'), keys.axis('KeyC', 'KeyV'), keys.axis('KeyB', 'KeyN')]) * ROTATION * k * dt
    if move.any() or turn.any():
        heading = np.array([[np.cos(yaw), -np.sin(yaw), 0.], [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
        action[arm] = [np.r_[heading @ move, heading @ turn].tolist()]
    action[f'{arm}_gripper'] = [[gap]]
    return action


class Server:
    """Serves the page, the camera stream, key events and a status line; ``page`` replaces the
    keyboard page (``__IMAGES__`` becomes the streams), ``gets`` and ``posts`` add paths:
    ``{path: function}`` returning ``(body bytes, content type)``, a POST's function given the JSON
    body."""

    def __init__(self, keys, views, port, page=PAGE, gets=None, posts=None):
        self.keys, self.views, self.frames, self.status = keys, list(views), {}, ''
        gets, posts = gets or {}, posts or {}
        self.condition = threading.Condition()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == '/':
                    images = ''.join(f'<img src="/stream/{name}" data-view="{name}">' for name in server.views)
                    return self.reply(page.replace('__IMAGES__', images).encode(), 'text/html')
                if self.path in gets:
                    return self.reply(*gets[self.path]())
                if self.path == '/status':
                    return self.reply(server.status.encode(), 'text/plain')
                if self.path.startswith('/stream/'):
                    return self.stream(self.path.split('/')[-1])
                self.send_error(404)

            def do_POST(self):
                if self.path != '/key' and self.path not in posts:
                    return self.send_error(404)
                event = json.loads(self.rfile.read(int(self.headers['Content-Length'])) or b'{}')
                if self.path in posts:
                    return self.reply(*posts[self.path](event))
                server.keys.event(event['key'], bool(event['down']))
                self.reply(b'ok', 'text/plain')

            def reply(self, body, kind):
                self.send_response(200)
                self.send_header('Content-Type', kind)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def stream(self, name):
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.end_headers()
                last = None
                try:
                    while True:
                        with server.condition:
                            server.condition.wait_for(lambda: server.frames.get(name) is not last, timeout=5)
                            frame = last = server.frames.get(name)
                        if frame is None:
                            continue
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                                         + str(len(frame)).encode() + b'\r\n\r\n' + frame + b'\r\n')
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.http = ThreadingHTTPServer(('0.0.0.0', port), Handler)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()

    def publish(self, images):
        from PIL import Image
        encoded = {}
        for name, image in images.items():
            buffer = io.BytesIO()
            Image.fromarray(image).save(buffer, format='JPEG', quality=80)
            encoded[name] = buffer.getvalue()
        with self.condition:
            self.frames.update(encoded)
            self.condition.notify_all()


def run(env, layout, output, port, views, render_every=5):
    """Teleoperate ``env`` until interrupted. Every ``render_every`` control steps the
    page gets ``views`` (the env's fixed views and robot cameras)."""
    from table_1000.sim import CONTROL_HZ
    dt = 1 / CONTROL_HZ
    keys = Keys()
    server = Server(keys, views, port)
    arms = list(env.spec.arms)
    arm, gaps = arms[0], {name: 0. for name in arms}
    log = (Path(output) / 'actions.jsonl').open('a', buffering=1)
    obs = env.reset(layout)
    step, wall = 0, time.perf_counter()
    costs = {'physics': 0., 'render': 0., 'total': 0.}
    before = dict(env.timings)
    print(f'teleop on http://<host>:{port}/', flush=True)
    while True:
        started = time.perf_counter()
        for key in keys.presses():
            if key == 'Tab':
                arm = arms[(arms.index(arm) + 1) % len(arms)]
            elif key == 'Space':
                gaps[arm] = env.spec.arms[arm].max_gap * .8 if gaps[arm] == 0. else 0.
            elif key == 'Backspace':
                obs = env.reset(layout)
                wall = time.perf_counter()
                log.write(json.dumps({'reset': True}) + '\n')
        # Keys for parts the robot lacks (a mobile base, a lift, a head) are ignored.
        yaw = obs['agent']['base_pose'][0, 2] if env.spec.base else 0.
        action = {group: value for group, value in action_from_keys(keys, arm, gaps[arm], dt, yaw).items()
                  if group in env.terms}
        obs = env.step(action)
        log.write(json.dumps(action) + '\n')
        step += 1
        base = obs['agent']['base_pose'][0]
        tcp = obs['agent']['tcp'][arm][0][:3, 3]
        server.status = (f'arm {arm}  gap {gaps[arm]:.3f}  t {env.time:7.2f}s  '
                         f'base ({base[0]:+.3f}, {base[1]:+.3f}, {np.degrees(base[2]):+.1f} deg)  '
                         f'tcp ({tcp[0]:+.3f}, {tcp[1]:+.3f}, {tcp[2]:+.3f})'
                         + ''.join(f'  {j} {value:.3f}' for j, value in zip(env.spec.lift, env.qpos(env.spec.lift)[0]))
                         + f'  realtime x{env.time / max(time.perf_counter() - wall, 1e-9):.2f}'
                         + '  ms/step physics {:.0f} render {:.0f} total {:.0f}'.format(
                             *(1e3 * costs[k] / max(step, 1) for k in ('physics', 'render', 'total'))))
        if step % render_every == 0:
            server.publish({name: outputs['rgb'][0] for name, outputs in env.render(views).items()})
        costs['physics'] = env.timings['step_s'] - before['step_s']
        costs['render'] = env.timings['render_s'] - before['render_s']
        costs['total'] += time.perf_counter() - started
        spare = dt - (time.perf_counter() - started)
        if spare > 0:
            time.sleep(spare)


def replay(env, layout, actions):
    """Run a recorded ``actions.jsonl`` from a reset; returns the final observation."""
    obs = env.reset(layout)
    for line in Path(actions).read_text().splitlines():
        entry = json.loads(line)
        obs = env.reset(layout) if entry.get('reset') else env.step(entry)
    return obs
