"""The expert-iteration website: lines of development batches, per-step search forests, media.

Serves runs/expert (or ``--runs``) live: a line's index is rebuilt in the
background when its batches change (``iteration.stale``), so a finished batch
shows up on reload. Media is rendered only when asked (a node, an edge, a step,
or everything the line lacks), unless the site setting ``auto_render`` is on,
which fills a line after each rebuild. Renders run render.py through gq on
recorded states and publish into the line's .iteration/media; their render
batches are removed afterwards.

    python -m table_1000.experts.audit.site --port 60011
"""
import argparse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import itertools
import json
import mimetypes
from pathlib import Path
import shlex
import shutil
import subprocess
import threading
import time
import traceback
from urllib.parse import parse_qs, urlparse

from table_1000.common.serialization import dumps
from table_1000.scenes import dev_batches
from . import iteration
from .iteration import CACHE, read, write

ROOT = Path(__file__).resolve().parents[3]
WEB = Path(__file__).with_name('web')
CAMERAS = ('top', 'left_wrist_view', 'right_wrist_view')
GQ = 'gq submit --mem 4 --cpus 2 --ram 8 --util 85 --owner iteration-viewer --name {name} --wait --'


class Site:
    """Lines, their indexes (built in background threads) and render jobs; thread safe."""

    def __init__(self, runs, *, python='python', launcher=GQ):
        self.runs, self.python, self.launcher = Path(runs).resolve(), python, launcher
        self.lock = threading.Lock()
        self.indexes, self.building, self.jobs, self.facts = {}, {}, [], {}
        self.ids = itertools.count(1)
        self.settings_file = self.runs/CACHE/'settings.json'

    # Lines and indexes.
    def lines(self):
        """``{id: directory}`` of every line (scene/robot/NNNN[-legacy-MMDD])."""
        directories = sorted({batch.parent for batch in dev_batches(self.runs)})
        return {str(d.relative_to(self.runs)): d for d in directories}

    def browsable(self, directory):
        """Whether the line's newest run records search-tree facts; cached per calls file version."""
        calls = iteration.newest_calls(directory)
        if calls is None:
            return False
        stamp = calls, calls.stat().st_mtime_ns
        if stamp not in self.facts:
            self.facts[stamp] = iteration.recorded_facts(calls)
        return self.facts[stamp]

    def line(self, identity):
        directory = self.lines().get(identity)
        if directory is None:
            raise KeyError(f'unknown line {identity!r}')
        if not self.browsable(directory):
            raise ValueError(f'{identity}: recorded before #114, it has no search-tree facts to browse')
        return directory

    def index(self, identity):
        """The line's current index (maybe stale while a rebuild runs) and the build state."""
        directory = self.line(identity)
        with self.lock:
            state = self.building.get(identity)
            if state is None and iteration.stale(directory):
                state = self.building[identity] = dict(state='building', done=0, total=None, batch=None, error=None)
                threading.Thread(target=self._build, args=(identity, directory), daemon=True).start()
            cached = self.indexes.get(identity)
        if cached is None and (directory/CACHE/'index.json').is_file():
            cached = read(directory/CACHE/'index.json')
            with self.lock:
                self.indexes[identity] = cached
        return cached, state

    def _build(self, identity, directory):
        def progress(done, total, batch):
            with self.lock:
                self.building[identity].update(done=done, total=total, batch=batch)
        try:
            value = iteration.line_index(directory, progress=progress)
            with self.lock:
                self.indexes[identity] = value
                del self.building[identity]
            if self.settings().get('auto_render'):
                self.render(identity, 'line')
        except Exception:
            with self.lock:
                self.building[identity].update(state='error', error=traceback.format_exc(limit=4))

    def summary(self, identity):
        index, state = self.index(identity)
        directory = self.line(identity)
        steps = []
        for step in (index or {}).get('steps', []):
            trees = iteration.forest(self._ready(directory, step))
            shown = [i for t in trees for i in [*t['nodes'], *t['edges']] if i.get('media')]
            steps.append(dict(key=key(step), step=step['step'], name=step['name'], deprecated=step['deprecated'],
                              replacement=step.get('replacement'), source_steps=step.get('source_steps'),
                              trees=len(trees), selected_nodes=sum(len(t['nodes']) for t in trees if t['selected']),
                              tree_nodes=sum(len(t['nodes']) for t in trees),
                              failures=dict(Counter(f['category'] for f in step['failures'])),
                              media=sum(i['media_ready'] for i in shown), media_total=len(shown)))
        meta = {k: index[k] for k in ('scene', 'robot', 'expert', 'layout', 'seed', 'joined', 'chain', 'unrecorded', 'before_facts')} if index else {}
        failures = Counter()
        for step in steps:
            failures.update(step['failures'])
        return dict(line=identity, build=state, steps=steps, failures=dict(failures), **meta)

    def step(self, identity, step_key):
        index, _ = self.index(identity)
        step = next(s for s in (index or {}).get('steps', []) if key(s) == step_key)
        return dict(key=step_key, step=step['step'], name=step['name'], deprecated=step['deprecated'],
                    trees=iteration.forest(self._ready(self.line(identity), step)))

    @staticmethod
    def _ready(directory, step):
        for item in [*step['nodes'], *step['edges']]:
            item['media_ready'] = bool(item.get('media')) and (directory/CACHE/item['media']).is_file()
        return step

    # Settings.
    def settings(self):
        return {'auto_render': False, **(read(self.settings_file) if self.settings_file.is_file() else {})}

    def update_settings(self, values):
        settings = self.settings()
        settings.update({k: bool(v) for k, v in values.items() if k == 'auto_render'})
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)
        write(self.settings_file, settings)
        return settings

    # Rendering.
    def render(self, identity, scope, *, step=None, media=()):
        """Queue the missing media of one item (``media`` paths), a step or the whole line."""
        directory = self.line(identity)
        index, _ = self.index(identity)
        if index is None:
            raise ValueError('the line is still being indexed')
        steps = [s for s in index['steps'] if scope == 'line' or (scope == 'step' and key(s) == step)]
        wanted = set(media) if scope == 'item' else {i['media'] for s in steps for i in [*s['nodes'], *s['edges']] if i.get('media')}
        with self.lock:
            pending = {path for job in self.jobs if job['line'] == identity and job['state'] not in ('done', 'failed')
                       for path in job['media']}
        spans = [s for s in index['media'] if s['path'] in wanted and s['path'] not in pending
                 and not (directory/CACHE/s['path']).is_file()]
        if not spans:
            return None
        job = dict(id=next(self.ids), line=identity, scope=scope, step=step, media=[s['path'] for s in spans],
                   stills=sum(s['kind'] == 'still' for s in spans), clips=sum(s['kind'] == 'clip' for s in spans),
                   state='queued', done=0, created=time.time(), error=None)
        with self.lock:
            self.jobs.append(job)
        threading.Thread(target=self._render, args=(job, directory, spans), daemon=True).start()
        return job

    def _render(self, job, directory, spans):
        work = directory/CACHE/'render'/str(job['id'])
        try:
            job['state'] = 'preparing'
            spans = iteration.prepare_media(directory, spans, work)
            name = f'iteration-{directory.parent.name}-{directory.name}-{job["id"]}'
            command = [self.python, str(ROOT/'scripts/scenes/render.py'), read(work/'summary.json')['scene'],
                       '--physics', 'isaac-cpu', '--batch', str(work), '--cameras', *CAMERAS, '--merge', '--captions',
                       '--jobs', '1', '--retries', '0', '--launcher', self.launcher.format(name=name)]
            job['state'] = 'rendering'
            with (work/'render.log').open('w') as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                while process.poll() is None:
                    job['done'] = sum((work/'videos'/str(s['seed'])/'merge.mp4').is_file() for s in spans)
                    time.sleep(5)
            if process.returncode:
                raise RuntimeError((work/'render.log').read_text()[-2000:])
            job['state'] = 'publishing'
            iteration.finish_media(work)
            shutil.rmtree(work)
            job.update(state='done', done=len(spans))
        except Exception as error:
            job.update(state='failed', error=str(error)[-2000:])

    def jobs_of(self, identity):
        with self.lock:
            return [dict(job) for job in self.jobs if identity is None or job['line'] == identity]


def key(step):
    """A step's stable key in URLs: s<number>, or d:<name> when deprecated."""
    return f'd:{step["name"]}' if step['deprecated'] else f's{step["step"]:02d}'


class Handler(BaseHTTPRequestHandler):
    site = None

    def log_message(self, *args):
        pass

    def reply(self, value, status=200):
        body = dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def file(self, path):
        if not path.is_file():
            return self.reply(dict(error='not found'), 404)
        body = path.read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        site = self.site
        try:
            if url.path in ('/', '/index.html'):
                return self.file(WEB/'index.html')
            if url.path in ('/app.js', '/style.css'):
                return self.file(WEB/url.path[1:])
            if url.path.startswith('/media/'):
                identity, _, name = url.path[len('/media/'):].rpartition('/')
                return self.file(site.line(identity)/CACHE/'media'/Path(name).name)
            if url.path == '/api/lines':
                return self.reply([dict(id=i, scene=i.split('/')[0], robot=i.split('/')[1], line=i.split('/')[2],
                                        batches=len(list(d.glob('*/summary.json'))), browsable=site.browsable(d),
                                        latest=iteration.latest(d))
                                   for i, d in site.lines().items()])
            if url.path == '/api/line':
                return self.reply(site.summary(query['id']))
            if url.path == '/api/step':
                return self.reply(site.step(query['line'], query['key']))
            if url.path == '/api/jobs':
                return self.reply(site.jobs_of(query.get('line')))
            if url.path == '/api/settings':
                return self.reply(site.settings())
            return self.reply(dict(error='not found'), 404)
        except Exception as error:
            return self.reply(dict(error=f'{type(error).__name__}: {error}'), 400)

    def do_POST(self):
        url = urlparse(self.path)
        length = int(self.headers.get('Content-Length') or 0)
        body = json.loads(self.rfile.read(length) or b'{}')
        try:
            if url.path == '/api/render':
                job = self.site.render(body['line'], body['scope'], step=body.get('step'), media=body.get('media', ()))
                return self.reply(dict(job=job))
            if url.path == '/api/settings':
                return self.reply(self.site.update_settings(body))
            return self.reply(dict(error='not found'), 404)
        except Exception as error:
            return self.reply(dict(error=f'{type(error).__name__}: {error}'), 400)


def serve(site, port, host='0.0.0.0'):
    handler = type('SiteHandler', (Handler,), dict(site=site))
    server = ThreadingHTTPServer((host, port), handler)
    server.serve_forever()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=Path, default=ROOT/'runs/expert', help='development runs (scene/robot/NNNN/batch)')
    parser.add_argument('--port', type=int, default=60011)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--python', default='python', help='interpreter for render.py (one the gq hosts can run)')
    parser.add_argument('--launcher', default=GQ, help='render launcher; {name} is the job name')
    args = parser.parse_args(argv)
    print(f'iteration site on http://{args.host}:{args.port}/ for {args.runs}', flush=True)
    serve(Site(args.runs, python=args.python, launcher=args.launcher), args.port, args.host)


if __name__ == '__main__':
    main()
