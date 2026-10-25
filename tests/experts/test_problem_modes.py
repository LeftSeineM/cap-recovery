"""Episode save policies and identical numerical decisions without Isaac."""
import gzip
import json
from types import SimpleNamespace

import numpy as np
import pytest

from table_1000.experts.generate import Generator
from table_1000.experts.context import ExpertFailed
from table_1000.experts.problems import Problems, decision, mark
from table_1000.experts.trace import traced
from table_1000.robots import load_spec
from table_1000.robots.kinematics import RobotModel


@pytest.mark.parametrize('profile', [False, True])
@pytest.mark.parametrize('status', ['success', 'failed', 'error'])
def test_episode_modes_keep_calls_decisions_and_random_streams(tmp_path, monkeypatch, status, profile):
    model = RobotModel(load_spec('x1pro_rm001'), tmp_path / 'planning')
    model.set_state(model.spec.keyframe('ready'))
    ready = model.q.copy()
    env = SimpleNamespace(spec=model.spec, joints=model.joints, steps=0, timings={'step_s': 0.},
                          contacts=SimpleNamespace(phase=None, intervals=lambda: []), targets=lambda: {},
                          privileged=lambda: {'joints': {}, 'poses': {}})
    def reset(state):
        env.steps = 0
        env.obs = {'agent': {'qpos': ready[None].copy(), 'base_pose': np.zeros((1, 3))}}
    env.reset_to_state = reset
    samples = []
    class Choice:
        def __init__(self, ctx):
            self.ctx, self.model = ctx, ctx.model
            self.held, self.reaches = {}, {}
            self.home, self.start_base = ready.copy(), np.zeros(3)
        @decision
        def select(self):
            target = model.tcp('left', world=True)
            unreachable = target.copy()
            unreachable[:3, 3] = 100.
            mark(self, 'IK', 0)
            rejected, _ = model.ik('left', unreachable, world=True)
            mark(self, 'IK', outcome='rejected')
            mark(self, 'IK', 1)
            accepted, q = model.ik('left', target, world=True)
            draw = self.ctx.rng.random(4)
            samples.append((rejected, accepted, q.copy(), draw))
            return {'q': q, 'draw': draw}
    @traced()
    def choose(ctx):
        ctx.solver('left')
        ctx._skills = Choice(ctx)
        result = ctx._skills.select()
        if status == 'failed':
            raise ExpertFailed(ctx.current, 'no candidate')
        if status == 'error':
            raise ValueError('expert bug')
        return result
    def solve(ctx):
        with ctx.subtask('choose'):
            choose(ctx)
    installs, diagnostics = [], []
    install, diagnose = Problems.install, Problems.diagnose
    def installing(record):
        installs.append(record)
        install(record)
    def diagnosing(record, *args):
        diagnostics.append(record)
        diagnose(record, *args)
    monkeypatch.setattr(Problems, 'install', installing)
    monkeypatch.setattr(Problems, 'diagnose', diagnosing)
    calls, decisions, following = {}, {}, {}
    for mode in ('none', 'failed', 'all'):
        generator = object.__new__(Generator)
        generator.name, generator.output, generator.assets = 'fake', tmp_path / mode, tmp_path
        generator.steps, generator.source, generator.problems = None, None, mode
        generator.profile_enabled = profile
        generator.skill_attempts = 1
        generator.env, generator.model, generator.spec = env, model, model.spec
        generator.env_args = {}
        generator.scene = SimpleNamespace(OBJECTS={}, ROBOT={'position': [0., 0., 0.]})
        generator.expert = SimpleNamespace(solve=solve)
        generator.recording = SimpleNamespace(begin=lambda *args: None, finish=lambda *args: None)
        (generator.output / 'episodes').mkdir(parents=True)
        before = len(installs), len(diagnostics)
        calls[mode], decisions[mode], following[mode] = [], [], []
        for repeat in (0, 1):
            report = generator.run(4, 2, {}, repeat=repeat)
            assert report['status'] == status, report.get('failure')
            saved = mode == 'all' or mode == 'failed' and status != 'success'
            name = f'4-{repeat}' if repeat else '4'
            assert report['solve_trace'] == (f'problems/{name}.json.gz' if saved else None)
            assert (generator.output / 'problems' / f'{name}.json.gz').exists() == saved
            with gzip.open(generator.output / report['call_trace'], 'rt') as file:
                calls[mode].append(json.load(file))
            decisions[mode].append(samples[-1])
            following[mode].append(model.ik('left', model.tcp('left', world=True), world=True))
        if mode == 'none':
            assert (len(installs), len(diagnostics)) == before
        else:
            assert len(installs) == before[0] + 2 and len(diagnostics) > before[1]
    for mode in ('failed', 'all'):
        assert calls[mode] == calls['none']
        for actual, expected in zip(decisions[mode], decisions['none']):
            assert actual[:2] == expected[:2]
            np.testing.assert_array_equal(actual[2], expected[2])
            np.testing.assert_array_equal(actual[3], expected[3])
        for actual, expected in zip(following[mode], following['none']):
            assert actual[0] == expected[0]
            np.testing.assert_array_equal(actual[1], expected[1])


def test_resolve_missing_problems_explains_collection(tmp_path, capsys):
    from scripts.experts.resolve import main
    batch = tmp_path / 'batch'
    (batch / 'episodes').mkdir(parents=True)
    (batch / 'episodes/4.json').write_text(json.dumps({'seed': 4, 'solve_trace': None}))
    with pytest.raises(SystemExit) as error:
        main([str(batch), '--seeds', '4', '--output', str(tmp_path / 'resolve')])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert '4: no solve_trace' in message and '--problems all' in message and '--from' in message
