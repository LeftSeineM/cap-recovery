"""Call frames, frozen requirements and replay without a simulator."""
import gzip
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from table_1000.experts.context import ExpertFailed, Unreachable
from table_1000.experts.grasps import Box, Cylinder
from table_1000.experts.skills.common import Target
from table_1000.experts.trace import CallTrace, decode, encode, frame, select_frame, snapshot_state, traced, restore, replay


def context(seed=4, repeat=0):
    ctx = SimpleNamespace(steps=0, current='put the pen', _skills=None, solvers={},
                          rng=np.random.default_rng(seed), scene=SimpleNamespace(OBJECTS={}),
                          env=SimpleNamespace(targets=lambda: {'gripper': np.array([[0.]])}, contacts=SimpleNamespace(phase=None)))
    ctx.trace = CallTrace(ctx, seed, 2, repeat)
    return ctx


@traced()
def pick(ctx, obj, point=None):
    ctx.steps += 3
    ctx.rng.random()
    raise Unreachable(ctx.current, 'no grasp', obj=obj)


def failed(ctx):
    with pytest.raises(Unreachable):
        with frame(ctx, 'step', 'move pen', {'index': 1}):
            with frame(ctx, 'subtask', 'pick pen'):
                pick(ctx, 'pen', point=np.array([1., 2., 3.]))


def test_tree_bound_arguments_rng_and_failure_path(tmp_path):
    ctx = context()
    original = encode(ctx.rng.bit_generator.state)
    failed(ctx)
    assert [(f['id'], f['parent'], f['kind']) for f in ctx.trace.frames] == [(0, None, 'step'), (1, 0, 'subtask'), (2, 1, 'skill')]
    entry, check = select_frame({'frames': ctx.trace.frames, 'failure_frame': ctx.trace.failure_frame})
    assert entry['id'] == 2 and not check
    assert entry['rng'] == original
    assert entry['enter'] == 0 and entry['exit'] == 3
    assert decode(entry['arguments'])['point'].tolist() == [1., 2., 3.]
    assert entry['exception']['unreachable'] and entry['exception']['details'] == {'obj': 'pen'}
    path = tmp_path / 'calls.json.gz'
    ctx.trace.save(path)
    with gzip.open(path, 'rt') as file:
        assert json.load(file)['frames'] == ctx.trace.frames


def test_checks_and_explicit_control_frame_select_last_concrete_skill():
    ctx = context()
    @traced()
    def release(ctx, arm):
        ctx.steps += 1
    with pytest.raises(ExpertFailed):
        with frame(ctx, 'step', 'place pen') as step:
            release(ctx, 'left')
            step['phase'] = 'check'
            raise ExpertFailed('place pen', 'not inside')
    tree = {'frames': ctx.trace.frames, 'failure_frame': ctx.trace.failure_frame}
    assert select_frame(tree)[0]['name'] == 'release' and select_frame(tree)[1]
    assert select_frame(tree, 0)[0]['id'] == 1
    assert select_frame(tree, 1)[1]  # Explicit concrete selection keeps the failed step check.


def test_caught_failed_attempt_does_not_steal_final_failure_path():
    ctx = context()
    with pytest.raises(Unreachable):
        with frame(ctx, 'step', 'step'):
            try:
                pick(ctx, 'old')
            except Unreachable:
                pass
            pick(ctx, 'new')
    assert ctx.trace.failure_frame == 2


def test_numpy_target_and_shape_roundtrip():
    target = Target('pen', ['pen:body'], 'pen:body', Cylinder(.01, .2), np.eye(4), np.ones((8, 3)))
    restored = decode(json.loads(json.dumps(encode({'target': target, 'supports': ['table'], 'grasp': {'pose': np.eye(4)}, 'shape': Box((1., 2., 3.))}))))
    assert restored['target'].shape == target.shape
    assert np.array_equal(restored['target'].points, target.points)
    assert restored['shape'].size == (1., 2., 3.)


def test_inhand_roundtrip_keeps_frozen_candidates_and_cached_verdicts():
    from table_1000.experts.skills.placing import Placing
    ctx = context()
    skill = Placing()
    skill.ctx = ctx
    skill.model = SimpleNamespace(base=np.eye(4), q=np.zeros(2))
    skill.in_reach_down = lambda *args: pytest.fail('cached verdict should avoid IK')
    ctx.skills = lambda: skill
    pose = np.eye(4)
    key = ('left', 0, 0, 0, 0, 0, 0, skill.model.base.round(3).tobytes(), skill.model.q.round(2).tobytes())
    recipe = {'obj': 'pen', 'target': None, 'options': [(pose, np.array([1., 2., 3.]))], 'clearance': .01,
              'above': .05, 'arms': ['left'], 'verdicts': [(key, True)], 'origin': {'region': 'evaluated'}}
    need = decode(json.loads(json.dumps(encode(skill.restore_need(recipe)))), ctx)
    assert need.obj == 'pen' and need.arms == ['left'] and need.accepts('left', pose)
    hands = need.hands('left', pose)
    assert hands[0][0][2, 3] == .01 and hands[0][1][2, 3] == pytest.approx(.06)


def test_restore_held_attachments_permissions_and_rng(monkeypatch):
    import mplib.pymp
    seeds = []
    monkeypatch.setattr(mplib.pymp, 'set_global_seed', seeds.append)
    ctx = context()
    operations = []
    world = SimpleNamespace(permissions={frozenset(('old', 'finger'))},
                            allow=lambda a,b,*args: operations.append(('allow', a,b,args)),
                            set_padding=lambda p: operations.append(('padding', p)))
    solver = SimpleNamespace(world=world, attach_at=lambda name, pose: operations.append(('attach', name, pose.tolist())))
    skill = SimpleNamespace(held={})
    ctx.skills, ctx.solver = lambda: skill, lambda arm: solver
    entry = {'held': encode({'left': {'target': Target('pen', ['pen'], 'pen', Box((1, 1, 1)), np.eye(4), np.zeros((1, 3))),
                                     'supports': ['table'], 'grasp': {'gap': .02}}}),
             'planners': {'left': {'extra_objects': [], 'padding': .005, 'permissions': [['pen', 'finger']],
                                   'attached': encode({'pen': np.eye(4)})}},
             'reaches': {}, 'current': 'place', 'mplib_seed': 31337, 'rng': encode(ctx.rng.bit_generator.state)}
    expected = ctx.rng.random()
    restore(ctx, entry)
    assert skill.held['left']['target'].obj == 'pen'
    assert operations[2][0:2] == ('attach', 'pen')
    assert operations[-1] == ('allow', ['pen'], ['finger'], ())
    assert ctx.current == 'place' and ctx.rng.random() == expected
    assert seeds == [31337]  # Shared restoration also seeds teleoperation's IK.


def test_restore_preserves_planning_obstacles_before_skill_sync():
    from table_1000.experts.trace import planner_state
    ctx = context()
    # A live scene may already have advanced; the planner still sees its last
    # sync. Rebuilding only from live poses would change the first IK decision.
    stale = np.eye(4)
    stale[0, 3] = .25
    world = SimpleNamespace(permissions=set(), padding=0., objects={},
                            poses={'cabinet:drawer': stale}, frame=np.diag([1., 1., 1., 1.]))
    restored = {}
    robot = SimpleNamespace(get_qpos=lambda: np.array([.2, .4]),
                            set_qpos=lambda q, full: restored.update(q=q.copy(), full=full))
    solver = SimpleNamespace(world=world, attachments={}, planner=SimpleNamespace(robot=robot))
    ctx.solvers = {'left': solver}
    saved = planner_state(ctx)
    world.poses = {}
    world.frame = np.zeros((4, 4))
    world.set_padding = lambda p: None
    world.set_pose = lambda name, pose: world.poses.update({name: pose.copy()})
    world.allow = lambda *args: None
    ctx.skills = lambda: SimpleNamespace()
    ctx.solver = lambda arm: solver
    restore(ctx, {'held': {}, 'reaches': {}, 'planners': saved, 'current': 'place'}, restore_rng=False)
    assert np.array_equal(world.poses['cabinet:drawer'], stale)
    assert np.array_equal(world.frame, np.eye(4))
    assert restored['q'].tolist() == [.2, .4] and restored['full']


@pytest.mark.parametrize('enter, expected', [(0, 10), (2, 21)])
def test_snapshot_takes_entry_state_and_command_not_failure_state(tmp_path, enter, expected):
    ctx = context()
    ctx.steps = enter
    failed(ctx)
    report = {'seed': 4, 'status': 'failed', 'step_names': ['move pen'], 'steps': [], 'call_trace': 'calls/4.json.gz',
              'failure': {'reason': 'no grasp'}}
    ctx.trace.save(tmp_path / report['call_trace'])
    with h5py.File(tmp_path / 'trajectories_failed.hdf5', 'w') as file:
        data = file.create_group('data')
        data.attrs['env_args'] = json.dumps({'action_layout': {'gripper': [0, 1]}})
        demo = data.create_group('demo_4')
        demo.create_dataset('initial_state/x', data=[[10]])
        demo.create_dataset('states/x', data=[[20], [21], [22], [23], [24]])
        demo.create_dataset('actions', data=np.ones((5, 1)))
    state, snapshot = snapshot_state(tmp_path, report)
    assert state['x'][0, 0] == expected
    assert state['targets']['gripper'][0, 0] == 0.
    assert snapshot['frame']['enter'] == enter
    assert json.loads(state['_expert_frame'])['failed_frame_id'] == 2


def test_replay_calls_only_original_skill_and_restores_first_rng(monkeypatch):
    from table_1000.experts import skills
    original = context()
    failed(original)
    entry = original.trace.frames[-1]
    snapshot = {'frame': entry, 'step': 1, 'check': False}
    monkeypatch.setattr(skills, 'pick', pick)
    expert = SimpleNamespace(STEPS=[SimpleNamespace(name='move pen', check=None)])
    for repeat in (0, 1):
        ctx = context(seed=99, repeat=repeat)
        ctx.skills = lambda: SimpleNamespace(held={})
        ctx.checkpoint = lambda *args: None
        with pytest.raises(Unreachable):
            replay(ctx, snapshot, expert)
        recorded = ctx.trace.frames[-1]
        assert recorded['source_id'] == entry['id']
        assert (recorded['rng'] == entry['rng']) == (repeat == 0)
        assert (recorded['mplib_seed'] == entry['mplib_seed']) == (repeat == 0)


def test_frame_snapshot_checkpoint_roundtrip_and_cli(tmp_path):
    from scripts.experts.failure_snapshots import main
    from table_1000.experts.generate import read_checkpoints
    ctx = context()
    failed(ctx)
    batch = tmp_path / 'batch'
    report = {'seed': 4, 'status': 'failed', 'step_names': ['move pen'], 'steps': [], 'call_trace': 'calls/4.json.gz',
              'failure': {'reason': 'no grasp', 'subtask': 'pick'}}
    ctx.trace.save(batch / report['call_trace'])
    (batch / 'episodes').mkdir()
    (batch / 'episodes/4.json').write_text(json.dumps(report))
    (batch / 'summary.json').write_text(json.dumps({'scene': '000002', 'robot': 'x1pro', 'physics': 'isaac-cpu'}))
    with h5py.File(batch / 'trajectories_failed.hdf5', 'w') as file:
        demo = file.create_group('data/demo_4')
        demo.attrs['num_samples'] = 3
        demo.create_dataset('initial_state/x', data=[[10]])
        demo.create_dataset('obs/qpos', data=np.zeros((3, 2)))
        demo.create_dataset('obs/base_pose', data=np.zeros((3, 3)))
    main([str(batch), '--output', str(tmp_path / 'snapshots')])
    path = next((tmp_path / 'snapshots').glob('000002/*/checkpoints/step0.hdf5'))
    _, states = read_checkpoints(path)
    state = states[4][1]
    assert state['x'][0, 0] == 10
    assert json.loads(state['_expert_frame'])['frame']['name'] == 'pick'
    assert states[4][2]['qpos'].shape == (2,)


def test_replay_check_failure_is_recorded_as_step_check(monkeypatch):
    from table_1000.experts import skills
    ctx = context()
    @traced()
    def release(ctx, arm):
        return {'arm': arm}
    with frame(ctx, 'step', 'put pen'):
        release(ctx, 'left')
    entry = ctx.trace.frames[-1]
    def check(ctx):
        raise ExpertFailed(ctx.current, 'not inside')
    expert = SimpleNamespace(STEPS=[SimpleNamespace(name='put pen', check=check)])
    new = context()
    new.skills = lambda: SimpleNamespace(held={})
    monkeypatch.setattr(skills, 'release', release)
    with pytest.raises(ExpertFailed, match='not inside'):
        replay(new, {'frame': entry, 'step': 1, 'check': True}, expert)
    assert new.trace.failure_frame == 0
    assert new.trace.frames[0]['phase'] == 'check'
    assert select_frame({'frames': new.trace.frames, 'failure_frame': 0})[0]['name'] == 'release'


def test_successful_skill_replay_does_not_mark_whole_step_complete(monkeypatch):
    from table_1000.experts import skills
    ctx = context()
    @traced()
    def release(ctx, arm):
        return {'arm': arm}
    with frame(ctx, 'step', 'step', {'index': 1}):
        release(ctx, 'left')
    entry = ctx.trace.frames[-1]
    new = context()
    new.skills = lambda: SimpleNamespace(held={})
    new.checkpoint = lambda *args: pytest.fail('a skill replay must not complete its entire step')
    monkeypatch.setattr(skills, 'release', release)
    assert replay(new, {'frame': entry, 'step': 1, 'check': False},
                  SimpleNamespace(STEPS=[SimpleNamespace(name='step')])) == {'arm': 'left'}


def test_snapshot_of_resumed_episode_keeps_original_robot_start(tmp_path):
    from scripts.experts.failure_snapshots import main
    from table_1000.experts.generate import read_checkpoints
    from table_1000.scenes.initial import write_tree
    ctx = context()
    failed(ctx)
    parent, batch = tmp_path / 'parent', tmp_path / 'batch'
    start = {'qpos': np.array([7., 8.]), 'base_pose': np.array([1., 2., 3.])}
    (parent / 'checkpoints').mkdir(parents=True)
    with h5py.File(parent / 'checkpoints/step0.hdf5', 'w') as file:
        group = file.create_group('4')
        group.attrs['attempt'] = 2
        write_tree(group.create_group('state'), {'x': np.zeros((1, 1))})
        write_tree(group.create_group('start'), start)
    report = {'seed': 4, 'status': 'failed', 'step_names': ['move pen'], 'steps': [], 'steps_run': [1, 1],
              'call_trace': 'calls/4.json.gz', 'failure': {'reason': 'no grasp', 'subtask': 'pick'}}
    ctx.trace.save(batch / report['call_trace'])
    (batch / 'episodes').mkdir()
    (batch / 'episodes/4.json').write_text(json.dumps(report))
    (batch / 'summary.json').write_text(json.dumps({'scene': '000002', 'robot': 'x1pro', 'physics': 'isaac-cpu',
                                                  'steps_run': [1, 1], 'from': str(parent)}))
    with h5py.File(batch / 'trajectories_failed.hdf5', 'w') as file:
        demo = file.create_group('data/demo_4')
        demo.attrs['num_samples'] = 0
        demo.create_dataset('initial_state/x', data=[[10]])
    main([str(batch), '--output', str(tmp_path / 'snapshots')])
    path = next((tmp_path / 'snapshots').glob('000002/*/checkpoints/step0.hdf5'))
    _, states = read_checkpoints(path)
    assert states[4][2]['qpos'].tolist() == [7., 8.]
    assert states[4][2]['base_pose'].tolist() == [1., 2., 3.]
