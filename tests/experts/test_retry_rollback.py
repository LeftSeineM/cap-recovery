"""Retries restore state and archive failed branches; first streams stay unchanged."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from table_1000.experts.context import Unreachable
from table_1000.experts.skills import retry
from table_1000.experts.skills.common import Unmet
from table_1000.experts.audit.trace import CallTrace, traced
from table_1000.experts.retrying import copy_recorded


def context(limit):
    state={'position':0,'commands':0,'steps':0,'record':[0]}
    env=SimpleNamespace(audit_state=lambda: deepcopy(state), targets=lambda:{'commands':state['commands']},contacts=SimpleNamespace(phase='pick'))
    skill=SimpleNamespace(held={},reaches={},log=[])
    ctx=SimpleNamespace(env=env,steps=0,current='pick',_skills=skill,solvers={},rng=np.random.default_rng(7),
        skills=lambda:skill,requirements=[],events=[],subtasks=[],subtask_wall_s=[],subtask_profiles=[],checkpoints=[],skill_attempts=limit,scene=SimpleNamespace(OBJECTS={}))
    ctx.trace=CallTrace(ctx,7,0)
    def capture():return deepcopy(state)
    def restore(saved):
        state.clear();state.update(deepcopy(saved));ctx.steps=state['steps']
    env.capture_retry=capture;env.restore_retry=restore;ctx.sync_model=lambda:None
    ctx.note=lambda text,**kw:ctx.events.append(dict(note=text,**kw))
    archives=[]
    def archive(control,attempt,failure):
        archives.append(dict(state=deepcopy(state),frames=deepcopy(ctx.trace.frames),failure=ctx.trace.failure_frame))
        return dict(batch=f'attempts/7/{control}-{attempt}',frame=ctx.trace.failure_frame,reason=failure.reason)
    ctx.archive_retry=archive
    return ctx,state,archives


@traced()
def move(ctx,state,fail):
    ctx.subtask_profiles.append({'move': {'wall_s': 1.}})
    seed=ctx.trace.frames[-1]['mplib_seed'];random=ctx.rng.random()
    assert state['position']==0 and state['commands']==0
    state.update(position=1,commands=2,steps=3);state['record'].append(1);ctx.steps=3;ctx.events.append({'note':'moved'});ctx._skills.log.append('move')
    if fail:raise Unreachable('pick','random disconnected branch')
    return seed,random


def test_rollback_has_continuous_main_prefix_and_new_streams():
    ctx,state,archives=context(2);count=0;first=[]
    def callback():
        nonlocal count
        count+=1
        first.append((ctx.trace.frames[-1]['id'],ctx.rng.bit_generator.state))
        return move(ctx,state,count==1)
    seed,random=retry(ctx,callback)
    assert count==2 and len(archives)==1
    assert state['record']==[0,1] and ctx.steps==3
    assert ctx._skills.log==['move']
    assert ctx.subtask_profiles==[{'move': {'wall_s': 1.}}]
    assert [f['name']for f in ctx.trace.frames]==['retry','move']
    assert ctx.trace.frames[1]['parent']==0 and ctx.trace.failure_frame is None
    assert archives[0]['frames'][1]['exception']['unreachable']
    assert archives[0]['state']['record']==[0,1]
    assert seed!=archives[0]['frames'][1]['mplib_seed']
    assert ctx.trace.frames[0]['attempts'][0]['frame']==1
    assert ctx.trace.retry_path==[]
    again,_,_=context(1)
    result=retry(again,lambda:move(again,{'position':0,'commands':0,'steps':0,'record':[0]},False))
    assert archives[0]['frames'][1]['mplib_seed']==again.trace.frames[1]['mplib_seed']
    assert archives[0]['frames'][1]['rng']==again.trace.frames[1]['rng']


def test_unmet_adapts_and_program_errors_propagate_without_retry():
    for error in [Unmet('pick','next action unmet'),ValueError('program error')]:
        ctx,state,archives=context(3);calls=[]
        def callback():calls.append(1);raise error
        with pytest.raises(type(error)):retry(ctx,callback)
        assert calls==[1] and archives==[]


def test_recorder_container_copy_is_independent_without_copying_cloned_values():
    value=np.array([2.]);data={'actions':[value],'states':{'q':[value]}}
    saved=copy_recorded(data);data['actions'].append(np.array([3.]));data['states']['q'].clear()
    assert len(saved['actions'])==1 and len(saved['states']['q'])==1
    assert saved['actions'][0]is value


def test_real_planning_world_is_restored_after_failed_attachment_changes(tmp_path):
    from table_1000.robots import load_spec
    from table_1000.robots.kinematics import RobotModel
    from table_1000.experts.solver import MotionSolver
    model=RobotModel(load_spec('x1pro_rm001'),tmp_path/'planning');model.set_state(model.spec.keyframe('ready'))
    ctx,state,archives=context(2);ctx.model=model;ctx.spec=model.spec
    ctx.solvers={arm:MotionSolver(ctx,arm)for arm in model.spec.arms}
    ctx.solver=lambda arm:ctx.solvers[arm]
    solver=ctx.solvers['left'];position=np.eye(4);position[:3,3]=[.7,.2,.8]
    solver.add_box('support',[.1,.1,.1],position);solver.add_box('held',[.01,.01,.01],position)
    solver.attach_at('held',np.eye(4));solver.world.set_padding(.01);solver.allow(['support'])
    ctx._skills.held={'left':{'target':None,'supports':['support'],'grasp':{'gap':.02}}}
    original_q=model.q.copy();permissions=solver.world.permissions.copy();padding=solver.world.object_padding.copy()
    capture=ctx.env.capture_retry;restore=ctx.env.restore_retry
    ctx.env.capture_retry=lambda:dict(physical=capture(),q=model.q.copy())
    def reset(saved):restore(saved['physical']);model.q=saved['q'].copy()
    ctx.env.restore_retry=reset
    calls=[]
    def callback():
        calls.append(1)
        if len(calls)==1:
            solver.detach('held');solver.world.allow(['left_arm_link1'],['right_arm_link1'])
            solver.add_box('extra',[1.,1.,1.],np.eye(4));solver.world.set_padding(.02)
            ctx._skills.held={};model.q+=.01
            raise Unreachable('pick','failed after mutation')
        assert set(solver.world.objects)=={'support','held'}
        assert solver.world.permissions==permissions and solver.world.object_padding==padding
        from mplib.collision_detection import AllowedCollision
        matrix=solver.planner.planning_world.get_allowed_collision_matrix()
        assert matrix.get_entry('left_arm_link1','right_arm_link1')!=AllowedCollision.ALWAYS
        assert solver.planner.planning_world.is_object_attached('held')
        np.testing.assert_array_equal(model.q,original_q)
        np.testing.assert_array_equal(solver.attachments['held'],np.eye(4))
        assert ctx._skills.held['left']['supports']==['support']
    retry(ctx,callback)
    assert len(calls)==2


def test_retry_restores_the_live_inhand_predicate_cache():
    from table_1000.experts.skills.place.strict import Placing
    from table_1000.experts.skills.common import Target
    from table_1000.experts.skills.hold.frames import Box
    ctx,state,archives=context(2)
    skill=Placing();skill.ctx=ctx;skill.model=SimpleNamespace(base=np.eye(4),q=np.zeros(2))
    skill.held,skill.reaches,skill.log={},{},[]
    world=SimpleNamespace(objects={},update=lambda poses:None)
    skill.solvers={'left':SimpleNamespace(world=world,planner=SimpleNamespace(planning_world=SimpleNamespace(is_object_attached=lambda name:False)))}
    ctx.poses=lambda:{'pen:body':np.eye(4)};ctx.skills=lambda:skill;ctx._skills=skill
    checks=[]
    def reachable(*args):checks.append(1);return len(checks)>1
    skill.in_reach_down=reachable
    target=Target('pen',[],'pen:body',Box((.01,.01,.1)),np.eye(4),np.zeros((1,3)))
    need=skill.restore_need(dict(obj='pen',target=target,options=[(np.eye(4),np.zeros(3))],
        clearance=.01,above=.08,arms=['left'],verdicts=[],origin={}))
    calls=[]
    def callback():
        calls.append(1)
        accepted=need.accepts('left',np.eye(4))
        if len(calls)==1:
            assert not accepted
            raise Unreachable('pick','sampled IK branch failed')
        assert accepted
    retry(ctx,callback)
    assert len(checks)==2 and ctx.requirements==[need]


def test_archived_failure_survives_successful_cleanup():
    from table_1000.experts.audit.trace import select_frame
    ctx,state,_=context(2)

    @traced()
    def cleanup(ctx):
        return None

    with ctx.trace.frame('skill','retry',concrete=False)as control:
        try:
            move(ctx,state,True)
        except Unreachable as error:
            cleanup(ctx)
            assert ctx.trace.failure_frame is None
            archived=ctx.trace.failed_branch(error,control['id'])
        selected,check=select_frame(archived)
        assert selected['name']=='move' and not check
        assert selected['exception']['reason']=='random disconnected branch'
        assert archived['frames'][control['id']]['exit']==3
        assert control['exit']is None


def test_callback_check_failure_archives_its_control_and_concrete_children():
    from table_1000.experts.context import ExpertFailed
    from table_1000.experts.audit.trace import select_frame
    ctx,state,_=context(2)
    with ctx.trace.frame('skill','retry',concrete=False)as control:
        move(ctx,state,False)
        archived=ctx.trace.failed_branch(ExpertFailed('pick','callback postcondition'),control['id'])
        assert archived['failure_frame']==control['id']
        assert archived['frames'][control['id']]['exception']['reason']=='callback postcondition'
        selected,check=select_frame(archived)
        assert selected['name']=='move' and not check


def test_numpy_failure_roundtrips_through_retry_archive_calls_and_episode(tmp_path):
    import json
    import time
    import h5py
    import torch
    from table_1000.common.serialization import attribute
    from table_1000.experts.audit.codec import load_document, save_document
    from table_1000.experts.context import ExpertFailed
    from table_1000.experts.generate import Generator

    # Exercise the real archive/export path without starting a physics app.
    # Only Isaac Lab's recorder storage is replaced; HDF5 attrs use the same
    # serializer as TrajectoryFile.write_episode.
    class Episode:
        def __init__(self):
            self.data = {'states': {'x': [torch.tensor([0.]), torch.tensor([1.])]}}

        def pre_export(self):
            self.data['states']['x'] = torch.stack(self.data['states']['x'])

    class File:
        def create(self, path):
            self.file = h5py.File(path + '.hdf5', 'w')

        def add_env_args(self, args):
            self.file.attrs['env_args'] = attribute(args)

        def write_episode(self, episode, seed):
            group = self.file.create_group(f'data/demo_{seed}')
            for key, value in episode.attrs.items():
                group.attrs[key] = attribute(value)

        def close(self):
            self.file.close()

    ctx, state, _ = context(2)
    ctx.start = {'qpos': np.zeros(2), 'base_pose': np.zeros(3)}
    ctx.initial_state, ctx.step_names, ctx.retry_archives = {'x': np.zeros((1, 1))}, ['pick'], []
    generator = Generator.__new__(Generator)
    generator.name, generator.output, generator.steps = '000002', tmp_path, (1, 1)
    generator.identity = dict(robot='x1pro_rm001', expert='0000', scope={'0': None}, layout=0, outside_scope=False)
    generator.spec = SimpleNamespace(name='x1pro_rm001')
    generator.env_args = dict(scene='000002', physics='isaac-cpu')
    generator.env = SimpleNamespace(env=SimpleNamespace(recorder_manager=SimpleNamespace(get_episode=lambda _: Episode())))
    generator.recording = SimpleNamespace(TrajectoryFile=File)
    ctx.archive_retry = lambda control, attempt, failure: generator.archive_retry(ctx, control, attempt, failure, 0)
    details = dict(area_held=np.bool_(False), rise=np.float64(.0002), points=np.array([[1., 2., 3.]]),
                   nested={'contacts': np.array([True, False]), 'count': np.int64(2)})
    expected = dict(area_held=False, rise=.0002, points=[[1., 2., 3.]], nested={'contacts': [True, False], 'count': 2})
    attempts = []

    def callback():
        attempts.append(1)
        if len(attempts) == 1:
            ctx.steps = state['steps'] = 3
            ctx.note('measured', details=details)
            raise ExpertFailed('pick', 'object not lifted', kind='check', criterion_id='lift', **details)
        return 'completed'

    with ctx.trace.frame('step', 'pick', {'index': 1}):
        assert retry(ctx, callback) == 'completed'
    assert len(attempts) == 2 and len(ctx.retry_archives) == 1
    archive = tmp_path / ctx.retry_archives[0]['batch']
    report = json.loads((archive / 'episodes/7.json').read_text())
    assert report['failure']['details'] == report['events'][0]['details'] == expected
    assert report['failure']['details']['area_held'] is False
    with h5py.File(archive / 'trajectories_failed.hdf5') as file:
        assert json.loads(file['data/demo_7'].attrs['failure'])['details'] == expected
    tree = load_document(archive / report['call_trace'])
    from table_1000.experts.audit.records import classify
    assert any(f['category'] == 'a3' for f in classify(tree))
    # Checks can also carry raw numerical values, not only trace.encode tags.
    tree['checks'].append(details)
    save_document(tmp_path / 'calls.json.gz', tree)
    assert load_document(tmp_path / 'calls.json.gz')['checks'][-1] == expected
    (tmp_path / 'episodes').mkdir()
    generator.save(dict(seed=7, repeat=0, failure={'details': details}), time.perf_counter())
    assert json.loads((tmp_path / 'episodes/7.json').read_text())['failure']['details'] == expected
    from table_1000.sim.isaaclab.replay import write_episode
    write_episode(tmp_path / 'replay.hdf5', [{'x': np.zeros((1, 1))}] * 2,
                  dict(failure=details, accepted=np.bool_(False)))
    with h5py.File(tmp_path / 'replay.hdf5') as file:
        assert json.loads(file['data/demo_0'].attrs['failure']) == expected
        assert not file['data/demo_0'].attrs['accepted']


@pytest.mark.parametrize('unavailable', ['missing region', 'flat', 'zero candidates'])
def test_retry_restores_an_unavailable_gravity_requirement_as_unsatisfiable(unavailable):
    from table_1000.experts.skills.chain import InHand
    from table_1000.experts.skills.place.strict import Placing
    from table_1000.experts.audit.trace import encode
    ctx, state, archives = context(2)
    skill = Placing()
    skill.ctx = ctx
    skill.held, skill.reaches, skill.log = {}, {}, []
    ctx.skills = lambda: skill
    ctx._skills = skill
    ctx.geometry = {'pen': SimpleNamespace(solid=lambda *args, **kwargs: True)}
    skill.target = lambda obj: SimpleNamespace()
    skill.low_release_options = lambda *args: []
    origin = dict(near=np.zeros(3), keep_out=[], step=.02, margin=.002)
    if unavailable != 'missing region':
        origin['region'] = (np.eye(4), np.zeros(2), np.ones(2))
    if unavailable == 'flat':
        origin['orientation'] = 'flat'
    recipe = dict(obj='pen', arms=['left'], gravity_release=True, hanging=True, origin=origin)
    need = InHand('pen', lambda arm, pose: True, ['left'], recipe=recipe)
    expected = encode(recipe)
    ctx.requirements.append(need)
    assert skill.low_release_need(need) is None  # optional creation still declines this relaxation
    calls = []
    def callback():
        calls.append(1)
        if len(calls) == 1:
            state['steps'] = ctx.steps = 3
            raise Unreachable('pick', 'release failed after motion')
        assert ctx.requirements == [need]
        assert need.obj == 'pen' and need.arms == ['left']
        assert need.hands is need.rank is None
        assert callable(need.accepts) and not need.accepts('left', np.eye(4))
        assert encode(need.recipe) == expected
    retry(ctx, callback)
    assert len(calls) == 2 and len(archives) == 1


def test_retry_archive_view_writes_the_same_logical_document_before_rollback(tmp_path):
    from table_1000.experts.audit.codec import save_document, load_document
    from table_1000.experts.audit.trace import encode
    ctx, state, _ = context(2)
    with ctx.trace.frame('skill', 'retry', concrete=False) as control:
        try:
            move(ctx, state, True)
        except Unreachable as error:
            payload = encode(np.arange(2400, dtype=float).reshape(800, 3))
            with ctx.trace.search.decision('lookahead', {'options': payload}):
                ctx.trace.search.mark('cache', origin=payload)
            ctx.trace.search.failures.append(dict(id=1, frame=control['id'], branch='main',
                                                  kind='check', state={'physical': payload}))
            expected_frames = deepcopy(ctx.trace.frames)
            expected_frames[error.skill_frame_id]['exception'] = dict(
                type=type(error).__name__, reason=error.reason, details=encode(error.details),
                kind=error.kind, branch=ctx.trace.search.branch, unreachable=True)
            for entry in expected_frames:
                entry.pop('_snapshot', None)
                entry.pop('_clock', None)
                if entry['exit'] is None:
                    entry['exit'] = ctx.steps
            expected = dict(version=2, failure_frame=error.skill_frame_id, frames=expected_frames,
                            **deepcopy(ctx.trace.search.document()))
            archived = ctx.trace.failed_branch(error, control['id'])
            assert archived == expected
            assert '_snapshot' in ctx.trace.frames[control['id']]
            assert ctx.trace.frames[control['id']]['exit'] is None
            path = tmp_path/'calls.json.gz'
            save_document(path, archived)
            # The synchronous writer has finished before rollback changes live metadata.
            ctx.trace.search.failures[0]['reason'] = 'changed after write'
            ctx.trace.search.decisions[0]['events'].append({'stage': 'later candidate'})
            assert load_document(path) == expected
