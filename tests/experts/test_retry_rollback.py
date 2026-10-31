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
    env=SimpleNamespace(targets=lambda:{'commands':state['commands']},contacts=SimpleNamespace(phase='pick'))
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
