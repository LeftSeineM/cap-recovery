"""A nearest release IK branch may have no above pose at its torso height."""
from types import SimpleNamespace
import numpy as np
from table_1000.experts.skills.place.strict import Placing


def test_need_and_place_keep_a_release_branch_that_can_reach_above():
    calls=[]
    def ik(arm,pose,q,**options):
        calls.append(options)
        if options.get('closest') is False:return 'Success',[np.array([0.,0.]),np.array([.5,0.])]
        if q[0]==0.:return 'IK Failed',None
        return 'Success',np.array([.5,1.])
    skill=object.__new__(Placing)
    skill.model=SimpleNamespace(q=np.zeros(2),ik=ik)
    skill.ctx=SimpleNamespace(trace=None)
    assert skill.in_reach_down('arm',np.eye(4),np.eye(4),.01,.08)
    goals=skill.down_goals('arm',np.eye(4),.08,True,20)
    np.testing.assert_allclose(goals,[[.5,1.]])
    assert calls[0]['attempts']==5 and calls[3]['attempts']==20
    assert all(not option.get('lift',False) for option in (calls[1],calls[2],calls[4],calls[5]))


def test_failed_transfer_plans_try_one_different_base_stance():
    target = SimpleNamespace(obj='dish', body='dish:body', points=np.array([[0.,0.,.2]]))
    skill = Placing()
    skill.held = {'left': {'target': target}}
    skill.ctx = SimpleNamespace(poses=lambda: {'dish:body': np.eye(4)},
                               spec=SimpleNamespace(base=object()), hold=lambda _: None, note=lambda *a, **kw: None)
    skill.solvers = {'left': SimpleNamespace(tcp=lambda: np.eye(4))}
    skill.placements = lambda *a, **kw: [(np.eye(4), np.zeros(3))]
    outcomes = iter([dict(done=False, planned=8), dict(done=True, planned=1, spot=np.zeros(3))])
    skill.put_down = lambda *a: next(outcomes)
    skill.stow = lambda: False
    moved = []
    skill.make_room = lambda *a, **kw: moved.append(kw)
    skill.release = lambda *a, **kw: None
    result = skill.place('dish', (np.eye(4), [-1., -1.], [1., 1.]), arm='left')
    assert result['arm'] == 'left'
    assert moved == [dict(need=1, touch=[], relocate=True)]


def test_high_destination_raises_live_payload_to_its_approach_plane():
    target = SimpleNamespace(obj='book', body='book:body', points=np.array([[0., 0., 0.]]))
    skill = Placing()
    skill.held = {'right': {'target': target}}
    pose = np.eye(4)
    skill.ctx = SimpleNamespace(poses=lambda: {'book:body': pose}, note=lambda *a, **kw: None,
                               spec=SimpleNamespace(base=object()), hold=lambda _: None)
    skill.solvers = {'right': SimpleNamespace(tcp=lambda: np.eye(4))}
    skill.placements = lambda *a, **kw: [(np.eye(4), np.zeros(3))]
    outcomes = iter([dict(done=False, planned=0), dict(done=True, planned=1, spot=np.zeros(3))])
    skill.put_down = lambda *a: next(outcomes)
    skill.stow = lambda: False
    moved = []
    def move(arm, displacement):
        moved.append((arm, displacement))
        pose[:3, 3] += displacement
    skill.move_by_displacement = move
    skill.make_room = lambda *a, **kw: (_ for _ in ()).throw(AssertionError('already reachable'))
    skill.release = lambda *a, **kw: None
    frame = np.eye(4)
    frame[2, 3] = .5
    skill.place('book', (frame, [-1., -1.], [1., 1.]), arm='right')
    assert moved[0][0] == 'right'
    np.testing.assert_allclose(moved[0][1], [0., 0., .58])
