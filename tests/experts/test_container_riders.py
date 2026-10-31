"""Native MPlib holds must carry supported contents, including protruding pens."""
from types import SimpleNamespace

import mplib
import numpy as np
import pytest

from table_1000.experts.grasps import Box
from table_1000.experts.skills.common import Target
from table_1000.experts.skills.grasping import Grasping
from table_1000.experts.skills.motion import Motion
from table_1000.experts.skills import semantic
from table_1000.experts.solver import MotionSolver
from table_1000.planning.screw import plan_screw
from table_1000.planning.world import PlanningObject, PlanningScene, pose
from table_1000.robots.planning_model import write_srdf
from tests.planning.test_model_world import robot


def geometry(parts):
    entries = []
    for name, size, centre in parts:
        frame = np.eye(4)
        frame[:3, 3] = centre
        entries.append(('body', name, frame, 'convex', Box(size).points()))
    return SimpleNamespace(shapes=entries, bodies=['body'],
        points=lambda *a: np.vstack([p @ t[:3, :3].T + t[:3, 3] for _, _, t, _, p in entries]))


def container_hold(tmp_path, selection, length, neighbour_gap=.002):
    source = tmp_path / 'source'
    source.mkdir()
    urdf = robot(source)
    srdf = tmp_path / 'robot.srdf'
    write_srdf(urdf, srdf)
    planner = mplib.Planner(str(urdf), 'l3', srdf=str(srdf))
    start = np.array([0., .05, 0.])
    planner.robot.set_qpos(start, True)
    pin = planner.robot.get_pinocchio_model()
    pin.compute_forward_kinematics(start)
    tcp = pin.get_link_pose(planner.link_name_2_idx['l3']).to_transformation_matrix()
    cup = geometry([('floor', (.084, .084, .01), (0., 0., 0.)),
                    ('left', (.004, .084, .08), (-.04, 0., .04)),
                    ('right', (.004, .084, .08), (.04, 0., .04)),
                    ('front', (.076, .004, .08), (0., -.04, .04)),
                    ('back', (.076, .004, .08), (0., .04, .04))])
    item = geometry([('body', (.006, .006, length), (0., 0., 0.))])
    floating = geometry([('body', (.005, .005, .005), (0., 0., 0.))])
    assets = dict(cup=cup, item=item, neighbour=cup, floating=floating)
    poses = {}
    for obj, xyz in [('cup', (.4, .3, .7)), ('item', (.4, .3, .706 + length / 2)),
                     ('neighbour', (.484 + neighbour_gap, .3, .7)), ('floating', (.42, .32, .75))]:
        poses[obj + ':body'] = np.eye(4)
        poses[obj + ':body'][:3, 3] = xyz
    ctx = SimpleNamespace(geometry=assets, poses=lambda: poses, scene=SimpleNamespace(OBJECTS=assets),
                          spec=SimpleNamespace(base=False), note=lambda *a, **kw: None, solvers={})
    solver = MotionSolver.__new__(MotionSolver)
    solver.ctx, solver.model, solver.planner = ctx, SimpleNamespace(base=np.eye(4)), planner
    solver.arm = SimpleNamespace(name='left', tcp_link='l3', finger_links=('l3',))
    solver.world, solver.attachments = PlanningScene(planner), {}
    solver.tcp, solver.finger_gap, solver.close_gripper = lambda: tcp, lambda: .006, lambda: None
    ctx.solvers['left'] = solver
    for obj, asset in assets.items():
        for body, part, frame, kind, points in asset.shapes:
            name = obj + ':' + part
            solver.world.add(PlanningObject(name, obj + ':' + body, [(frame, kind, points)]), poses[obj + ':body'])

    class Skill(Grasping, Motion):
        pass

    skill = Skill()
    skill.ctx, skill.solvers, skill.arms, skill.held, skill.log = ctx, ctx.solvers, ['left'], {}, []
    skill.rejected = dict.fromkeys(('grasp IK', 'pre-grasp IK', 'then', 'need'), 0)
    skill.constrained, skill.unprepared_flat_thin = lambda _: False, lambda _: False
    skill.check_closure = lambda *a: {'held': True}
    names = [n for n in solver.world.objects if n.startswith('cup:')]
    target = Target('cup', names if selection == 'whole' else ['cup:left'], 'cup:body',
                    Box((.084, .084, .08)), np.eye(4), cup.points(), container=True)
    grasp = SimpleNamespace(pose=tcp, open_gap=.05, width=.006, backoff=0., cost=0., info={}, target=None)
    skill.reach = lambda *a, **kw: ('left', grasp, [])
    assert not semantic.inside(ctx, 'neighbour', 'cup')
    if length == .18:
        assert not semantic.inside(ctx, 'item', 'cup')  # centre above the mouth
    skill.grasp(target)
    assert skill.held['left']['riders'] == ['item:body']
    assert planner.planning_world.is_object_attached('item:body')
    assert not any(planner.planning_world.is_object_attached(n) for n in solver.world.objects
                   if n.startswith(('neighbour:', 'floating:')))
    return skill, solver, planner, start, tcp


@pytest.mark.parametrize('selection', ['whole', 'wall'])
@pytest.mark.parametrize('length', [.04, .18])
def test_native_loaded_container_plans_a_one_centimetre_lift(tmp_path, selection, length):
    _, _, planner, start, tcp = container_hold(tmp_path, selection, length)
    goal = tcp.copy()
    goal[2, 3] += .01
    result = plan_screw(planner, pose(goal), start)
    assert result['status'] == 'Success', str(result)
    planner.robot.set_qpos(result['position'][-1], True)
    assert not planner.planning_world.check_collision()



def test_submillimetre_outside_neighbour_remains_an_unattached_obstacle(tmp_path):
    _, solver, planner, _, _ = container_hold(tmp_path, 'whole', .18, neighbour_gap=.00042)
    assert all(not planner.planning_world.is_object_attached(name)
               for name in solver.world.objects if name.startswith('neighbour:'))
