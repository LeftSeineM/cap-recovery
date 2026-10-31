"""Handles, pull directions and places to put things down, derived from asset geometry alone."""

from table_1000.experts.skills.drawer import joints as drawer_joints
from table_1000.experts.skills.place import regions
from table_1000.experts.world import geometry as world_geometry

import numpy as np
import pytest

from table_1000.experts.skills import semantic
from table_1000.scenes.geometry import AssetGeometry
from table_1000.scenes.layout import yaw_pose


def box(low, high):
    return [[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])]


def cabinet_model(pull):
    """A box cabinet (300 x 300 x 110 mm, open at its -Y face) with one drawer sliding out along
    ``pull`` (a unit vector in the cabinet frame, here -Y or +X with the opening turned there):
    the drawer frame turns with the opening, so the drawer is the same in its own frame."""
    yaw = np.arctan2(pull[1], pull[0]) + np.pi / 2       # the turn taking -Y to pull
    turn = yaw_pose(0., 0., 0., yaw)
    nodes = [{'name': 'cab', 'parent': None, 'matrix': np.eye(4).tolist(), 'rigid': False}]

    def body(name, matrix, parts):
        nodes.append({'name': name, 'parent': 'cab', 'matrix': np.asarray(matrix).tolist(), 'rigid': True})
        for part, (low, high) in parts.items():
            nodes.append({'name': f'{name}.{part}', 'parent': name, 'matrix': np.eye(4).tolist(), 'rigid': False})
            nodes.append({'name': f'{name}.{part}.collision', 'parent': f'{name}.{part}', 'role': 'collision',
                          'matrix': np.eye(4).tolist(), 'rigid': False, 'vertices': box(low, high)})

    # Housing walls in the turned frame (opening toward -Y there), drawer in the opening.
    walls = {'top': ((-.15, -.15, .10), (.15, .15, .11)), 'base': ((-.15, -.15, 0.), (.15, .15, .01)),
             'left': ((-.15, -.15, .01), (-.14, .15, .10)), 'right': ((.14, -.15, .01), (.15, .15, .10)),
             'back': ((-.14, .14, .01), (.14, .15, .10))}
    body('cab.housing', turn, {k: v for k, v in walls.items()})
    drawer = turn @ yaw_pose(0., -.15, .02, 0.)
    body('cab.drawer', drawer, {'bottom': ((-.13, 0., 0.), (.13, .28, .01)), 'front': ((-.135, -.01, 0.), (.135, 0., .075)),
                                'side1': ((-.13, 0., .01), (-.12, .28, .075)), 'side2': ((.12, 0., .01), (.13, .28, .075)),
                                'rear': ((-.12, .27, .01), (.12, .28, .075)), 'handle': ((-.04, -.03, .03), (.04, -.01, .05))})
    joint = turn @ yaw_pose(0., -.15, .02, -np.pi / 2)    # x axis along -Y of the turned frame: out
    return {'nodes': nodes, 'joints': [{'name': 'cab.drawer.slide', 'type': 'SLIDER', 'body1': 'cab.housing',
                                        'body2': 'cab.drawer', 'matrix': joint.tolist(), 'limits': [0., .25]}]}


def posed(geometry, transform, joint, opening):
    poses = geometry.body_poses(transform)
    poses[joint.child] = poses[joint.child].copy()
    poses[joint.child][:3, 3] += joint.direction(poses[joint.child]) * opening
    return poses


TABLE = {'size': (1., .8), 'height': .7, 'top_thickness': .03, 'center': (0., 0.)}


@pytest.mark.parametrize('pull', [(0., -1., 0.), (1., 0., 0.)])
def test_drawer_handle_and_pull_direction(pull):
    geometry = AssetGeometry(cabinet_model(pull))
    joint = drawer_joints.Joint.of(geometry, 'cab.drawer.slide')
    assert (joint.kind, joint.parent, joint.child, joint.handle) == ('SLIDER', 'cab.housing', 'cab.drawer',
                                                                    'cab.drawer.handle')
    transform = yaw_pose(.1, .2, TABLE['height'], .4)
    assert np.allclose(joint.direction(geometry.body_poses(transform)['cab.drawer']),
                       transform[:3, :3] @ pull, atol=1e-9)


@pytest.mark.parametrize('pull', [(0., -1., 0.), (1., 0., 0.)])
def test_drawer_floor_out_of_the_cabinet(pull):
    """Only the part of the drawer's floor pulled out of the housing is a place to put things in."""
    geometry = AssetGeometry(cabinet_model(pull))
    joint = drawer_joints.Joint.of(geometry, 'cab.drawer.slide')
    transform = yaw_pose(-.1, .1, TABLE['height'], .3)
    closed = world_geometry.asset_shapes('cab', geometry, posed(geometry, transform, joint, 0.))
    # Shut, only its front wall and handle are out of the housing: no floor (at 0.03) to put things on.
    region, _ = world_geometry.support_region(closed, 'cab', 'cab.drawer', floor=True)
    assert (np.linalg.inv(transform) @ region[0])[2, 3] > .05
    poses = posed(geometry, transform, joint, .18)
    (frame, low, high), keep_out = world_geometry.support_region(world_geometry.asset_shapes('cab', geometry, poses), 'cab',
                                                           'cab.drawer', floor=True)
    step = world_geometry.SUPPORT_STEP
    local = np.linalg.inv(poses['cab.drawer']) @ frame
    assert np.allclose(local[:3, :3], np.eye(3)) and local[2, 3] == pytest.approx(.01, abs=1e-6)
    # Between the side walls, from the front wall to the housing's face (0.18 out of it).
    assert np.allclose(low, [-.12, 0.], atol=step / 2 + 1e-9)
    assert np.allclose(high, [.12, .18], atol=step / 2 + 1e-9)
    assert keep_out == []


def test_table_keeps_out_what_stands_on_it():
    geometry = AssetGeometry(cabinet_model((0., -1., 0.)))
    joint = drawer_joints.Joint.of(geometry, 'cab.drawer.slide')
    transform = yaw_pose(.2, .1, TABLE['height'], .5)
    poses = posed(geometry, transform, joint, .1)
    shapes = [world_geometry.table_shape(TABLE), *world_geometry.asset_shapes('cab', geometry, poses)]
    (frame, low, high), keep_out = world_geometry.support_region(shapes, 'table')
    assert frame[2, 3] == pytest.approx(TABLE['height'])
    assert np.allclose(high, [.5, .4], atol=world_geometry.SUPPORT_STEP) and np.allclose(low, [-.5, -.4],
                                                                                 atol=world_geometry.SUPPORT_STEP)
    # The housing and the drawer pulled out of it, each by its collision box in its own frame.
    assert {tuple(np.round(f[:3, 3], 6)) for f, _, _ in keep_out} == {
        tuple(np.round(poses[body][:3, 3], 6)) for body in ('cab.housing', 'cab.drawer')}
    housing = next((l, h) for f, l, h in keep_out if np.allclose(f, poses['cab.housing']))
    assert np.allclose(housing, geometry.bounds('cab.housing', 'cab.housing'))


def test_cast_hits_the_nearest_face():
    shapes = [world_geometry.Shape('a', 'a', 'a', np.eye(4), np.array(box((0., 0., 0.), (1., 1., 1.)))),
              world_geometry.Shape('b', 'b', 'b', np.eye(4), np.array(box((0., 0., 2.), (1., 1., 3.))))]
    first, distance, normal = world_geometry.cast(shapes, [[.5, .5, 5.], [.5, .5, 1.5], [3., 3., 5.]], [0., 0., -1.])
    assert first.tolist() == [1, 0, -1]
    assert distance[:2] == pytest.approx([2., .5]) and distance[2] == np.inf
    assert np.allclose(normal[:2], [[0., 0., 1.]] * 2)


def test_what_hangs_high_over_a_floor_is_no_obstacle():
    """A board 0.11 m over the front part of the drawer's floor hides it, unless what is put down
    comes down from lower than that (as a drawer's handle over the one below); the housing's top,
    0.07 m over the floor, still hides the part inside."""
    geometry = AssetGeometry(cabinet_model((0., -1., 0.)))
    joint = drawer_joints.Joint.of(geometry, 'cab.drawer.slide')
    poses = posed(geometry, np.eye(4), joint, .18)
    drawer = poses['cab.drawer']
    corners = np.array(box((-.2, 0., .12), (.2, .09, .13))) @ drawer[:3, :3].T + drawer[:3, 3]
    shapes = [*world_geometry.asset_shapes('cab', geometry, poses),
              world_geometry.Shape('board', 'board', 'board', np.eye(4), corners)]
    (_, low, _), keep_out = world_geometry.support_region(shapes, 'cab', 'cab.drawer', floor=True)
    assert low[1] == pytest.approx(.09, abs=world_geometry.SUPPORT_STEP / 2 + 1e-9)
    (_, low, high), keep_out = world_geometry.support_region(shapes, 'cab', 'cab.drawer', floor=True, headroom=.09)
    assert np.allclose([low[1], high[1]], [0., .18], atol=world_geometry.SUPPORT_STEP / 2 + 1e-9) and keep_out == []


def test_put_on_keeps_room_at_the_edges():
    """On the table, an object stays EDGE inside its edges (fingers reach past it when picking it up)."""
    region, _ = world_geometry.support_region([world_geometry.table_shape(TABLE)], 'table')
    frame, low, high = regions.around(regions.inset(region, regions.EDGE), [0., -.3, TABLE['height']])
    assert np.allclose(low, [-.3, -.4 + regions.EDGE], atol=world_geometry.SUPPORT_STEP)
    assert np.allclose(high, [.3, 0.], atol=world_geometry.SUPPORT_STEP)


def test_near_off_the_surface_takes_its_nearest_part():
    """A near point off the table (the robot turned away) still leaves room: its nearest corner."""
    region, _ = world_geometry.support_region([world_geometry.table_shape(TABLE)], 'table')
    _, low, high = regions.around(regions.inset(region, regions.EDGE), [-.9, -.7, TABLE['height']])
    corner = np.array([-.5, -.4]) + regions.EDGE
    assert np.allclose(low, corner, atol=world_geometry.SUPPORT_STEP)
    assert np.allclose(high, corner + regions.NEAR, atol=world_geometry.SUPPORT_STEP)
    _, low, high = regions.inset((np.eye(4), np.zeros(2), np.array([.05, .3])), regions.EDGE)
    assert np.allclose(low, [.025, .04]) and np.allclose(high, [.025, .26])
