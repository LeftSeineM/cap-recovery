"""Opening and withdrawal determine height before any IK is attempted."""
from types import SimpleNamespace
import numpy as np
import pytest

from table_1000.experts.skills.place.flat import FlatReleaseGeometry
from table_1000.experts.skills.place.strict import convex_keepout


def box(size, centre=(0.,0.,0.)):
    return np.array([[x,y,z] for x in (-size[0]/2,size[0]/2)
                     for y in (-size[1]/2,size[1]/2) for z in (-size[2]/2,size[2]/2)])+centre


def obstacle(name, points):
    return name, convex_keepout(points), (points.min(0),points.max(0))


def hand():
    def frames(gap):
        frame=np.eye(4);frame[0,3]=gap
        return {'finger':frame}
    return SimpleNamespace(shapes={'finger':[(np.eye(4),'box',(.01,.01,.01),True)]},frames=frames)


def test_open_support_uses_mm_and_container_height_bounds_release():
    assert FlatReleaseGeometry([]).minimum(hand(),np.eye(4),np.eye(4),0.,.04,.08)==.002
    assert FlatReleaseGeometry([], .093).maximum==pytest.approx(.153)


def test_measured_opening_caps_release_without_raising_the_lowest_clear_height():
    wall = obstacle('drawer:wall', box((.01, .02, .08), (.02, 0., .04)))
    geometry = FlatReleaseGeometry([wall], .08, ceiling=.06)
    assert geometry.maximum == pytest.approx(.06)
    assert geometry.minimum(hand(), np.eye(4), np.eye(4), 0., 0., .08) == .002
    assert geometry.minimum(hand(), np.eye(4), np.eye(4), 0., .04, .08) is None


def test_opening_checks_mid_gap_and_selects_lowest_clear_height():
    wall=obstacle('container:wall',box((.01,.02,.08),(.02,0.,.04)))
    geometry=FlatReleaseGeometry([wall],.08)
    assert geometry.minimum(hand(),np.eye(4),np.eye(4),0.,0.,.08)==.002
    value=geometry.minimum(hand(),np.eye(4),np.eye(4),0.,.04,.08)
    assert .084 <= value <= .088
    assert 'container:wall' in geometry.blocked_by
    assert geometry.minimum(hand(),np.eye(4),np.eye(4),0.,.04,.08,heights=(.081,)) is None
    assert geometry.minimum(hand(),np.eye(4),np.eye(4),0.,.04,.08,heights=(.088,))==.088


def test_withdrawal_sweep_and_upper_bound_reject_incompatible_hold():
    bridge=obstacle('container:rim',box((.01,.02,.01),(0.,0.,.06)))
    geometry=FlatReleaseGeometry([bridge],.08)
    assert geometry.minimum(hand(),np.eye(4),np.eye(4),0.,0.,0.)==.002
    assert geometry.minimum(hand(),np.eye(4),np.eye(4),0.,0.,.08)>.065
    tall=obstacle('container:wall',box((.01,.02,.2),(.02,0.,.1)))
    incompatible=FlatReleaseGeometry([tall],.05)
    assert incompatible.minimum(hand(),np.eye(4),np.eye(4),0.,.04,.08) is None
    assert incompatible.blocked_by=={'container:wall'}


def test_released_object_stays_behind_and_must_clear_the_open_withdrawal():
    # Closed pad contact is expected; an open finger to the side clears it.
    body = box((.02,.02,.02))
    geometry = FlatReleaseGeometry([], .1, [('held:cover', body)])
    assert geometry.minimum(hand(),np.eye(4),np.eye(4),0.,.04,.08)==.002
    # This cover is clear of the open hand at release, but blocks it above.
    cover = box((.02,.02,.01),(.04,0.,.04))
    blocked = FlatReleaseGeometry([], .1, [('held:cover', cover)])
    assert blocked.minimum(hand(),np.eye(4),np.eye(4),0.,.04,0.)==.002
    assert blocked.minimum(hand(),np.eye(4),np.eye(4),0.,.04,.08) is None
    assert blocked.blocked_by=={'held:cover'}
    # Back out of the cover before rising; raising the release cannot fix
    # this relative collision, while this screened retreat can.
    assert blocked.minimum(hand(),np.eye(4),np.eye(4),0.,.04,.08,.05)==.002
    configured = FlatReleaseGeometry([], .1, [('held:cover', cover)], backoff=.05)
    assert configured.minimum(hand(),np.eye(4),np.eye(4),0.,.04,.08)==.002


def test_interval_search_keeps_the_first_gap_below_overhead_obstacles():
    # Independent axis-aligned intersection intervals, including multiple
    # disjoint blocked height bands and a clear window below an overhead box.
    rng = np.random.default_rng(95)
    channels = [(np.array(lo),np.array(hi)) for lo,hi in [
        ((-.005,-.005,-.005),(.045,.005,.005)),
        ((-.015,-.005,-.005),(.045,.005,.005)),
        ((-.015,-.005,-.005),(-.005,.005,.035))]]
    for _ in range(20):
        obstacles = [obstacle(f'wall{i}',box((.02,.02,rng.uniform(.005,.03)),
                                              (rng.uniform(-.06,.06),0.,rng.uniform(.01,.18))))
                     for i in range(8)]
        geometry = FlatReleaseGeometry(obstacles,.08)
        levels = np.r_[np.arange(.002,geometry.maximum-1e-9,.002),geometry.maximum]
        expected = next((float(height) for height in levels if not any(
            np.all(np.minimum(high+[0.,0.,height],bounds[1])
                   - np.maximum(low+[0.,0.,height],bounds[0]) > 1e-8)
            for low,high in channels for _,_,bounds in obstacles)),None)
        actual = geometry.minimum(hand(),np.eye(4),np.eye(4),0.,.04,.03,.05)
        assert actual is None if expected is None else actual == pytest.approx(expected)
