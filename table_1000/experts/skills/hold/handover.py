"""Handover: the held object passed from one hand to the other, held as the next action needs.

After WRS (wanweiwei07/wrs, MIT), ``manipulation/placement/handover.py``: a handover pose of the
object is kept where the giving hand reaches it with its hold, there is a receiving grasp whose
hand is clear of the giving one, and the receiving arm reaches that grasp with the giving arm at
its pose. WRS screens such poses offline into a library searched as a graph; here they are
screened as needed, one handover ahead, in front of the body between the shoulders (``REACH`` ahead,
``SIDES`` toward the giving one, ``HEIGHTS`` above where the object is, the object turned by ``TURNS``
about the vertical).

The giving arm plans to the first pose whose screening passes, the receiving one to the pre-grasp
and straight in, closes and must hold; then the giving hand lets go and backs out
(:meth:`~table_1000.experts.skills.push.motion.Motion.release`). The torso lift, shared by the
arms, moves only with the giving arm, before the receiving one reaches.
"""

import itertools
from dataclasses import replace

from table_1000.common.timing import timed

import numpy as np
from scipy.spatial.transform import Rotation

from table_1000.experts.skills.hold.frames import grasp_candidates, standoff
from table_1000.experts.skills.common import APPROACH, CANDIDATE_PLANS, IK_ATTEMPTS, Unmet, collisions, normalized, upright_need
from table_1000.experts.solver import PlanningFailed
from table_1000.experts.audit.problems import decision, mark

REACH = (.30, .20)                  # m ahead of the point between the shoulders
HEIGHTS = (0., .10)                 # m above the object's height now, or TABLE_CLEARANCE above the table
TABLE_CLEARANCE = .12               # m
SIDES = (.15, .30)                  # m from between the shoulders toward the giving one
TURNS = np.radians([0., 90., -90., 180.])   # of the object about the vertical, from how it is held
MAX_RECEIVING_IK = 8                # receiving grasps checked by IK per handover pose
MAX_HANDOVER_IK = 30                # poses and grasps checked by IK in all
# The holds of the first NEED_HOLDS receiving grasps at the first pose in reach are asked of the need
# before any IK: where it takes none, it takes no handover (the base must move instead).
NEED_HOLDS = 12
MAX_HANDOVER_PLANS = 3              # handovers whose motion is tried


class Handing:
    """The handover skill of :class:`~table_1000.experts.skills.Skills`."""

    def handover_poses(self, body, giver):
        """World poses of the held object's body ``body`` (4x4, now) to hand it over at from arm
        ``giver``, the nearest first."""
        shoulders = [self.shoulder(arm) for arm in self.arms]
        centre = np.mean(shoulders, axis=0)
        ahead = normalized(self.model.base[:3, 0] - [0., 0., self.model.base[2, 0]])
        aside = self.shoulder(giver) - centre
        aside = normalized(aside - [0., 0., aside[2]])
        low = max(body[2, 3], self.ctx.scene.TABLE['height'] + TABLE_CLEARANCE)
        poses = []
        for reach in REACH:
            for height in HEIGHTS:
                for side in SIDES:
                    for turn in TURNS:
                        pose = np.eye(4)
                        pose[:3, :3] = Rotation.from_euler('z', turn).as_matrix() @ body[:3, :3]
                        pose[:3, 3] = centre + ahead * reach + aside * side
                        pose[2, 3] = low + height
                        poses.append((float(np.linalg.norm(pose[:3, 3] - body[:3, 3])) + abs(turn) * .3, len(poses),
                                      pose))
        return [pose for *_, pose in sorted(poses, key=lambda option: option[:2])]

    @decision
    @timed('handover_search')
    def handoffs(self, target, giver, to, need):
        """Feasible handovers ``(pose, q_give, grasp)``, the first grasp of each pose, as they are found
        (see the module docstring)."""
        ctx = self.ctx
        giving, receiving = self.solvers[giver], self.solvers[to]
        body = ctx.poses()[target.body]
        com_local = body[:3, :3].T @ (ctx.env.center_of_mass(target.body)[0] - body[:3, 3])
        tcp_object = np.linalg.inv(giving.tcp()) @ body
        gap = giving.finger_gap()
        side = ctx.spec.arms[to]
        checked = 0
        self.rejected = {'giving IK': 0, 'receiving IK': 0, 'pre-grasp IK': 0, 'need': 0}
        asked = need is None
        for index, pose in enumerate(self.handover_poses(body, giver)):
            mark(self, 'candidate', index, pose=pose)
            if checked >= MAX_HANDOVER_IK:
                return
            checked += 1
            give = pose @ np.linalg.inv(tcp_object)
            mark(self, 'giving IK')
            status, q_give = self.model.ik(giver, give, self.model.q, world=True, lift=True, attempts=IK_ATTEMPTS)
            if status != 'Success':
                mark(self, 'giving IK', outcome='rejected', status=status)
                self.rejected['giving IK'] += 1
                continue
            transform = pose @ target.offset
            com = pose[:3, :3] @ com_local + pose[:3, 3]
            obstacles, _ = self.obstacles(to, target, transform)
            obstacles.append((self.hands[giver].fcl(give, gap), 0.))
            preferred = normalized(transform[:3, 3] - self.shoulder(to))
            end_grasps = upright_need(need)
            if not asked:
                asked = True
                holds = itertools.islice(grasp_candidates(target.shape, transform, self.hands[to], obstacles, preferred,
                                                          mass_center=com, weight=self.grasp_weight(target),
                                                          approach_distance=APPROACH, end_grasps=end_grasps), NEED_HOLDS)
                if not any(need.accepts(to, np.linalg.inv(grasp.pose) @ pose) for grasp in holds):
                    mark(self, 'need', outcome='rejected')
                    self.rejected['need'] += NEED_HOLDS
                    return
            moved = pose @ np.linalg.inv(body)
            receiving.allow(target.names)
            try:
                found = None
                for count, grasp in enumerate(grasp_candidates(target.shape, transform, self.hands[to], obstacles,
                                                               preferred, mass_center=com, weight=self.grasp_weight(target),
                                                               approach_distance=APPROACH, end_grasps=end_grasps)):
                    if count >= MAX_RECEIVING_IK or checked >= MAX_HANDOVER_IK:
                        break
                    checked += 1
                    # The object where it would be handed over, in the receiving arm's world (a need's
                    # check puts it back).
                    receiving.sync()
                    poses = ctx.poses()
                    for name in target.names:
                        receiving.world.set_pose(name, moved @ poses[receiving.world.objects[name].body])
                    start = self.model.vector(side.finger_targets(grasp.open_gap), q_give)
                    mark(self, 'receiving IK', f'{index}/{count}', grasp=grasp)
                    status, q_grasp = self.model.ik(to, grasp.pose, start, world=True, attempts=IK_ATTEMPTS)
                    if status != 'Success':
                        mark(self, 'receiving IK', outcome='rejected', status=status)
                        self.rejected['receiving IK'] += 1
                        continue
                    mark(self, 'pre-grasp IK')
                    status, _ = self.model.ik(to, standoff(grasp.pose, APPROACH), q_grasp, world=True,
                                              attempts=IK_ATTEMPTS)
                    if status != 'Success':
                        mark(self, 'pre-grasp IK', outcome='rejected', status=status)
                        self.rejected['pre-grasp IK'] += 1
                        continue
                    mark(self, 'need')
                    if need is not None and not need.accepts(to, np.linalg.inv(grasp.pose) @ pose):
                        mark(self, 'need', outcome='rejected')
                        self.rejected['need'] += 1
                        continue
                    mark(self, 'screen', outcome='accepted')
                    found = grasp
                    break
            finally:
                receiving.allow(target.names, False)
                receiving.sync()
            if found is not None:
                yield pose, q_give, found

    def handover(self, obj, to=None, need=None):
        """Hand the held ``obj`` to arm ``to`` (default: the other one ``need`` takes), held as
        ``need`` (an :class:`~table_1000.experts.skills.chain.InHand`) asks; see the module
        docstring. :class:`Unmet` where no handover is in reach. Returns ``{'arm', 'giver', 'grasp',
        'tcp_object', 'pose'}`` (the object's where it was handed over).
        Loaded rim grasps are unsupported: keep the current arm and place the
        container before changing hands; handover turns can spill its contents."""
        ctx = self.ctx
        giver = self.holding(obj)
        if giver is None:
            raise ValueError(f'{obj!r} is not held')
        if to is None:
            to = next(arm for arm in (need.arms if need is not None else self.arms) if arm != giver)
        held = self.held[giver]
        if held['riders']:
            raise ValueError('loaded container handover is unsupported; keep the current arm and place it before changing hands')
        target = held['target']
        if held['carried']:
            # A real-part grip still transfers the asset's entire collision
            # assembly in receiver previews and after the giving hand releases.
            target = replace(target, names=list(dict.fromkeys([*target.names, *held['carried']])))
        giving, receiving = self.solvers[giver], self.solvers[to]
        # In the receiving arm's world the object is not held: it touches the giving fingers.
        receiving.world.allow(target.names, ctx.spec.arms[giver].finger_links)
        try:
            return self.hand_over(target, giver, to, need)
        finally:
            receiving.world.allow(target.names, ctx.spec.arms[giver].finger_links, False)

    def hand_over(self, target, giver, to, need):
        ctx = self.ctx
        giving, receiving = self.solvers[giver], self.solvers[to]
        planned, done = 0, False
        for pose, q_give, grasp in self.handoffs(target, giver, to, need):
            planned += 1
            try:
                giving.move_to_pose_with_RRTConnect(None, lift=True, goals=[q_give], attempts=CANDIDATE_PLANS)
                # Where the object is now, rather than where it was meant to be.
                taken = ctx.poses()[target.body] @ np.linalg.inv(pose) @ grasp.pose
                receiving.open_gripper(grasp.open_gap)
                receiving.move_to_pose_with_RRTConnect(standoff(taken, APPROACH), attempts=CANDIDATE_PLANS)
                receiving.allow(target.names)
                receiving.move_to_pose_with_screw(taken)
                done = True
                break
            except PlanningFailed as error:
                receiving.allow(target.names, False)
                ctx.note('next handover', giver=giver, arm=to, reason=error.reason, **collisions(error))
                if planned >= MAX_HANDOVER_PLANS:
                    break
        if not done:
            ctx.fail('no handover', Unmet, giver=giver, arm=to, planned=planned, rejected=self.rejected)
        receiving.close_gripper()
        gap = receiving.finger_gap()
        checked = self.check_closure(target, to, grasp)
        ctx.note('closure checked', arm=to, **checked)
        if not checked['held']:
            ctx.fail('object not handed over', finger_gap=gap, width=grasp.width)
        poses = ctx.poses()
        previous_permissions = set(receiving.world.permissions)
        for name in target.names:
            receiving.attach(name, poses[receiving.world.objects[name].body])
        receiving.world.allow(target.names, target.names)
        chosen = {'pose': taken.tolist(), 'open_gap': grasp.open_gap, 'width': grasp.width,
                  'backoff': grasp.backoff, 'cost': grasp.cost, **grasp.info}
        self.held[to] = {'target': target, 'supports': [], 'grasp': chosen,
                        'carried': [], 'riders': [], 'carry_contacts': [],
                        'carry_permissions': [list(pair) for pair in receiving.world.permissions - previous_permissions]}
        # Now in the giving arm's world the object touches the receiving fingers (as it backs out, and on
        # while they hold it).
        giving.world.allow(target.names, ctx.spec.arms[to].finger_links)
        self.release(giver)
        tcp_object = np.linalg.inv(receiving.tcp()) @ ctx.poses()[target.body]
        ctx.note('handed over', giver=giver, arm=to, target=target.names[0], finger_gap=receiving.finger_gap(),
                 **grasp.info)
        return {'arm': to, 'giver': giver, 'grasp': chosen, 'tcp_object': tcp_object.tolist(), 'pose': pose.tolist()}
