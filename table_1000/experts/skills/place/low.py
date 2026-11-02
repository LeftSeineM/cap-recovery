"""Bounded gravity releases after strict placement fails.

The object-to-floor gap is the drop distance, measured again before opening.
The final semantic predicate remains the caller's responsibility. Free-space
and descent use the existing full-robot/payload collision checks. A conservative
swept envelope excludes other objects throughout the predicted tip onto the
floor; loaded containers are excluded rather than tipped.
"""
from dataclasses import replace

import numpy as np

from table_1000.experts.context import Unreachable, remaining_wall
from table_1000.experts.audit.problems import decision, mark
from table_1000.experts.skills.chain import InHand


class LowRelease:
    def release_floor(self, region):
        """The target container's floor in the current observed scene."""
        from table_1000.experts.world.geometry import scene_shapes, selects, support_region
        metadata = region[3] if len(region) > 3 else {}
        container, part = metadata.get('release_container'), metadata.get('release_part')
        if not container:
            return None
        walls = [s for s in scene_shapes(self.ctx) if selects(s, container, part)]
        return support_region(walls, container, part, floor=True)[0]

    def low_release_options(self, target, region, near, keep_out, step, margin):
        from table_1000.experts.world.geometry import scene_shapes, selects, support_region
        from table_1000.experts.skills.place.strict import MAX_PLACE_IK
        metadata = region[3] if len(region) > 3 else {}
        container = metadata.get('release_container')
        if not container or metadata.get('release_support') is not None:
            return []
        limit = self.ctx.geometry[target.obj].max_drop_m
        if limit <= 0.:
            return []
        shapes = scene_shapes(self.ctx, exclude=(target.obj,))
        walls = [s for s in shapes if selects(s, container, metadata.get('release_part'))]
        floor, _ = support_region(walls, container, metadata.get('release_part'), floor=True)
        if floor is None:
            return []
        frame, low, high = region[:3]
        # A release region must still be on the live floor. Elevated expert
        # release planes cannot masquerade as a low drop.
        if not np.allclose(frame, floor[0], atol=1e-5):
            return []
        whole = self.target(target.obj)
        current = self.ctx.poses()[whole.body]
        bounds = np.ptp(whole.points, axis=0)
        slender = bounds.max() >= 2 * np.partition(bounds, -2)[-2]
        # Keep cups upright; never deliberately turn an articulated container.
        modes = [None, 'upright'] if slender and not target.container else [None]
        released_region = (frame, low, high)  # resting-height bound does not constrain release tilt
        options = []
        for mode in modes:
            options += self.placements(whole, current, released_region, near, keep_out,
                                       step, margin, mode)
        local = np.linalg.inv(frame)
        floor_low, floor_high = np.asarray(floor[1]), np.asarray(floor[2])
        other = [s for s in shapes if s.obj not in (container, 'table')]
        wall_top = max(((s.points @ local[:3, :3].T + local[:3, 3])[:, 2].max()
                        for s in walls), default=0.)
        other_bounds = [(p.min(0), p.max(0)) for s in other
                        for p in [s.points @ local[:3, :3].T + local[:3, 3]]]
        safe = []
        for placed, spot in options:
            remaining_wall(self.ctx)
            points = whole.points @ placed[:3, :3].T + placed[:3, 3]
            com_local = (np.linalg.inv(current) @ np.r_[self.ctx.env.center_of_mass(whole.body)[0], 1.])[:3]
            com = placed[:3, :3] @ com_local + placed[:3, 3]
            # Worst horizontal reach when this shape tips about its lowest end.
            # Require room in at least one inward direction, and exclude loose
            # objects from the full envelope, including those already inside.
            height = float(np.ptp(points @ frame[:3, 2]))
            footprint = points @ local[:3, :3].T + local[:3, 3]
            centre = (local @ np.r_[com, 1.])[:2]
            radius = float(np.hypot(height, np.linalg.norm(np.ptp(footprint[:, :2], axis=0))))
            if slender:
                destinations = [centre + direction * height / 2 for direction in
                                (np.array([1.,0.]), np.array([-1.,0.]), np.array([0.,1.]), np.array([0.,-1.]))]
                if not any(np.all(p >= floor_low + margin) and np.all(p <= floor_high - margin)
                           for p in destinations):
                    continue
            # Several small bounded release gaps allow the fingers to clear
            # the rim; no candidate above the authored/default maximum exists.
            for gap in sorted({min(gap, limit) for gap in (.002, .01, .025, .05, .10, .15, limit)}):
                # The walls confine material below their rim. Above it use a
                # sphere containing every possible tip, swept from the floor
                # to this gap. Include the descent before tipping; evaluate
                # each gap so overhead objects do not block low releases.
                pivot = np.r_[centre, gap]
                floor_pivot = np.r_[centre, 0.]
                blocked = False
                for a, b in other_bounds:
                    for low_z, high_z, confine in ((0., wall_top, True), (wall_top, np.inf, False)):
                        lo, hi = a.copy(), b.copy()
                        lo[2], hi[2] = max(lo[2], low_z), min(hi[2], high_z)
                        if confine:
                            lo[:2], hi[:2] = np.maximum(lo[:2], floor_low), np.minimum(hi[:2], floor_high)
                        separation = np.maximum(np.maximum(lo-pivot, floor_pivot-hi), 0.)
                        if np.all(lo <= hi) and np.linalg.norm(separation) <= radius:
                            blocked = True
                            break
                    if blocked:
                        break
                if blocked:
                    continue
                safe.append((placed, spot, gap))
                if len(safe) >= MAX_PLACE_IK:
                    return safe
        return safe

    def low_release_need(self, need):
        """Propose a bounded-release hold constraint, or None when unavailable.

        This optional relaxation may fail. Restoring an existing requirement is
        handled by restore_need, which must retain it even when no hold works.
        """
        origin = (need.recipe or {}).get('origin', {})
        region = origin.get('region')
        if region is None or origin.get('orientation') == 'flat':
            return None
        target = self.target(need.obj)
        geometry = self.ctx.geometry[need.obj]
        if not geometry.solid(None, narrow=0.):
            # Nonconvex tools are not containers. Use the same cavity test as
            # ordinary pickup instead of equating a nonconvex hull with one.
            from table_1000.experts.skills.hold.targets import rim_sections
            rim_sections(self, need.obj, target, geometry)
        options = self.low_release_options(target, region, np.asarray(origin['near'])[:3],
                                            origin['keep_out'], origin['step'], origin['margin'])
        if not options:
            return None
        # Check hand opening at release geometrically, leaving place free to
        # choose a different stance. This chooses holds that can hang vertically.
        obstacles = self.insertion_obstacles(target)
        def accepts(arm, tcp_object):
            gap = self.flat_open_gap(target, arm, tcp_object)
            for placed, _, height in options:
                release = placed.copy()
                release[:3, 3] += np.asarray(region[0])[:3, 2] * height
                if not self.insertion_blockers(arm, release @ np.linalg.inv(tcp_object), .02, gap, obstacles):
                    return True
            return False
        bounds = np.ptp(target.points, axis=0)
        hanging = bounds.max() >= 2 * np.partition(bounds, -2)[-2] and not target.container
        hanging = need.recipe.get('hanging', hanging)
        recipe = {**need.recipe, 'gravity_release': True, 'hanging': hanging}
        relaxed = InHand(need.obj, accepts, need.arms, recipe=recipe)
        self.ctx.requirements.append(relaxed)
        return relaxed

    def release_opening(self, target, region, arm, placed, tcp_object):
        """Return the live opening geometry and its bounded-release verdict."""
        opening = self.flat_release_geometry(target, region)
        clear = opening.minimum(self.hands[arm], placed, tcp_object,
                                self.solvers[arm].finger_gap(), self.held[arm]['grasp']['open_gap'], 0.,
                                backoff=0., heights=(0.,), check_budget=lambda: remaining_wall(self.ctx))
        return opening, clear is not None

    @decision
    def low_release(self, obj, region, arm, near, keep_out, step, margin, reason, orientation=None):
        from table_1000.experts.skills.common import ROOM_PLACES
        from table_1000.experts.skills.semantic import inside
        ctx, solver = self.ctx, self.solvers[arm]
        if orientation == 'flat':
            # Flat placement has additional level, COM and minimum-opening
            # conditions. Retain its strict path until those can be preserved
            # across an intentional change of release attitude.
            ctx.fail('bounded gravity release does not replace flat release checks', Unreachable, kind='planning', obj=obj)
        held = self.held[arm]
        if held.get('riders'):
            ctx.fail('bounded release would risk container contents', Unreachable, kind='planning', obj=obj)
        # Part grasps do not necessarily register the container's contents as
        # carried riders. Check live containment too before tilting anything.
        contents = [other for other in ctx.scene.OBJECTS
                    if other != obj and inside(ctx, other, obj)]
        if contents:
            ctx.fail('bounded release would risk container contents', Unreachable, kind='planning', obj=obj, contents=contents)
        target = held['target']
        options = self.low_release_options(target, region, near, keep_out, step, margin)
        ctx.note('placement relaxation attempted', obj=obj, level=1, relaxation='bounded gravity release',
                 strict_failure=reason, candidates=len(options), max_drop_m=ctx.geometry[obj].max_drop_m)
        if not options:
            ctx.fail('no bounded gravity release candidates', Unreachable, kind='planning', obj=obj, strict_failure=reason)
        names = [name for name in solver.world.objects if name.startswith(obj + ':')]
        held['target'] = target = replace(self.target(obj), names=names)
        def measured_drop():
            current = ctx.poses()[target.body]
            points = target.points @ current[:3,:3].T + current[:3,3]
            floor = self.release_floor(region)
            if floor is None:
                ctx.fail('no live container floor for gravity release', Unreachable, kind='planning', obj=obj)
            floor_frame = np.asarray(floor[0])
            drop = float(((points-floor_frame[:3,3]) @ floor_frame[:3,2]).min())
            limit = ctx.geometry[obj].max_drop_m
            if not np.isfinite(drop) or drop < 0. or drop > limit:
                ctx.fail('measured gravity release exceeds drop limit', Unreachable, kind='check',
                         obj=obj, drop_m=drop, max_drop_m=limit)
            return drop
        def attempt():
            tcp_object = np.linalg.inv(solver.tcp()) @ ctx.poses()[target.body]
            for name in names:
                solver.attach(name, ctx.poses()[solver.world.objects[name].body])
            result = {'done': False, 'ik_checked': 0, 'planned': 0, 'opening_checked': 0,
                      'last': 'no bounded release in reach'}
            for index, (placed, spot, height) in enumerate(options):
                mark(self, 'release candidate', str(index), placed=placed, drop_m=height)
                # A reachable closed hand may not be able to open by the rim.
                # Screen that channel before moving, so one blocked opening
                # cannot stop the search for another bounded release.
                nominal = placed.copy(); nominal[2, 3] += height
                tcp_object = np.linalg.inv(solver.tcp()) @ ctx.poses()[target.body]
                opening, clear = self.release_opening(target, region, arm, nominal, tcp_object)
                result['opening_checked'] += 1
                if not clear:
                    result['last'] = 'bounded release opening blocked'
                    mark(self, 'release candidate', outcome='rejected', reason=result['last'],
                         blocked_by=sorted(opening.blocked_by))
                    continue
                if self.lower(arm, placed, spot, tcp_object, height, .02, True, result, 'gravity'):
                    # The payload can settle in the pads during transport.
                    # A blocked measured opening rejects this candidate too;
                    # retain the object and plan the next pose from here.
                    measured_drop()
                    current = ctx.poses()[target.body]
                    tcp_object = np.linalg.inv(solver.tcp()) @ current
                    opening, clear = self.release_opening(target, region, arm, current, tcp_object)
                    if clear:
                        return result
                    result.update(done=False, last='bounded gravity release opening is blocked',
                                  blocked_by=sorted(opening.blocked_by))
                    mark(self, 'release candidate', outcome='rejected', reason=result['last'],
                         blocked_by=result['blocked_by'], measured=True)
                    ctx.note('next placement', arm=arm, spot=spot.tolist(), stage='opening',
                             reason=result['last'], blocked_by=result['blocked_by'])
            return result
        result = attempt()
        if not result['done'] and ctx.spec.base:
            tcp_object = np.linalg.inv(solver.tcp()) @ ctx.poses()[target.body]
            releases = []
            # The prefix may contain only the preserved tilt and tiny gaps.
            # Spread the existing stance budget over both hanging ends and
            # release heights, as need() already does for ordinary placement.
            indices = np.unique(np.linspace(0, len(options) - 1, min(ROOM_PLACES, len(options))).astype(int))
            for index in indices:
                placed, _, height = options[index]
                release = placed.copy();release[:3,3] += region[0][:3,2]*height
                release = release @ np.linalg.inv(tcp_object)
                above = release.copy();above[2,3] += .02
                releases.append([release, above])
            self.make_room(releases, [arm], need=1, touch=[], relocate=True)
            result = attempt()
        if not result['done']:
            ctx.fail('no reachable bounded gravity release', Unreachable, kind='planning', obj=obj,
                     options=len(options), **{k:v for k,v in result.items() if k!='done'})
        current = ctx.poses()[target.body]
        drop = measured_drop()
        limit = ctx.geometry[obj].max_drop_m
        ctx.note('placement relaxation used', obj=obj, level=1, relaxation='bounded gravity release',
                 drop_m=drop, max_drop_m=limit, release_pose=current.tolist())
        self.release(arm, settle=1.)
        frame = np.asarray(region[0])  # retain the original final-pose constraint
        metadata = region[3] if len(region) > 3 else {}
        if metadata.get('maximum_height') is not None:
            settled = ctx.poses()[target.body]
            points = target.points @ settled[:3, :3].T + settled[:3, 3]
            top = float(((points - frame[:3, 3]) @ frame[:3, 2]).max())
            if not np.isfinite(top) or top > metadata['maximum_height']:
                ctx.fail('object above target height after gravity release; pick it again before placing', kind='check',
                         obj=obj, top_m=top, maximum_height_m=metadata['maximum_height'], recoverable=True)
        if orientation == 'upright' and 'upright_container' in metadata:
            from table_1000.experts.skills.semantic import storage_result
            ctx.hold(.3)
            container = metadata['upright_container']
            if not storage_result(ctx, obj, container, metadata['part']):
                ctx.fail('object not supported inside after release; pick it again before placing', kind='check',
                         obj=obj, container=container, recoverable=True)
        return {'arm':arm, 'spot':result['spot'].tolist(), 'object':ctx.poses()[target.body].tolist(),
                'relaxation_level':1, 'drop_m':drop}
