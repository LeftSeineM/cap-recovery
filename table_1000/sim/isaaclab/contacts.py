"""PhysX contact reports kept as time intervals per labelled body pair (environment 0).

Isaac Lab's contact sensor reports forces, not how deep two bodies overlap,
which the robot checks need; this reads CPU contact events or GPU detailed
contact tensors. The spawners activate contact reporting on the robot and the
scene's shapes (see :mod:`table_1000.sim.isaaclab.scene`).
"""

from contextlib import contextmanager


class Contacts:
    """Contact intervals of labelled bodies, timed by physics steps of ``dt``.

    ``labels`` maps prim path prefixes to names; pairs are kept when either side's
    label starts with one of ``watched``. Normal impulses are summed per pair until
    :meth:`take_impulses`.
    """

    TOUCH = .0005

    def __init__(self, labels, watched, dt, device='cpu'):
        import carb
        from omni.physx import get_physx_interface, get_physx_simulation_interface
        # Isaac Lab turns PhysX contact processing off unless one of its contact sensors needs it.
        carb.settings.get_settings().set_bool('/physics/disableContactProcessing', False)
        self.labels, self.watched, self.dt = labels, tuple(watched), dt
        self.time, self.phase = 0., 'start'
        self.events = []
        self.open = {}
        self.impulses = {}
        self.surface_contacts = {}
        self.impulse_observers = []
        self.tensor_view = None
        self.subscription = None
        if device == 'cpu':
            self.subscription = get_physx_simulation_interface().subscribe_contact_report_events(self.on_report)
        else:
            # GPU PhysX does not emit the CPU contact-report event stream.
            # Detailed tensor contacts retain separation and body-pair identity.
            import omni.usd
            from pxr import UsdPhysics
            from isaacsim.core.simulation_manager import SimulationManager
            stage = omni.usd.get_context().get_stage()
            bodies = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)
                      and ('/env_' not in str(p.GetPath()) or '/env_0/' in str(p.GetPath()))]
            static = []
            for prim in stage.Traverse():
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                parent = prim
                while parent and not parent.HasAPI(UsdPhysics.RigidBodyAPI):
                    parent = parent.GetParent()
                if not parent:
                    static.append(str(prim.GetPath()))
            self.filters = bodies + static
            sensors = [path for path in bodies if self.label(path).startswith(self.watched)]
            self.tensor_view = SimulationManager.get_physics_sim_view().create_rigid_contact_view(
                sensors, [self.filters for _ in sensors], max_contact_data_count=65536)
        self.clock = get_physx_interface().subscribe_physics_step_events(self.on_step)

    def on_step(self, dt):
        self.time += dt
        if self.tensor_view is not None:
            self.on_tensor_step()

    def on_tensor_step(self):
        import numpy as np
        forces, positions, _, distances, counts, starts = self.tensor_view.get_contact_data(self.dt)
        distances = distances.detach().cpu().numpy().ravel()
        forces = forces.detach().cpu().numpy().ravel()
        counts = counts.detach().cpu().numpy()
        starts = starts.detach().cpu().numpy()
        positions = positions.detach().cpu().numpy().reshape(-1, 3)
        touching = {}
        self.surface_contacts.clear()
        for i, j in np.argwhere(counts > 0):
            start, count = int(starts[i, j]), int(counts[i, j])
            separation = float(distances[start:start+count].min())
            pair = tuple(sorted((self.label(self.tensor_view.sensor_paths[i]), self.label(self.filters[j]))))
            self.record_impulse(pair, float(np.abs(forces[start:start+count]).sum()) * self.dt)
            if separation < self.TOUCH:
                touching[pair] = min(touching.get(pair, float('inf')), separation)
                paths = [self.tensor_view.sensor_paths[i], self.filters[j]]
                self.surface_contacts[tuple(paths)] = {'bodies': list(map(self.label, paths)),
                    'colliders': paths, 'source': 'tensor', 'separation_m': separation,
                    'points': positions[start:start+count].tolist()}
        for pair, separation in touching.items():
            entry = self.open.setdefault(pair, {'pair': pair, 'start': self.time, 'phase': self.phase,
                                                'max_penetration_m': 0.})
            entry['max_penetration_m'] = max(entry['max_penetration_m'], -separation)
        for pair in self.open.keys() - touching.keys():
            entry = self.open.pop(pair)
            entry['end'] = self.time
            self.events.append(entry)

    def label(self, path):
        """Robot link (``robot.<link>``), object part (``<body key>.<part>``) or fixture."""
        for prefix, name in self.labels.items():
            if path == prefix or path.startswith(prefix + '/'):
                first = path[len(prefix):].strip('/').split('/')[0]
                if name == 'robot' or (first and name not in ('table', 'ground')):
                    return f'{name}.{first}' if first else name
                return name
        return path

    def on_report(self, headers, data):
        """Track intervals in which a watched body actually touches something.

        PhysX reports shape pairs within their contact offset; a pair counts as
        touching while its closest contact point has separation below TOUCH.
        """
        from pxr import PhysicsSchemaTools
        for header in headers:
            colliders = [str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                         str(PhysicsSchemaTools.intToSdfPath(header.collider1))]
            a, b = map(self.label, colliders)
            if not (a.startswith(self.watched) or b.startswith(self.watched)):
                continue
            pair = tuple(sorted((a, b)))
            points = data[header.contact_data_offset:header.contact_data_offset + header.num_contact_data]
            separation = min((point.separation for point in points), default=float('inf'))
            self.record_impulse(pair, sum(
                abs(sum(point.impulse[k] * point.normal[k] for k in range(3))) for point in points))
            touching = 'LOST' not in str(header.type) and separation < self.TOUCH
            key = tuple(colliders)
            if touching:
                self.surface_contacts[key] = {'bodies': [a, b], 'colliders': colliders,
                    'separation_m': float(separation),
                    'points': [[float(point.position[k]) for k in range(3)] for point in points],
                    'normals': [[float(point.normal[k]) for k in range(3)] for point in points],
                    'impulses_ns': [[float(point.impulse[k]) for k in range(3)] for point in points],
                    'physics_dt_s': self.dt}
            else:
                self.surface_contacts.pop(key, None)
            if touching:
                entry = self.open.setdefault(pair, {'pair': pair, 'start': self.time, 'phase': self.phase,
                                                    'max_penetration_m': 0.})
                entry['max_penetration_m'] = max(entry['max_penetration_m'], -float(separation))
            elif pair in self.open:
                entry = self.open.pop(pair)
                entry['end'] = self.time
                self.events.append(entry)

    def record_impulse(self, pair, impulse):
        self.impulses[pair] = self.impulses.get(pair, 0.) + impulse
        for observer in self.impulse_observers:
            observer(self.time, pair, impulse)

    @contextmanager
    def observe_impulses(self, observer):
        """Observe fresh physics-step impulses without consuming other readers' totals."""
        self.impulse_observers.append(observer)
        try:
            yield
        finally:
            self.impulse_observers.remove(observer)

    def take_impulses(self):
        """Normal impulse (N s) summed per pair since the last call."""
        impulses, self.impulses = self.impulses, {}
        return impulses

    def intervals(self):
        return self.events + [dict(entry, end=None) for entry in self.open.values()]

    def close(self):
        self.subscription = self.clock = None
        self.tensor_view = None

    def clear(self):
        """Forget every interval and restart the clock (after a reset)."""
        self.events.clear()
        self.open.clear()
        self.impulses.clear()
        self.surface_contacts.clear()
        self.time = 0.
