"""Scene sources, robot placements and numbered experts. Simulator-independent helpers."""

import importlib.util
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
# The one instruction of every Table-1000 scene; experts divide it into subtasks.
INSTRUCTION = 'tidy up the table'


def source_dir(name, root=ROOT):
    return Path(root) / 'asset_sources' / 'scenes' / name


def load(name, module='scene', root=ROOT, *, robot=None, expert=None):
    """Load a scene module, or its selected robot's numbered expert."""
    if module == 'expert':
        return load_expert(name, robot, expert, root)
    path = source_dir(name, root) / f'{module}.py'
    spec = importlib.util.spec_from_file_location(f'scene_{name.replace("-", "_")}_{module}', path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def robot_config(scene, robot=None):
    """Resolve a supported robot's asset, placement and starting joints from a scene module."""
    if scene.ROBOT not in scene.ROBOTS:
        raise ValueError(f'default robot {scene.ROBOT!r} is not in ROBOTS')
    name = robot or scene.ROBOT
    if name not in scene.ROBOTS:
        raise ValueError(f'robot {name!r} is not in ROBOTS: {list(scene.ROBOTS)}')
    return {**scene.ROBOTS[name], 'asset': f'robots/{name}'}


def initial_dir(base, scene, robot, layout=0):
    """Initial-state products below an assets/scenes root."""
    path = Path(base) / scene / robot
    return path / 'layouts' / str(layout) if layout else path


def data_dir(base, scene, robot, expert, backend, layout=0):
    """A robot/expert's batch below a data/scenes root, optionally for another layout."""
    path = Path(base) / scene / robot / expert / backend
    return path / 'layouts' / str(layout) if layout else path


def case_dir(base, scene, robot, expert, name):
    return Path(base) / scene / robot / expert / name


def parse_scope(scope):
    """Validate layout → seed lists; None means every available seed of that layout."""
    if not isinstance(scope, dict) or not scope:
        raise ValueError('SCOPE must be a nonempty layout → seed list/None mapping')
    for layout, seeds in scope.items():
        if type(layout) is not int or layout < 0:
            raise ValueError(f'SCOPE layout must be a nonnegative integer: {layout!r}')
        if seeds is not None and (not isinstance(seeds, list) or not seeds or
                                  any(type(seed) is not int or seed < 0 for seed in seeds) or
                                  len(set(seeds)) != len(seeds)):
            raise ValueError(f'SCOPE[{layout}] must be a nonempty list of distinct seeds or None')
    return {layout: None if seeds is None else list(seeds) for layout, seeds in sorted(scope.items())}


def outside_scope(scope, layout, seeds):
    """Whether a layout/seed request falls outside the declared scope."""
    return layout not in scope or (scope[layout] is not None and not set(seeds) <= set(scope[layout]))


def load_expert(name, robot=None, expert=None, root=ROOT):
    scene = load(name, root=root)
    robot_config(scene, robot)
    robot = robot or scene.ROBOT
    directory = source_dir(name, root) / 'experts' / robot
    numbers = sorted(path.stem for path in directory.glob('[0-9][0-9][0-9][0-9].py'))
    number = expert if expert is not None else next(iter(numbers), None)
    if number not in numbers:
        raise ValueError(f'no expert {number!r} for {name}/{robot}; available: {numbers}')
    path = directory / f'{number}.py'
    # Namespace packages permit ``from ..common import ...`` without scene-specific sys.path edits.
    package = f'table1000_scene_{name}_{abs(hash(str(Path(root).resolve()))):x}'
    for module_name, directory in [(package, source_dir(name, root)),
                                   (f'{package}.experts', source_dir(name, root) / 'experts'),
                                   (f'{package}.experts.{robot}', directory)]:
        if module_name not in sys.modules:
            namespace = types.ModuleType(module_name)
            namespace.__path__ = [str(directory)]
            sys.modules[module_name] = namespace
    spec = importlib.util.spec_from_file_location(f'{package}.experts.{robot}.expert_{number}', path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    if loaded.ROBOT != robot:
        raise ValueError(f'{path}: ROBOT {loaded.ROBOT!r} differs from directory {robot!r}')
    loaded.SCOPE = parse_scope(loaded.SCOPE)
    loaded.NUMBER = number
    if not hasattr(loaded, 'solve'):
        from table_1000.experts.context import run_steps
        loaded.solve = lambda ctx: run_steps(ctx, loaded.STEPS)
    return loaded


def scene_names(root=ROOT):
    return sorted(path.parent.name for path in (Path(root) / 'asset_sources' / 'scenes').glob('*/scene.py'))


def primitive_surface(spec, pose, count=400):
    """Points on the surface of a primitive (``{'shape', 'size'}``, as for :func:`add_primitive`) posed in the world."""
    import trimesh
    if spec['shape'] == 'sphere':
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=spec['size'])
    elif spec['shape'] == 'box':
        mesh = trimesh.creation.box(spec['size'])
    else:
        mesh = trimesh.creation.cylinder(radius=spec['size'][0], height=spec['size'][1])
    points, _ = trimesh.sample.sample_surface_even(mesh, count, seed=0)
    return points @ pose[:3, :3].T + pose[:3, 3]
