"""Replaying recorded episodes (:mod:`.recording`) by their states, for rendering.

Restoring the recorded state of a step (``InteractiveScene.reset_to``) puts
every body exactly where it was, without simulating: images rendered from a
trajectory file match the episode that was recorded, whatever the physics
would do on a second run.
"""

import json

import h5py
import torch

from table_1000.scenes.recordings import decoded_attrs, subtask


def read_tree(group, index=None):
    """Nested dict of ``[1, ...]`` tensors from an HDF5 group (row ``index`` of each dataset, or the
    whole dataset when it holds one row)."""
    out = {}
    for key, value in group.items():
        if isinstance(value, h5py.Group):
            out[key] = read_tree(value, index)
        else:
            row = value[()] if index is None else value[index:index + 1]
            out[key] = torch.as_tensor(row)
    return out


class Episode:
    """One ``demo_<id>`` group: its attributes and the state before each step."""

    def __init__(self, group):
        self.group, self.name = group, group.name.rsplit('/', 1)[-1]
        self.attrs = decoded_attrs(group)
        self.length = int(group.attrs['num_samples'])

    def state(self, t):
        """Scene state before step ``t`` (``t = length``: after the last step)."""
        return read_tree(self.group['initial_state']) if t == 0 else read_tree(self.group['states'], t - 1)

    def subtask(self, t):
        """Instruction of the subtask step ``t`` belongs to, or None."""
        return subtask(self.attrs, t, self.length)


def episodes(path):
    """``(file, {name: Episode})`` of a trajectory file, in demo order (keep the file open while reading)."""
    file = h5py.File(path, 'r')
    names = sorted(file['data'], key=lambda name: int(name.split('_')[1]))
    return file, {name: Episode(file['data'][name]) for name in names}


def env_args(file):
    return json.loads(file['data'].attrs['env_args'])


def write_episode(path, states, attrs):
    """Write one episode, ``data/demo_0``, readable by :func:`episodes`: ``states`` are the scene states
    before the first step and after each step (NumPy trees, as :func:`numpy_state` gives); ``attrs``
    are JSON-encoded where they are not numbers or strings."""
    import numpy as np

    def write(group, trees):
        for key, value in trees[0].items():
            if isinstance(value, dict):
                write(group.create_group(key), [tree[key] for tree in trees])
            else:
                group.create_dataset(key, data=np.concatenate([tree[key] for tree in trees]))

    with h5py.File(path, 'w') as file:
        data = file.create_group('data')
        data.attrs['env_args'] = json.dumps({key: value for key, value in attrs.items()})
        demo = data.create_group('demo_0')
        demo.attrs['num_samples'] = len(states) - 1
        for key, value in attrs.items():
            demo.attrs[key] = value if isinstance(value, (int, float, str)) else json.dumps(value)
        write(demo.create_group('initial_state'), states[:1])
        write(demo.create_group('states'), states[1:])


def numpy_state(env):
    """The scene state of ``env`` (environment axis kept) as NumPy, for :func:`write_episode`."""
    def convert(tree):
        return {key: convert(value) if isinstance(value, dict) else value.cpu().numpy().copy() for key, value in tree.items()}
    return convert(env.env.scene.get_state(is_relative=True))


def set_state(env, state):
    """Put ``env`` (a :class:`~table_1000.sim.isaaclab.env.RobotEnv`, environment 0) in ``state``."""
    device = env.device

    def to_device(tree):
        return {key: to_device(value) if isinstance(value, dict) else value.to(device) for key, value in tree.items()}

    from omni.physx import get_physx_interface
    state = to_device(state)
    env.env.scene.reset_to(state, is_relative=True)
    env.place_kinematic(state)
    env.env.sim.forward()
    # Without a physics step nothing writes the new poses to USD, which the renderer reads.
    get_physx_interface().update_transformations(False, True)
