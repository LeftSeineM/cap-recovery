"""Trajectory datasets of a :class:`~table_1000.sim.isaaclab.env.RobotEnv`: Isaac Lab's recorders and HDF5 files.

Isaac Lab's recorder manager writes one group per episode, ``data/demo_<id>``:

- ``initial_state``: the scene state the episode starts from
  (``InteractiveScene.get_state(is_relative=True)``: every articulation's root pose,
  root velocity, joint positions and velocities, every rigid object's root pose
  and velocity)
- ``actions`` ``[T, A]``: the flat action of each control step (``RobotEnv.action_layout``
  gives the slice of every action group)
- ``states``: the scene state after each step, as ``initial_state`` with a leading ``T``
- ``obs/<term>``: the robot observation (``agent``) before each step

so ``env.reset_to(initial_state)`` followed by the actions replays an episode
through physics, and ``scene.reset_to(states[t])`` restores step ``t`` exactly
(for rendering). Succeeded episodes go to ``<name>.hdf5``, failed ones to
``<name>_failed.hdf5``. :func:`finish` adds the episode's attributes (scene,
seed, instruction, subtasks, ...) to its group, JSON-encoded where they are not
numbers or strings; ``data`` carries the environment's (``env_args``).
"""

import json

import torch

from isaaclab.envs.mdp.recorders.recorders_cfg import (PostStepStatesRecorderCfg, PreStepActionsRecorderCfg)
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.managers.recorder_manager import DatasetExportMode, RecorderManagerBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets import HDF5DatasetFileHandler


class AgentObservationRecorder(RecorderTerm):
    """The ``agent`` observation group before each step (Isaac Lab's own term records ``policy``)."""

    def record_pre_step(self):
        return 'obs', dict(self._env.obs_buf['agent'])


@configclass
class AgentObservationRecorderCfg(RecorderTermCfg):
    class_type: type = AgentObservationRecorder


class TrajectoryFile(HDF5DatasetFileHandler):
    """Isaac Lab's HDF5 file, plus the episode's ``attrs`` (set by :func:`finish`) on its group."""

    def write_episode(self, episode, demo_id=None):
        name = f'demo_{self._demo_count if demo_id is None else demo_id}'
        super().write_episode(episode, demo_id)
        group = self._hdf5_data_group[name]
        for key, value in getattr(episode, 'attrs', {}).items():
            group.attrs[key] = value if isinstance(value, (int, float, str)) else json.dumps(value)


@configclass
class TrajectoryRecorderCfg(RecorderManagerBaseCfg):
    dataset_file_handler_class_type: type = TrajectoryFile
    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES
    # Episodes are exported by finish(), with their success and attributes.
    export_in_record_pre_reset: bool = False
    record_pre_step_actions = PreStepActionsRecorderCfg()
    record_pre_step_agent_observations = AgentObservationRecorderCfg()
    record_post_step_states = PostStepStatesRecorderCfg()


def recorder_cfg(directory, name='trajectories'):
    """Recorders writing ``<directory>/<name>.hdf5`` and ``<name>_failed.hdf5``."""
    return TrajectoryRecorderCfg(dataset_export_dir_path=str(directory), dataset_filename=name)


def redirect(env, directory, name='trajectories'):
    """Write later episodes to ``<directory>/<name>.hdf5`` and ``<name>_failed.hdf5`` (the files
    written so far are closed)."""
    import os
    manager = env.env.recorder_manager
    for attribute, suffix in (('_dataset_file_handler', ''), ('_failed_episode_dataset_file_handler', '_failed')):
        handler = getattr(manager, attribute)
        handler.close()
        handler.create(os.path.join(str(directory), f'{name}{suffix}'), env_name=getattr(env.env.cfg, 'env_name', None))
    manager.cfg.dataset_export_dir_path = str(directory)


def begin(env):
    """Start an episode at the current state: drop what was recorded since the reset (settling)."""
    manager = env.env.recorder_manager
    manager.reset()
    manager.add_to_episodes('initial_state', env.env.scene.get_state(is_relative=True))


def finish(env, demo_id, success, attrs, env_args):
    """Export the episode (environment 0) as ``demo_<demo_id>`` with ``attrs``; ``env_args`` go to
    the file (through Isaac Lab's episode metadata hook)."""
    manager = env.env.recorder_manager
    manager.set_success_to_episodes([0], torch.tensor([[bool(success)]], device=env.device))
    manager.get_episode(0).attrs = attrs
    env.env.cfg.get_ep_meta = lambda: env_args
    manager.export_episodes([0], demo_ids=[demo_id])
