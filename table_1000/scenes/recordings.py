"""A batch of trajectories in ``data/scenes/<scene>/<robot>/<expert>/<backend>/``, without changing Isaac Lab's HDF5 format.

``trajectories.hdf5`` and ``trajectories_failed.hdf5`` hold one ``data/demo_<seed>``
group per episode; the processes of a batch write their own files, merged at the
end (:func:`merge`). Videos sit next to them, those of succeeded episodes in
``videos/``, of failed ones in ``fails/``: ``<seed>/<camera>.mp4`` per camera,
``<seed>/merge.mp4`` for the cameras merged into one. Each is encoded
elsewhere and moved in when complete, so an interrupted render leaves nothing
half-written.
"""

import json
from pathlib import Path


def merge(sources, target, keep=lambda source, seed: True):
    """Copy the episodes of the Isaac Lab trajectory files ``sources`` into ``target``, in seed order,
    those for which ``keep(source, seed)``; returns the number of episodes. ``data`` keeps the attributes
    of the first file with an episode (Isaac Lab completes ``env_args`` at the first export; a process
    that stopped before it left only ``env_name`` and ``type``), with the summed ``total``."""
    import h5py
    episodes, attrs = [], {}
    sources = [Path(path) for path in sources if Path(path).is_file()]
    files = [h5py.File(path, 'r') for path in sources]
    try:
        for source, file in zip(sources, files):
            if not attrs and len(file['data']):
                attrs = dict(file['data'].attrs)
            for name in file['data']:
                seed = int(name.split('_')[1])
                if keep(source, seed):
                    episodes.append((seed, file, name))
        attrs = attrs or (dict(files[0]['data'].attrs) if files else {})
        with h5py.File(target, 'w') as out:
            data = out.create_group('data')
            data.attrs.update(attrs)
            total = 0
            for _, file, name in sorted(episodes, key=lambda episode: episode[0]):
                file.copy(file['data'][name], data, name)
                total += int(file['data'][name].attrs.get('num_samples', 0))
            data.attrs['total'] = total
    finally:
        for file in files:
            file.close()
    return len(episodes)


def clear_batch(directory, *, keep=()):
    """Empty a batch directory for a new batch, preserving named sibling product trees."""
    import shutil
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.name in keep:
            continue
        shutil.rmtree(path) if path.is_dir() else path.unlink()


def decoded_attrs(group):
    """An episode group's attributes, JSON-encoded ones (lists, dicts) decoded."""
    return {key: json.loads(value) if isinstance(value, str) and value[:1] in ('[', '{') else value
            for key, value in group.attrs.items()}


def demo_name(seed):
    return f'demo_{seed}'


# The name of the merged video, in place of a camera's.
MERGED = 'merge'


def video_dir(directory, failed=False):
    return Path(directory) / ('fails' if failed else 'videos')


def video_path(directory, seed, camera=None, failed=False):
    """A camera's video of episode ``seed``, or the merged one (``camera`` None)."""
    return video_dir(directory, failed) / str(seed) / f'{camera or MERGED}.mp4'


def videos(directory):
    """Seeds with a video in a batch directory, ``{'videos': {camera: seeds}, 'fails': {...}}``,
    the merged videos under :data:`MERGED`."""
    found = {}
    for failed, key in ((False, 'videos'), (True, 'fails')):
        kinds = {}
        for path in video_dir(directory, failed).glob('*/*.mp4'):
            if path.parent.name.isdigit():
                kinds.setdefault(path.stem, []).append(int(path.parent.name))
        found[key] = {kind: sorted(seeds) for kind, seeds in sorted(kinds.items())}
    return found


def subtask(attrs, t, length):
    """Instruction of the subtask at step t."""
    entries = attrs.get('subtasks', [])
    if isinstance(entries, str):
        entries = json.loads(entries)
    for entry in entries:
        if entry['start'] <= t < (entry['end'] if entry['end'] is not None else length):
            return entry['instruction']
    return None


def concatenate(sources, target, name):
    """Write ``data/<name>`` in the open HDF5 file ``target``: the episodes ``sources``
    (``[(group, length)]``, in order) one after the other, each cut to its first ``length`` steps;
    the first's ``initial_state`` and attributes, ``num_samples`` the sum."""
    import h5py
    import numpy as np

    def join(groups, out, skip=None):
        for key, value in groups[0][0].items():
            if key == skip:
                continue
            if isinstance(value, h5py.Group):
                join([(group[key], length) for group, length in groups], out.create_group(key))
            else:
                out.create_dataset(key, data=np.concatenate([group[key][:length] for group, length in groups]))

    first = sources[0][0]
    group = target['data'].create_group(name)
    group.attrs.update(dict(first.attrs))
    first.file.copy(first['initial_state'], group, 'initial_state')
    join(sources, group, skip='initial_state')
    group.attrs['num_samples'] = sum(length for _, length in sources)
    return group
