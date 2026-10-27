"""Grasp trial state recording; replay uses the scene renderer's state and video tools."""

from pathlib import Path

from table_1000.sim import CONTROL_HZ

VIEWS = {'wide': {'position': (1.35, -1.2, 1.55), 'target': (.45, -.1, .85),
                  'focal_length_mm': 22, 'resolution': (960, 720)}}


def frame(env, text, path, first=False):
    from PIL import Image
    from scripts.scenes.render import FIRST_FRAME_RENDERS
    from table_1000.sim.isaaclab.video import caption, save_frame
    for _ in range(FIRST_FRAME_RENDERS - 1 if first else 0):
        env.render([])
    image = Image.fromarray(env.render(['wide'])['wide']['rgb'][0])
    save_frame(caption(image, text), path)


class Recorder:
    """State after each control step, including all attempts and their reset jumps."""

    def __init__(self, env, frames=None):
        self.env, self.frames = env, frames
        self.states, self.captions = [], []
        if frames is not None:
            frames.mkdir(parents=True)

    def capture(self, text):
        from table_1000.sim.isaaclab.replay import numpy_state
        self.states.append(numpy_state(self.env))
        self.captions.append(text)
        if self.frames is not None:
            frame(self.env, text, self.frames / f'{len(self.states) - 1:05d}.jpg', first=len(self.states) == 1)

    def save(self, path, attrs):
        from table_1000.sim.isaaclab.replay import write_episode
        path.parent.mkdir(parents=True, exist_ok=True)
        # Episode.state(t+1) is the state shown in frame t. The initial state is
        # duplicated because capture runs after the first control step.
        write_episode(path, [self.states[0], *self.states],
                      {**attrs, 'captions': self.captions, 'sampling': 'post_control_step',
                       'control_hz': CONTROL_HZ, 'views': VIEWS})


def render(evaluator, trajectory, video, frames):
    from table_1000.sim.isaaclab import replay
    from table_1000.sim.isaaclab.video import encode
    frames = Path(frames)
    frames.mkdir(parents=True)
    file, episodes = replay.episodes(trajectory)
    try:
        episode = episodes['demo_0']
        for t in range(episode.length):
            replay.set_state(evaluator.env, episode.state(t + 1))
            frame(evaluator.env, episode.attrs['captions'][t], frames / f'{t:05d}.jpg', first=t == 0)
        video.parent.mkdir(parents=True, exist_ok=True)
        encode(frames, episode.attrs['control_hz'], video)
    finally:
        file.close()
