"""Robot/expert provenance checks for stored batches and checkpoints."""


def check_identity(saved, robot, expert, source):
    for key, expected in [('robot', robot), ('expert', expert)]:
        if not saved.get(key):
            raise ValueError(f'{source}: batch predates robot/expert tags; rerun it on the current code')
        if saved[key] != expected:
            raise ValueError(f'{source}: {key} {saved[key]!r} differs from current {key} {expected!r}')
