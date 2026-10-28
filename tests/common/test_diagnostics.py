import faulthandler
import sys
import threading

from table_1000.common.diagnostics import cancel_stack_dumps, dump_stacks_every


def test_python_timer_replaces_the_previous_destination_and_cancels(tmp_path, monkeypatch):
    dumped = threading.Event()
    destinations = []
    def capture(*, file, all_threads):
        destinations.append(file.name)
        assert all_threads
        dumped.set()
    monkeypatch.setattr(faulthandler, 'dump_traceback', capture)
    try:
        with (tmp_path / 'old.log').open('w') as old, (tmp_path / 'new.log').open('w') as new:
            dump_stacks_every(60, file=old)
            dump_stacks_every(.01, file=new)
            assert dumped.wait(2)
            cancel_stack_dumps()
            assert destinations and set(destinations) == {str(tmp_path / 'new.log')}
            assert faulthandler.is_enabled()
    finally:
        cancel_stack_dumps()
        faulthandler.enable(file=sys.__stderr__, all_threads=True)
