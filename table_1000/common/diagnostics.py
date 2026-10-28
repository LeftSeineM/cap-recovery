"""Periodic Python stack dumps without the native faulthandler watchdog."""

import faulthandler
import sys
import threading

_stop = None
_thread = None


def cancel_stack_dumps():
    """Cancel the timer most recently started by :func:`dump_stacks_every`."""
    global _stop, _thread
    if _stop is not None:
        _stop.set()
        _thread.join()
        _stop = None
        _thread = None


def dump_stacks_every(seconds=300, file=None):
    """Dump all threads from a Python thread, replacing any earlier timer.

    The destination must stay open until cancellation. A C call holding the
    GIL can delay this diagnostic; fatal-signal capture remains enabled.
    """
    global _stop, _thread
    if seconds <= 0:
        raise ValueError('seconds must be positive')
    cancel_stack_dumps()
    destination = sys.stderr if file is None else file
    faulthandler.enable(file=destination, all_threads=True)
    stop = _stop = threading.Event()

    def dump():
        while not stop.wait(seconds):
            faulthandler.dump_traceback(file=destination, all_threads=True)

    _thread = threading.Thread(target=dump, name='python-stack-dump', daemon=True)
    _thread.start()
