import threading
import faulthandler
import os
import sys


class kill_if_hangs:
    """Kill the whole process if the with-block doesn't finish in `seconds`."""

    def __init__(self, seconds: int):
        self.seconds = seconds

    def __enter__(self):
        faulthandler.enable()
        self._t = threading.Timer(self.seconds, self._boom)
        self._t.daemon = True
        self._t.start()

    def __exit__(self, exc_type, exc, tb):
        self._t.cancel()  # finished in time → cancel

    def _boom(self):
        print(
            f"[FATAL] Watchdog timeout ({self.seconds}s). Dumping stacks & exiting.",
            file=sys.stderr,
            flush=True,
        )
        faulthandler.dump_traceback()
        os._exit(1)  # hard exit, even if some thread is hung

