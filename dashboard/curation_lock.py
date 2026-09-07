"""
Reusable atomic exclusive-create file lock for one-off curation build scripts
(curate_*_step23.py / curate_*_step4.py style drivers that run outside the
Streamlit app itself to build a curated_matches/<id>/bundle.json).

Uses os.open with O_CREAT|O_EXCL, a single kernel syscall that either
creates the file or fails with FileExistsError - there is no check-then-
create window for two racing processes/sessions to both see "lock doesn't
exist yet" and both proceed. That race (a plain `if not os.path.exists(...)`
check followed by a separate create) is what corrupted the barca_madrid_pt1
build's shared progress file, and separately let two run_cv_analysis.py
subprocesses launch against the same --output-dir/--match-name.

Generalized out of that build's inline, throwaway lock code so a future
curation run doesn't have to reinvent it - import directly:

    from curation_lock import ExclusiveLock
    lock = ExclusiveLock(path)
    if not lock.acquire():
        print("FATAL: lock already held"); sys.exit(1)
    try:
        ...
    finally:
        lock.release()
"""
import os


class ExclusiveLock:
    def __init__(self, path):
        self.path = path
        self._fd = None

    def acquire(self):
        try:
            self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(self._fd, str(os.getpid()).encode())
            os.close(self._fd)
            return True
        except FileExistsError:
            return False

    def release(self):
        try:
            os.remove(self.path)
        except OSError:
            pass
