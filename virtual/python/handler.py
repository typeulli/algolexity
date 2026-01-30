from pathlib import Path
from threading import Thread
import json
import subprocess
import traceback
import time
import select
import logging

path_here = Path(__file__).parent.resolve()
path_log = path_here / "stream" / "python.log"
logger = logging.getLogger("handler.python")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(path_log, encoding="utf-8")
logger.addHandler(fh)

def wrap_exception(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            with path_log.open("a", encoding="utf-8") as f:
                logger.error(f"Exception in handler: {type(e).__name__}: {e}")
                traceback.print_exc(file=f)
    return wrapper
@wrap_exception
def _run(data: str) -> None:
    data = json.loads(data)
    stdin = data["stdin"]
    name = data['target']
    session = data['session']
    data["code"] = Path(path_here / "stream" / f"{name}.py").read_text(encoding="utf-8")
    del data["stdin"]
    del data["target"]
    del data["session"]
    
    start_time = time.time()
    procs[session].stdin.write(json.dumps(data) + "\n")
    procs[session].stdin.flush()
    rlist, _, _ = select.select([procs[session].stdout], [], [], 4.0)
    if rlist:
        stdout = procs[session].stdout.readline()
        stderr = ""
    else:
        stdout, stderr = "<timeout>", ""
    end_time = time.time()
    
    with path_log.open("a", encoding="utf-8") as f:
        f.write(
            f"Handler run for {name} completed in {end_time - start_time} seconds.\n"
        )
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError:
        result = {
            "result": "timeout",
            "message": "process timeout or invalid output.",
            "data": {"stdout": stdout}
        }
    if stderr:
        result = {
            "result": "error_internal",
            "message": "stderr is provided.",
            "data": {"stdout": stdout, "stderr": stderr}
        }
    (path_here / "stream" / f"{name}.report.json").write_text(json.dumps(result))
    


procs = {}
if __name__ == "__main__":
    path_here = Path(__file__).parent

    (path_here / "stream").mkdir(parents=True, exist_ok=True)
    while True:
        for path_new_session in (path_here / "stream").glob("*.open"):
            if not path_new_session.exists():
                continue
            uid = path_new_session.stem
            procs[uid] = subprocess.Popen(
                [
                    "python", 
                    str(path_here / "tracker.py")
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            path_new_session.unlink()
            logger.info(f"Session {uid} started.")
        for path_close_session in (path_here / "stream").glob("*.close"):
            if not path_close_session.exists():
                continue
            uid = path_close_session.stem
            proc = procs.get(uid, None)
            if proc is not None:
                proc.terminate()
                proc.wait(timeout=2.0)
                del procs[uid]
            path_close_session.unlink()
            logger.info(f"Session {uid} closed.")
        for path_request in (path_here / "stream").glob("*.request.json"):
            if not path_request.exists():
                continue
            data = path_request.read_text()
            Thread(target=_run, args=(data,), daemon=True).start()
            path_request.unlink()