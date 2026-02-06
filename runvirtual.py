from pathlib import Path
from typing import Literal, overload, Self
from data import AlgorithmExecutor, AlgorithmScopeData, SandboxResult, parse_report, log_process, ExecutionError, AlgorithmSession

import json
import subprocess
import tempfile
import time

path_here = Path(__file__).parent.resolve()
path_temp = path_here / "temp"
if not path_temp.exists():
    path_temp.mkdir(parents=True, exist_ok=True)

class PyAlgorithmSession(AlgorithmSession["PyAlgorithmExecutor"]):
    def __init__(self, executor: "PyAlgorithmExecutor", uid: str) -> None:
        super().__init__(executor, uid)

    def open(self) -> None:
        path = (path_temp / "python" / f"{self.uid}.open")
        path.write_text("open")
        while path.exists():
            time.sleep(0.01)
        
    def close(self) -> None:
        path = (path_temp / "python" / f"{self.uid}.close")
        path.write_text("close")
        while path.exists():
            time.sleep(0.01)
    
    @overload
    def run(self, code: str, timeout: float, all: Literal[True]) -> SandboxResult[AlgorithmScopeData]: pass
    @overload
    def run(self, code: str, timeout: float, all: Literal[False] = False) -> SandboxResult[int]: pass
    def run(self, code: str, timeout: float, all: bool = False) -> SandboxResult[AlgorithmScopeData] | SandboxResult[int]:
        with tempfile.TemporaryFile(mode="w", dir=path_temp / "python", suffix=".py") as tmp:
            tmp.write(code)
            tmp.flush()

            path_request = path_temp / "python" / f"{Path(tmp.name).stem}.request.json"
            path_request.write_text(json.dumps({
                "target": Path(tmp.name).stem,
                "session": self.uid,
                "request_all": all,
                "stdin": "",
                
            }))


            path_json = path_temp / "python" / f"{Path(tmp.name).stem}.report.json"
            sleep_time: float = 0

            while sleep_time < timeout:
                if not path_json.exists():
                    time.sleep(0.01)
                    sleep_time += 0.01
                    continue
                try:
                    raw_data = json.loads(path_json.read_text())
                    report = parse_report(raw_data)
                    break
                except json.JSONDecodeError:
                    pass
            else:
                path_json.unlink(missing_ok=True)
                raise TimeoutError("Timeout waiting for sandboxed code execution.")
            time.sleep(0.01)
            path_json.unlink(missing_ok=False)
            
            if report.result == "error_internal":
                print(raw_data)
                raise ExecutionError("InternalError", report.message)
            elif report.result == "error":
                result = SandboxResult(
                    data = -1,
                    time = sleep_time,
                    error_type = report.message.split(":", 1)[0],
                    error_message = ":".join(report.message.split(":", 1)[1:]) if ":" in report.message else ""
                )
            elif report.result == "timeout":
                raise TimeoutError("Sandboxed code execution timeout.")
            elif all:
                assert type(report.data_call) == dict
                result = SandboxResult(
                    data=AlgorithmScopeData.fromJson(
                        report.data_call if report.result == "success" else {"type": "module", "name": "<module>", "stack": []}),
                    time=sleep_time,
                    error_type="",
                    error_message=""
                )
            else:
                assert type(report.data_call) == int, raw_data
                result = SandboxResult(
                    data=report.data_call if report.result == "success" else -1,
                    time=sleep_time,
                    error_type="",
                    error_message=""
                )
            return result

class PyAlgorithmExecutor(AlgorithmExecutor):
    instance: "PyAlgorithmExecutor | None" = None

    def __new__(cls):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance
    
    def __init__(self):
        cmd = [
            "docker.exe", "run", "--rm",
            "--network=none",
            "--memory=64m",
            "--cpus=4",
            "--pids-limit=64",
            "--read-only",
            "-v", f"{(path_here / 'virtual' / 'python' / 'tracker.py')}:/sandbox/tracker.py:ro",
            "-v", f"{(path_here / 'virtual' / 'python' / 'handler.py')}:/sandbox/handler.py:ro",
            "-v", f"{(path_here / 'data.py')}:/sandbox/data.py:ro",
            "-v", f"{path_temp / 'python'}:/sandbox/stream:rw",
            "algolexity-python"
        ]

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        log_process(self.process)
    
    def session(self, uid: str) -> AlgorithmSession["PyAlgorithmExecutor"]:
        return PyAlgorithmSession(self, uid)
        
        
PyExecutor = PyAlgorithmExecutor()


if __name__ == "__main__":
    code = """
def func(n):
    cnt = 0
    for i in range(n//1000):
        for j in range(n//1000):
            cnt += 1
    print(cnt)
    return cnt
func(100000)
"""
    with PyExecutor.session("test") as session:
        result = session.run(code, 5.0, all=True)
        if result.error_message:
            print("Error:", result.error_type, result.error_message)
        print(result.data.total_calls())