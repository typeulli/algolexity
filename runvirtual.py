from pathlib import Path
from typing import Literal, overload
from data import AlgorithmExecutor, AlgorithmScopeData, SandboxResult, parse_report

import json
import subprocess
import tempfile
import time

path_here = Path(__file__).parent.resolve()
path_temp = path_here / "temp"
if not path_temp.exists():
    path_temp.mkdir(parents=True, exist_ok=True)

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
            "-v", f"{path_temp / 'in'}:/sandbox/in:ro",
            "-v", f"{(path_here / 'virtual' / 'python' / 'tracker.py')}:/sandbox/tracker.py:ro",
            "-v", f"{(path_here / 'data.py')}:/sandbox/data.py:ro",
            "-v", f"{path_temp / 'out'}:/sandbox/out:rw",
            "algolexity-python"
        ]

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

    @overload
    def run(self, code: str, all: Literal[True]) -> SandboxResult[AlgorithmScopeData]: pass
    @overload
    def run(self, code: str, all: Literal[False] = False) -> SandboxResult[int]: pass
    def run(self, code: str, all: bool = False) -> SandboxResult[AlgorithmScopeData] | SandboxResult[int]:
        with tempfile.TemporaryFile(mode="w", dir=path_temp / "in", suffix=".py") as tmp:
            tmp.write(code)
            tmp.flush()

            path_request = path_temp / "in" / f"{Path(tmp.name).stem}.request.json"
            path_request.write_text(json.dumps({
                "target": Path(tmp.name).stem,
                "request_all": all
            }))


            path_json = path_temp / "out" / f"{Path(tmp.name).stem}.report.json"
            sleep_time = 0
            while sleep_time < 5:
                try:
                    report = parse_report(json.loads(path_json.read_text()))
                    break
                except (FileNotFoundError, json.JSONDecodeError):
                    pass
                time.sleep(0.05)
                sleep_time += 0.05
            else:
                
                path_request.unlink(missing_ok=True)
                path_json.unlink(missing_ok=True)
                raise TimeoutError("Timeout waiting for sandboxed code execution.")
            path_request.unlink(missing_ok=True)
            time.sleep(0.1)
            path_json.unlink(missing_ok=False)
            
            if all:
                assert type(report.data) == dict
                result = SandboxResult(
                    data=AlgorithmScopeData.fromJson(
                        report.data if report.result == "success" else {"type": "module", "name": "<module>", "stack": []}),
                    error_type="",
                    error_message=""
                )
            else:
                assert type(report.data) == int
                result = SandboxResult(
                    data=report.data if report.result == "success" else -1,
                    error_type="",
                    error_message=""
                )
            if report.result == "error":
                result.error_type = report.message.split(":", 1)[0]
                result.error_message = ":".join(report.message.split(":", 1)[1:]) if ":" in report.message else ""
            return result
                
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
    result = PyAlgorithmExecutor().run(code, True)
    if result.error_message:
        print("Error:", result.error_type, result.error_message)
    print(result.data.total_calls())