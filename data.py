from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from subprocess import Popen
from typing import Generic, Iterable, Literal, TypeVar, overload, Self
from threading import Thread
import time

T = TypeVar("T")

@dataclass
class AlgorithmScopeData:
    scope_type: str
    name: str
    stack: "list[AlgorithmScopeData | int]"
    
    def total_calls(self) -> int:
        total = 0
        for item in self.stack:
            if isinstance(item, AlgorithmScopeData):
                total += item.total_calls()
            else:
                total += item
        return total
    
    def toJson(self) -> dict:
        return {
            "type": self.scope_type,
            "name": self.name,
            "stack": [
                item.toJson() if isinstance(item, AlgorithmScopeData) else item
                for item in self.stack
            ]
        }
    
    @classmethod
    def fromJson(cls, json_data: dict) -> "AlgorithmScopeData":
        stack = []
        for item in json_data.get("stack", []):
            if isinstance(item, dict):
                stack.append(cls.fromJson(item))
            else:
                stack.append(item)
        return cls(
            scope_type=json_data.get("type", ""),
            name=json_data.get("name", ""),
            stack=stack
        )

@dataclass
class SandboxResult(Generic[T]):    
    data: T
    time: float
    error_type: str
    error_message: str

def log_process(process: Popen) -> None:
    def report() -> None:
        while True:
            stdout, stderr = process.communicate()
            if stderr:
                print("Sandbox stderr:", stderr)
            time.sleep(1)
            
    Thread(target=report, daemon=True).start()

Executor = TypeVar("Executor", bound="AlgorithmExecutor")
class AlgorithmSession(Generic[Executor], metaclass=ABCMeta):
    def __init__(self, executor: Executor, uid: str) -> None:
        self.executor = executor
        self.uid = uid
    @abstractmethod
    def open(self) -> None: pass
    @abstractmethod
    def close(self) -> None: pass
    def __enter__(self):
        self.open()
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    @overload
    def run(self, code: Iterable[str], timeout: float, all: Literal[True]) -> SandboxResult[AlgorithmScopeData]: pass
    @overload
    def run(self, code: Iterable[str], timeout: float, all: Literal[False] = False) -> SandboxResult[int]: pass
    @abstractmethod
    def run(self, code: Iterable[str], timeout: float, all: bool = False) -> SandboxResult[AlgorithmScopeData] | SandboxResult[int]: pass
    def __call__(self, code: Iterable[str], timeout: float = 5.0, all: bool = False):
        return self.run(code, timeout, all=all)
    
class AlgorithmExecutor(metaclass=ABCMeta):
    @abstractmethod
    def session(self, uid: str) -> AlgorithmSession[Self]: pass


class ExecutionError(Exception):
    def __init__(self, error_type: str, error_message: str):
        self.error_type = error_type
        self.error_message = error_message
        super().__init__(f"{error_type}: {error_message}")
        
    @classmethod
    def wrap_ignore(cls, func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except cls:
                pass
        return wrapper

class EvaluateContext(metaclass=ABCMeta):
    @abstractmethod
    def on_track_start(self, total_calls: int) -> None: pass
    @abstractmethod
    def on_track(self, idx: int, n: int) -> None: pass
    @abstractmethod
    def on_track_end(self) -> None: pass
    @abstractmethod
    def on_complexity_fit_start(self, model_cnt: int, epoch_per_model: int) -> None: pass
    @abstractmethod
    def on_complexity_fit_enter_model(self, idx: int, model_name: str) -> None: pass
    @abstractmethod
    def on_complexity_fit_train_step(self, epoch: int, loss: float) -> None: pass
    @abstractmethod
    def on_complexity_fit_exit_model(self, idx: int, model_name: str, error: float) -> None: pass
    @abstractmethod
    def on_complexity_fit_end(self) -> None: pass
    @abstractmethod
    def on_done(self, TestResult) -> None: pass
    def on_execution_error(self, error_type: str, error_message: str) -> None:
        raise ExecutionError(error_type, error_message)
    def on_internal_error(self, error: Exception) -> None:
        raise error

@dataclass
class AlgorithmExecutionReport:
    result: Literal["success", "error"]
    message: str
    data: int | dict

def parse_report(json_data: dict) -> AlgorithmExecutionReport:
    return AlgorithmExecutionReport(
        result=json_data.get("result", "error"),
        message=json_data.get("message", ""),
        data=json_data.get("data", 0)
    )