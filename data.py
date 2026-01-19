from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar, overload

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
    error_type: str
    error_message: str

class AlgorithmExecutor(metaclass=ABCMeta):
    @overload
    def run(self, code: str, all: Literal[True]) -> SandboxResult[AlgorithmScopeData]: pass
    @overload
    def run(self, code: str, all: Literal[False] = False) -> SandboxResult[int]: pass
    @abstractmethod
    def run(self, code: str, all: bool = False) -> SandboxResult[AlgorithmScopeData] | SandboxResult[int]: pass
    def __call__(self, code: str, all: bool = False):
        return self.run(code, all=all)

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