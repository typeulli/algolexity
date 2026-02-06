from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Iterable
from matplotlib import pyplot as plt
from uuid import uuid4
import math
import numpy as np
import torch
import torch.nn as nn

from data import AlgorithmExecutor, AlgorithmScopeData, EmptyEvaluateContext, EvaluateContext, ExecutionError
from runvirtual import PyExecutor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ComplexityModel(nn.Module):
    @abstractmethod
    def prefit(self, x: np.ndarray, y: np.ndarray) -> None: pass
    @abstractmethod
    def OString(self, mx: float = 1.0, my: float = 1.0) -> str: pass


class ComplexityN(ComplexityModel):
    def __init__(self):
        super().__init__()
        self.c = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        
    def prefit(self, x: np.ndarray, y: np.ndarray) -> None:
        poly = np.polyfit(x, y, 1)
        self.c = nn.Parameter(torch.tensor(poly[0]))
        self.b = nn.Parameter(torch.tensor(poly[1]))
    
    def forward(self, x):
        return self.c * x + self.b
    
    def OString(self, mx: float = 1.0, my: float = 1.0) -> str:
        return f"O({self.c * mx * my:.2f}N)"

class ComplexityLogN(ComplexityModel):
    def __init__(self):
        super().__init__()
        self.c = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
    
    def prefit(self, x: np.ndarray, y: np.ndarray) -> None:
        lx = np.log2(x)

        poly = np.polyfit(lx, y, 1)

        self.c = nn.Parameter(torch.tensor(poly[0], dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(poly[1], dtype=torch.float32))
    
    def forward(self, x):
        return self.c * torch.log2(x) + self.b

    def OString(self, mx: float = 1.0, my: float = 1.0) -> str:
        return f"O({self.c * math.log2(mx) * my:.2f}log N)"

class ComplexityNLogN(ComplexityModel):
    def __init__(self):
        super().__init__()
        self.c = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
    
    def prefit(self, x: np.ndarray, y: np.ndarray) -> None:
        xlogx = x * np.log2(x)
        
        poly = np.polyfit(xlogx, y, 1)

        self.c = nn.Parameter(torch.tensor(poly[0], dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(poly[1], dtype=torch.float32))
    
    def forward(self, x):
        return self.c * x * torch.log2(x) + self.b
    
    def OString(self, mx: float = 1.0, my: float = 1.0) -> str:
        return f"O({self.c * mx * math.log2(mx) * my:.2f}N log N)"

class ComplexityNSquared(ComplexityModel):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.c = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        
    def prefit(self, x: np.ndarray, y: np.ndarray) -> None:
        poly = np.polyfit(x, y, 2)
        self.a = nn.Parameter(torch.tensor(poly[0], dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(poly[1], dtype=torch.float32))
        self.c = nn.Parameter(torch.tensor(poly[2], dtype=torch.float32))
    
    def forward(self, x):
        return self.a * x**2 + self.b * x + self.c
    
    def OString(self, mx: float = 1.0, my: float = 1.0) -> str:
        return f"O({self.a * mx**2 * my:.2f}N^2)"
    
class ComplexityConstant(ComplexityModel):
    def __init__(self):
        super().__init__()
        self.c = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        
    
    def prefit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.c = nn.Parameter(torch.tensor(np.mean(y), dtype=torch.float32))
    
    def forward(self, x):
        return self.c * torch.ones_like(x)
    
    def OString(self, mx: float = 1.0, my: float = 1.0) -> str:
        return f"O({self.c * my:.2f})"

class ComplexitySqrtN(ComplexityModel):
    def __init__(self):
        super().__init__()
        self.c = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
    
    def prefit(self, x: np.ndarray, y: np.ndarray) -> None:
        sx = np.sqrt(x)

        poly = np.polyfit(sx, y, 1)

        self.c = nn.Parameter(torch.tensor(poly[0], dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(poly[1], dtype=torch.float32))
    
    def forward(self, x):
        return self.c * torch.sqrt(x) + self.b
    
    def OString(self, mx: float = 1.0, my: float = 1.0) -> str:
        return f"O({self.c * math.sqrt(mx) * my:.2f}sqrt N)"
    
@dataclass
class ComplexityFitResult:
    name: str
    model: ComplexityModel
    loss: float


def fit_complexity(x: np.ndarray, y: np.ndarray, ctx: EvaluateContext = EmptyEvaluateContext()) -> list[ComplexityFitResult]:
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    criterion = nn.MSELoss()

    candidates: dict[str, ComplexityModel] = {
        "O(1)": ComplexityConstant(),
        "O(log n)": ComplexityLogN(),
        "O(sqrt n)": ComplexitySqrtN(),
        "O(n)": ComplexityN(),
        "O(n log n)": ComplexityNLogN(),
        "O(n^2)": ComplexityNSquared(),
    }

    results: list[ComplexityFitResult] = []
    
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda")) # type: ignore[PylancereportPrivateImportUsage]

    LEARN_LIMIT_PER_MODEL = 1000
    ctx.on_complexity_fit_start(len(candidates), LEARN_LIMIT_PER_MODEL)
    for name, model in candidates.items():
        ctx.on_complexity_fit_enter_model(len(results), name)
        model.prefit(x, y)
        model = model.to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

        last_loss = float('inf')
        for epoch in range(LEARN_LIMIT_PER_MODEL):
            optimizer.zero_grad()
            
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")): # type: ignore[PylancereportPrivateImportUsage]
                y_pred = model(x_tensor)
                loss = criterion(y_pred, y_tensor)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            cur_loss = loss.item()
            if abs(last_loss - cur_loss) < 1e-8:
                ctx.on_complexity_fit_exit_model(len(results), name, cur_loss)
                break
            last_loss = cur_loss
            if epoch % 10 == 0:
                ctx.on_complexity_fit_train_step(epoch, last_loss)
        else:
            ctx.on_complexity_fit_exit_model(len(results), name, last_loss)

        model.eval()
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")): # type: ignore[PylancereportPrivateImportUsage]
                y_pred = model(x_tensor)
            error = criterion(y_pred, y_tensor).item()
            results.append(ComplexityFitResult(name, model, error))
    ctx.on_complexity_fit_end()
    return results

@dataclass
class TestResult:
    x: np.ndarray
    y: np.ndarray
    normalizer_x: float
    normalizer_y: float
    results: list[ComplexityFitResult]
    @property
    def multiplier_x(self) -> float:
        return 1.0 / self.normalizer_x
    @property
    def multiplier_y(self) -> float:
        return 1.0 / self.normalizer_y


@dataclass
class EvaluateSetting:
    target_n: Iterable[int] = field(default_factory=lambda: range(1, 1001, 100))
    uid: str = field(default_factory=lambda: str(uuid4()))

def gen_data(code: str, track: AlgorithmExecutor, ctx: EvaluateContext = EmptyEvaluateContext(), setting: EvaluateSetting | None = None) -> TestResult:
    eval_setting = setting if setting is not None else EvaluateSetting()
    x = np.array(eval_setting.target_n, dtype=np.int32)
    
    ctx.on_track_start(len(x))
    y = [0] * len(x)

    try:
        with track.session(eval_setting.uid) as session:
            for i, n in enumerate(x):
                code_run = code.format(n=n)
                report = session(code_run, 5.0, all=False)
                if report.error_type != "":
                    raise ExecutionError(report.error_type, report.error_message)
                data = report.data
                if type(data) == int:
                    ctx.on_track(i, data)
                    y[i] = data
                else:
                    assert type(data) == AlgorithmScopeData
                    cnt = data.total_calls()
                    ctx.on_track(i, cnt)
                    y[i] = cnt
                    
    except ExecutionError as e:
        ctx.on_execution_error(e.error_type, e.error_message)
        raise e
    except Exception as e:
        ctx.on_internal_error(e)
        raise e
    ctx.on_track_end()
    normalizer_x = float(1.0 / max(x))
    normalizer_y = float(1.0 / max(y))
    x_normalized = (x.astype(np.float32) * normalizer_x)
    y_normalized = (np.array(y, dtype=np.float32) * normalizer_y)
    
    results = fit_complexity(x_normalized, y_normalized, ctx)
    
    result_obj = TestResult(x_normalized, y_normalized, normalizer_x, normalizer_y, results)
    ctx.on_done(result_obj)
    return result_obj

if __name__ == "__main__":
    code_sqrt = """
def s(x):
    return x * (x + 1) // 2
def func(k):
    cnt = 0
    n = 0
    sum = 0
    dp = 0
    while k > s(n + 1):
        n += 1
        dp += n
        sum += dp
        cnt += 1
    sum += s(k - s(n))
    return cnt
func({n})
    """
    code_sqaure = """
def func(n):
    cnt = 0
    for i in range(n//100):
        for j in range(n//100):
            cnt += 1
    return cnt
func({n})
    """
    code_nlogn = """
def func(n):
    cnt = 0
    i = 1
    while i < n:
        for j in range(n):
            cnt += 1
        i *= 2
    return cnt

func({n})
    """
    code = code_nlogn
    test_result = gen_data(code, PyExecutor, setting=EvaluateSetting(target_n=range(1000, 10001, 500)))
    x = test_result.x
    y = test_result.y
    results = test_result.results

    best = min(results, key=lambda r: r.loss)
    print("Best Time Complexity:", best.name)
    print("\nAll Results:")
    for result in results:
        print(f"{result.name:10s} : {result.loss:.2e} : {result.model.OString(test_result.multiplier_x, test_result.multiplier_y)}")

    plt.figure(figsize=(10, 6))
    x_tensor = torch.arange(min(x), max(x)+1, max(x)/100, dtype=torch.float32, device=device)
    x_num = x_tensor.cpu().numpy()
    plt.scatter(x, y, label="Actual Data", color='black', s=10)
    for result in results:
        with torch.no_grad():
            y_pred = result.model(x_tensor).cpu().numpy()
            plt.plot(x_num, y_pred, label=result.name)
    plt.xlim(0, max(x))
    plt.ylim(0, max(y))
    plt.legend()
    plt.show()
