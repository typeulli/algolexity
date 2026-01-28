import asyncio
import json
import logging
import time
import datetime
import docker
import torch
from data import ExecutionError
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from runvirtual import PyExecutor
from tester import *
from threading import Thread
from pydantic import BaseModel
from pathlib import Path
from uuid import uuid4

path_here = Path(__file__).parent
path_log = path_here / "logs" / f"log-api.python.{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
path_log.parent.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("api.algolexity")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(path_log, encoding="utf-8")
logger.addHandler(fh)



# def predict_n(code: str) -> int:
#     low = 1
#     high = 100000
#     while low < high:
#         mid = (low + high + 1) // 2
#         try:
#             result = PyExecutor.run(code.format(n=mid), timeout=2.0, all=False)
#             if result.error_type == "":
#                 low = mid
#             else:
#                 high = mid - 1
#         except TimeoutError:
#             high = mid - 1
#     return low

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EvaluateEvent(EvaluateContext):
    def __init__(self, loop):
        super().__init__()
        self.queue = asyncio.Queue()
        self.loop = loop
        self.closed = False

    def message(self, state, **kwargs):
        self.loop.call_soon_threadsafe(
            self.queue.put_nowait,
            f"data: {json.dumps({'state': state, **kwargs})}\n\n"
        )
        
    def on_track_start(self, n):
        self.message("track_start", total=n)
    
    def on_track(self, idx, n):
        self.message("track", index=idx, cnt=n)
        
    def on_track_end(self):
        self.message("track_end")

    def on_complexity_fit_start(self, total_models: int, learn_limit_per_model: int):
        self.message("fit_start", total_models=total_models, learn_limit_per_model=learn_limit_per_model)

    def on_complexity_fit_enter_model(self, model_index: int, model_name: str):
        self.message("fit_enter_model", model_index=model_index, model_name=model_name)

    def on_complexity_fit_train_step(self, epoch: int, loss: float):
        self.message("fit_train_step", epoch=epoch, loss=loss)

    def on_complexity_fit_exit_model(self, model_index: int, model_name: str, final_loss: float):
        self.message("fit_exit_model", model_index=model_index, model_name=model_name, final_loss=final_loss)

    def on_complexity_fit_end(self):
        self.message("fit_end")
    
    def on_done(self, result: TestResult) -> None:
        x_tensor = torch.Tensor(result.x).to(device)
        x_num = x_tensor.cpu().numpy()
        dictionary = {"real_x": result.x.tolist(), "real_y": result.y.tolist()}
        model_predictions = {}
        for case in result.results:
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    model_predictions[case.name] = case.model(x_tensor).cpu().numpy()
        model_graph = []
        for i in range(len(x_num)):
            model_graph.append({
                "n": x_num[i].item(),
                **{case.name: model_predictions[case.name][i].item() for case in result.results}
            })
        dictionary["model_graph"] = model_graph
        
        self.message("plot_data", data=dictionary)
                
        best = min(result.results, key=lambda r: r.loss)
        for case in result.results:
            self.message("result", name=case.name, loss=case.loss, complexity=case.model.OString(result.multiplier_x, result.multiplier_y))
        self.message("best", name=best.name, loss=best.loss)
        self.message("done")
        self.closed = True

    def on_execution_error(self, error_type: str, error_message: str) -> None:
        self.message("error", error_type=error_type, error_message=error_message)
        self.closed = True
    
    def on_internal_error(self, error: Exception) -> None:
        self.message("error", error_type=type(error).__name__, error_message=str(error))
        print(f"Internal Error: {type(error).__name__}: {str(error)}")
        self.closed = True

class EvalRequest(BaseModel):
    code: str

async def event_generator(ctx: EvaluateEvent):
    while not ctx.closed or not ctx.queue.empty():
        while not ctx.queue.empty():
            yield await ctx.queue.get()
        await asyncio.sleep(0.05)
    

        

contexts = {}
@app.post("/evaluate")
async def post_evaluate_algorithm(request: EvalRequest):
    ctx = EvaluateEvent(asyncio.get_event_loop())
    testing_thread = Thread(target=ExecutionError.wrap_ignore(gen_data), args=(request.code, PyExecutor, ctx, EvaluateSetting(target_n=range(1, 10001, 100))), daemon=False)
    testing_thread.start()
    uid = str(uuid4())
    contexts[uid] = ctx
    return {"uuid": uid}

@app.get("/evaluate/stream/{uuid}")
async def sse_evaluate_algorithm(uuid: str):
    ctx = contexts.get(uuid)
    if not ctx:
        return {"error": "Invalid ID"}
    return StreamingResponse(
        event_generator(ctx),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

class TimedCache:
    def __init__(self, timeout_seconds: int):
        self.timeout_seconds = timeout_seconds
        self.value = None
        self.timestamp = 0

    def has(self):
        return (time.time() - self.timestamp) < self.timeout_seconds
    def get(self):
        return self.value
    def set(self, value):
        self.value = value
        self.timestamp = time.time()

CACHE_DOCKER_STATUS = TimedCache(timeout_seconds=3)
@app.get("/status/docker")
def find_docker_container():
    if CACHE_DOCKER_STATUS.has():
        return CACHE_DOCKER_STATUS.get()
    client = docker.from_env()
    containers = client.containers.list()  # running containers only
    languages = ["python"]
    available_languages = set()
    for c in containers:
        for lang in languages:
            if any(tag.startswith(f"algolexity-{lang}") for tag in c.image.tags):
                available_languages.add(lang)
    response = {lang: (lang in available_languages) for lang in languages}
    CACHE_DOCKER_STATUS.set(response)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=7004, reload=False, proxy_headers=True, forwarded_allow_ips="*")