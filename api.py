import asyncio
import json
from fastapi.responses import StreamingResponse
import torch
from runvirtual import PyAlgorithmExecutor, PyExecutor
from tester import EvaluateContext, gen_data, TestResult, device
from threading import Thread
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from uuid import uuid4

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
        dictionary = {"real_x": result.x.tolist(), "real_y": result.y.tolist(), }
        model_predictions = {}
        for case in result.results:
            with torch.no_grad():
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
    testing_thread = Thread(target=gen_data, args=(request.code, PyExecutor, 10000, ctx), daemon=False)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=7004, reload=False)