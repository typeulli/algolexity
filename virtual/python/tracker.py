import ast
import json
from multiprocessing import Process
from pathlib import Path
from typing import Callable, TypedDict
from data import AlgorithmScopeData

def count_stmt(node: ast.expr) -> int:
    match type(node):
        case ast.BinOp:
            assert type(node) == ast.BinOp
            return count_stmt(node.left) + count_stmt(node.right) + 1
        case ast.UnaryOp:
            assert type(node) == ast.UnaryOp
            return count_stmt(node.operand) + 1
        case ast.Call:
            assert type(node) == ast.Call
            total = 0
            for arg in node.args:
                total += count_stmt(arg)
            for keyword in node.keywords:
                total += count_stmt(keyword.value)
            return total
        case ast.Attribute:
            assert type(node) == ast.Attribute
            return count_stmt(node.value) + 1
        case ast.Subscript:
            assert type(node) == ast.Subscript
            return count_stmt(node.value) + count_stmt(node.slice) + 1
        case _:
            return 0

class InstrumentingVisitor(ast.NodeTransformer):
    def make_on_call(self, n):
        return ast.Expr(
            value=ast.Call(
                func=ast.Name(id='on_call', ctx=ast.Load()),
                args=[ast.Constant(n)],
                keywords=[]
            )
        )

    def make_on_scope_enter(self, scope_type, name):
        return ast.Expr(
            value=ast.Call(
                func=ast.Name(id='on_scope_enter', ctx=ast.Load()),
                args=[ast.Constant(scope_type), ast.Constant(name)],
                keywords=[]
            )
        )
    def make_on_scope_exit(self):
        return ast.Expr(
            value=ast.Call(
                func=ast.Name(id='on_scope_exit', ctx=ast.Load()),
                args=[],
                keywords=[]
            )
        )

    def instrument_block(self, stmts):
        new_body = []
        counts = 0
        def flush():
            nonlocal counts
            if counts > 0:
                new_body.append(self.make_on_call(counts))
                counts = 0

        for stmt in stmts:
            if isinstance(stmt, ast.Assign) or isinstance(stmt, ast.AugAssign):
                new_body.append(stmt)
                counts += 1 + count_stmt(stmt.value)
            elif type(stmt) in (ast.Return, ast.Raise, ast.Break, ast.Continue):
                flush()
                new_body.append(self.make_on_scope_exit())
                new_body.append(stmt)
            elif type(stmt) in (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.AsyncWith, ast.AsyncFor, ast.Match, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef):
                flush()
                new_body.append(stmt)
            else:
                flush()
                new_body.append(stmt)
                counts += count_stmt(stmt)
                flush()

        flush()
        return new_body
    
    def visit_Module(self, node):
        self.generic_visit(node)
        node.body = self.instrument_block(node.body)
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.body = [
            self.make_on_scope_enter("function", node.name),
            *self.instrument_block(node.body),
             self.make_on_scope_exit()
        ]
        return node

    def visit_While(self, node):
        self.generic_visit(node)
        node.body = [
            self.make_on_scope_enter("while", "<while>"),
            *self.instrument_block(node.body),
            self.make_on_scope_exit()
        ]
        return node

    def visit_For(self, node):
        self.generic_visit(node)
        node.body = [
            self.make_on_scope_enter("for", "<for>"),
            *self.instrument_block(node.body),
            self.make_on_scope_exit()
        ]
        return node

    

class AlgorithmTracker:
    def __init__(self, compact: bool) -> None:
        self.module_scope: AlgorithmScopeData = AlgorithmScopeData("module", "<module>", [])
        self.current_scope: AlgorithmScopeData = self.module_scope
        self.scope_stack: list[AlgorithmScopeData] = []

        self.compact = compact
        if self.compact:
            self.current_scope.stack.append(0)

    def on_scope_enter(self, scope_type: str, name: str) -> None:
        if self.compact: return
        self.scope_stack.append(self.current_scope)
        new_scope = AlgorithmScopeData(scope_type, name, [])
        self.current_scope.stack.append(new_scope)
        self.current_scope = new_scope
        
    def on_scope_exit(self) -> None:
        if self.compact: return
        self.current_scope = self.scope_stack.pop()

    def on_call(self, n: int) -> None:
        if self.compact:
            self.current_scope.stack[-1] += n # type: ignore
            return
        if self.current_scope.stack and isinstance(self.current_scope.stack[-1], int):
            self.current_scope.stack[-1] += n
        else:
            self.current_scope.stack.append(n)

def pyTrack(code: str, compact: bool) -> AlgorithmTracker:
    tree = ast.parse(code)
    visitor = InstrumentingVisitor()
    tree = visitor.visit(tree)
    ast.fix_missing_locations(tree)
    tracker = AlgorithmTracker(compact)
    exec(compile(tree, filename="<ast>", mode="exec"), {'on_scope_enter': tracker.on_scope_enter, 'on_scope_exit': tracker.on_scope_exit, 'on_call': tracker.on_call})
    return tracker



class Setting(TypedDict):
    target: str
    request_all: bool
    callback: Callable[[dict], None]


def _run(code: str, setting: Setting) -> None:
    result = {}
    try:
        tracker = pyTrack(code, compact = not setting["request_all"])
        result = {
            "result": "success",
            "message": "",
            "data": tracker.module_scope.toJson() if setting["request_all"] else tracker.module_scope.total_calls()
        }
    except Exception as e:
        result = {
            "result": "error",
            "message": type(e).__name__ + ":" + str(e),
            "data": {} if setting["request_all"] else -1
        }
    setting["callback"](result)

path_here = Path(__file__).parent.resolve()


if __name__ == "__main__":
    path_here = Path(__file__).parent
    path_log = path_here / "out" / "log.python.txt"

    (path_here / "out").mkdir(parents=True, exist_ok=True)
    while True:
        for path_request in (path_here / "in").glob("*.request.json"):
            try:
                setting = {"target": "", "request_all": False, "callback": lambda x: None}
                setting.update(json.loads(path_request.read_text()))

                if not setting["target"]:
                    continue
                
                setting["callback"] = lambda result: (path_here / "out" / f"{setting['target']}.report.json").write_text(json.dumps(result))

                path_target = path_here / "in" / f"{setting['target']}.py"
                code = path_target.read_text()

                p = Process(
                    target=_run,
                    args=(code, setting),
                    daemon=True
                )
                p.start()

            except Exception as e:
                path_log.parent.mkdir(parents=True, exist_ok=True)
                with path_log.open("a", encoding="utf-8") as f:
                    f.write(
                        f"Error processing {path_request}: "
                        f"{type(e).__name__}:{e}\n"
                    )