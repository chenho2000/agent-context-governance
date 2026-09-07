import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import sys
import threading
from types import SimpleNamespace
import pytest
from context_lab.cache import PrefixCache, commit_value
from context_lab.codemap import CodeMap
from context_lab.store import Store
from context_lab.providers import Ollama, Lingua
from context_lab.compression import compress, cascade
from context_lab.types import BudgetError
from context_lab.flywheel import evaluate_trial, export
from context_lab.kv import demo as kv_demo
from context_lab.demo import run
from context_lab.cli import main


def test_cache_ttl_scope_prefix_and_cost():
    cache = PrefixCache(ttl=10)
    first = cache.request("a b c", now=0)
    assert cache.request("a b c", now=1)["read"] == first["units"]
    assert cache.request("a b d", now=2)["read"] == 4
    assert cache.request("a b c", now=3, model="other")["read"] == 0
    assert cache.request("a b c", now=20)["read"] == 0
    assert commit_value(5000, 4500) < 0
    assert commit_value(5000, 4500, expired=True) > 0
    assert commit_value(5000, 4500, future_calls=200) > 0
    assert kv_demo()["append_prefix_reusable"] and kv_demo()["edit_changes_suffix"]


def test_codemap_version_range_and_scope(tmp_path):
    root = tmp_path / "code"
    root.mkdir()
    source = root / "a.py"
    source.write_text("class A:\n    def f(self):\n        return 42\n\ndef g():\n    pass\n")
    index = CodeMap(Store(tmp_path / "db"), root)
    index.build()
    result = index.get_symbol("a.py", "A.f")
    assert result["text"] == "    def f(self):\n        return 42\n"
    source.write_text("def newer():\n    pass\n")
    with pytest.raises(ValueError, match="stale"):
        index.get_symbol("a.py", "A.f")
    index.build()
    with pytest.raises(ValueError, match="absent"):
        index.get_symbol("a.py", "A.f")
    with pytest.raises(ValueError, match="outside"):
        index.get_symbol("../outside.py", "x")
    source.unlink()
    assert index.build() == {}


def test_ollama_transport_contract():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {"message": {"content": '{"summary":"ok"}'}, "prompt_eval_count": 20, "eval_count": 4}
                ).encode()
            )

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        model = Ollama("test", f"http://127.0.0.1:{server.server_port}")
        assert model.generate("summarize", {"text": "data"}) == {"summary": "ok"}
        assert requests[0]["stream"] is False and requests[0]["format"] == "json"
        assert model.last_usage["eval_count"] == 4
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_lingua_adapter_contract_no_weights(monkeypatch):
    calls = []

    class Fake:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def compress_prompt(self, context, **kwargs):
            calls.append(kwargs)
            return {"compressed_prompt": context[0][:10]}

    monkeypatch.setitem(sys.modules, "llmlingua", SimpleNamespace(PromptCompressor=Fake))
    for method in ["llmlingua", "longllmlingua", "llmlingua2"]:
        adapter = Lingua(method)
        assert adapter.compress("a long string", question="why")["compressed_prompt"] == "a long str"
    assert any(c.get("use_llmlingua2") for c in calls)
    assert any(c.get("rank_method") == "longllmlingua" for c in calls)


def test_compression_sparse_and_unmet_budget():
    text = 'Traceback\n  File "a.py", line 1\nerror'
    assert compress(text, budget=1)["text"] == text
    with pytest.raises(BudgetError):
        cascade(text, 1)
    assert cascade("INFO same\n" * 20, 30)["text"] == "INFO same\n"


def test_flywheel_no_future_in_sft_and_matched_dpo(tmp_path):
    prefix = {"visible_ids": ["a"]}
    future = {"required_ids": ["a"]}
    good = evaluate_trial(prefix, "keep", future, ["a"])
    bad = evaluate_trial(prefix, "prune", future, [])
    other = evaluate_trial({"different": True}, "prune", future, [])
    result = export([good, bad, other], tmp_path)
    assert result["dpo"] == 1 and result["skipped"]
    assert "required_ids" not in (tmp_path / "sft.jsonl").read_text()


def test_end_to_end_and_cli_restart(tmp_path, capsys):
    output = run(tmp_path / "runs")
    trace = json.loads((output / "trace.json").read_text())
    assert all(trace["assertions"].values())
    assert trace["commit"]["committed"]
    assert trace["flywheel"]["dpo"] == 1
    main(["inspect", "--state", str(output / "state"), "--session", "payment"])
    assert "Never charge twice" in capsys.readouterr().out
    main(["query", "--state", str(output / "state"), "--session", "memory", "Ada"])
    assert "memories" in capsys.readouterr().out
