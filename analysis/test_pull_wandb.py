"""Tests for analysis/pull_wandb.py (no network / no wandb needed).

Run with pytest, or standalone: `python analysis/test_pull_wandb.py`.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pull_wandb as P


# --- fakes (stand in for wandb Api / Run, no network) ---------------------------

class FakeRun:
    def __init__(self, rid, history=None, raise_history=False, summary=None):
        self.id = rid
        self.name = f"run-{rid}"
        self.group = "grp"
        self.tags = ["sft", "fr-en"]
        self.state = "finished"
        self.created_at = "2026-09-29T00:00:00"
        self.url = f"http://wandb/{rid}"
        self.config = {"model": {"lora_r": 8}, "train": {"learning_rate": 2e-4}}
        self._history = history if history is not None else [{"train/global_step": 1}]
        self._raise_history = raise_history
        self.summary = type("S", (), {"_json_dict": summary or {"eval/bleu": 64.0}})()

    def scan_history(self):
        if self._raise_history:
            raise RuntimeError("history fetch failed")
        return list(self._history)


class FakeApi:
    def __init__(self):
        self.default_entity = "defent"
        self.calls = {}

    def run(self, path):
        return FakeRun(path.split("/")[-1])

    def runs(self, path, filters=None, order=None):
        self.calls = {"path": path, "filters": filters, "order": order}
        return [FakeRun(f"r{i}") for i in range(100)]


# --- resolve_runs ---------------------------------------------------------------

def test_resolve_runs_run_ids_bypass_filters():
    api = FakeApi()
    runs = P.resolve_runs(api, "ent", "hfmt", run_ids=["abc", "def"], limit=1)
    assert [r.id for r in runs] == ["abc", "def"]
    assert api.calls == {}  # api.runs not consulted


def test_resolve_runs_filters_and_limit():
    api = FakeApi()
    runs = P.resolve_runs(api, "ent", "hfmt", group="g1", state="finished", limit=3)
    assert api.calls["path"] == "ent/hfmt"
    assert api.calls["filters"] == {"group": "g1", "state": "finished"}
    assert api.calls["order"] == "-created_at"
    assert len(runs) == 3


def test_resolve_runs_state_any_drops_state_filter():
    api = FakeApi()
    P.resolve_runs(api, "ent", "hfmt", state="any", limit=1)
    assert "state" not in (api.calls["filters"] or {})


# --- collect_runs / collect_run -------------------------------------------------

def test_collect_runs_skips_bad_run():
    runs = [FakeRun("good"), FakeRun("bad", raise_history=True)]
    got = P.collect_runs(runs)
    assert [r["id"] for r in got] == ["good"]


def test_collect_run_shape_and_empty_history():
    r = P.collect_run(FakeRun("x", history=[]))
    assert r["id"] == "x" and r["history"] == []
    assert r["config"]["train"]["learning_rate"] == 2e-4
    assert r["summary"]["eval/bleu"] == 64.0
    assert set(["id", "name", "group", "tags", "state", "created_at", "url",
                "config", "summary", "history"]).issubset(r)


# --- build_export / write_export / serialization --------------------------------

def test_build_export_shape():
    collected = [P.collect_run(FakeRun("a"))]
    exp = P.build_export(collected, project="hfmt", entity="ent",
                         filters={"group": None, "state": "finished"})
    assert exp["meta"]["num_runs"] == 1
    assert exp["meta"]["project"] == "hfmt" and exp["meta"]["entity"] == "ent"
    assert "exported_at" in exp["meta"]
    assert exp["runs"] is collected


def test_write_export_roundtrip_makedirs():
    exp = P.build_export([P.collect_run(FakeRun("a"))],
                         project="hfmt", entity="ent", filters={})
    out = os.path.join(tempfile.mkdtemp(), "sub", "wandb_export.json")  # nested -> makedirs
    size = P.write_export(exp, out)
    assert size > 0
    data = json.load(open(out))
    assert data["meta"]["num_runs"] == 1 and data["runs"][0]["id"] == "a"


def test_json_default_coercion():
    class FakeScalar:  # np.float32-like
        def item(self): return 3.5
        def tolist(self): return 3.5

    class FakeArray:   # np.ndarray-like
        def tolist(self): return [1, 2, 3]
        def item(self): raise ValueError("more than one element")

    class Weird:
        def __repr__(self): return "WEIRD"

    assert P._json_default(FakeScalar()) == 3.5
    assert P._json_default(FakeArray()) == [1, 2, 3]
    assert P._json_default(Weird()) == "WEIRD"
    # and it works as the json.dumps hook
    assert json.loads(json.dumps({"a": FakeArray()}, default=P._json_default)) == {"a": [1, 2, 3]}


# --- error classifier -----------------------------------------------------------

def test_friendly_wandb_error_classification():
    class AuthenticationError(Exception):
        pass

    conn = Exception("Connection refused")
    type(conn).__name__  # generic; rely on message
    assert "authentication" in P._friendly_wandb_error(AuthenticationError("x")).lower()
    assert "network" in P._friendly_wandb_error(Exception("Request timed out")).lower()
    assert P._friendly_wandb_error(Exception("weird")).startswith("W&B error")


def test_die_exits_nonzero():
    try:
        P._die("boom")
    except SystemExit as se:
        assert se.code == 1
    else:
        raise AssertionError("_die did not exit")


# --- reduction ------------------------------------------------------------------

def test_sigfig():
    assert P._sigfig(0.5889937400817871) == 0.588994
    assert P._sigfig(1.20634567, 3) == 1.21
    assert P._sigfig(0) == 0
    assert P._sigfig(True) is True        # bool untouched
    assert P._sigfig("x") == "x"          # non-float untouched
    assert P._sigfig(123) == 123          # int untouched


def test_metric_series_per_metric_and_offset():
    # eval/loss on steps 50,100 (trainer); eval/bleu on 60,110 (callback) -> offset
    hist = [
        {"train/global_step": 50, "eval/loss": 0.49},
        {"train/global_step": 60, "eval/bleu": 61.07},
        {"train/global_step": 100, "eval/loss": 0.47},
        {"train/global_step": 110, "eval/bleu": 62.24},
    ]
    s = P._metric_series(hist, P.EVAL_METRICS)
    assert s["loss"] == [[50, 0.49], [100, 0.47]]       # own grid
    assert s["bleu"] == [[60, 61.07], [110, 62.24]]     # own (offset) grid, no None
    assert "chrf" not in s                               # never-logged metric omitted


def test_reduce_run_shape_and_config_fallback():
    hist = [{"train/global_step": 10, "train/loss": 0.8},
            {"train/global_step": 50, "eval/loss": 0.49, "eval/bleu": 64.0}]
    run = {"id": "a", "name": "n", "group": "g", "tags": [], "state": "finished",
           "created_at": "t", "url": "u",
           "config": {"max_steps": 800, "eval_steps": 50, "logging_steps": 10,
                      "hfmt": {"model": {"lora_r": 8}}},
           "history": hist}
    red = P.reduce_run(run)
    assert red["config"] == {"model": {"lora_r": 8}}     # config.hfmt only
    assert red["steps"]["max_step"] == 800 and red["steps"]["eval_steps"] == 50
    assert red["trajectory"]["train"]["loss"] == [[10, 0.8]]
    assert red["trajectory"]["eval"]["loss"] == [[50, 0.49]]
    # no hfmt block -> fall back to the full config, read cadence from hfmt.train
    run2 = {"config": {"hfmt": {"train": {"eval_steps": 25}}}, "history": []}
    assert P.reduce_run(run2)["steps"]["eval_steps"] == 25


def test_build_export_reduced_flag():
    assert P.build_export([], project="p", entity="e", filters={})["meta"]["reduced"] is True
    assert P.build_export([], project="p", entity="e", filters={}, reduced=False)["meta"]["reduced"] is False


def test_dumps_export_inlines_series():
    obj = {"meta": {"a": 1},
           "runs": [{"trajectory": {"eval": {"loss": [[50, 0.49], [100, 0.47]]}}}]}
    text = P.dumps_export(obj)
    assert "[[50,0.49],[100,0.47]]" in text               # series on one compact line
    assert json.loads(text) == obj                          # round-trips exactly
    assert "\n" in text                                     # structure still indented
    assert text.count("\n") < json.dumps(obj, indent=2).count("\n")  # far fewer newlines


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
