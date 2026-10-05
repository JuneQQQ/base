#!/usr/bin/env python3
"""Turn the 4,407 training reports into soft labels (+ per-cell loss weights).

Backends (all emit one status per finding, then knee.report_labels.statuses_to_soft):
  regex   multilingual lexicon, CPU, ~5 s.  ~0.82 macro AUC vs the 58 gold (untuned baseline)
  claude  Claude via the Message Batches API (50% price, async) with structured JSON output;
          refusals/errors are retried synchronously with server-side fallbacks
  vllm    local open-weights model (e.g. Qwen) with JSON-schema guided decoding (needs a GPU)
  table   import a public soft-label CSV (e.g. Kaggle datasets stevenleehans/rsna-knee-llm-report-labels,
          pilkwang/rsna-knee-llm-labels) and normalise its columns
Then `merge` averages several label files: two independent readers beat either one alone.

The host confirmed (discussion 733965) that sending report text to a commercial LLM API is
allowed for this competition, provided the service is reasonably accessible and low-cost.

Examples
  python scripts/02_extract_labels.py regex  --train-csv train.csv --out labels_regex.csv
  python scripts/02_extract_labels.py claude --train-csv train.csv --out labels_claude.csv
  python scripts/02_extract_labels.py vllm   --train-csv train.csv --out labels_qwen.csv --model Qwen/Qwen3-14B
  python scripts/02_extract_labels.py table  --train-csv train.csv --out labels_pub.csv --table llm_labels_v4_blend.csv
  python scripts/02_extract_labels.py merge  --train-csv train.csv --out labels_final.csv --inputs a.csv b.csv
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from knee.config import LABELS  # noqa: E402
from knee.report_labels import (RUBRIC, USER_TEMPLATE, evaluate_against_gold, output_schema,  # noqa: E402
                                parse_llm_json, regex_statuses, statuses_to_soft)


def save(soft: pd.DataFrame, out: str, train: pd.DataFrame, statuses: pd.DataFrame | None = None):
    soft.index.name = "StudyInstanceUID"
    soft.to_csv(out)
    if statuses is not None:
        statuses.to_csv(out.replace(".csv", "_status.csv"))
    print(f"wrote {out}: {len(soft)} studies")
    print("agreement with the 58 radiologist-labelled studies (AUC):")
    print(evaluate_against_gold(soft, train).round(3).to_string())


# ------------------------------------------------------------------ claude
def run_claude(a, train):
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic(max_retries=5)
    schema = output_schema(with_evidence=not a.no_evidence)
    system = [{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral"}}]
    output_config = {"effort": a.effort, "format": {"type": "json_schema", "schema": schema}}
    raw_path = a.out.replace(".csv", "_raw.jsonl")
    done = {}
    if os.path.exists(raw_path):
        for line in open(raw_path):
            r = json.loads(line)
            if r["obj"] is not None:  # failed rows are retried
                done[r["uid"]] = r
    todo = train[~train["StudyInstanceUID"].isin(done)].reset_index(drop=True)
    cid2uid = {f"s{i}": u for i, u in enumerate(todo["StudyInstanceUID"])}  # custom_id: [A-Za-z0-9_-]{1,64}

    def params(report):
        return dict(model=a.model, max_tokens=8000, system=system, output_config=output_config,
                    messages=[{"role": "user", "content": USER_TEMPLATE.format(report=report)}])

    def record(uid, obj=None, error=None):
        r = {"uid": uid, "obj": obj, "error": error}
        done[uid] = r
        with open(raw_path, "a") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    retry = []
    if len(todo) and a.mode == "batch":
        state = a.out.replace(".csv", "_batch.json")
        if os.path.exists(state):
            batch_id = json.load(open(state))["id"]
            cid2uid = json.load(open(state))["cid2uid"]
            print("resuming batch", batch_id)
        else:
            reqs = [Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**params(rep)))
                    for cid, rep in zip(cid2uid, todo["Report"])]
            batch_id = client.messages.batches.create(requests=reqs).id
            json.dump({"id": batch_id, "cid2uid": cid2uid}, open(state, "w"))
            print("created batch", batch_id, "with", len(reqs), "requests")
        while True:
            b = client.messages.batches.retrieve(batch_id)
            if b.processing_status == "ended":
                break
            print(f"  {b.processing_status}: {b.request_counts.processing} processing, "
                  f"{b.request_counts.succeeded} succeeded", flush=True)
            time.sleep(60)
        for res in client.messages.batches.results(batch_id):  # arrives in any order: key by custom_id
            uid = cid2uid[res.custom_id]
            if res.result.type != "succeeded":
                retry.append(uid)
                continue
            msg = res.result.message
            if msg.stop_reason == "refusal":  # Batches API doesn't take `fallbacks`; retry below
                retry.append(uid)
                continue
            text = next((blk.text for blk in msg.content if blk.type == "text"), "")
            try:
                record(uid, json.loads(text))
            except json.JSONDecodeError:
                retry.append(uid)
        os.remove(state)
    elif len(todo):
        retry = todo["StudyInstanceUID"].tolist()

    # synchronous path (mode=sync, or batch leftovers): server-side fallback on refusals
    rep = train.set_index("StudyInstanceUID")["Report"]

    def one(uid):
        try:
            msg = client.beta.messages.create(**params(rep[uid]),
                                              betas=["server-side-fallback-2026-07-01"],
                                              fallbacks="default")
        except anthropic.APIStatusError as e:
            return uid, None, f"{e.status_code}: {e.message}"
        except anthropic.APIConnectionError as e:
            return uid, None, f"connection: {e}"
        if msg.stop_reason == "refusal":
            return uid, None, "refusal"
        text = next((blk.text for blk in msg.content if blk.type == "text"), "")
        try:
            return uid, json.loads(text), None
        except json.JSONDecodeError:
            return uid, None, f"bad json (stop_reason={msg.stop_reason})"

    if retry:
        print(f"synchronous requests: {len(retry)}")
        with ThreadPoolExecutor(a.threads) as ex:
            for uid, obj, err in ex.map(one, retry):
                record(uid, obj, err)
    failed = [u for u, r in done.items() if r["obj"] is None]
    if failed:
        print(f"{len(failed)} studies failed (kept as not_mentioned); rerun the command to retry them")
        for u in failed:  # let a rerun pick them up again
            done.pop(u)
    rows = {u: parse_llm_json(r["obj"]) for u, r in done.items()}
    return pd.DataFrame.from_dict(rows, orient="index").reindex(train["StudyInstanceUID"])


# ------------------------------------------------------------------ vllm (local, free)
def run_vllm(a, train):
    from vllm import LLM, SamplingParams
    schema = output_schema(with_evidence=not a.no_evidence)
    try:  # vLLM renamed guided decoding to structured outputs; support both
        from vllm.sampling_params import StructuredOutputsParams
        sp = SamplingParams(temperature=0.0, max_tokens=1500, structured_outputs=StructuredOutputsParams(json=schema))
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        sp = SamplingParams(temperature=0.0, max_tokens=1500, guided_decoding=GuidedDecodingParams(json=schema))
    llm = LLM(model=a.model, max_model_len=8192, tensor_parallel_size=a.tp, gpu_memory_utilization=0.9)
    convs = [[{"role": "system", "content": RUBRIC},
              {"role": "user", "content": USER_TEMPLATE.format(report=r)}] for r in train["Report"]]
    try:  # Qwen3-style templates: thinking blocks break JSON-constrained decoding
        outs = llm.chat(convs, sp, chat_template_kwargs={"enable_thinking": False})
    except TypeError:
        outs = llm.chat(convs, sp)
    rows, bad = {}, 0
    for uid, o in zip(train["StudyInstanceUID"], outs):
        try:
            rows[uid] = parse_llm_json(json.loads(o.outputs[0].text))
        except Exception:
            rows[uid] = parse_llm_json({})
            bad += 1
    print(f"unparsed outputs: {bad}")
    return pd.DataFrame.from_dict(rows, orient="index")


# ------------------------------------------------------------------ public table / merge
def _canon(c):
    return str(c).lower().replace("_", " ").replace("'", "").replace("’", "").strip()


def load_table(path):
    df = pd.read_csv(path)
    id_col = next(c for c in df.columns if _canon(c) in ("studyinstanceuid", "study instance uid", "study id"))
    df = df.set_index(df[id_col].astype(str))
    out = pd.DataFrame(index=df.index)
    for lab in LABELS:
        col = next((c for c in df.columns if _canon(c) == _canon(lab)), None)
        if col is None:
            raise SystemExit(f"{path}: no column for {lab}")
        out[lab] = pd.to_numeric(df[col], errors="coerce")
        wcol = next((c for c in df.columns if _canon(c) == _canon(f"w_{lab}")), None)
        out[f"w_{lab}"] = df[wcol] if wcol else np.where(out[lab].notna(), 1.0, 0.0)
        out[lab] = out[lab].fillna(0.5)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backend", choices=["regex", "claude", "vllm", "table", "merge"])
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None, help="claude: model id (default claude-opus-5-5); "
                    "vllm: HF model path, e.g. Qwen/Qwen3-14B")
    ap.add_argument("--mode", default="batch", choices=["batch", "sync"], help="claude only")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high"],
                    help="claude only: report reading is extraction, low effort is usually enough")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tp", type=int, default=1, help="vllm tensor parallel size")
    ap.add_argument("--no-evidence", action="store_true", help="drop the quoted evidence field (fewer tokens)")
    ap.add_argument("--table", default=None)
    ap.add_argument("--inputs", nargs="+", default=None)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    train = pd.read_csv(a.train_csv)
    train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)
    if a.limit:
        train = train.head(a.limit)

    if a.backend == "regex":
        st = pd.DataFrame([regex_statuses(r) for r in train["Report"]], index=train["StudyInstanceUID"])
        save(statuses_to_soft(st), a.out, train, st)
    elif a.backend == "claude":
        a.model = a.model or "claude-opus-5-5"
        st = run_claude(a, train)
        save(statuses_to_soft(st), a.out, train, st)
    elif a.backend == "vllm":
        if not a.model:
            raise SystemExit("--model is required for vllm")
        st = run_vllm(a, train)
        save(statuses_to_soft(st), a.out, train, st)
    elif a.backend == "table":
        save(load_table(a.table), a.out, train)
    else:
        tabs = [load_table(p) for p in a.inputs]
        w = np.asarray(a.weights or [1.0] * len(tabs), float)
        w = w / w.sum()
        idx = tabs[0].index
        for t in tabs[1:]:
            idx = idx.intersection(t.index)
        out = sum(wi * t.loc[idx] for wi, t in zip(w, tabs))
        save(out, a.out, train)


if __name__ == "__main__":
    main()
