"""RAG 回归评测脚本（真实 Ollama 嵌入，检索侧指标；不调用对话模型，保证可重复）。

用法：
    python evals/run_eval.py

- 自动把数据目录隔离到 evals/runtime/（不污染生产 data/）；
- 入库 3 份种子文档 + 1 份越权测试文档；
- 跑三种配置：纯向量 / 混合(RRF) / 混合+重排；
- 输出 Recall@6、MRR、离题拒答率、引用正确率，落盘 evals/results/。
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 必须在 import app.config 之前设置隔离数据目录
RUNTIME = ROOT / "evals" / "runtime"
RUNTIME.mkdir(parents=True, exist_ok=True)
import shutil
# 每次评测干净起步，保证可复现
if RUNTIME.exists():
    shutil.rmtree(RUNTIME)
RUNTIME.mkdir(parents=True, exist_ok=True)
os.environ["RAG_DATA_DIR"] = str(RUNTIME)

TOP_K = 6
EVAL_USER = "eval_a"

CONFIGS = [
    {"name": "A_纯向量", "use_hybrid": False, "use_rerank": False},
    {"name": "B_混合RRF", "use_hybrid": True, "use_rerank": False},
    {"name": "C_混合+重排", "use_hybrid": True, "use_rerank": True},
]


def setup_kb():
    from app import knowledge
    from app import config
    from app.db import init_db
    init_db()
    # 种子文档归到评测用户（默认 seed_documents 用 local_user，评测需换身份）
    cat_map = {"尺码推荐": "尺码", "洗涤养护": "洗涤", "颜色选择": "颜色"}
    n_seed = 0
    for f in sorted(Path(config.SEED_DOC_DIR).glob("*.txt")):
        r = knowledge.ingest_text(EVAL_USER, f.name, f.read_text(encoding="utf-8"),
                                  category=cat_map.get(f.stem, ""))
        if r["status"] in ("ok", "skipped"):
            n_seed += 1
    chunks = knowledge.rebuild_index()
    return n_seed, chunks


def run_config(dataset, cfg):
    from app import knowledge
    answerable = dataset["answerable"]
    off_topic = dataset["off_topic"]
    unauth = dataset["unauthorized"]

    recalls, rranks = [], []
    citation_correct, citation_total = 0, 0

    # 可答问题
    per_q = []
    for item in answerable:
        hits = knowledge.search(
            EVAL_USER, item["q"], top_k=TOP_K,
            use_hybrid=cfg["use_hybrid"], use_rerank=cfg["use_rerank"],
        )
        sources = [h.get("source") for h in hits]
        rank = next((i + 1 for i, s in enumerate(sources) if s == item["expect_source"]), 0)
        recalls.append(1.0 if rank > 0 else 0.0)
        rranks.append(1.0 / rank if rank > 0 else 0.0)
        citation_total += len(sources)
        if rank > 0:
            citation_correct += 1
        per_q.append({"q": item["q"], "expect": item["expect_source"],
                      "rank": rank, "top_sources": sources})

    # 离题问题：0 命中 = 拒答成功
    refused = 0
    off_detail = []
    for q in off_topic:
        hits = knowledge.search(EVAL_USER, q, top_k=TOP_K,
                                use_hybrid=cfg["use_hybrid"], use_rerank=cfg["use_rerank"])
        ok = len(hits) == 0
        refused += 1 if ok else 0
        off_detail.append({"q": q, "hits": len(hits), "refused": ok})

    # 越权：攻击者问受害者文档内容，必须 0 命中
    atk_hits = knowledge.search(unauth["attacker_user"], unauth["attacker_question"],
                                top_k=TOP_K, use_hybrid=cfg["use_hybrid"], use_rerank=cfg["use_rerank"])
    # 受害者自己必须能命中（对照组）
    vic_hits = knowledge.search(unauth["victim_user"], unauth["attacker_question"],
                                top_k=TOP_K, use_hybrid=cfg["use_hybrid"], use_rerank=cfg["use_rerank"])

    n = len(answerable)
    return {
        "config": cfg["name"],
        "recall_at_6": round(sum(recalls) / n, 4),
        "mrr": round(sum(rranks) / n, 4),
        "answerable_n": n,
        "citation_accuracy": round(citation_correct / n, 4),
        "off_topic_refusal_rate": round(refused / len(off_topic), 4),
        "off_topic_n": len(off_topic),
        "unauthorized_blocked": len(atk_hits) == 0,
        "victim_self_hit": len(vic_hits) > 0,
        "per_question": per_q,
        "off_detail": off_detail,
    }


def main():
    with open(ROOT / "evals" / "dataset.json", encoding="utf-8") as f:
        dataset = json.load(f)

    n_seed, n_chunks = setup_kb()

    # 越权对照文档
    from app import knowledge
    un = dataset["unauthorized"]
    knowledge.ingest_text(un["victim_user"], un["victim_doc"], un["victim_text"])

    results = []
    for cfg in CONFIGS:
        print(f"\n=== 评测配置：{cfg['name']} ===")
        r = run_config(dataset, cfg)
        results.append(r)
        print(f"Recall@{TOP_K} = {r['recall_at_6']}  MRR = {r['mrr']}  "
              f"引用正确率 = {r['citation_accuracy']}  离题拒答率 = {r['off_topic_refusal_rate']}  "
              f"越权拦截 = {r['unauthorized_blocked']}")

    out = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "top_k": TOP_K,
        "seed_docs": n_seed, "chunks_indexed": n_chunks,
        "configs": results,
    }
    res_dir = ROOT / "evals" / "results"
    res_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = res_dir / f"eval_{stamp}.json"
    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["# RAG 评测结果\n",
          f"- 时间：{out['timestamp']}　TopK={TOP_K}　种子文档 {n_seed} 份 / 索引块 {n_chunks}",
          f"- 命令：`python evals/run_eval.py`（真实 Ollama qwen3-embedding:4b 嵌入；检索侧指标，不调用对话模型）\n",
          "| 配置 | Recall@6 | MRR | 引用正确率 | 离题拒答率 | 越权拦截 |",
          "|---|---|---|---|---|---|"]
    for r in results:
        md.append(f"| {r['config']} | {r['recall_at_6']} | {r['mrr']} | "
                  f"{r['citation_accuracy']} | {r['off_topic_refusal_rate']} | {r['unauthorized_blocked']} |")
    md.append("")
    md_path = res_dir / "latest.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"\n结果已落盘：{json_path}")
    print(f"摘要：{md_path}")


if __name__ == "__main__":
    main()
