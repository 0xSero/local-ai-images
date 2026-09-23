"""Hot-token list for the MTP draft head: the N most frequent tokens of the served tokenizer over UltraChat test_sft
(English chat, assistant turns weighted x2), Python stdlib sources and SGLang docs, plus every special/added token. Output: torch list file for --speculative-token-map."""
import glob, os, sys, collections, torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer
model, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
tok = AutoTokenizer.from_pretrained(model)
cnt = collections.Counter()
def add(text, w=1):
    for i in tok(text, add_special_tokens=False)["input_ids"]:
        cnt[i] += w
t = pq.read_table(glob.glob("/w/runs/tokenmap/ultrachat/data/*.parquet")[0], columns=["messages"]).to_pylist()
for row in t:
    for m in row["messages"]:
        add(m["content"], 2 if m["role"] == "assistant" else 1)
print("ultrachat done", sum(cnt.values()), flush=True)
for f in glob.glob("/usr/lib/python3.12/**/*.py", recursive=True)[:4000]:
    try: add(open(f, errors="ignore").read())
    except Exception: pass
for f in glob.glob("/sgl-workspace/sglang/docs/**/*.md", recursive=True) + glob.glob("/sgl-workspace/sglang/docs/**/*.mdx", recursive=True):
    try: add(open(f, errors="ignore").read())
    except Exception: pass
print("corpus tokens", sum(cnt.values()), "distinct", len(cnt), flush=True)
hot = [i for i, _ in cnt.most_common(n)]
must = set(tok.all_special_ids) | {v for k, v in tok.get_added_vocab().items()}
vocab = tok.get_vocab()
hs = set(hot)
extra = [i for i in sorted(must) if i not in hs]
hot = hot[: n - len(extra)] + extra
total = sum(cnt.values()); covered = sum(cnt[i] for i in hot)
print(f"hot {len(hot)} tokens cover {100*covered/total:.2f} % of corpus tokens; forced {len(extra)} special/byte tokens")
torch.save(sorted(hot), out)
