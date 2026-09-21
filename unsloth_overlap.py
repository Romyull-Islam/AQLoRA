"""How much does the budgeted rule overlap Unsloth's hand-curated selection?

Both protect layers in fp16 on Qwen2.5-7B and both reach 89.0 on the 8-task
average, but they are not the same layers. Reproduces the comparison in 7.2.

usage: python unsloth_overlap.py
"""
import json

UNSLOTH = "results/unsloth_skip_qwen7b.json"
OURS = "results/results-cs-qwen2.5-7b-aqslora-q/plan.json"
TASKS = ["boolq", "piqa", "social_i_qa", "hellaswag",
         "winogrande", "ARC-Easy", "ARC-Challenge", "openbookqa"]

theirs = set(json.load(open(UNSLOTH)))
# our plan names projections (…self_attn.k_proj); Unsloth names the parent module
ours = {n.rsplit(".", 1)[0] for n in json.load(open(OURS))["skip_modules"]}
shared = ours & theirs

print(f"Unsloth modules : {len(theirs)}")
print(f"ours            : {len(ours)}")
print(f"shared          : {len(shared)}")
print(f"\nin both:")
for m in sorted(shared):
    print(f"  {m}")
print(f"\ntheirs only:")
for m in sorted(theirs - ours):
    print(f"  {m}")

for run in ("results-cs-qwen2.5-7b-unslothskip", "results-cs-qwen2.5-7b-aqslora-q"):
    c = json.load(open(f"results/{run}/report.json"))["commonsense"]
    print(f"\n{run}: {sum(c[t] for t in TASKS) / 8 * 100:.2f}")
