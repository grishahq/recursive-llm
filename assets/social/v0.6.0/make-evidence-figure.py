"""Render the actual differential control results as a shareable figure."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "experiments/publication-20261102/offline-controls.json"
data = json.loads(SOURCE.read_text())
tampered = {
    "forged_quote",
    "forged_hash",
    "wrong_page",
    "shifted_offset",
    "boolean_page",
    "unknown_document",
    "stale_same_id",
}


def records(variant):
    return {item["fixture"]: item for item in data["records"] if item["variant"] == variant}


old = records("head-default")
custom = records("head-with-source-validator")
new = records("current-document-api")
for name in tampered:
    assert old[name]["status"] == "accepted" and not old[name]["grade"]["passed"]
    for variant in (custom, new):
        assert variant[name]["grade"]["passed"]
        assert sum(e["kind"] == "final_answer_rejected" for e in variant[name]["trajectory"]) == 1
assert old["native_json"]["status"] == "rejected_or_failed"
assert new["native_json"]["grade"]["passed"]
assert all(items["wrong_claim_real_quote"]["status"] == "accepted" for items in (old, custom, new))
empty = {item["variant"]: item for item in data["empty_cli"]}
assert empty["head"]["status"] == "rejected_or_failed" and empty["current"]["status"] == "accepted"
loader = {item["variant"]: item for item in data["loader"]}
assert loader["head_companion_default"]["status"] == "rejected"
assert loader["current_companion_long"]["page_count"] == 157

rows = [
    ["Altered source reference\n7 controlled cases", "7 accepted\nwithout source checks", "7 rejected\nValid retry accepted"],
    ["Standalone final JSON\n1-call allowance", "No accepted final", "Accepted"],
    ["Empty CLI text,\nthen valid answer", "Transport error", "Recovered in 2 calls"],
    ["157-page PDF\nExplicit long profile", "Upload rejected", "157 pages preserved"],
    ["Wrong claim,\nreal quotation", "Accepted", "Accepted\nSemantic limit"],
]
background, ink, blue = "#fbfaf7", "#0b2144", "#1559c5"
fig = plt.figure(figsize=(8, 8), dpi=200, facecolor=background)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_facecolor(background)
ax.axis("off")
ax.text(.055, .945, "recursive-llm", fontsize=22, weight="bold", color=ink)
ax.text(.055, .875, "Source checks, under test", fontsize=27, weight="bold", color=ink)
ax.text(.055, .826, "Recorded controls  |  2 November 2026", fontsize=12, color="#52627a")
table = ax.table(
    cellText=rows,
    colLabels=["Behavior under test", "v0.5.0 default", "v0.6.0"],
    cellLoc="left", colLoc="left",
    colWidths=[.36, .31, .33],
    bbox=[.055, .26, .89, .52],
)
table.auto_set_font_size(False)
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#d7dfe9")
    cell.set_linewidth(.6)
    cell.PAD = .12
    cell.set_text_props(fontsize=11.7, color=ink)
    if row == 0:
        cell.set_facecolor(ink)
        cell.set_text_props(weight="bold", color="white", fontsize=12)
    else:
        cell.set_facecolor("#ffffff" if row % 2 else "#f0f4fa")
        if col == 2:
            cell.set_text_props(color=blue, weight="bold")
        if row == len(rows):
            cell.set_facecolor("#fff1db")
            cell.set_text_props(color="#794b12", fontsize=11.7)
ax.text(
    .055, .202,
    "The old release + a custom source validator also rejects all 7 corruptions.\n"
    "The new document API integrates that source contract.",
    fontsize=11.5, color=ink, linespacing=1.55,
)
ax.text(
    .055, .095,
    "Quote matching does not prove the claim.\n"
    "Fixed-response controls: corrected replies supplied, no live model calls.",
    fontsize=11.5, weight="bold", color=ink, linespacing=1.55,
)
output = Path(__file__).with_name("recursive-llm-control-evidence.png")
fig.savefig(output, dpi=200, facecolor=background)
plt.close(fig)
print(output)
