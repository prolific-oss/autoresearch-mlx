"""
Convert prolific_pair_grid.json → simple-annotation-app's scenarios.json format.

Pairwise scenarios use `response_a` + `response_b` fields (canonical A/B per
the grid file; the app randomizes display per participant).

Scenario IDs are opaque short hashes (not the model names from the grid file)
so participants can't see model identities in URLs. The mapping from
opaque_id → (prompt_id, a_model, b_model) is kept in the scenario record's
metadata fields for server-side analysis, AND saved separately as
scenario_id_map.json so the original grid can be reconstructed from results.

Usage:
    python build_app_scenarios.py [output_path]

Default output: ../../../prolific-oss/simple-annotation-app/data/scenarios.json
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
# autoresearch-mlx is at ~/Projects/experiments/research-agent-autoresearch/autoresearch-mlx
# simple-annotation-app is at ~/Projects/prolific-oss/simple-annotation-app
DEFAULT_OUT = HERE.parent.parent.parent.parent.parent / "prolific-oss/simple-annotation-app/data/scenarios.json"

grid_path = HERE / "prolific_pair_grid.json"
out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
# Decoder stays in autoresearch-mlx/eval_data/phase2/decoder/ (local-only).
# Never write it into the deployable app's data/ dir — it'd leak model identities.
map_path = HERE / "decoder" / "scenario_id_map.json"
map_path.parent.mkdir(parents=True, exist_ok=True)


def opaque_id(raw_id: str) -> str:
    """Stable 10-char hex hash → opaque scenario ID that doesn't leak model names."""
    h = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()
    return f"p2_{h[:10]}"


grid = json.load(open(grid_path))

scenarios = []
id_map = {}
for entry in grid:
    raw_id = entry["id"]
    opaque = opaque_id(raw_id)
    assert opaque not in id_map, f"opaque ID collision for {raw_id}"
    id_map[opaque] = {
        "raw_id": raw_id,
        "prompt_id": entry["prompt_id"],
        "axis": entry["axis"],
        "a_model": entry["a_model"],
        "b_model": entry["b_model"],
    }
    scenarios.append({
        "scenario_id": opaque,
        "prompt": entry["prompt"],
        "response_a": entry["a_text"],
        "response_b": entry["b_text"],
        # Metadata kept for analysis (the app records it back per submission).
        # These fields are NOT shown to the participant.
        "prompt_id": entry["prompt_id"],
        "axis": entry["axis"],
        "a_model": entry["a_model"],
        "b_model": entry["b_model"],
    })

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(scenarios, indent=2))
map_path.write_text(json.dumps(id_map, indent=2))
print(f"Wrote {len(scenarios)} pairwise scenarios → {out_path}")
print(f"  {len(set(s['prompt_id'] for s in scenarios))} unique prompts")
print(f"  {len(set(s['axis'] for s in scenarios))} unique model-pair axes")
print(f"  Opaque ID map → {map_path}")
