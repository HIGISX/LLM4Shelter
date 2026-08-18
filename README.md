# LLMforShelter — minimal reproducibility package



## Research scope

LLMforShelter connects a natural-language planning interface to a deterministic facility-location model:

```text
Natural-language planning request
    -> schema-constrained parsing
    -> deterministic validation gate
    -> validated scenario parameters
    -> MHA-PM / Gurobi
    -> deterministic shelter configuration
```

The LLM parses planning intent; it does not select shelter locations. MHA-PM is a weighted P-Median formulation with:

- time-specific normalized mobility activity weights;
- road-network distance as accessibility cost;
- hazard scenarios represented as candidate-feasibility restrictions; and
- deterministic Gurobi settings.

The mobility variable is a **mobility activity proxy**, not actual population or observed evacuation demand. Hazard restrictions do not model road damage, and model assignments are normative rather than observed evacuee behavior.

## Included study data

The package contains only the processed columns needed for numerical reproduction:

| Object | Size | Role |
|---|---:|---|
| Demand locations | 61 × 3 time scenarios | Mobility activity proxy and within-scenario demand weights |
| Candidate shelters | 438 | Candidate identity and baseline/moderate/strict feasibility |
| OD pairs | 26,718 | Road-network distance and reachability |
| Optimization scenarios | 27 | 3 times × 3 hazards × p in {5, 10, 15} |
| LLM benchmark | 60 requests | 20 simple, 20 composite, 20 ambiguous/invalid |

All 26,718 OD pairs are reachable in the released analytical table. Candidate counts are 438 (baseline), 394 (moderate), and 379 (strict).

## Directory layout

```text
code/llm_for_shelter/          MHA-PM and minimal LLM-interface code
data/processed/                reduced analytical input tables
results/                       frozen E1–E4 outputs and verified result cache
results/llm_experiment_E5/     benchmark and original L1/L2 prediction table
llm_experiment_final/          frozen L3-Final validator, gate, evaluator, outputs
scripts/verify_release.py      data, result, benchmark, and hash checks
DATA_DICTIONARY.md             released-column definitions
DATA_SOURCES.md                source provenance and attribution notes
OPEN_SOURCE_AUDIT.md           inclusion/exclusion and release checklist
```

## Environment

Tested on Python 3.12 with Gurobi 13.0.1. A working Gurobi installation and license are required only to solve MHA-PM instances again. Cached results and release verification do not require a new optimization run.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Gurobi is proprietary solver software and is not distributed in this package. Academic users may obtain a license separately from Gurobi.

## Verify the release without solving

From the package root:

```powershell
python scripts/verify_release.py
```

The check verifies file hashes, the 61/438/26,718 data dimensions, hazard-feasibility nesting, OD completeness, 27 optimal result records, assignment/objective consistency, the 60-request benchmark composition, frozen L2 input integrity, and validator-rule hashes.

## Re-run MHA-PM experiments (E1–E4)

PowerShell:

```powershell
$env:PYTHONPATH = "code"
python -m llm_for_shelter.main --workspace-root . --results-dir reproduced_results
```

Linux/macOS:

```bash
PYTHONPATH=code python -m llm_for_shelter.main --workspace-root . --results-dir reproduced_results
```

The command discovers the five processed input tables, performs data QA, solves all 27 deterministic scenarios, and exports E1–E4 tables. The original reference outputs remain under `results/`.

## Reproduce E5-Final from frozen L2 predictions

E5-Final deliberately reuses the same frozen L2 structured predictions and does not call the LLM again:

```powershell
python llm_experiment_final/validation/run_gate.py --workspace-root .
python llm_experiment_final/evaluation/evaluate_final.py --workspace-root .
```

The first command runs the gold-blind deterministic gate. The second command loads gold labels only after gate execution, uses the verified 27-scenario optimization cache, and regenerates the final E5 CSV outputs. The frozen validator verifies its own code and rule hashes before execution.

The local inference configuration used `qwen3.5:9b` through Ollama with temperature 0, top-p 1, and seed 42. Model weights are not included. The package includes the prompts, schema, inference client, benchmark, and saved predictions needed to audit the interface experiment; no paid API is required.

## Frozen model and experiment boundaries

This release does not add capacity, population, vulnerability, road failure, congestion, evacuation behavior, GA, RL, HRL, or alternative objectives. The implemented objective remains `weighted_p_median`. E1–E5 settings and reported results are frozen.

## Data provenance, terms, and citation

The reduced analytical tables were derived from public transportation, facility, hazard-zone, and street-network sources. Source links, temporal coverage, transformation notes, and redistribution cautions are in `DATA_SOURCES.md`. Users should cite the original providers in addition to the paper and this repository.

Before publishing this directory, the authors still need to:

1. select a code license;
2. confirm redistribution/attribution wording for every source, especially the MTA-derived aggregate;
3. add final author, paper, repository, and DOI metadata; and
4. archive a versioned release rather than uploading the full working directory.

## 中文快速说明

本目录只保留复现实验所需的核心代码、精简处理数据、27 组确定性优化结果缓存、60 条 E5 benchmark、冻结的 L2 prediction 与 L3-Final validator。运行 `scripts/verify_release.py` 可先检查文件完整性和主要 QA；有 Gurobi 许可证时可重新运行 E1–E4。正式上传 GitHub/Zenodo 前仍需由作者确定代码许可证，并复核各公共数据源的转载与署名要求。

