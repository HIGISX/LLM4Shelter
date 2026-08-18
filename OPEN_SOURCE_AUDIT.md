# Minimal open-source audit

## Included

- deterministic MHA-PM implementation and shared data-loading, metrics, QA, experiment, and export modules;
- five reduced processed input tables containing only analytical and display fields;
- frozen E1–E4 CSV outputs, all-scenario summaries, selected sites, assignments, and QA records;
- the frozen 60-request E5 benchmark;
- L1/L2 prompts, JSON schema, minimal local Ollama client, and saved original prediction table;
- frozen L2 inputs, L3-Final validator rules/code/hashes, gold-blind gate, gold-aware evaluator, and final E5 CSV outputs; and
- a release-integrity and numerical-consistency verification script.

## Excluded from the minimal package

- manuscript DOCX/PDF files, drafts, correspondence, and working notes;
- complete upstream MTA, TLC, NYC Open Data, LION, and GIS source downloads;
- map boundaries, shapefile sidecars, map exports, manuscript figures, TIFF/PNG files, and plotting intermediates;
- virtual environments, Python caches, temporary outputs, logs, and local absolute-path artifacts;
- the unrelated `text2location` project and other exploratory code;
- the preliminary L3 parsing/repair variant and its internal debugging artifacts;
- Ollama/Qwen model weights and Gurobi software/license files; and
- Excel workbooks that duplicate released CSV content.

## Automated checks completed

- no `__pycache__`, `.pyc`, model weights, office documents, GIS source bundles, or image exports included;
- no obvious API keys, passwords, bearer tokens, or user-home absolute paths detected;
- reduced data preserve 61 demand locations per time, 438 candidates, and 26,718 unique OD pairs;
- candidate feasibility remains nested at 438/394/379;
- 27 reference scenarios are optimal and cached objectives equal assignment recalculation within floating-point tolerance;
- a release-directory smoke solve reproduced Morning–baseline–p=10 with objective 1654.034470174065 and 10 selected shelters;
- a clean 27-scenario run from the release directory reproduced every frozen objective, selected shelter set, demand assignment, and assigned distance exactly;
- benchmark and frozen validator hashes are preserved; and
- E5-Final remains a gate applied to the same frozen L2 predictions rather than a second LLM parse.

## Author decisions required before publication

1. **Code license:** choose and add a license (for example MIT, BSD-3-Clause, or Apache-2.0). No license is selected in this draft package.
2. **Data terms:** confirm provider-specific redistribution and attribution requirements, especially for the MTA-derived aggregate, and insert any exact required NYC disclaimer.
3. **Citation metadata:** add final authors, paper title/venue, repository URL, release date, and DOI/archival identifier.
4. **Privacy/content review:** candidate names and addresses are public-facility information, but the author should approve their inclusion.
5. **Independent solver note:** HiGHS verification was not completed in the frozen experiment because a supported environment was unavailable; do not claim cross-solver verification.

Until items 1–3 are resolved, treat this directory as a release candidate rather than a legally complete open-source repository.
