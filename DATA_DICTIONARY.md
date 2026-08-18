# Released analytical data dictionary

Only columns needed for optimization reproduction, audit, and basic spatial display are retained. Engineering fields not used by the released workflow were removed.

## Demand tables

Files:

- `data/processed/02_Demand_Points_Morning_Peak.csv`
- `data/processed/03_Demand_Points_Evening_Peak.csv`
- `data/processed/04_Demand_Points_Night.csv`

| Column | Meaning |
|---|---|
| `demand_id` | Stable demand-location identifier; mathematical index i |
| `zone_name` | Taxi Zone name for audit/display |
| `time_scena` | Source time-scenario label |
| `mta_norm` | Scenario-specific normalized subway activity component |
| `taxi_norm` | Scenario-specific normalized Green Taxi activity component |
| `mobility_i` | Fused mobility activity proxy q_it; normalized again by the loader to w_it |
| `x_32118`, `y_32118` | Demand-point coordinates in EPSG:32118 (metres) |

The time windows are Morning Peak (07:00–09:59), Evening Peak (16:00–18:59), and Night (22:00–23:59). Scenario-specific normalization represents relative spatial activity within each time scenario; it is not an estimate of population or absolute cross-time demand volume.

## Candidate shelter table

File: `data/processed/01_Shelter_Candidates_Hazard_438.csv`

| Column | Meaning |
|---|---|
| `candidate_id` | Stable candidate-site identifier; mathematical index j |
| `candidate_name` | Public facility name |
| `candidate_type` | School or official evacuation-center candidate |
| `candidate_source` | Source class used to construct the candidate set |
| `longitude`, `latitude` | Geographic coordinates |
| `safe_z12` | Moderate-scenario candidate-feasibility indicator a_jh |
| `safe_z123` | Strict-scenario candidate-feasibility indicator a_jh |
| `x_32118`, `y_32118` | Candidate coordinates in EPSG:32118 (metres) |

Baseline sets a_jh = 1 for every candidate. The moderate and strict fields restrict candidate feasibility only; they do not encode road failure probabilities or road closures.

## OD network-distance table

File: `data/processed/od/OD.csv`

| Column | Meaning |
|---|---|
| `demand_id` | Demand-location identifier i |
| `candidate_id` | Candidate-site identifier j |
| `network_distance_m` | Road-network distance d_ij in metres |
| `reachable` | Network-reachability indicator r_ij |

The table contains every 61 × 438 pair exactly once. Network distance is not travel time and contains no congestion or dynamic-disruption component.

## Result tables

`results/all_scenario_summaries.csv` contains one row per time × hazard × p scenario. `results/all_selected_shelters.csv` and `results/all_assignments.csv` keep selected sites and demand assignments separate from summary metrics. E1–E4 subdirectories contain the corresponding derived comparison tables.

