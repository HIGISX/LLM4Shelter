# Data provenance and release notes

This package distributes reduced, study-specific analytical tables rather than the full upstream downloads. Users seeking the original geometry, full attributes, or later data versions should retrieve them from the official providers.

| Component | Study use | Official source |
|---|---|---|
| MTA Subway Hourly Ridership, 2020–2024 | Subway activity component; October 2024 weekdays | https://data.ny.gov/en/Transportation/MTA-Subway-Hourly-Ridership-2020-2024/wujg-7c2s/data |
| NYC TLC Green Taxi Trip Records | Pickup/drop-off activity component; October 2024 weekdays | https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page |
| NYC TLC Taxi Zones | Spatial aggregation units | https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page |
| NYC School Point Locations | School candidate sites | https://data.cityofnewyork.us/Education/School-Point-Locations/jfju-ynrr |
| Hurricane Evacuation Centers | Official-center candidate sites | https://data.cityofnewyork.us/Public-Safety/Hurricane-Evacuation-Centers-Map-/ayer-cga7 |
| Hurricane Evacuation Zones | Candidate-feasibility overlay | https://www.nyc.gov/content/planning/pages/resources/datasets/hurricane-evacuation-zones |
| NYC LION street centerline | Road graph and shortest-path distance | https://www.nyc.gov/content/planning/pages/resources/datasets/lion |

## Transformations represented in this package

- Subway and Green Taxi activity were aggregated to Brooklyn Taxi Zones and the three defined time scenarios.
- Each transport component underwent `log1p` transformation followed by scenario-specific min–max normalization.
- The two normalized components were combined with equal weights to form the mobility activity proxy.
- School and official-center records were integrated into 438 physical candidate sites.
- Hurricane Evacuation Zones were converted into nested moderate and strict candidate-feasibility indicators.
- Demand and candidate points were connected to a processed LION road graph, and the complete 61 × 438 shortest-path OD matrix was generated in EPSG:32118.

## Terms and attribution review

NYC and TLC pages describe their datasets as informational and provide no warranty of accuracy or completeness. The NYC Data Mine terms also specify disclaimer wording for applications that present modified NYC data. Before public release, copy the exact current wording from the official terms page and confirm that the repository presentation satisfies it:

https://www.nyc.gov/html/datamine/html/data/terms.html?dataSetJs=raw

The MTA dataset page currently lists its license as unspecified. Although this package contains only an aggregated and normalized research derivative, the authors should confirm the appropriate attribution and redistribution statement with the provider or institutional repository before publication.

This document is a provenance checklist, not legal advice. Data providers retain their own terms, and inclusion in a public portal does not transfer ownership to this project.

