# OhNoFire — Terzigno Wildfire Burn Severity Analysis

Earth Observation AI Engineer take-home assignment. Burn-severity mapping and decision-intelligence
analysis for the August 2025 wildfire in Vesuvius National Park (Terzigno, Mount Somma slope),
using Sentinel-2 imagery, plus two bonus components: an MCP-wrapped agentic tool and a geospatial
foundation model comparison.

## Repository structure

```
.
├── fire_burn_severity_with_Prithvi.ipynb   # Core notebook: required deliverables + GFM bonus
├── MCP server + LLM Demo/                  # Bonus: MCP server + LLM tool-calling demo
│   ├── burn_pipeline.py                    # Core pipeline, refactored as one callable function
│   ├── mcp_server.py                       # MCP server exposing the pipeline as a tool
│   ├── demo_client.py                      # Client wiring an LLM to the MCP server
│   └── README.md                           # Setup/run instructions for this component
└── README.md                               # This file
```

## What's in the notebook

`fire_burn_severity_with_Prithvi.ipynb` is self-contained and narrated inline (markdown cells explain
each method choice as it's made). It covers, in order:

1. **Approach, backend, and assumptions** — why Copernicus Data Space Ecosystem (CDSE) via STAC
   rather than Google Earth Engine, AOI/date-window choices, and what's assumed vs. verified.
2. **Core pipeline** — Sentinel-2 L2A ingestion, SCL-based cloud/shadow masking, NDVI/NBR/BAIS2/NDMI
   computation, time-series plot, pre/post composites, dNBR + dBAIS2 cross-check severity
   classification, class-based vectorization to GeoJSON, GeoTIFF/COG exports, and an interactive map.
3. **Trade-offs, limitations, and next steps.**
4. **Decision intelligence** — a concrete stakeholder question (erosion-control prioritization for
   park/civil-protection authorities), the rule mapping severity classes to a priority flag, a ranked
   table, an alert GeoJSON, and a plain-language summary.
5. **Bonus: geospatial foundation model** — Prithvi-EO-2.0 (via TerraTorch) used as a frozen feature
   extractor to compute an embedding-change signal between pre/post composites, compared against the
   band-math dNBR output.

## Running the notebook

Requires a free Copernicus Data Space Ecosystem account and S3 keys (generated separately from your
account login, at the S3 Keys Manager) — the notebook prompts for these at runtime rather than storing
them in the file, so it's safe to run from this public repo.

```bash
pip install pystac-client odc-stac rioxarray rasterio xarray geopandas shapely \
            matplotlib pandas numpy hvplot holoviews geoviews cartopy \
            ipywidgets panel bokeh jupyter_bokeh dask contextily terratorch
```

Run top to bottom. CDSE's public S3 gateway rate-limits under concurrent load, so a full run can take
several minutes; retry/backoff handling is already built into the ingestion cells.

## Bonus: MCP server + LLM

See `MCP server + LLM Demo/README.md` for setup and run instructions. In short: the core pipeline is
wrapped as a single MCP tool (arbitrary AOI + date range, not hardcoded to this fire), and
`demo_client.py` shows an LLM choosing the tool and arguments from a plain-English question, then
answering grounded in the pipeline's actual output. There is also a console report as a PDF from a sample run.

## Scope notes

Both bonus goals were attempted only after the required core was complete and correct, per the
assignment's own guidance. The MCP tool is deliberately scoped to burn-severity analysis specifically
(parameterized across AOI/dates within that domain), not a general-purpose multi-domain EO tool, a
time-budget-driven decision, stated explicitly rather than left implicit.

