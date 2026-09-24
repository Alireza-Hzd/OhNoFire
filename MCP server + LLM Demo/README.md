# Bonus A: MCP Server + LLM Integration

Wraps the burn-severity pipeline from the main notebook as a single MCP tool, and demonstrates an LLM (configured via OpenRouter / OpenAI-compatible API) calling it with model-chosen arguments to answer a plain-English question, grounded in the pipeline's actual output.

## Scope

The tool is parameterized (arbitrary AOI bbox, arbitrary pre/post date ranges) so it isn't hardcoded to the Terzigno fire—but it stays scoped to *burn-severity analysis* specifically (same dNBR/dBAIS2 pipeline as the notebook), rather than a general-purpose "analyze any EO phenomenon" tool. This keeps the design lean and focused.

## Files

- `burn_pipeline.py` -- The core analysis as one callable function. Uses a synchronous Dask scheduler to prevent CDSE API rate-limiting during tile loading.
- `mcp_server.py` -- FastMCP/MCPServer exposing the pipeline function as an MCP tool. Includes log-suppression filters for subprocess execution.
- `demo_client.py` -- Sets up `PROJ_LIB` and GDAL retry settings, spawns the MCP server via stdio, exposes tool schemas to an OpenRouter model via the `openai` SDK, and streams the grounded English result.

## Setup

```bash
pip install "mcp>=2,<3" openai pystac-client odc-stac rioxarray rasterio xarray geopandas shapely pandas numpy boto3 pyproj dask
```


## Environment variables required:

```bash
export OPENROUTER_API_KEY=sk-or-v1-...  # OpenRouter API key
export AWS_ACCESS_KEY_ID=...            # CDSE S3 access key
export AWS_SECRET_ACCESS_KEY=...        # CDSE S3 secret key
```



## To Run
```bash
cd path_to_folder
python demo_client.py
```

This script will:

Automatically configure local GDAL/PROJ environment paths and retry limits.

Connect to mcp_server.py over stdio and fetch registered tool schemas.

Send the prompt to OpenRouter (openrouter/free or specified model) with explicit instructions to respond in English.

Execute the burn pipeline tool against CDSE when requested by the model.

## A PDF of the terminal log showing the output has also been provided.
