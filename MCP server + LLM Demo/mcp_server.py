"""
mcp_server.py

Minimal MCP server exposing burn_pipeline.analyze_burn_severity as one tool,
using the official Python MCP SDK's FastMCP helper.

Run standalone for local testing:
    python mcp_server.py

Normally this is launched as a subprocess by an MCP client (see demo_client.py),
which talks to it over stdio.
"""

from mcp.server.mcpserver import MCPServer  # mcp SDK v2 (FastMCP was renamed to MCPServer)
from burn_pipeline import analyze_burn_severity
import logging

mcp = MCPServer("burn-severity-eo")


# Mute botocore log spam inside the MCP server subprocess
for logger_name in ["botocore", "botocore.credentials", "urllib3", "s3transfer"]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.ERROR)
    logger.disabled = True
    logger.propagate = False

@mcp.tool()
def analyze_burn_severity_tool(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    pre_start: str,
    pre_end: str,
    post_start: str,
    post_end: str,
    priority_min_area_ha: float = 0.5,
) -> dict:
    """
    Analyze wildfire burn severity for an area of interest using Sentinel-2 imagery.

    Computes dNBR and dBAIS2 burn-severity indices from pre- and post-fire Sentinel-2
    composites, classifies burned area into severity classes (low / moderate-low /
    moderate-high / high), and flags priority patches for erosion-control intervention
    based on severity and patch size. Works for any fire, not just a specific one --
    supply the AOI and date ranges for the event you want analyzed.

    Args:
        min_lon: Minimum longitude of the area of interest (WGS84).
        min_lat: Minimum latitude of the area of interest (WGS84).
        max_lon: Maximum longitude of the area of interest (WGS84).
        max_lat: Maximum latitude of the area of interest (WGS84).
        pre_start: Start of the pre-fire search window, YYYY-MM-DD.
        pre_end: End of the pre-fire search window, YYYY-MM-DD (should be before ignition).
        post_start: Start of the post-fire search window, YYYY-MM-DD (after containment,
            ideally with enough buffer for smoke to clear).
        post_end: End of the post-fire search window, YYYY-MM-DD.
        priority_min_area_ha: Minimum patch size (hectares) to flag as a priority
            intervention area, given at least moderate-high severity. Default 0.5.

    Returns:
        A dict with pre/post acquisition counts used, total burned area in hectares,
        a breakdown by severity class (count and area per class), the number and area
        of priority-flagged patches, and the local path to a GeoJSON file containing
        every burned patch geometry.
    """
    return analyze_burn_severity(
        bbox=(min_lon, min_lat, max_lon, max_lat),
        pre_start=pre_start,
        pre_end=pre_end,
        post_start=post_start,
        post_end=post_end,
        priority_min_area_ha=priority_min_area_ha,
    )


if __name__ == "__main__":
    mcp.run()
