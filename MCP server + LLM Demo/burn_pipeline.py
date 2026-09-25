"""
burn_pipeline.py

Standalone, parameterized version of the burn-severity core pipeline from
fire_burn_severity.ipynb, refactored into one callable function so it can be
wrapped as an MCP tool. Same indices, thresholds, and decision rule as the
notebook (dNBR classes, dBAIS2 cross-check, class-based vectorization to avoid
the mean-dilution problem on large contiguous burn scars).

This deliberately duplicates logic that also lives in the notebook, rather
than having the notebook import from here — the notebook needs to stay
self-contained and narrated inline per the assignment's own instructions. In
a production setting you'd factor this out into one shared module instead.

Requires CDSE S3 credentials as environment variables:
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
(see the notebook's §1 runtime note for how to generate these)
"""

import os
from pathlib import Path
import dask
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray  # noqa: F401  registers the .rio accessor
import rasterio
from rasterio.features import shapes as rio_shapes
from shapely.geometry import box, shape

import pystac_client
import odc.stac
import os
os.environ["AWS_S3_ENDPOINT"] = "eodata.dataspace.copernicus.eu"
os.environ["AWS_VIRTUAL_HOSTING"] = "FALSE"
STAC_URL = "https://stac.dataspace.copernicus.eu/v1"
# Configure GDAL HTTP client to wait and retry on 429 errors
os.environ["GDAL_HTTP_MAX_RETRY"] = "10"
os.environ["GDAL_HTTP_RETRY_DELAY"] = "3"
DNBR_CLASS_BOUNDS = [0.10, 0.27, 0.44, 0.66]
DNBR_CLASS_LABELS = ["unburned", "low", "moderate-low", "moderate-high", "high"]

BANDS = ["B02_10m", "B03_10m", "B04_10m", "B06_20m", "B07_20m",
         "B08_10m", "B8A_20m", "B11_20m", "B12_20m", "SCL_20m"]
BAND_RENAME = {
    "B02_10m": "blue", "B03_10m": "green", "B04_10m": "red",
    "B06_20m": "rededge2", "B07_20m": "rededge3", "B08_10m": "nir",
    "B8A_20m": "nir_narrow", "B12_20m": "swir22", "B11_20m": "swir16",
    "SCL_20m": "scl",
}
SCL_KEEP = [2, 4, 5, 6, 7, 11]


def _classify_severity(mean_dnbr: float) -> str:
    for bound, label in zip(DNBR_CLASS_BOUNDS, DNBR_CLASS_LABELS):
        if mean_dnbr < bound:
            return label
    return DNBR_CLASS_LABELS[-1]


def _load_period(catalog, bbox, start, end, cloud_threshold, min_valid_fraction=0.90):
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": cloud_threshold}},
    )
    items = list(search.item_collection())
    if not items:
        return None, 0
    ds = odc.stac.load(
        items, bands=BANDS, bbox=bbox, crs="EPSG:32633", resolution=10,
        chunks={"x": 1024, "y": 1024}, groupby="solar_day",
    ).rename(BAND_RENAME)

    scl = ds["scl"]
    valid_mask = scl.isin(SCL_KEEP)
    refl_bands = ["blue", "green", "red", "rededge2", "rededge3", "nir", "nir_narrow", "swir22", "swir16"]
    ds_refl = (ds[refl_bands] / 10000.0).where(valid_mask)

    valid_frac = valid_mask.mean(dim=["x", "y"]).compute()
    keep_times = valid_frac.where(valid_frac >= min_valid_fraction, drop=True).time.values
    ds_refl = ds_refl.sel(time=keep_times)
    num_times = ds_refl.sizes.get("time", 0)

    if num_times == 0:
        return None, 0

    with dask.config.set(scheduler="synchronous"):
        return ds_refl.load(), num_times


def _calc_nbr(ds):
    return (ds["nir"] - ds["swir22"]) / (ds["nir"] + ds["swir22"])


def _calc_bais2(ds):
    return (1 - np.sqrt(ds["rededge2"] * ds["rededge3"] * ds["nir_narrow"] / ds["red"])) * (
        (ds["swir22"] - ds["nir_narrow"]) / np.sqrt(ds["swir22"] + ds["nir_narrow"]) + 1
    )


def analyze_burn_severity(
    bbox: tuple[float, float, float, float],
    pre_start: str,
    pre_end: str,
    post_start: str,
    post_end: str,
    priority_min_area_ha: float = 0.5,
    min_patch_area_m2: float = 2000,
    min_valid_fraction: float = 0.90,
    cloud_threshold: int = 10,    output_dir: str = "mcp_outputs",
) -> dict:
    """
    Run the full burn-severity pipeline for an arbitrary AOI and pre/post date range.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat), WGS84.
        pre_start, pre_end: date range (YYYY-MM-DD) to search for pre-fire imagery.
        post_start, post_end: date range (YYYY-MM-DD) to search for post-fire imagery.
        priority_min_area_ha: minimum patch size to flag as a priority intervention area.
        min_patch_area_m2: minimum patch size to keep at all (smaller = noise).
        cloud_threshold: max scene-level cloud cover (%) for the initial STAC search filter.
        output_dir: local directory for the GeoJSON output.

    Returns:
        dict summary: acquisition counts, total burned area, per-class breakdown,
        priority patch count/area, and the GeoJSON file path.
    """
    catalog = pystac_client.Client.open(STAC_URL)

    pre_ds, n_pre = _load_period(catalog, bbox, pre_start, pre_end, cloud_threshold)
    post_ds, n_post = _load_period(catalog, bbox, post_start, post_end, cloud_threshold)

    if pre_ds is None or post_ds is None:
        return {
            "error": "Insufficient usable imagery in one or both periods after cloud/quality filtering.",
            "n_acquisitions_pre": n_pre, "n_acquisitions_post": n_post,
        }

    pre_median = pre_ds.median(dim="time", skipna=True)
    post_median = post_ds.median(dim="time", skipna=True)

    nbr_pre, nbr_post = _calc_nbr(pre_median), _calc_nbr(post_median)
    dnbr = (nbr_pre - nbr_post).compute()

    bais2_pre, bais2_post = _calc_bais2(pre_median), _calc_bais2(post_median)
    dbais2 = (bais2_post - bais2_pre).compute()

    dnbr_array = dnbr.values
    valid = np.isfinite(dnbr_array)
    class_codes = np.zeros(dnbr_array.shape, dtype=np.uint8)
    for idx, bound in enumerate(DNBR_CLASS_BOUNDS):
        class_codes = np.where(valid & (dnbr_array >= bound), idx + 1, class_codes)
    severity_class = np.where(valid, class_codes, 255).astype(np.uint8)

    dbais2_threshold = float(np.nanpercentile(dbais2.values, 80))
    dbais2_agrees = dbais2.values >= dbais2_threshold

    severity_for_vec = np.where(dbais2_agrees, severity_class, 0)
    severity_for_vec = np.where(severity_class == 255, 255, severity_for_vec)
    class_mask = (severity_for_vec > 0) & (severity_for_vec < 255)

    transform = post_ds.rio.transform()
    polygons = [
        {"geometry": shape(geom), "class_code": int(value)}
        for geom, value in rio_shapes(severity_for_vec, mask=class_mask, transform=transform)
    ]

    records = []
    for i, item in enumerate(polygons):
        geom = item["geometry"]
        if geom.area < min_patch_area_m2:
            continue
        records.append({
            "patch_id": i,
            "area_ha": geom.area / 10_000,
            "severity_class": DNBR_CLASS_LABELS[item["class_code"]],
            "geometry": geom,
        })

    if not records:
        return {
            "n_acquisitions_pre": n_pre, "n_acquisitions_post": n_post,
            "total_burned_area_ha": 0.0, "patches_by_severity": {},
            "n_priority_patches": 0, "priority_area_ha": 0.0,
            "geojson_path": None,
        }

    burned_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=post_ds.rio.crs)
    burned_gdf["priority_flag"] = (
        burned_gdf["severity_class"].isin({"moderate-high", "high"}) &
        (burned_gdf["area_ha"] >= priority_min_area_ha)
    )

    Path(output_dir).mkdir(exist_ok=True)
    geojson_path = str(Path(output_dir) / "burned_patches.geojson")
    burned_gdf.to_crs("EPSG:4326").to_file(geojson_path, driver="GeoJSON")

    by_class = burned_gdf.groupby("severity_class")["area_ha"].agg(["count", "sum"]).round(2)
    patches_by_severity = {
        cls: {"count": int(row["count"]), "area_ha": float(row["sum"])}
        for cls, row in by_class.iterrows()
    }

    return {
        "n_acquisitions_pre": n_pre,
        "n_acquisitions_post": n_post,
        "total_burned_area_ha": round(float(burned_gdf["area_ha"].sum()), 2),
        "patches_by_severity": patches_by_severity,
        "n_priority_patches": int(burned_gdf["priority_flag"].sum()),
        "priority_area_ha": round(float(burned_gdf.loc[burned_gdf["priority_flag"], "area_ha"].sum()), 2),
        "geojson_path": geojson_path,
    }
