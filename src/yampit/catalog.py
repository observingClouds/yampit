import numpy as np
import xarray as xr
import requests
from functools import lru_cache
import intake
from gribscan.gridutils import LambertLam
from typing import Dict
import datetime as dt
import isodate
import pandas as pd
import pyproj
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Parameter IDs commonly used for DMI catalog variables
DMI_VARIDS = [129, 130, 134, 151, 159, 165, 166, 167, 172, 3073, 3074, 3075,
              174096, 228023, 228024, 228141, 228164, 228235, 228236,
              231045, 231046, 231047, 231048, 231049, 231067, 231070,
              260109, 260242]


@lru_cache(maxsize=None)
def get_param_info(paramid):
    return requests.get(f"https://codes.ecmwf.int/parameter-database/api/v1/param/{paramid}/?format=json").json()

@lru_cache(maxsize=None)
def get_units():
    return {e["id"]: e["name"] for e in requests.get("https://codes.ecmwf.int/parameter-database/api/v1/unit/?format=json").json()}


def prefetch_param_info(param_ids):
    """Pre-fetch parameter info for multiple IDs in parallel."""
    def fetch_single(paramid):
        try:
            return get_param_info(paramid)
        except Exception as e:
            logger.warning(f"Failed to fetch param info for {paramid}: {e}")
            return None
    
    # Pre-fetch units
    get_units()
    
    # Fetch all params in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(fetch_single, param_ids))

def param_info_to_var_metadata(param_info):
    return {
        "id": param_info["id"],
        "attrs": {
            "long_name": param_info["name"],
            "units": get_units()[param_info["unit_id"]],
            # "coordinates": "lat lon",
        }
    }

def convert_gsv_cat_entry(entry):
    args = entry["args"]
    metadata = entry["metadata"]
    if "levelist" in args["request"]:
        leveldim = {"levelist": np.array(args["request"]["levelist"]) }
    else:
        leveldim = {}

    variables = {}
    for varid in metadata["variables"]:
        pinfo = get_param_info(varid)
        variables[pinfo["shortname"]] = {
            "dims": ("time", *leveldim, "cell"),
            **param_info_to_var_metadata(pinfo),
        }

    return {
        "base_request": {k: v for k, v in args["request"].items() if k not in ["levelist"]},
        "coords": {
            "time": xr.date_range(args["data_start_date"], args["data_end_date"], freq=args["savefreq"]),
            "cell": source_grids[metadata["source_grid_name"]],
            **leveldim,
        },
        "variables": variables,
        "internal_dims": ["cell"],
    }

def _read_destine_catalog(url, prefix=None):
    import fsspec
    import yaml
    import jinja2
    from urllib.parse import urljoin


    catalog_dir = urljoin(url, '.')
    prefix = prefix or []
    with fsspec.open(url) as f:
        cat = yaml.safe_load(f)

    for name, config in cat.get("sources", {}).items():
        match config.get("driver"):
            case "gsv":
                try:
                    yield ".".join(prefix + [name]), convert_gsv_cat_entry(config)
                except Exception as e:
                    print("WARNING: could not read " + ".".join(prefix + [name]))
                    print(e)
            case "yaml_file_cat":
                cat_url = jinja2.Template(config["args"]["path"]).render(CATALOG_DIR=catalog_dir)
                yield from _read_destine_catalog(cat_url, prefix + [name])

def read_destine_catalog():
    root = "https://raw.githubusercontent.com/DestinE-Climate-DT/Climate-DT-catalog/refs/heads/main/catalogs/climatedt-phase1/catalog.yaml"
    return {name: config
            for name, config in _read_destine_catalog(root)}


def _create_projection(config) -> Dict:
    # Calculate SW corner of projection
    corners = get_domain_properties(config)
    kwargs={
        'Ni': int(config['config.domain.nimax']),
        'Nj': int(config['config.domain.njmax']),
        'shapeOfTheEarth': 6,
        'edition': 2,
        'DxInMetres': int(config['config.domain.xdx']),
        'DyInMetres': int(config['config.domain.xdy']),
        'LaDInDegrees': float(config['config.domain.xlatcen']),
        'LoVInDegrees': float(config['config.domain.xloncen']),
        'Latin1InDegrees': float(config['config.domain.xlat0']),
        'Latin2InDegrees': float(config['config.domain.xlat0']),
        'longitudeOfFirstGridPointInDegrees': corners['minlon'],  # does not match GRIB message
        'latitudeOfFirstGridPointInDegrees': corners['minlat'],
        'iScansPositively': 1,
        'jScansPositively': 1,
        'radiusInMetres': None # not set for GRIB2
    }
    proj = LambertLam()
    p = proj.compute_coords(**kwargs)
    
    return p


def _build_coords_from_exp_config(config, use_proj=True, flatten=True) -> Dict:
    start = dt.datetime.strptime(config['config.general.times.start'], "%Y-%m-%dT%H:%M:%SZ")
    period = isodate.parse_duration(config['config.general.times.forecast_range'])
    end = start + period

# Fix deprecated pandas frequency notation: 'H' -> 'h', 'T' -> 'min', 'S' -> 's'
    freq_str = config['config.general.output_settings.fullpos'].replace('PT', '')
    freq_str = freq_str.replace('H', 'h').replace('T', 'min').replace('S', 's').replace('M', 'ME')

    if flatten:
        coords = {
            "time": pd.date_range(start, end, freq=freq_str),  # TODO check FDB output freq
            "cell": range(int(config['config.domain.nimax'])*int(config['config.domain.njmax'])),
        }
    else:
        coords = {
            "time": pd.date_range(start, end, freq=freq_str),  # TODO check FDB output freq
            "x": np.arange(int(config['config.domain.nimax'])),
            "y": np.arange(int(config['config.domain.njmax'])),
        }

    if use_proj:
        geocoords = _create_projection(config)
        if flatten:
            geocoords["lat"] = geocoords["lat"].flatten().astype('<f4')
            geocoords["lon"] = geocoords["lon"].flatten().astype('<f4')
            coords.update(geocoords)
        else:
            coords.update(_create_projection(config))


    return coords


def get_domain_properties(config: dict) -> dict:
    """Get domain properties.

    Args:
        domain_spec (dict): Domain specification

    Returns:
        dict: Domain properties
    
    NOTE: This function is mostly a copy from Deode Workflow but without rounding the outputs.
    """
    domain_spec = {
        "nlon": config["config.domain.nimax"],
        "nlat": config["config.domain.njmax"],
        "latc": config["config.domain.xlatcen"],
        "lonc": config["config.domain.xloncen"],
        "lat0": config["config.domain.xlat0"],
        "lon0": config["config.domain.xlon0"],
        "gsize": config["config.domain.xdx"],
    }

    # Validate projection parameters are not NaN or None
    required_params = ["latc", "lonc", "lat0", "lon0"]
    for param in required_params:
        value = domain_spec[param]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            raise ValueError(f"Invalid projection parameter '{param}': {value}")

    lonc = domain_spec["lonc"]
    latc = domain_spec["latc"]
    nlon = int(domain_spec["nlon"])
    nlat = int(domain_spec["nlat"])
    gsize = domain_spec["gsize"]

    proj_string = (
            f"+proj=lcc +lat_0={domain_spec['lat0']!s} +lon_0={domain_spec['lon0']!s} "
            f"+lat_1={domain_spec['lat0']!s} +lat_2={domain_spec['lat0']!s} "
            f"+units=m +no_defs +R={6371229!s}"
        )

    xloncen, xlatcen = pyproj.Transformer.from_crs(
        pyproj.CRS.from_string("EPSG:4326"), pyproj.CRS.from_string(proj_string), always_xy=True
    ).transform(lonc, latc)

    x_0 = float(xloncen) - (0.5 * ((float(nlon) - 1.0) * gsize))
    y_0 = float(xlatcen) - (0.5 * ((float(nlat) - 1.0) * gsize))

    xxx = np.empty([nlon])
    yyy = np.empty([nlat])
    for i in range(nlon):
        xxx[i] = x_0 + (float(i) * gsize)
    for j in range(nlat):
        yyy[j] = y_0 + (float(j) * gsize)

    x_v, y_v = np.meshgrid(xxx, yyy)
    lons, lats = pyproj.Transformer.from_crs(
        pyproj.CRS.from_string(proj_string), pyproj.CRS.from_string("EPSG:4326"), always_xy=True
    ).transform(x_v, y_v)

    minlat = np.min(lats)
    minlon = np.min(lons)
    maxlat = np.max(lats)
    maxlon = np.max(lons)

    minlat = np.max([minlat, -90])
    minlon = np.max([minlon, -180])
    maxlat = np.min([maxlat, 90])
    maxlon = np.min([maxlon, 180])

    domain_properties = {
        "minlat": minlat,
        "minlon": minlon,
        "maxlat": maxlat,
        "maxlon": maxlon,
    }
    return domain_properties


def _decode_dmi_catalog_entry(cat_entry, flatten=True):
    base_request = cat_entry["fdb"]["fdb_request"]
    base_request['levtype'] = 'sfc'
    coords = _build_coords_from_exp_config(cat_entry, use_proj=True, flatten=flatten)
    # Use module-level DMI_VARIDS (single source of truth)
    
    # Determine polytope configuration based on stores
    if cat_entry.get('fdb', {}).get('data_briges', {}).get('lumi', False):
        polytope_config = {
            'host': 'polytope.lumi.apps.dte.destination-earth.eu',
            'collection': 'destination-earth'
        }
    else:
        polytope_config = {
            'host': 'polytope.ecmwf.int',
            'collection': 'deode'
        }
    
    if flatten:
        # build variables using cached param lookups (avoid duplicate get_param_info calls)
        variables = {}
        for varid in DMI_VARIDS:
            pinfo = get_param_info(varid)
            variables[pinfo["shortname"]] = {
                "dims": ("time", "cell"),
                **param_info_to_var_metadata(pinfo),
            }
        if "lat" in coords:
            variables["lat"] = {"dims": ("cell",), "attrs": {"long_name": "latitude", "units": "degrees_north", "standard_name": "latitude"}}
        if "lon" in coords:
            variables["lon"] = {"dims": ("cell",), "attrs": {"long_name": "longitude", "units": "degrees_east", "standard_name": "longitude"}}
        internal_dims = ["cell", "lat", "lon"]
    else:
        # reuse the same varid list and cached lookups
        variables = {}
        for varid in DMI_VARIDS:
            pinfo = get_param_info(varid)
            variables[pinfo["shortname"]] = {
                "dims": ("time", "x", "y"),
                **param_info_to_var_metadata(pinfo),
            }
        if "lat" in coords:
            variables["lat"] = {"dims": ("y", "x"), "attrs": {"long_name": "latitude", "units": "degrees_north", "standard_name": "latitude", "axis": "Y"}}
        if "lon" in coords:
            variables["lon"] = {"dims": ("y", "x"), "attrs": {"long_name": "longitude", "units": "degrees_east", "standard_name": "longitude", "axis": "X"}}
        internal_dims = ["x", "y", "lat", "lon"]

    result = {
        "base_request": base_request,
        "coords": coords,
        "variables": variables,
        "internal_dims": internal_dims,
        "polytope_config": polytope_config,
    }

    return result


def _decode_dmi_catalog_entry_both(cat_entry):
    """Compute and return (nonflat_result, flat_result) in a single pass.

    Shares projection and parameter-info lookups to avoid duplicate work when
    both flattened and non-flattened representations are needed.
    """
    base_request = cat_entry["fdb"]["fdb_request"]
    base_request['levtype'] = 'sfc'

    # Compute non-flattened coords once (contains 2D lat/lon if georef present)
    coords_nonflat = _build_coords_from_exp_config(cat_entry, use_proj=True, flatten=False)

    # Derive flattened coords from non-flat coords (avoid recomputing projection)
    nx = len(coords_nonflat.get("x", []))
    ny = len(coords_nonflat.get("y", []))
    cell_count = nx * ny if nx and ny else int(cat_entry["config"]["domain"]["nimax"]) * int(cat_entry["config"]["domain"]["njmax"])
    coords_flat = {
        "time": coords_nonflat["time"],
        "cell": range(cell_count),
    }
    if "lat" in coords_nonflat:
        coords_flat["lat"] = coords_nonflat["lat"].flatten().astype('<f4')
        coords_flat["lon"] = coords_nonflat["lon"].flatten().astype('<f4')

    # polytope config (same logic as single-decoder)
    if cat_entry.get('fdb', {}).get('data_briges', {}).get('lumi', False):
        polytope_config = {
            'host': 'polytope.lumi.apps.dte.destination-earth.eu',
            'collection': 'destination-earth'
        }
    else:
        polytope_config = {
            'host': 'polytope.ecmwf.int',
            'collection': 'deode'
        }

    # Build variable metadata for both flattened and non-flattened using a single
    # loop over DMI_VARIDS (reuses cached get_param_info calls).
    variables_flat = {}
    variables_nonflat = {}
    for varid in DMI_VARIDS:
        pinfo = get_param_info(varid)
        variables_flat[pinfo["shortname"]] = {
            "dims": ("time", "cell"),
            **param_info_to_var_metadata(pinfo),
        }
        variables_nonflat[pinfo["shortname"]] = {
            "dims": ("time", "x", "y"),
            **param_info_to_var_metadata(pinfo),
        }

    if "lat" in coords_flat:
        variables_flat["lat"] = {"dims": ("cell",), "attrs": {"long_name": "latitude", "units": "degrees_north", "standard_name": "latitude"}}
    if "lon" in coords_flat:
        variables_flat["lon"] = {"dims": ("cell",), "attrs": {"long_name": "longitude", "units": "degrees_east", "standard_name": "longitude"}}

    if "lat" in coords_nonflat:
        variables_nonflat["lat"] = {"dims": ("y", "x"), "attrs": {"long_name": "latitude", "units": "degrees_north", "standard_name": "latitude", "axis": "Y"}}
    if "lon" in coords_nonflat:
        variables_nonflat["lon"] = {"dims": ("y", "x"), "attrs": {"long_name": "longitude", "units": "degrees_east", "standard_name": "longitude", "axis": "X"}}

    nonflat_result = {
        "base_request": base_request,
        "coords": coords_nonflat,
        "variables": variables_nonflat,
        "internal_dims": ["x", "y", "lat", "lon"],
        "polytope_config": polytope_config,
    }

    flat_result = {
        "base_request": base_request,
        "coords": coords_flat,
        "variables": variables_flat,
        "internal_dims": ["cell", "lat", "lon"],
        "polytope_config": polytope_config,
    }

    return nonflat_result, flat_result


def read_dmi_catalog(flatten=True):
    intake_esm_url = "https://object-store.os-api.cci1.ecmwf.int/deode-dcmdb/catalog/deode_intake_esm_catalog-rc.json"
    cat = intake.open_esm_datastore(
            intake_esm_url,
            columns_with_iterables=["fdb", "variables", "stores"],
            sep="/",
        )

    ds_collection = {}
    for name, exp in tqdm(cat.items(), desc=f"Loading DMI catalog (flatten={flatten})"):
        if exp.df.iloc[0]["fdb"] is not {} and "fdb_request" in exp.df.iloc[0]["fdb"] and "georef" in exp.df.iloc[0]["fdb"]["fdb_request"]:
            ds_name = name
            try:
                ds_collection[ds_name] = _decode_dmi_catalog_entry(exp.df.iloc[0], flatten=flatten)
            except Exception as e:
                logger.warning(f"Skipping catalog entry '{ds_name}' due to error: {type(e).__name__}: {e}")
                continue
    return ds_collection


# if __name__ == "__main__":
#     print(read_dmi_catalog())


def init_catalog():
    """Initialize catalog by fetching remote data once and processing both flatten modes."""
    import time
    from tqdm import tqdm
    start_time = time.time()
    
    print("[YAMPIT] Starting catalog initialization...", flush=True)
    logger.info("Starting catalog initialization...")
    
    # Pre-fetch all parameter info in parallel to speed up processing
    param_ids = DMI_VARIDS
    print(f"[YAMPIT] Pre-fetching parameter info for {len(param_ids)} parameters...", flush=True)
    prefetch_start = time.time()
    prefetch_param_info(param_ids)
    print(f"[YAMPIT] Parameter prefetch completed in {time.time()-prefetch_start:.2f}s", flush=True)
    
    # Fetch catalog once
    intake_esm_url = "https://object-store.os-api.cci1.ecmwf.int/deode-dcmdb/catalog/deode_intake_esm_catalog-rc.json"
    print(f"[YAMPIT] Fetching catalog from remote...", flush=True)
    cat_start = time.time()
    cat = intake.open_esm_datastore(
            intake_esm_url,
            columns_with_iterables=["fdb", "variables", "stores"],
            sep="/",
        )
    print(f"[YAMPIT] Catalog fetch completed in {time.time()-cat_start:.2f}s", flush=True)
    
    # Process both flatten modes in one pass
    datasets = {}
    flatdatasets = {}
    
    valid_entries = [(f"{r[1]['case']}/{r[1]['experiment']}", r) for r in cat.df.iterrows()
                     if r[1]["fdb"] is not {} 
                     and "fdb_request" in r[1]["fdb"] 
                     and "georef" in r[1]["fdb"]["fdb_request"]
                     ]
    
    print(f"[YAMPIT] Processing {len(valid_entries)} catalog entries...", flush=True)
    process_start = time.time()
    
    # Parallelize per-entry decoding using threads (shares cached param lookups)
    import os
    from concurrent.futures import as_completed

    max_workers = min(8, (os.cpu_count() or 1) * 2)
    print(f"[YAMPIT] Processing entries with max_workers={max_workers}...", flush=True)

    def _process_single_entry(name, row):
        entry_data = row[1]
        try:
            ds, flatds = _decode_dmi_catalog_entry_both(entry_data)
        except Exception as e:
            logger.warning(f"Skipping catalog entry '{name}' due to error during combined decoding: {type(e).__name__}: {e}")
            return name, None, None
        return name, ds, flatds

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_process_single_entry, name, exp): name for name, exp in valid_entries}
        for i, fut in enumerate(tqdm(as_completed(futures), total=len(futures), desc=f"Processing DMI catalog entries"), 1):
            name = futures[fut]
            try:
                n, ds, flatds = fut.result()
                if ds is not None:
                    datasets[n] = ds
                if flatds is not None:
                    flatdatasets[n] = flatds
            except Exception as e:
                logger.warning(f"Skipping catalog entry '{name}' due to error during processing: {type(e).__name__}: {e}")
            if i % 10 == 0:
                print(f"[YAMPIT] Processed {i}/{len(valid_entries)} entries...", flush=True)
    
    print(f"[YAMPIT] Entry processing completed in {time.time()-process_start:.2f}s", flush=True)
    print(f"[YAMPIT] Catalog initialization complete. Loaded {len(datasets)} datasets, {len(flatdatasets)} flat datasets", flush=True)
    print(f"[YAMPIT] Total initialization time: {time.time()-start_time:.2f}s", flush=True)
    logger.info(f"Catalog initialization complete. Loaded {len(datasets)} datasets, {len(flatdatasets)} flat datasets in {time.time()-start_time:.2f}s")
    return datasets, flatdatasets
