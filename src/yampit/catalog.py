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

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def get_param_info(paramid):
    return requests.get(f"https://codes.ecmwf.int/parameter-database/api/v1/param/{paramid}/?format=json").json()

@lru_cache(maxsize=None)
def get_units():
    return {e["id"]: e["name"] for e in requests.get("https://codes.ecmwf.int/parameter-database/api/v1/unit/?format=json").json()}

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
        leveldim = {"levelist": np.array(args["request"]["levelist"])}
    else:
        leveldim = {}

    return {
        "base_request": {k: v for k, v in args["request"].items() if k not in ["levelist"]},
        "coords": {
            "time": xr.date_range(args["data_start_date"], args["data_end_date"], freq=args["savefreq"]),
            "cell": source_grids[metadata["source_grid_name"]],
            **leveldim,
        },
        "variables": {
            get_param_info(varid)["shortname"]: {
                "dims": ("time", *leveldim, "cell"),
                **param_info_to_var_metadata(get_param_info(varid)),
            }
            for varid in metadata["variables"]
        },
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
    freq_str = freq_str.replace('H', 'h').replace('T', 'min').replace('S', 's')

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

    lonc = domain_spec["lonc"]
    latc = domain_spec["latc"]
    nlon = domain_spec["nlon"]
    nlat = domain_spec["nlat"]
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
    if flatten:
        variables = {
            get_param_info(varid)["shortname"]: {
                "dims": ("time", "cell"),
                **param_info_to_var_metadata(get_param_info(varid)),
            }
            for varid in [129, 130, 134, 151, 159, 165, 166, 167, 172, 3073, 3074, 3075, 174096, 228023, 228024, 228141, 228164, 228235, 228236, 231045, 231046, 231047, 231048, 231049, 231067, 231070, 260109, 260242]
        }
        if "lat" in coords:
            variables["lat"] = {"dims": ("cell",), "attrs": {"long_name": "latitude", "units": "degrees_north", "standard_name": "latitude"}}
        if "lon" in coords:
            variables["lon"] = {"dims": ("cell",), "attrs": {"long_name": "longitude", "units": "degrees_east", "standard_name": "longitude"}}
        internal_dims = ["cell", "lat", "lon"]
    else:
        variables = {
            get_param_info(varid)["shortname"]: {
            "dims": ("time", "x", "y"),
            **param_info_to_var_metadata(get_param_info(varid)),
            }
            for varid in [129, 130, 134, 151, 159, 165, 166, 167, 172, 3073, 3074, 3075, 174096, 228023, 228024, 228141, 228164, 228235, 228236, 231045, 231046, 231047, 231048, 231049, 231067, 231070, 260109, 260242]
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
    }

    return result


def read_dmi_catalog(flatten=True):
    intake_esm_url = "https://object-store.os-api.cci1.ecmwf.int/deode-dcmdb/catalog/catalog-fdb.json"
    cat = intake.open_esm_datastore(
            intake_esm_url,
            columns_with_iterables=["fdb", "variables", "stores"],
            sep="/",
        )

    ds_collection = {}
    for name, exp in cat.items():
        if exp.df.iloc[0]["fdb"] is not {} and "fdb_request" in exp.df.iloc[0]["fdb"] and "georef" in exp.df.iloc[0]["fdb"]["fdb_request"]:
            ds_name = name
            try:
                ds_collection[ds_name] = _decode_dmi_catalog_entry(exp.df.iloc[0], flatten=flatten)
            except Exception as e:
                logger.warning(f"Skipping catalog entry '{ds_name}' due to error: {type(e).__name__}: {e}")
                continue
    return ds_collection


if __name__ == "__main__":
    print(read_dmi_catalog())
