from sanic import Sanic, exceptions
from sanic.response import raw, json
from sanic.worker.manager import WorkerManager

import eccodes

from .catalog import read_destine_catalog, read_dmi_catalog, init_catalog
from .mapper import MarsDataset
from .async_polytope_request_handler import AsyncPolytopeRequestHandler
from .exceptions import NoSuchData

app = Sanic("YAMPIT_Server")

WorkerManager.THRESHOLD = 10000

@app.before_server_start
async def setup_catalog(app, loop):
    """Initialize catalog once when server starts (runs in each worker)."""
    # Optimized: fetch catalog only once and process both flatten modes
    datasets, flatdatasets = init_catalog()
    app.ctx.datasets = {k: MarsDataset(**v) for k, v in datasets.items()}
    app.ctx.flatdatasets = {k: MarsDataset(**v) for k, v in flatdatasets.items()}
    app.ctx.request_handler = AsyncPolytopeRequestHandler("polytope.lumi.apps.dte.destination-earth.eu", "destination-earth")

def is_meta(key):
    return key.split("/")[-1].startswith(".z")

@app.get("/ds")
async def list_datasets(request):
    return json(list(sorted(app.ctx.flatdatasets)))

@app.get("/api/v1/reload")
async def reload_catalog(request):
    datasets, flatdatasets = init_catalog()
    app.ctx.datasets = {k: MarsDataset(**v) for k, v in datasets.items()}
    app.ctx.flatdatasets = {k: MarsDataset(**v) for k, v in flatdatasets.items()}
    return json({"status": "reloaded"})

@app.get("/flatds/<dsid1>/<dsid2>/<key:path>")
async def get_flattened_chunk(request, dsid1, dsid2, key):
    dsid = f"{dsid1}/{dsid2}"
    kind, request = app.ctx.flatdatasets[dsid].key2request(key)

    if is_meta(key):
        content_type="application/json"
    else:
        content_type = "application/octet-stream"

    headers = {
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": "public, max-age=604800",
    }

    if kind == 'inline':
        return raw(request, content_type=content_type, headers=headers)
    elif kind == 'request':
        try:
            data = await app.ctx.request_handler.get(request)
        except NoSuchData:
            raise exceptions.NotFound("Could not find data for MARS request " + str(request))

        mid = eccodes.codes_new_from_message(data)
        data = eccodes.codes_get_array(mid, "values")
        eccodes.codes_release(mid)
        return raw(bytes(data.astype("<f4")), content_type=content_type, headers=headers)
    else:
        raise NotImplementedError(f"kind {kind}")

@app.get("/ds/<dsid1>/<dsid2>/<key:path>")
async def get_chunk(request, dsid1, dsid2, key):
    dsid = f"{dsid1}/{dsid2}"
    kind, request = app.ctx.datasets[dsid].key2request(key)

    if is_meta(key):
        content_type="application/json"
    else:
        content_type = "application/octet-stream"

    headers = {
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": "public, max-age=604800",
    }

    if kind == 'inline':
        return raw(request, content_type=content_type, headers=headers)
    elif kind == 'request':
        try:
            data = await app.ctx.request_handler.get(request)
        except NoSuchData:
            raise exceptions.NotFound("Could not find data for MARS request " + str(request))

        mid = eccodes.codes_new_from_message(data)
        data = eccodes.codes_get_array(mid, "values")
        eccodes.codes_release(mid)
        return raw(bytes(data.astype("<f4")), content_type=content_type, headers=headers)
    else:
        raise NotImplementedError(f"kind {kind}")
