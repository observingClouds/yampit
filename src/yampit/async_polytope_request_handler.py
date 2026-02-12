import json
import logging
import os
from urllib.parse import urljoin
import aiohttp
import asyncio
from .exceptions import NoSuchData


logger = logging.getLogger(__name__)


async def get_client(**kwargs):
    return aiohttp.ClientSession(**kwargs)


class AsyncPolytopeRequestHandler:
    def __init__(self, server, collection):
        from polytope.api import Client
        self.client = Client(address=server)

        self.server = server
        self.collection = collection
        self.get_client = get_client
        self.max_poll_retries = 100
        self._session = None

        # Select appropriate API key based on server
        if "polytope-test.ecmwf.int" in server:
            api_key = f"Bearer {os.environ.get("POLYTOPE_USER_KEY_ATOS")}"
            if not api_key:
                logger.warning("POLYTOPE_USER_KEY_ATOS not found, falling back to polytope client auth")
                api_key = ", ".join(self.client.auth.get_auth_headers())
        elif "polytope.lumi.apps.dte.destination-earth.eu" in server:
            api_key = f"Bearer {os.environ.get("POLYTOPE_USER_KEY_LUMI")}"
            if not api_key:
                logger.warning("POLYTOPE_USER_KEY_LUMI not found, falling back to polytope client auth")
                api_key = ", ".join(self.client.auth.get_auth_headers())
        else:
            # Unknown server, use polytope client auth
            api_key = f"Bearer {self.client.auth.get_auth_headers()}"
        
        self.auth_headers = {"Authorization": api_key}


    async def set_session(self):
        if self._session is None:
            self._session = await self.get_client()
        return self._session

    def __del__(self):
        if self._session:
            self._session.close()

    async def _poll_get(self, url):
        session = await self.set_session()

        for i in range(self.max_poll_retries):
            async with session.get(url, headers=self.auth_headers) as r:
                match r.status:
                    case 200:  # OK, direct download
                        return await r.read()
                    case 202:  # Accepted, scheduled. Needs polling
                        url = urljoin(url, r.headers.get("Location"))
                        wait = int(r.headers.get("Retry-After", 0))
                        await asyncio.sleep(wait)
                        continue
                    case 303:  # redirect to direct download
                        raise NotImplementedError("direct download")
                    case 400:
                        raise NoSuchData("request can't be fullfilled")
                r.raise_for_status()
        else:
            raise RuntimeError("max poll retries exceeded")


    async def get(self, request):
        request_object = {"verb": "retrieve", "request": json.dumps(request)}
        url = self.client.config.get_url("requests", collection_id=self.collection)
        poll_url = None

        session = await self.set_session()
        async with session.post(url, headers=self.auth_headers, json=request_object) as r:
            match r.status:
                case 200:  # OK, direct download
                    res = await r.read()
                case 202:  # Accepted, scheduled. Needs polling
                    poll_url = urljoin(url, r.headers.get("Location"))
                    wait = int(r.headers.get("Retry-After", 0))
                    await asyncio.sleep(wait)
                    res = await self._poll_get(poll_url)
                case 303:  # redirect to direct download
                    raise NotImplementedError("direct download")
                case 400:
                    raise NoSuchData("request can't be fullfilled")
            r.raise_for_status()

        if poll_url:
            async with session.delete(poll_url, headers=self.auth_headers) as r:
                if not r.ok:
                    logger = logging.getLogger(__name__)
                    logger.warn("couldn't DELETE %s: %s %s", poll_url, r.status, r.reason)
        return res
