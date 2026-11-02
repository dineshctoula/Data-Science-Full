"""Day 91 — the forecast over HTTP, standard library only.

    GET  /health
    GET  /forecast?date=YYYY-MM-DD
    POST /forecast          {"date": "YYYY-MM-DD"}
    POST /forecast          {"date": "...", "weather": [24 rows]}

The handler speaks JSON in both directions. A failure the client caused
is an ApiError and comes back as {"error": "..."}. Anything else is a
500 with a fixed sentence; the traceback stays on the server, because
the message is the whole API and a stack trace would leak paths.
"""

from __future__ import annotations

import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from contract import ApiError, parse_body, parse_query

# a day of weather JSON is under 2 KB; anything larger is not this API
MAX_BODY = 8_192


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        # HTTP/1.0 ends the exchange after one response. Keep-alive would
        # need a timeout the tests and the one-shot client don't have.
        protocol_version = "HTTP/1.0"

        def log_message(self, fmt, *args):
            # the walkthrough prints the JSON itself; an access line per
            # request would bury the 24 hours
            return

        def do_GET(self):
            self._route("GET")

        def do_POST(self):
            self._route("POST")

        def do_PUT(self):
            self._method_not_allowed()

        def do_DELETE(self):
            self._method_not_allowed()

        def _method_not_allowed(self):
            self._send(405, {"error": "only GET and POST are served"}, allow="GET, POST")

        def _route(self, method: str):
            try:
                parsed = urlparse(self.path)
                if parsed.path == "/health":
                    self._health(method, parsed.query)
                elif parsed.path == "/forecast":
                    self._forecast(method, parsed.query)
                else:
                    # no trailing-slash alias: /forecast and /forecast/ would be two resources
                    raise ApiError(404, "unknown path")
            except ApiError as exc:
                allow = "GET, POST" if exc.status == 405 else None
                self._send(exc.status, {"error": str(exc)}, allow=allow)
            except Exception:
                traceback.print_exc()
                self._send(500, {"error": "the forecast failed"})

        def _health(self, method: str, query: str):
            if method != "GET":
                raise ApiError(405, "health only answers GET")
            if query:
                raise ApiError(400, "health takes no query")
            self._send(200, service.health())

        def _forecast(self, method: str, query: str):
            if method == "GET":
                self._send(200, service.forecast(parse_query(query), None))
                return
            if method != "POST":
                raise ApiError(405, "forecast only answers GET and POST")
            if query:
                raise ApiError(400, "put date in the JSON body, not the query")
            self._send(200, service.forecast(*self._json_body()))

        def _json_body(self):
            media = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if media != "application/json":
                # a form post would otherwise be parsed as JSON and fail opaquely
                raise ApiError(415, "Content-Type should be application/json")
            return parse_body(self._read_body())

        def _read_body(self) -> bytes:
            raw_len = self.headers.get("Content-Length")
            if raw_len is None:
                raise ApiError(411, "Content-Length is required")
            try:
                length = int(raw_len)
            except ValueError:
                raise ApiError(400, "Content-Length is not an integer") from None
            if length < 0:
                raise ApiError(400, "Content-Length is not an integer")
            if length > MAX_BODY:
                # not read: the connection closes, so the leftover bytes
                # cannot be read as the next request
                raise ApiError(413, f"body is larger than {MAX_BODY} bytes")
            return self.rfile.read(length)

        def _send(self, status: int, payload: dict, allow: str | None = None):
            try:
                # allow_nan so a NaN count can't leave as invalid JSON
                body = json.dumps(payload, allow_nan=False).encode("utf-8")
            except ValueError:
                status = 500
                body = b'{"error":"the forecast failed"}'
                allow = None
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            # a forecast for a date is not a cacheable page; yesterday's
            # reply should not be reused for a later ask
            self.send_header("Cache-Control", "no-store")
            if allow:
                self.send_header("Allow", allow)
            self.end_headers()
            self.wfile.write(body)

    return Handler


def bind(service, port: int = 8765) -> ThreadingHTTPServer:
    """Listen on 127.0.0.1 only.

    A script on this machine can call it. Binding every interface would
    publish the model with no separate decision to do that.
    """
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(service))
