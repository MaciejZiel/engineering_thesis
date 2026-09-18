"""Loopback web server. Requests enqueue existing application actions only."""
from __future__ import annotations

import asyncio
import secrets
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .bridge import WebBridge


def create_app(bridge: WebBridge, frontend: Path | None = None) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])

    @app.get("/api/v1/state")
    def state():
        return bridge.snapshot()

    @app.get("/api/v1/frame")
    def frame():
        with bridge.lock:
            data = bridge.jpeg
        if not data:
            return Response(status_code=204)
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/v1/commands/{name}")
    async def command(name: str, request: Request):
        origin = request.headers.get("origin")
        if origin and urlparse(origin).netloc != request.headers.get("host"):
            raise HTTPException(403, "Origin not allowed")
        if not secrets.compare_digest(request.headers.get("x-session-token", ""), bridge.token):
            raise HTTPException(403, "Invalid session")
        try:
            bridge.command(name)
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        return {"accepted": True, "message": "Action queued; waiting for the tracking engine."}

    @app.websocket("/api/v1/telemetry")
    async def telemetry(socket: WebSocket):
        origin = socket.headers.get("origin")
        if origin and urlparse(origin).netloc != socket.headers.get("host"):
            await socket.close(code=1008)
            return
        await socket.accept()
        try:
            while not bridge.closed:
                await asyncio.wait_for(socket.send_json(bridge.snapshot()), timeout=2)
                await asyncio.sleep(1 / 15)
        except (WebSocketDisconnect, RuntimeError, OSError, asyncio.TimeoutError):
            return

    if frontend and (frontend / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=frontend / "assets"), name="assets")

        @app.get("/")
        def index():
            return FileResponse(frontend / "index.html")

    return app
