"""Dashboard process: FastAPI app that bridges the MetricsQueue to WebSockets.

A background asyncio task drains the multiprocessing MetricsQueue and broadcasts
each metrics dict as JSON to every connected browser. The dashboard is fully
decoupled from training: if it crashes, training continues; the queue simply
fills and is drained here.
"""

import asyncio
import json
import logging
import os
import queue as queue_mod

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

_HTML = os.path.join(os.path.dirname(__file__), "dashboard.html")


def build_app(metrics_queue):
    app = FastAPI()
    clients: set[WebSocket] = set()

    @app.get("/")
    async def index():
        return FileResponse(_HTML)

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        clients.add(websocket)
        logger.debug(f"Client connected; {len(clients)} total")
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            clients.discard(websocket)
            logger.debug(f"Client disconnected; {len(clients)} remaining")

    async def pump():
        loop = asyncio.get_event_loop()
        while True:
            try:
                payload = await loop.run_in_executor(
                    None, lambda: metrics_queue.get(timeout=1.0)
                )
            except queue_mod.Empty:
                await asyncio.sleep(0)
                continue
            except Exception:
                await asyncio.sleep(0.1)
                continue

            dead = []
            data = json.dumps(payload)
            for c in list(clients):
                try:
                    await c.send_text(data)
                except Exception:
                    dead.append(c)
            for c in dead:
                clients.discard(c)

    @app.on_event("startup")
    async def _startup():
        asyncio.create_task(pump())

    return app


def dashboard_main(config, metrics_queue) -> None:
    app = build_app(metrics_queue)
    uvicorn.run(app, host="127.0.0.1", port=config.dashboard_port, log_level="warning")
