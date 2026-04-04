import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image

from src.services.overlay_service import create_overlay_service
from src.utils.settings import Settings

app = FastAPI()


@asynccontextmanager
async def lifespan(app: FastAPI):

    settings = Settings.from_sources("configs/service.yaml")
    overlay_service = create_overlay_service(settings=settings)

    app.state.overlay_service = overlay_service

    yield


@app.get("/")
def root():
    return FileResponse("src/ui/index.html")


@app.post("/overlay")
async def overlay(file: UploadFile = File(...), mask: str = Form("default")):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    result = app.state.overlay_service.process_image(image, mask)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
