"""
FastAPI backend.

    GET  /health   liveness + which model is active
    POST /caption  upload an image, get {"caption": "..."}
    GET  /         Gradio UI, mounted here for easy testing

Run: uvicorn app.main:app --host 0.0.0.0 --port 7860

Gradio is mounted on the same app/port so the whole thing is one
container, one port, works fine on HF Spaces / Render / Fly.io etc.
"""
import os

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.inference import caption_image_bytes, get_captioner, MODE
from app.schemas import CaptionResponse, HealthResponse

app = FastAPI(
    title="Image Caption Generator API",
    description="Generates natural-language captions for images "
                 "(Flickr8k-trained CNN+RNN+Attention, or pretrained fallback).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "mode": MODE}


@app.post("/caption", response_model=CaptionResponse)
async def caption(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")
    image_bytes = await file.read()
    try:
        text = caption_image_bytes(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Captioning failed: {e}")
    return {"caption": text, "model_mode": MODE}


# warm up the model at startup so the first request isn't slow
@app.on_event("startup")
def warmup():
    try:
        get_captioner()
    except Exception as e:
        print(f"[startup] Warning: model warmup failed: {e}")


# mount the Gradio UI at "/"
from app.gradio_app import build_demo  # noqa: E402
import gradio as gr  # noqa: E402

demo = build_demo()
app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
