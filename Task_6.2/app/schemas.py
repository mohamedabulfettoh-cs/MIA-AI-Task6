from pydantic import BaseModel


class CaptionResponse(BaseModel):
    caption: str
    model_mode: str


class HealthResponse(BaseModel):
    status: str
    mode: str
