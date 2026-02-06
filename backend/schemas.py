from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel

# ---------- MODELOS (entrada) ----------
class GenerateIn(BaseModel):
    steps: int
    sleepHours: int
    sleepMinutes: int
    drawingStyle: str
    survey: Dict[str, Any]
    createdAt: Optional[str] = None


# ---------- “RECETA” (salida) ----------
ShapeType = Literal["circle", "rect", "line"]


class Shape(BaseModel):
    type: ShapeType
    # props opcionales según tipo
    cx: Optional[float] = None
    cy: Optional[float] = None
    r: Optional[float] = None

    x: Optional[float] = None
    y: Optional[float] = None
    w: Optional[float] = None
    h: Optional[float] = None

    x1: Optional[float] = None
    y1: Optional[float] = None
    x2: Optional[float] = None
    y2: Optional[float] = None

    opacity: float = 1.0
    strokeWidth: float = 2.0


class Recipe(BaseModel):
    canvas: Dict[str, int]
    shapes: List[Shape]
    meta: Dict[str, Any]


class MakeDatasetIn(BaseModel):
    samples: int = 60


class ImageGenIn(BaseModel):
    steps: int
    sleepHours: int
    sleepMinutes: int
    drawingStyle: str
    survey: Dict[str, Any]
    prompt_extra: Optional[str] = None
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    num_inference_steps: int = 35
    guidance_scale: float = 8.5
    width: int = 512
    height: int = 512
    trace: bool = False
    trace_every: int = 5


# Training input model: a pair of input + expected Recipe
class TrainPair(BaseModel):
    input: GenerateIn
    target: Recipe
