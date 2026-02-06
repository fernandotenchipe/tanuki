from typing import List
import random

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from schemas import GenerateIn, ImageGenIn, MakeDatasetIn, Recipe, TrainPair
from alivee_pipeline import (
    PIPELINE,
    baseline_recipe,
    clamp,
    make_recipe,
    recipe_to_descriptor,
    recipe_to_output_vector,
)
from diffusion import run_image_generation
from diffusion import stream_image_generation
from fastapi.responses import StreamingResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ping")
def ping():
    return {"ok": True}


@app.post("/generate", response_model=Recipe)
def generate(data: GenerateIn):
    return make_recipe(data)


@app.post("/train")
def train(pairs: List[TrainPair], epochs: int = 50):
    """Pre-entrena el agente drawer con pares {input, target}."""
    dataset = []
    for p in pairs:
        steps = p.input.steps
        sleep_total = p.input.sleepHours + (p.input.sleepMinutes / 60.0)
        energy = clamp(steps / 12000.0)
        rest = clamp(sleep_total / 8.0)
        style_flag = 1.0 if p.input.drawingStyle and p.input.drawingStyle.lower().startswith('minimal') else 0.0
        inputs = [energy, rest, style_flag]
        outputs = recipe_to_output_vector(p.target)
        dataset.append({'inputs': inputs, 'outputs': outputs})

    try:
        PIPELINE.pretrain('drawer', dataset, epochs=epochs)
    except Exception as e:
        return {'ok': False, 'error': str(e)}
    return {'ok': True, 'samples': len(dataset), 'epochs': epochs}


@app.post('/debug')
def debug_input(data: GenerateIn):
    sleep_total = data.sleepHours + (data.sleepMinutes / 60.0)
    energy = clamp(data.steps / 12000.0)
    rest = clamp(sleep_total / 8.0)
    style_flag = 1.0 if data.drawingStyle and data.drawingStyle.lower().startswith('minimal') else 0.0
    inputs = {'steps_norm': energy, 'sleep_norm': rest, 'style_flag': style_flag}
    snapshot = PIPELINE.update(inputs=inputs)
    drawer_data = snapshot.get('drawer', {}).get('data', {})
    return {'ok': True, 'drawer': drawer_data, 'inputs': inputs}


@app.post("/make-dataset")
def make_dataset(body: MakeDatasetIn):
    pairs = []
    styles = ["Abstracto", "Minimalista", "Manga", "Realista", "Surrealista"]

    rnd = random.Random(123)

    for _ in range(body.samples):
        steps = rnd.randint(0, 15000)
        sleep_h = rnd.randint(3, 9)
        sleep_m = rnd.choice([0, 15, 30, 45])
        style = rnd.choice(styles)

        inp = GenerateIn(
            steps=steps,
            sleepHours=sleep_h,
            sleepMinutes=sleep_m,
            drawingStyle=style,
            survey={"tanukiName": "Fer"},
        )
        target = baseline_recipe(inp)
        pairs.append({"input": inp.model_dump(), "target": target.model_dump()})

    return {"ok": True, "pairs": pairs}


@app.post("/generate-image")
def generate_image(body: ImageGenIn):
    gen_in = GenerateIn(
        steps=body.steps,
        sleepHours=body.sleepHours,
        sleepMinutes=body.sleepMinutes,
        drawingStyle=body.drawingStyle,
        survey=body.survey,
    )
    recipe = make_recipe(gen_in)
    descriptor = recipe_to_descriptor(recipe)
    return run_image_generation(body, descriptor=descriptor, recipe=recipe)


@app.post("/generate-image-stream")
def generate_image_stream(body: ImageGenIn):
    """Stream image generation frames as SSE (text/event-stream).

    The client should POST and read the response as an event stream.
    """
    gen_in = GenerateIn(
        steps=body.steps,
        sleepHours=body.sleepHours,
        sleepMinutes=body.sleepMinutes,
        drawingStyle=body.drawingStyle,
        survey=body.survey,
    )
    recipe = make_recipe(gen_in)
    descriptor = recipe_to_descriptor(recipe)
    gen = stream_image_generation(body, descriptor=descriptor, recipe=recipe)

    return StreamingResponse(gen, media_type='text/event-stream')

