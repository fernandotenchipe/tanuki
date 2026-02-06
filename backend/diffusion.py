import base64
import io
import os
from typing import Any, Dict, List, Optional

try:
    import torch
except ImportError:
    torch = None  # type: ignore

DEFAULT_NEG_PROMPT = (
    "blurry, low quality, lowres, bad anatomy, ugly, deformed, extra limbs, fused fingers, "
    "mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed face, "
    "watermark, text, logo, signature, username, artist name, cropped, out of frame, "
    "worst quality, low quality, normal quality, jpeg artifacts, duplicate, morbid, mutilated"
)
PROMPT_GENERAL = "high quality, well composed, balanced composition, clear visual intent, aesthetically pleasing"

from alivee_pipeline import clamp
from schemas import ImageGenIn, Recipe
from typing import Generator
import queue
import threading
import json


def get_image_pipeline():
    """Lazily load a Stable Diffusion pipeline."""
    global _IMAGE_PIPELINE
    if '_IMAGE_PIPELINE' in globals() and _IMAGE_PIPELINE is not None:
        return _IMAGE_PIPELINE
    if torch is None:
        raise ImportError("torch no está instalado; instala torch (idealmente con CUDA).")
    try:
        from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
    except Exception as e:
        raise ImportError("Falta diffusers: pip install diffusers transformers safetensors accelerate") from e

    model_id = os.environ.get("MODEL_ID", "runwayml/stable-diffusion-v1-5")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    if device == "cuda":
        pipe = pipe.to(device)
    else:
        pipe.enable_attention_slicing()

    _IMAGE_PIPELINE = pipe
    return _IMAGE_PIPELINE


def build_image_prompt(data: ImageGenIn, descriptor: Optional[str] = None) -> str:
    """Construye un prompt combinando el cuestionario con la guía de composición."""
    survey = data.survey or {}
    sleep_total = data.sleepHours + (data.sleepMinutes / 60.0)
    energy = clamp(data.steps / 12000.0)
    rest = clamp(sleep_total / 8.0)
    wellbeing = (energy + rest) / 2.0

    def norm(v: Optional[str]) -> str:
        return (v or "").strip().lower()

    style_map = {
        "realista": "realistic rendering, lifelike lighting",
        "artístico / pictórico": "painterly style, visible brush strokes",
        "ilustrado": "digital illustration, clean outlines",
        "abstracto": "abstract composition, expressive shapes",
    }

    tema_map = {
        "persona / retrato": "portrait of one person, face focus, detailed facial features",
        "mascota / animal": "animal portrait, clear subject, expressive features",
        "paisaje / entorno": "landscape scene, environmental depth",
        "objeto / símbolo": "still life, single object focus, strong composition",
        "formas abstractas": "abstract forms, geometric and organic shapes",
    }

    paleta_map = {
        "colores cálidos": "warm color palette",
        "colores fríos": "cool color palette",
        "neutros": "neutral tones",
        "blanco y negro": "black and white, high contrast",
    }

    detalle_map = {
        "sencillo": "sketch, lineart, minimal shading, clean surfaces",
        "equilibrado": "moderate detail, balanced textures",
        "muy detallado": "highly detailed, intricate textures",
    }

    fondo_map = {
        "liso": "plain background",
        "con textura": "textured background",
        "escena / ambiente": "environmental background with context",
    }

    luz_map = {
        "suave": "soft lighting",
        "dramática": "dramatic lighting, strong contrast",
        "natural": "natural light",
    }

    libertad_map = {
        "muy fiel a lo pedido": "strictly follow the described subject",
        "interpretación artística": "artistic interpretation allowed",
        "creativa / libre": "creative freedom, imaginative elements allowed",
    }

    filtro_map = {
        "kawaii / tierno": "kawaii, cute, pastel mood",
        "anime / ilustración japonesa": "anime illustration, crisp lines, cel shading",
        "realista": "realistic mood and rendering",
        "oscuro / de miedo": "dark, moody, ominous atmosphere",
        "triste / melancólico": "melancholic, subdued",
        "profundo / introspectivo": "introspective, thoughtful mood",
        "alegre / luminoso": "bright, cheerful, optimistic",
        "misterioso": "mysterious, enigmatic",
        "épico / cinematográfico": "epic, cinematic scale",
        "surreal / onírico": "surreal, dreamlike",
    }

    parts = [PROMPT_GENERAL]

    style_choice = norm(survey.get('styleVisual'))
    tema_choice = norm(survey.get('tema'))
    paleta_choice = norm(survey.get('paleta'))
    detalle_choice = norm(survey.get('detalle'))
    fondo_choice = norm(survey.get('fondo'))
    luz_choice = norm(survey.get('iluminacion'))
    libertad_choice = norm(survey.get('libertad'))
    filtro_choice = norm(survey.get('filtro'))

    drawing_style = norm(data.drawingStyle)
    drawing_map = {
        "abstracto": "abstract painting, geometric shapes, soft gradients",
        "minimalista": "minimalist poster, clean lines, whitespace",
        "manga": "manga style, crisp lines, anime shading",
        "realista": "realistic illustration",
        "surrealista": "surreal art, dreamy forms",
    }

    for choice, mapping in [
        (style_choice, style_map),
        (tema_choice, tema_map),
        (paleta_choice, paleta_map),
        (detalle_choice, detalle_map),
        (fondo_choice, fondo_map),
        (luz_choice, luz_map),
        (libertad_choice, libertad_map),
    ]:
        if choice and choice not in ("aleatorio / auto", "aleatorio"):
            mapped = mapping.get(choice)
            if mapped:
                parts.append(mapped)

    if (not style_choice or style_choice.startswith("aleatorio")) and drawing_style:
        mapped = drawing_map.get(drawing_style)
        if mapped:
            parts.append(mapped)

    if wellbeing > 0.75:
        parts.append("masterpiece, highly detailed, sharp focus, professional quality")
        if energy > 0.66:
            parts.append("dynamic energy")
    elif wellbeing > 0.5:
        parts.append("detailed, good quality")
    else:
        parts.append("simple, soft rendering")

    if filtro_choice and filtro_choice not in ("aleatorio / auto", "aleatorio"):
        filtro_text = filtro_map.get(filtro_choice)
        if filtro_text:
            parts.append(filtro_text)

    extra = data.prompt_extra.strip() if data.prompt_extra else ""
    if extra:
        parts.append(extra)

    prompt = ", ".join([p for p in parts if p])

    if descriptor and tema_choice not in ("persona / retrato", "mascota / animal"):
        prompt += f", composition: {descriptor}"

    return prompt


def run_image_generation(body: ImageGenIn, descriptor: str, recipe: Recipe) -> Dict[str, Any]:
    try:
        pipe = get_image_pipeline()
    except Exception as e:
        return {"ok": False, "error": str(e)}

    prompt = build_image_prompt(body, descriptor=descriptor)
    neg_prompt = body.negative_prompt.strip() if body.negative_prompt else DEFAULT_NEG_PROMPT

    def _snap(x: int) -> int:
        return max(256, min(1024, int(x // 8 * 8)))

    width = _snap(body.width)
    height = _snap(body.height)
    generator = None
    if body.seed is not None and torch is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(int(body.seed))

    frames: List[str] = []

    def _callback(step: int, timestep: int, latents):
        # be defensive: body.trace may be falsy or body.trace_every may be None
        if not getattr(body, 'trace', False):
            return
        # always emit frames (avoid modulo entirely to prevent TypeError)
        try:
            with torch.no_grad():
                imgs = pipe.decode_latents(latents)
                pil = pipe.image_processor.postprocess(imgs, output_type="pil")[0]
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            frames.append(base64.b64encode(buf.getvalue()).decode())
        except Exception:
            pass

    try:
        out = pipe(
            prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=max(5, body.num_inference_steps),
            guidance_scale=max(1.0, body.guidance_scale),
            width=width,
            height=height,
            generator=generator,
            callback=_callback if body.trace else None,
        )
    except Exception as e:
        return {"ok": False, "error": str(e)}

    if not out.images:
        return {"ok": False, "error": "No se generó imagen"}

    img = out.images[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "ok": True,
        "prompt": prompt,
        "image": img_b64,
        "frames": frames if body.trace else [],
        "descriptor": descriptor,
        "recipe": recipe.model_dump(),
        "seed": body.seed,
        "model_id": os.environ.get("MODEL_ID", "runwayml/stable-diffusion-v1-5"),
        "steps": body.num_inference_steps,
        "guidance": body.guidance_scale,
    }


def stream_image_generation(body: ImageGenIn, descriptor: str, recipe: Recipe) -> Generator[str, None, None]:
    """Run generation and stream frames as Server-Sent Events (SSE).

    Yields text/event-stream chunks like: data: {...}\n\n
    The consumer should POST to the endpoint and parse SSE events.
    """
    if 'torch' not in globals() or torch is None:
        yield f"data: {json.dumps({'type':'error','msg':'torch no instalado'})}\n\n"
        return

    try:
        pipe = get_image_pipeline()
    except Exception as e:
        yield f"data: {json.dumps({'type':'error','msg': str(e)})}\n\n"
        return

    prompt = build_image_prompt(body, descriptor=descriptor)
    neg_prompt = body.negative_prompt.strip() if body.negative_prompt else DEFAULT_NEG_PROMPT

    q: "queue.Queue" = queue.Queue()

    def _callback(step: int, timestep: int, latents):
        # Only emit frames when tracing is requested; be defensive about trace flag
        if not getattr(body, 'trace', False):
            return
        # always emit frames (avoid modulo entirely to prevent TypeError)
        try:
            with torch.no_grad():
                imgs = pipe.decode_latents(latents)
                pil = pipe.image_processor.postprocess(imgs, output_type="pil")[0]
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            q.put({'type': 'frame', 'data': b64, 'step': step})
        except Exception:
            pass

    def _run():
        try:
            out = pipe(
                prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=max(5, body.num_inference_steps),
                guidance_scale=max(1.0, body.guidance_scale),
                width=max(256, min(1024, int(body.width // 8 * 8))),
                height=max(256, min(1024, int(body.height // 8 * 8))),
                generator=(torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(body.seed)) if body.seed is not None and torch is not None else None),
                callback=_callback,
            )

            if out and out.images:
                img = out.images[0]
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                final_b64 = base64.b64encode(buf.getvalue()).decode()
                q.put({'type': 'done', 'data': final_b64, 'prompt': prompt})
            else:
                q.put({'type': 'error', 'msg': 'No se generó imagen'})
        except Exception as e:
            q.put({'type': 'error', 'msg': str(e)})
        finally:
            q.put({'type': 'end'})

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    # stream events as they arrive
    while True:
        item = q.get()
        try:
            yield f"data: {json.dumps(item)}\n\n"
        except Exception:
            # ensure we don't crash generator
            pass
        if item.get('type') == 'end':
            break
