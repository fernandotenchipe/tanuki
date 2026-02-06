import math
import random
from typing import Dict, List, Optional

try:
    from library.alivee import Pipeline
except Exception as e:
    raise ImportError('library.alivee import failed: ' + str(e))

from schemas import GenerateIn, Recipe, Shape

# --------- helpers ----------
def clamp(x: float, a: float = 0.0, b: float = 1.0) -> float:
    return max(a, min(b, x))


def recipe_to_descriptor(recipe: Recipe) -> str:
    """Transforma la receta (shapes) en un descriptor textual breve."""
    circles = [s for s in recipe.shapes if s.type == 'circle']
    rects = [s for s in recipe.shapes if s.type == 'rect']
    lines = [s for s in recipe.shapes if s.type == 'line']

    parts: List[str] = []

    def size_label(val: float, small: float, medium: float) -> str:
        if val < small:
            return 'small'
        if val < medium:
            return 'medium'
        return 'large'

    # Circles summary
    if circles:
        rs = [c.r or 0 for c in circles]
        avg_r = sum(rs) / len(rs) if rs else 0
        lab = size_label(avg_r, 30, 80)
        parts.append(f"{len(circles)} circles, {lab} radius")

        cx_mean = sum(c.cx or 0 for c in circles) / len(circles)
        cy_mean = sum(c.cy or 0 for c in circles) / len(circles)
        W = recipe.canvas.get('w', 360)
        H = recipe.canvas.get('h', 360)
        if abs(cx_mean - W/2) < W*0.15 and abs(cy_mean - H/2) < H*0.15:
            parts.append("circles clustered near center")

    # Rects summary
    if rects:
        ws = [r.w or 0 for r in rects]
        hs = [r.h or 0 for r in rects]
        avg_w = sum(ws) / len(ws) if ws else 0
        avg_h = sum(hs) / len(hs) if hs else 0
        lab_w = size_label(avg_w, 60, 140)
        lab_h = size_label(avg_h, 40, 120)
        parts.append(f"{len(rects)} rectangles, {lab_w} width, {lab_h} height")

        ys = [r.y or 0 for r in rects]
        H = recipe.canvas.get('h', 360)
        if ys:
            y_avg = sum(ys) / len(ys)
            if y_avg < H * 0.33:
                parts.append("rectangles in upper band")
            elif y_avg > H * 0.66:
                parts.append("rectangles in lower band")
            else:
                parts.append("rectangles around center band")

    # Lines summary
    if lines:
        def orientation(l: Shape) -> str:
            if l.x1 is None or l.x2 is None or l.y1 is None or l.y2 is None:
                return 'diag'
            dx = abs(l.x2 - l.x1)
            dy = abs(l.y2 - l.y1)
            if dx > dy * 2:
                return 'horizontal'
            if dy > dx * 2:
                return 'vertical'
            return 'diag'

        ori_counts = {'horizontal': 0, 'vertical': 0, 'diag': 0}
        for l in lines:
            ori_counts[orientation(l)] += 1
        dominant = max(ori_counts, key=lambda i: ori_counts[i])
        parts.append(f"{len(lines)} lines, mostly {dominant}")

    if not parts:
        return "simple composition"

    return "; ".join(parts)


# --------- alivee pipeline setup ----------
MAX_SHAPES = 8
PIPELINE = Pipeline(use_sys_sensors=False)
OUTPUT_LABELS: List[str] = []
for i in range(MAX_SHAPES):
    OUTPUT_LABELS.extend([
        f's{i}_t0', f's{i}_t1', f's{i}_t2',
        f's{i}_cx', f's{i}_cy', f's{i}_w', f's{i}_h', f's{i}_op'
    ])
PIPELINE.create_entity('drawer').senses(['steps_norm', 'sleep_norm', 'style_flag']).expresses(OUTPUT_LABELS).build()


def recipe_to_output_vector(recipe: Recipe) -> List[float]:
    """Convert a `Recipe` into the drawer output vector expected by the agent."""
    vec: List[float] = [0.0] * (MAX_SHAPES * 8)
    for i, s in enumerate(recipe.shapes[:MAX_SHAPES]):
        base = i * 8
        if s.type == 'circle':
            t0, t1, t2 = 5.0, -5.0, -5.0
        elif s.type == 'rect':
            t0, t1, t2 = -5.0, 5.0, -5.0
        else:
            t0, t1, t2 = -5.0, -5.0, 5.0

        cx = (s.cx if s.cx is not None else (s.x1 if s.x1 is not None else 0.0)) / 360.0
        cy = (s.cy if s.cy is not None else (s.y1 if s.y1 is not None else 0.0)) / 360.0
        if s.type == 'circle':
            w = (s.r or 0.0) / 180.0
            h = w
        elif s.type == 'rect':
            w = (s.w or 0.0) / 360.0
            h = (s.h or 0.0) / 360.0
        else:
            w = (s.x2 or 0.0) / 360.0
            h = (s.y2 or 0.0) / 360.0

        op = clamp(s.opacity if s.opacity is not None else 1.0)

        vec[base:base+8] = [t0, t1, t2, cx, cy, w, h, op]
    return vec


def _softmax3(t0: float, t1: float, t2: float) -> int:
    m = max(t0, t1, t2)
    e0, e1, e2 = math.exp(t0 - m), math.exp(t1 - m), math.exp(t2 - m)
    probs = [e0, e1, e2]
    return int(max(range(3), key=lambda i: probs[i]))


def output_vector_to_shapes(vec: List[float]) -> List[Shape]:
    shapes: List[Shape] = []
    W = 360.0
    H = 360.0

    for i in range(0, min(len(vec) // 8, MAX_SHAPES)):
        base = i * 8
        t0 = float(vec[base + 0])
        t1 = float(vec[base + 1])
        t2 = float(vec[base + 2])
        cx = clamp(float(vec[base + 3]))
        cy = clamp(float(vec[base + 4]))
        wv = clamp(float(vec[base + 5]))
        hv = clamp(float(vec[base + 6]))
        op = clamp(float(vec[base + 7]), 0.15, 1.0)

        stype_idx = _softmax3(t0, t1, t2)

        if stype_idx == 0:
            r = max(6.0, wv * 140.0)
            cx_px = cx * (W - 2*r) + r
            cy_px = cy * (H - 2*r) + r
            shapes.append(Shape(type='circle', cx=cx_px, cy=cy_px, r=r, opacity=op, strokeWidth=2.0))
        elif stype_idx == 1:
            w = max(10.0, wv * W)
            h = max(10.0, hv * H)
            x = cx * (W - w)
            y = cy * (H - h)
            shapes.append(Shape(type='rect', x=x, y=y, w=w, h=h, opacity=op, strokeWidth=2.0))
        else:
            x1 = cx * W
            y1 = cy * H
            x2 = wv * W
            y2 = hv * H
            shapes.append(Shape(type='line', x1=x1, y1=y1, x2=x2, y2=y2, opacity=op, strokeWidth=2.0))

    return shapes


def make_recipe(data: GenerateIn) -> Recipe:
    sleep_total = data.sleepHours + (data.sleepMinutes / 60.0)
    energy = clamp(data.steps / 12000.0)
    rest = clamp(sleep_total / 8.0)

    style_flag = 1.0 if data.drawingStyle and data.drawingStyle.lower().startswith('minimal') else 0.0
    inputs = {'steps_norm': energy, 'sleep_norm': rest, 'style_flag': style_flag}

    snapshot = PIPELINE.update(inputs=inputs)
    drawer_data = snapshot.get('drawer', {}).get('data', {})

    vec: List[float] = []
    for key in OUTPUT_LABELS:
        v = drawer_data.get(key, 0.0)
        try:
            vec.append(float(v))
        except Exception:
            vec.append(0.0)

    shapes = output_vector_to_shapes(vec)

    return Recipe(
        canvas={"w": 360, "h": 360},
        shapes=shapes,
        meta={
            "energy": energy,
            "rest": rest,
            "drawingStyle": data.drawingStyle,
            "tanukiName": data.survey.get('tanukiName'),
        },
    )


def baseline_recipe(data: GenerateIn) -> Recipe:
    sleep_total = data.sleepHours + (data.sleepMinutes / 60.0)
    energy = clamp(data.steps / 12000.0)
    rest = clamp(sleep_total / 8.0)
    wellbeing = (energy + rest) / 2.0

    W, H = 360.0, 360.0
    cx, cy = W/2, H/2

    rnd = random.Random(f"{data.steps}-{data.sleepHours}-{data.sleepMinutes}-{data.drawingStyle}-{data.survey.get('tanukiName','')}")
    jitter = (1.0 - rest) * 40.0

    shapes: List[Shape] = []
    style = (data.drawingStyle or "").lower()

    base_circles = 1 if energy < 0.35 else 2 if energy < 0.75 else 3
    for i in range(base_circles):
        r = 18 + energy * (60 + i * 15)
        ox = rnd.uniform(-35, 35) + rnd.uniform(-jitter, jitter) * 0.25
        oy = rnd.uniform(-35, 35) + rnd.uniform(-jitter, jitter) * 0.25
        shapes.append(
            Shape(
                type="circle",
                cx=clamp(cx + ox, r, W - r),
                cy=clamp(cy + oy, r, H - r),
                r=r,
                opacity=0.35 + rest * 0.55,
                strokeWidth=2 + energy * 2,
            )
        )

    n_lines = 4 + int(energy * 10)
    for _ in range(n_lines):
        x1 = cx + rnd.uniform(-40, 40)
        y1 = cy + rnd.uniform(-40, 40)
        x2 = cx + rnd.uniform(-160, 160) + rnd.uniform(-jitter, jitter)
        y2 = cy + rnd.uniform(-160, 160) + rnd.uniform(-jitter, jitter)
        shapes.append(
            Shape(
                type="line",
                x1=clamp(x1, 0, W), y1=clamp(y1, 0, H),
                x2=clamp(x2, 0, W), y2=clamp(y2, 0, H),
                opacity=0.15 + energy * 0.65,
                strokeWidth=1.5 + energy * 3.5,
            )
        )

    if "minimal" in style:
        for _ in range(5):
            w = rnd.uniform(40, 140) * (0.7 + rest * 0.6)
            h = rnd.uniform(18, 90) * (0.7 + rest * 0.6)
            x = rnd.uniform(0, W - w)
            y = rnd.uniform(0, H - h)
            shapes.append(Shape(type="rect", x=x, y=y, w=w, h=h, opacity=0.25 + rest * 0.6, strokeWidth=2))
    else:
        for _ in range(2 + int(energy * 3)):
            w = rnd.uniform(30, 120)
            h = rnd.uniform(20, 110)
            x = rnd.uniform(0, W - w)
            y = rnd.uniform(0, H - h)
            shapes.append(Shape(type="rect", x=x, y=y, w=w, h=h, opacity=0.15 + rest * 0.55, strokeWidth=1.8))

    shapes = shapes[:MAX_SHAPES]

    return Recipe(
        canvas={"w": int(W), "h": int(H)},
        shapes=shapes,
        meta={"energy": energy, "rest": rest, "style": data.drawingStyle},
    )
