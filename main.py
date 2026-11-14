"""Provenance backend: image forensics API (CPU-only, no GPU needed).
Optional AI detector: set DETECTOR_URL (+ DETECTOR_TOKEN) to a hosted model endpoint
(e.g. a Hugging Face Space / Inference Endpoint on a GPU). Without it, that check is reported as "Not run"."""
import io, json, os, re, shutil, subprocess, tempfile
import numpy as np, requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageChops

ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
MAX_MB = int(os.getenv("MAX_MB", "20"))
app = FastAPI(title="Provenance API")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["*"], allow_headers=["*"])

SCREENS = {(1170,2532),(1179,2556),(1284,2778),(1290,2796),(1080,1920),(1080,2340),(1080,2400),(1440,2560),
           (1920,1080),(2560,1440),(1366,768),(1440,900),(1536,864),(1920,1200),(2880,1800),(3840,2160),(750,1334),(828,1792)}
GENS = [("Midjourney", r"midjourney"), ("DALL·E / OpenAI", r"dall[-·. ]?e|openai"),
        ("Stable Diffusion", r"stable diffusion|sd-webui|Steps: \d+, Sampler|Negative prompt"),
        ("ComfyUI", r"comfyui|\"class_type\""), ("Adobe Firefly", r"firefly"), ("Google Imagen / Gemini", r"imagen|synthid|gemini")]
EDITORS = r"photoshop|lightroom|gimp|canva|snapseed|affinity|pixelmator|capture one|illustrator"

def exiftool(data: bytes) -> dict:
    if not shutil.which("exiftool"): return {}
    try:
        p = subprocess.run(["exiftool", "-json", "-n", "-G1", "-a", "-u", "-"], input=data, capture_output=True, timeout=20)
        return json.loads(p.stdout)[0]
    except Exception: return {}

def c2pa_check(data: bytes, suffix: str):
    try:
        import c2pa
    except Exception: return None  # library unavailable
    with tempfile.NamedTemporaryFile(suffix=suffix) as f:
        f.write(data); f.flush()
        try:
            r = c2pa.Reader.from_file(f.name); m = json.loads(r.json())
            gen = m.get("active_manifest"); return {"present": True, "active": gen, "validation": m.get("validation_status", [])}
        except Exception as e:
            return {"present": False, "error": str(e)[:120]}

def ela(img: Image.Image, q=90):
    rgb = img.convert("RGB"); buf = io.BytesIO(); rgb.save(buf, "JPEG", quality=q); buf.seek(0)
    d = np.asarray(ImageChops.difference(rgb, Image.open(buf)).convert("L"), dtype=float)
    return float(d.mean()), float(d.std()), float(np.percentile(d, 99))

def detectors(data: bytes):
    """Calls the detector service (CNNDetection, UniversalFakeDetect, TruFor). Returns dict or None."""
    url = os.getenv("DETECTORS_URL")
    if not url: return None
    try:
        r = requests.post(url.rstrip("/") + "/detect", files={"file": ("image", data)},
                          data={"trufor": os.getenv("RUN_TRUFOR", "false")}, timeout=240)
        r.raise_for_status(); return r.json()
    except Exception as e: return {"error": str(e)[:120]}

def pct(d, k):
    v = (d or {}).get(k)
    if isinstance(v, dict) and "ai_prob" in v: return f"{v['ai_prob']:.0%}"
    return "Unavailable" if v else "Not run"

def card(t, s, rows=None, **kw): return {"t": t, "s": s, "rows": rows or [], **kw}

@app.get("/")
def root(): return {"service": "Provenance API", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health(): return {"ok": True, "exiftool": bool(shutil.which("exiftool")), "detectors": bool(os.getenv("DETECTORS_URL"))}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), include_gps: bool = Form(False)):
    data = await file.read()
    if len(data) > MAX_MB * 1048576: raise HTTPException(413, "File too large")
    try: img = Image.open(io.BytesIO(data)); img.load()
    except Exception: raise HTTPException(400, "Not a valid image")
    fmt, (w, h) = img.format, img.size
    txt = data.decode("latin1"); ex = exiftool(data)
    g = lambda *k: next((ex[x] for x in ex if any(x.endswith(":" + n) for n in k)), None)
    make, model, soft = g("Make"), g("Model"), g("Software")
    # AI markers
    trained = bool(re.search("trainedAlgorithmicMedia", txt)); sd = bool(re.search(r"parameters\x00?[\s\S]{0,40}(Steps:|Negative prompt)", txt))
    gens = [n for n, p in GENS if re.search(p, txt, re.I)]
    det = detectors(data)
    cnn_p = ((det or {}).get('cnndetection') or {}).get('ai_prob'); ufd_p = ((det or {}).get('ufd') or {}).get('ai_prob')
    tf = (det or {}).get('trufor') or {}; ai_score = max([p for p in (cnn_p, ufd_p) if p is not None], default=None)
    # provenance, editing
    c2 = c2pa_check(data, "." + (fmt or "jpg").lower()); c2_marker = bool(re.search("c2pa|jumb|contentauth", txt, re.I))
    tools = sorted({*re.findall(r'stEvt:softwareAgent="([^"]+)"', txt), *re.findall(r'xmp:CreatorTool(?:>|=")([^<"]+)', txt), *([soft] if soft else [])})
    editors = [t for t in tools if re.search(EDITORS, t, re.I)]
    hist = "xmpMM:History" in txt
    # compression / screenshot
    bpp = round(len(data) * 8 / (w * h), 2); sc, why_sc = 0, []
    if not make and fmt == "PNG": sc += 1; why_sc.append("PNG with no camera metadata")
    if (w, h) in SCREENS or (h, w) in SCREENS: sc += 1; why_sc.append(f"Dimensions match a common screen size ({w}×{h})")
    if re.search("screenshot|screen shot", file.filename or "", re.I): sc += 1; why_sc.append("File name contains 'screenshot'")
    # ELA
    em, es, ep = ela(img); ela_flag = ep > 25
    # verdict
    if trained or sd: v, c, why = "Likely AI-generated", 90, "Embedded metadata declares AI generation."
    elif (ufd_p or 0) > .85 or (cnn_p is not None and ufd_p is not None and min(cnn_p, ufd_p) > .7): v, c, why = "Likely AI-generated", int(max(cnn_p or 0, ufd_p or 0) * 100), "Research detector models scored this image as synthetic. These models can be wrong, especially on newer generators."
    elif gens: v, c, why = "Possibly AI-generated", 62, "A generator name appears in the file data."
    elif c2 and c2.get("present"): v, c, why = "Provenance present", 70, "A C2PA manifest was found; review its validation status."
    elif sc >= 2: v, c, why = "Likely screenshot", 66, "Several screenshot traits were found."
    elif editors: v, c, why = "Edited", 55, "Editing software is recorded: " + ", ".join(editors) + "."
    elif make and model: v, c, why = "Likely camera photo", 48, "Camera make and model present. Weak evidence: metadata can be forged."
    else: v, c, why = "Inconclusive", 18, "No decisive evidence. Stripped metadata is common and proves nothing either way."
    if ai_score is None: why += " The AI-detector models were not run."
    gps = (g("GPSLatitude"), g("GPSLongitude"))
    cards = [
      card("AI detection evidence", "strong" if trained or sd else "mod" if gens or (ai_score or 0) > .5 else "weak" if ai_score is not None else "na", [
        ["CNNDetection (ResNet-50)", f"{cnn_p:.0%} synthetic" if cnn_p is not None else "Not run"],
        ["UniversalFakeDetect (CLIP)", f"{ufd_p:.0%} synthetic" if ufd_p is not None else "Not run"],
        ["Metadata AI markers", "IPTC trained-algorithmic-media flag" if trained else "Stable Diffusion parameters" if sd else "None found"],
        ["Possible generator", ", ".join(gens) or "No generator signature found"]]),
      card("Metadata findings", "mod" if make else "weak" if ex else "none",
        [["Format", fmt], ["Dimensions", f"{w} × {h}"], ["Size", f"{len(data)/1024:.1f} KB"], ["Make", make or "—"], ["Model", model or "—"],
         ["Software", soft or "—"], ["Captured", g("DateTimeOriginal") or "—"], ["Lens", g("LensModel", "LensID") or "—"],
         ["ISO", g("ISO") or "—"], ["Exposure", g("ExposureTime", "ShutterSpeed") or "—"], ["Aperture", g("FNumber") or "—"]]),
      card("Provenance status", "mod" if c2 and c2.get("present") else "na", [
        ["C2PA manifest", "Valid manifest read" if c2 and c2.get("present") else "Marker only, not parseable" if c2_marker else "Not found"],
        ["Validation issues", str(len(c2.get("validation", []))) if c2 and c2.get("present") else "—"]]),
      card("Screenshot & compression", "strong" if sc >= 2 else "weak" if sc else "none", [
        ["Container", f"{fmt} ({'lossless' if fmt == 'PNG' else 'lossy'})"], ["Bits per pixel", bpp],
        ["ELA mean / 99th pct", f"{em:.1f} / {ep:.0f}"], ["Screenshot signals", "; ".join(why_sc) or "None"]]),
      card("Editing history", "mod" if editors or hist else "none", [
        ["Software recorded", ", ".join(tools) or "None"], ["XMP edit history", "Present" if hist else "Not recorded"]]),
      card("Manipulation investigation", "mod" if tf.get("score", 0) > .5 else "weak" if ela_flag else "na", [
        ["Error-level analysis", "Uneven recompression error, worth a closer look" if ela_flag else "No strong ELA anomaly"],
        ["TruFor manipulation score", f"{tf['score']:.0%} (area flagged {tf['flagged_area']:.0%})" if "score" in tf else "Not run"]]),
      card("Location (GPS)", "mod" if all(gps) else "none", [["GPS",
        (f"{gps[0]}, {gps[1]}" if include_gps else "Found but hidden (consent needed)") if all(gps) else "None embedded"]]),
    ]
    return {"V": {"v": v, "c": c, "why": why}, "S": cards}
