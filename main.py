"""Provenance backend: image forensics API. CPU-only, no GPU needed.

Local checks (always on): metadata/EXIF/GPS, C2PA, generator signatures, editing history,
screenshot + compression traces, error-level analysis.
Optional AI models: set DETECTORS_URL to the detector service (CNNDetection, UniversalFakeDetect, TruFor).
Env: ALLOWED_ORIGINS, MAX_MB, MAX_CONCURRENT, DETECTORS_URL, DETECTOR_TIMEOUT, RUN_TRUFOR.
"""
import io, json, os, re, shutil, subprocess, tempfile, threading
import numpy as np, requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageChops

Image.MAX_IMAGE_PIXELS = 60_000_000
ORIGINS = [o.strip().rstrip("/") for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
MAX_MB = int(os.getenv("MAX_MB", "20"))
DET_TIMEOUT = int(os.getenv("DETECTOR_TIMEOUT", "120"))
_gate = threading.BoundedSemaphore(int(os.getenv("MAX_CONCURRENT", "2")))
app = FastAPI(title="Provenance API")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["*"], allow_headers=["*"])

SCREENS = {(1170,2532),(1179,2556),(1284,2778),(1290,2796),(1080,1920),(1080,2340),(1080,2400),(1440,2560),
           (1920,1080),(2560,1440),(1366,768),(1440,900),(1536,864),(1920,1200),(2880,1800),(3840,2160),(750,1334),(828,1792)}
GENS = [("Midjourney", r"midjourney"), ("DALL·E / OpenAI", r"dall[^a-z]{0,3}e\b|openai"),
        ("Stable Diffusion", r"stable diffusion|sd-webui|steps: \d+, sampler|negative prompt"),
        ("ComfyUI", r"comfyui|\"class_type\""), ("Adobe Firefly", r"firefly"), ("Google Imagen / Gemini", r"imagen|synthid|gemini")]
EDITORS = r"photoshop|lightroom|gimp|canva|snapseed|affinity|pixelmator|capture one|illustrator"
MIME = {"JPEG": "image/jpeg", "MPO": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "TIFF": "image/tiff", "GIF": "image/gif", "AVIF": "image/avif", "HEIF": "image/heic"}
LOSSLESS = {"PNG", "BMP", "TIFF", "GIF"}
STD_LUMA = sum([16,11,10,16,24,40,51,61,12,12,14,19,26,58,60,55,14,13,16,24,40,57,69,56,14,17,22,29,51,87,80,62,
                18,22,37,56,68,109,103,77,24,35,55,64,81,104,113,92,49,64,78,87,103,121,120,101,72,92,95,98,112,100,103,99])

# ---------- metadata ----------
def exiftool(data: bytes) -> dict:
    if not shutil.which("exiftool"): return {}
    try:
        p = subprocess.run(["exiftool", "-json", "-n", "-G1", "-a", "-u", "-"], input=data, capture_output=True, timeout=20)
        return json.loads(p.stdout)[0]
    except Exception: return {}

def num(v):
    try: return float(v)
    except Exception: return None

def meta_exiftool(ex: dict) -> dict:
    def g(*names):
        for n in names:
            for k, v in ex.items():
                if k.endswith(":" + n) and v not in ("", None): return v
    lat, lon = num(ex.get("Composite:GPSLatitude")), num(ex.get("Composite:GPSLongitude"))
    if lat is None and g("GPSLatitude") is not None:
        lat = num(g("GPSLatitude")); lat = -lat if lat is not None and g("GPSLatitudeRef") == "S" else lat
    if lon is None and g("GPSLongitude") is not None:
        lon = num(g("GPSLongitude")); lon = -lon if lon is not None and g("GPSLongitudeRef") == "W" else lon
    return {"Make": g("Make"), "Model": g("Model"), "Software": g("Software"), "DateTimeOriginal": g("DateTimeOriginal"),
            "ModifyDate": g("ModifyDate"), "Lens": g("LensModel", "LensID"), "ISO": g("ISO"), "ExposureTime": num(g("ExposureTime")),
            "FNumber": num(g("FNumber")), "lat": lat, "lon": lon}

def meta_pillow(img) -> dict:  # fallback when exiftool is not installed
    out = dict.fromkeys(["Make", "Model", "Software", "DateTimeOriginal", "ModifyDate", "Lens", "ISO", "ExposureTime", "FNumber", "lat", "lon"])
    try: ex = img.getexif()
    except Exception: return out
    s = lambda x: str(x).strip("\x00 ") or None
    out.update(Make=s(ex.get(0x010F, "")), Model=s(ex.get(0x0110, "")), Software=s(ex.get(0x0131, "")), ModifyDate=s(ex.get(0x0132, "")))
    try:
        sub = ex.get_ifd(0x8769)
        out.update(DateTimeOriginal=s(sub.get(0x9003, "")), Lens=s(sub.get(0xA434, "")), ISO=sub.get(0x8827),
                   ExposureTime=num(sub.get(0x829A)), FNumber=num(sub.get(0x829D)))
    except Exception: pass
    try:
        gp = ex.get_ifd(0x8825)
        dms = lambda t, ref, neg: (lambda d: -d if ref == neg else d)(sum(float(x) / 60 ** i for i, x in enumerate(t)))
        if 2 in gp and 4 in gp: out["lat"], out["lon"] = dms(gp[2], gp.get(1), "S"), dms(gp[4], gp.get(3), "W")
    except Exception: pass
    return out

# ---------- provenance ----------
def c2pa_check(data: bytes, fmt):
    mime = MIME.get(fmt or "")
    try: import c2pa
    except Exception: return None
    if not mime: return None
    try:
        raw = c2pa.Reader(mime, io.BytesIO(data)).json()
        m = json.loads(raw); man = (m.get("manifests") or {}).get(m.get("active_manifest")) or {}; sig = man.get("signature_info") or {}
        return {"present": True, "ai": "trainedAlgorithmicMedia" in raw, "state": m.get("validation_state"),
                "issues": [x.get("code", "?") for x in (m.get("validation_status") or [])],
                "generator": (man.get("claim_generator_info") or [{}])[0].get("name") or man.get("claim_generator"),
                "issuer": sig.get("issuer") or sig.get("common_name")}
    except Exception as e:
        return {"present": False, "missing": "ManifestNotFound" in type(e).__name__ + str(e), "error": str(e)[:120]}

# ---------- pixels ----------
def ela(img: Image.Image, q=90):
    rgb = img.convert("RGB"); buf = io.BytesIO(); rgb.save(buf, "JPEG", quality=q); buf.seek(0)
    d = np.asarray(ImageChops.difference(rgb, Image.open(buf)).convert("L"))
    return float(d.mean()), float(np.percentile(d, 99))

def jpeg_quality(img):
    try:
        t = img.quantization[0]; s = sum(t) / STD_LUMA * 100
        return int(max(1, min(100, 5000 / s if s >= 100 else (200 - s) / 2)))
    except Exception: return None

# ---------- detector service ----------
def detectors(data: bytes):
    url = os.getenv("DETECTORS_URL")
    if not url: return None
    try:
        r = requests.post(url.rstrip("/") + "/detect", files={"file": ("image", data)},
                          data={"trufor": os.getenv("RUN_TRUFOR", "false")}, timeout=(10, DET_TIMEOUT))
        r.raise_for_status(); return r.json()
    except Exception as e: return {"error": type(e).__name__ + ": " + str(e)[:100]}

def line(det, key, fmt):
    if det is None: return "Not run"
    if key not in det: return "Service error: " + det.get("error", "no response")
    d = det[key] or {}
    return fmt(d) if "error" not in d else "Error: " + str(d["error"])[:80]

def card(t, s, rows): return {"t": t, "s": s, "rows": rows}

@app.get("/")
def root(): return {"service": "Provenance API", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health(): return {"ok": True, "exiftool": bool(shutil.which("exiftool")), "detectors": bool(os.getenv("DETECTORS_URL"))}

@app.post("/analyze")
def analyze(file: UploadFile = File(...), include_gps: bool = Form(False)):  # sync: runs in a worker thread, never blocks the server
    data = file.file.read(MAX_MB * 1048576 + 1)
    if len(data) > MAX_MB * 1048576: raise HTTPException(413, f"File too large (max {MAX_MB} MB)")
    if not _gate.acquire(timeout=60): raise HTTPException(503, "Server busy, try again shortly")
    try: return _analyze(data, file.filename or "", include_gps)
    finally: _gate.release()

def _analyze(data: bytes, filename: str, include_gps: bool):
    try: img = Image.open(io.BytesIO(data))
    except Exception: raise HTTPException(400, "Not a valid or supported image")
    if img.size[0] * img.size[1] > Image.MAX_IMAGE_PIXELS: raise HTTPException(413, "Image dimensions too large (max 60 megapixels)")
    try: img.load()
    except Exception: raise HTTPException(400, "Image could not be decoded")
    fmt, (w, h) = img.format, img.size
    low = data.decode("latin1").lower()
    ex = exiftool(data); M = meta_exiftool(ex) if ex else meta_pillow(img)
    make, model, soft = M["Make"], M["Model"], M["Software"]
    # AI markers in file data
    trained = "trainedalgorithmicmedia" in low
    sd = bool(re.search(r"parameters\x00?[\s\S]{0,40}(steps:|negative prompt)", low))
    gens = [n for n, p in GENS if re.search(p, low)]
    # C2PA, editing history
    c2 = c2pa_check(data, fmt); c2ok = bool(c2 and c2.get("present")); c2_marker = "c2pa" in low or "contentauth" in low
    tools = sorted({*re.findall(r'stevt:softwareagent="([^"]+)"', low), *re.findall(r'xmp:creatortool(?:>|=")([^<"]+)', low), *([soft.lower()] if soft else [])})
    editors = [t for t in tools if re.search(EDITORS, t)]
    hist = "xmpmm:history" in low
    ts_diff = bool(M["DateTimeOriginal"] and M["ModifyDate"] and M["DateTimeOriginal"] != M["ModifyDate"])
    # compression / screenshot
    bpp = round(len(data) * 8 / (w * h), 2); sc, why_sc = 0, []
    if not make and fmt == "PNG": sc += 1; why_sc.append("PNG with no camera metadata")
    if (w, h) in SCREENS or (h, w) in SCREENS: sc += 1; why_sc.append(f"Dimensions match a common screen size ({w}×{h})")
    if re.search("screenshot|screen shot", filename, re.I): sc += 1; why_sc.append("File name contains 'screenshot'")
    em, ep = ela(img); ela_flag = ep > 25; jq = jpeg_quality(img)
    # AI models
    det = detectors(data); cnn_p = ((det or {}).get("cnndetection") or {}).get("ai_prob"); ufd_p = ((det or {}).get("ufd") or {}).get("ai_prob")
    tf = (det or {}).get("trufor") or {}; probs = [p for p in (cnn_p, ufd_p) if p is not None]
    ai_score = max(probs, default=None); low_ai = ai_score is not None and ai_score < .3
    # verdict
    if trained or sd or (c2ok and c2["ai"]): v, c, why = "Likely AI-generated", 92, "Embedded metadata or signed content credentials declare AI generation."
    elif (ufd_p or 0) > .85 or (len(probs) == 2 and min(probs) > .7):
        v, c, why = "Likely AI-generated", int(ai_score * 100), "Research detector models scored this image as synthetic. These models can be wrong, especially on newer generators."
    elif gens: v, c, why = "Possibly AI-generated", 62, "A generator name appears in the file data."
    elif c2ok and c2.get("state") == "Invalid": v, c, why = "Credentials failed validation", 75, "Content credentials were found but fail validation: the file may have been changed after signing, or the signature is broken."
    elif c2ok: v, c, why = "Provenance present", 70, "A C2PA manifest was found; review its validation status."
    elif sc >= 2: v, c, why = "Likely screenshot", 66, "Several screenshot traits were found."
    elif editors: v, c, why = "Edited", 55, "Editing software is recorded: " + ", ".join(editors) + "."
    elif make and model: v, c, why = "Likely camera photo", 48 + (12 if low_ai else 0), "Camera make and model present. Weak evidence: metadata can be forged."
    elif low_ai: v, c, why = "No AI signs found", min(55, int((1 - ai_score) * 60)), "The detector models did not flag it, but they miss many newer generators."
    else: v, c, why = "Inconclusive", 25, "No decisive evidence. Stripped metadata is common and proves nothing either way."
    if det is None: why += " The AI-detector models were not run."
    elif "error" in det: why += " The AI-detector service could not be reached."
    # cards
    has_meta = any(M[k] for k in ("Make", "Model", "Software", "DateTimeOriginal"))
    fx = lambda: f"1/{round(1 / M['ExposureTime'])} s" if M["ExposureTime"] and M["ExposureTime"] < 1 else (f"{M['ExposureTime']} s" if M["ExposureTime"] else "—")
    has_gps = M["lat"] is not None and M["lon"] is not None
    c2line = ("Marker found, C2PA library unavailable" if c2 is None and c2_marker else "Could not be read: " + c2["error"] if c2 and not c2.get("missing") and c2_marker else "Not found") if not c2ok else \
        (f"Manifest found ({c2['state']})" if c2.get("state") else "Manifest found") + (": " + ", ".join(c2["issues"][:3]) if c2["issues"] else "")
    cards = [
      card("AI detection evidence", "strong" if trained or sd else "mod" if gens or (ai_score or 0) > .5 else "weak" if ai_score is not None else "na", [
        ["CNNDetection (ResNet-50)", line(det, "cnndetection", lambda d: f"{d['ai_prob']:.0%} synthetic")],
        ["UniversalFakeDetect (CLIP)", line(det, "ufd", lambda d: f"{d['ai_prob']:.0%} synthetic")],
        ["Metadata AI markers", "IPTC trained-algorithmic-media flag" if trained else "Stable Diffusion parameters" if sd else "None found"],
        ["Possible generator", ", ".join(gens) or "No generator signature found"]]),
      card("Metadata findings", "mod" if make else "weak" if has_meta else "none", [
        ["Format", fmt], ["Dimensions", f"{w} × {h}"], ["Size", f"{len(data)/1024:.1f} KB"], ["Make", make or "—"], ["Model", model or "—"],
        ["Software", soft or "—"], ["Captured", M["DateTimeOriginal"] or "—"], ["Lens", M["Lens"] or "—"], ["ISO", M["ISO"] or "—"],
        ["Exposure", fx()], ["Aperture", f"f/{M['FNumber']:.1f}" if M["FNumber"] else "—"]]),
      card("Provenance status", ("strong" if c2.get("state") in ("Valid", "Trusted") and not c2["issues"] else "mod") if c2ok else "na", [
        ["C2PA manifest", c2line], ["Claim generator", (c2 or {}).get("generator") or "—"], ["Signer", (c2 or {}).get("issuer") or "—"]]),
      card("Screenshot & compression", "strong" if sc >= 2 else "weak" if sc else "none", [
        ["Container", f"{fmt} ({'lossless' if fmt in LOSSLESS else 'lossy' if fmt in ('JPEG', 'MPO') else 'lossy or lossless'})"],
        ["Bits per pixel", bpp], ["Estimated JPEG quality", f"~{jq}" if jq else "—"], ["ELA mean / 99th pct", f"{em:.1f} / {ep:.0f}"],
        ["Screenshot signals", "; ".join(why_sc) or "None"]]),
      card("Editing history", "mod" if editors or hist or ts_diff else "none", [
        ["Software recorded", ", ".join(tools) or "None"], ["XMP edit history", "Present" if hist else "Not recorded"],
        ["Timestamp mismatch", "Capture and modify times differ" if ts_diff else "No"]]),
      card("Manipulation investigation", "mod" if tf.get("score", 0) > .5 else "weak" if ela_flag else "na", [
        ["Error-level analysis", "Uneven recompression error, worth a closer look" if ela_flag else "No strong ELA anomaly"],
        ["TruFor manipulation score", line(det, "trufor", lambda d: f"{d['score']:.0%} (area flagged {d['flagged_area']:.0%})") if "trufor" in (det or {}) else "Not run"]]),
      card("Location (GPS)", "mod" if has_gps else "none", [["GPS",
        (f"{M['lat']:.5f}, {M['lon']:.5f}" if include_gps else "Found but hidden (consent needed)") if has_gps else "None embedded"]]),
    ]
    return {"V": {"v": v, "c": c, "why": why, "model_run": ai_score is not None}, "S": cards}
