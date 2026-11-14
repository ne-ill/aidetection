# Provenance: image forensics (no laptop needed)

## Deploy the backend (Render, free)
1. Create a GitHub repo and upload this whole folder (GitHub web: *Add file → Upload files*).
2. render.com → New → Blueprint → pick the repo. It reads `render.yaml` and builds `backend/Dockerfile`.
3. Copy the service URL, e.g. https://provenance-api.onrender.com. Open `/health` to check it.

## Deploy the frontend (Vercel/Netlify/Cloudflare Pages, free)
1. In `frontend/index.html` set `const API_URL='https://provenance-api.onrender.com';` (edit on GitHub web).
2. Import the repo on Vercel/Netlify with root directory `frontend`.
3. In Render, set `ALLOWED_ORIGINS` to your frontend URL.

## Optional: real AI detection (GPU)
Host a detector model (Hugging Face Space/Endpoint or Modal), then set Render env vars
`DETECTOR_URL` and `DETECTOR_TOKEN`. Expected reply: `[{"label":"artificial","score":0.93}]`.

Free Render servers sleep when idle; the first request can take ~1 minute.
