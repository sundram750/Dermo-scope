# Dermo-Scope Lite

This is a lighter Vercel-ready rebuild of the original `sid` project. The
original app uses Streamlit plus Python TFLite inference. This version serves a
static browser app and sends image tensors to a small Vercel Python function
that runs `saved_model.tflite` with LiteRT.

## What is improved

- No Streamlit server or full TensorFlow runtime.
- No install or build step required for the app shell.
- Upload and camera capture run directly in the browser.
- The existing TFLite model is reused from `models/dermo-scope.tflite`.
- The inference API uses `ai-edge-litert` instead of full TensorFlow.
- The UI includes risk grouping, class probabilities, and downloadable reports.
- A clearly marked demo fallback appears if the browser cannot load the ML
  API.

## Project layout

```text
dermo-scope-next/
  index.html
  pyproject.toml
  requirements.txt
  vercel.json
  api/
    index.py
  models/
    dermo-scope.tflite
  src/
    main.js
    styles.css
```

## Run locally

From this folder:

```bash
python -m http.server 4173
```

Open:

```text
http://localhost:4173
```

The static local server does not run the Vercel Python API, so the UI will use
demo scores locally unless you run through Vercel's Python function runtime.
Camera capture requires a secure context in most browsers. It works on
`localhost` and on HTTPS deployments such as Vercel.

## Deploy to Vercel

Import this folder as a Vercel project or run `vercel` from this directory.
There is no frontend build command. Vercel serves the app as static files and
deploys `api/index.py` as the Python Function entrypoint.

## Runtime notes

The API dependencies are intentionally small:

- `ai-edge-litert==2.1.4`
- `fastapi`
- `numpy`
- `Pillow`

The browser only loads Lucide icons from CDN. No React, Streamlit, TensorFlow,
or OpenCV is needed for the deployed app shell.

The original Streamlit Grad-CAM promise is not reproduced because the deployed
artifact is a TFLite model and this lightweight API does not expose gradients.

## Medical disclaimer

This is an educational project only. It is not a medical diagnostic tool. Any
concerning lesion should be reviewed by a qualified clinician.
