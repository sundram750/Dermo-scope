import base64
import json
from http.server import BaseHTTPRequestHandler
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import ai_edge_litert.interpreter as tflite
except ImportError:  # pragma: no cover - reported by the health endpoint.
    tflite = None


IMG_SIZE = (224, 224)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "dermo-scope.tflite"

CLASS_INFO = {
    "akiec": {
        "full_name": "Actinic Keratoses / Intraepithelial Carcinoma",
        "risk": "High",
    },
    "bcc": {
        "full_name": "Basal Cell Carcinoma",
        "risk": "High",
    },
    "bkl": {
        "full_name": "Benign Keratosis",
        "risk": "Low",
    },
    "df": {
        "full_name": "Dermatofibroma",
        "risk": "Low",
    },
    "mel": {
        "full_name": "Melanoma",
        "risk": "High",
    },
    "nv": {
        "full_name": "Melanocytic Nevi",
        "risk": "Low",
    },
    "vasc": {
        "full_name": "Vascular Lesions",
        "risk": "Low",
    },
}

CLASS_NAMES = sorted(CLASS_INFO.keys())
_interpreter = None
_input_details = None
_output_details = None


def get_interpreter():
    global _interpreter, _input_details, _output_details

    if tflite is None:
        raise RuntimeError("ai-edge-litert is not installed.")
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file is missing: {MODEL_PATH}")

    if _interpreter is None:
        _interpreter = tflite.Interpreter(model_path=str(MODEL_PATH))
        _interpreter.allocate_tensors()
        _input_details = _interpreter.get_input_details()
        _output_details = _interpreter.get_output_details()

    return _interpreter, _input_details, _output_details


def decode_image(image_value):
    if not isinstance(image_value, str):
        raise ValueError("Expected an image data URL.")

    encoded = image_value.split(",", 1)[1] if "," in image_value else image_value
    raw = base64.b64decode(encoded)
    image = Image.open(BytesIO(raw)).convert("RGB")
    image = image.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def adapt_input(array, input_detail):
    target_dtype = input_detail["dtype"]
    if target_dtype == np.float32:
        return array.astype(np.float32)

    scale, zero_point = input_detail.get("quantization", (0.0, 0))
    if not scale:
        return array.astype(target_dtype)

    quantized = array / scale + zero_point
    limits = np.iinfo(target_dtype)
    quantized = np.clip(np.rint(quantized), limits.min, limits.max)
    return quantized.astype(target_dtype)


def adapt_output(array, output_detail):
    output = array.astype(np.float32).reshape(-1)
    scale, zero_point = output_detail.get("quantization", (0.0, 0))
    if scale:
        output = (output - zero_point) * scale

    output = np.maximum(output, 0)
    total = float(output.sum())
    if total <= 0:
        return np.ones(len(CLASS_NAMES), dtype=np.float32) / len(CLASS_NAMES)
    return output / total


def predict(image_value):
    interpreter, input_details, output_details = get_interpreter()
    input_array = decode_image(image_value)
    input_tensor = adapt_input(input_array, input_details[0])

    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()

    raw_output = interpreter.get_tensor(output_details[0]["index"])[0]
    scores = adapt_output(raw_output, output_details[0])
    ranked = sorted(
        ((class_name, float(scores[index])) for index, class_name in enumerate(CLASS_NAMES)),
        key=lambda item: item[1],
        reverse=True,
    )
    top_class, confidence = ranked[0]
    info = CLASS_INFO[top_class]

    return {
        "predicted_class": top_class,
        "full_name": info["full_name"],
        "confidence": confidence,
        "risk": info["risk"],
        "all_probs": ranked,
    }


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        ok = tflite is not None and MODEL_PATH.exists()
        payload = {
            "ok": ok,
            "runtime": "ai-edge-litert",
            "model": MODEL_PATH.name,
            "model_bytes": MODEL_PATH.stat().st_size if MODEL_PATH.exists() else 0,
        }
        if not ok:
            payload["error"] = "LiteRT package or model file is unavailable."
        self._json(200 if ok else 503, payload)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))
            image_value = payload.get("image")
            if not image_value:
                self._json(400, {"error": "Missing image field."})
                return

            self._json(200, predict(image_value))
        except Exception as exc:  # Keep failures readable in the client.
            self._json(500, {"error": str(exc)})

    def _json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
