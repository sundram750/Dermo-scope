const API_URL = "./api/predict";
const IMAGE_SIZE = 224;
const LUCIDE_VERSION = "0.515.0";

const CLASS_INFO = {
  akiec: {
    label: "Actinic Keratoses / Intraepithelial Carcinoma",
    risk: "High",
    description:
      "A high-review class associated with sun-damaged or pre-cancerous lesions.",
    recommendation:
      "Arrange professional review promptly, especially if the lesion is changing, bleeding, or painful.",
  },
  bcc: {
    label: "Basal Cell Carcinoma",
    risk: "High",
    description:
      "A high-review class associated with a common form of skin cancer.",
    recommendation:
      "Use this as a reason to seek clinician review rather than as confirmation.",
  },
  bkl: {
    label: "Benign Keratosis",
    risk: "Low",
    description: "A lower-review class commonly associated with benign growths.",
    recommendation:
      "Monitor for change and seek care if the lesion evolves or becomes symptomatic.",
  },
  df: {
    label: "Dermatofibroma",
    risk: "Low",
    description: "A lower-review class often associated with firm benign growths.",
    recommendation:
      "Monitor over time and compare against the ABCDE checklist.",
  },
  mel: {
    label: "Melanoma",
    risk: "High",
    description:
      "A high-review class associated with the most serious type of skin cancer.",
    recommendation:
      "Seek professional medical review urgently if this aligns with clinical signs.",
  },
  nv: {
    label: "Melanocytic Nevi",
    risk: "Low",
    description: "A lower-review class commonly associated with moles.",
    recommendation:
      "Continue routine skin checks and watch for asymmetry, border, color, size, or change.",
  },
  vasc: {
    label: "Vascular Lesions",
    risk: "Low",
    description: "A lower-review class commonly associated with vascular marks.",
    recommendation:
      "Seek care if it changes quickly, bleeds frequently, or becomes painful.",
  },
};

const CLASS_NAMES = Object.keys(CLASS_INFO).sort();

const state = {
  activeMode: "upload",
  cameraStream: null,
  hasImage: false,
  isBusy: false,
  lastResult: null,
  apiReady: false,
  demoMode: false,
};

const els = {
  analyzeButton: document.querySelector("#analyzeButton"),
  cameraPanel: document.querySelector("#cameraPanel"),
  cameraPreview: document.querySelector("#cameraPreview"),
  captureButton: document.querySelector("#captureButton"),
  confidenceMetric: document.querySelector("#confidenceMetric"),
  downloadButton: document.querySelector("#downloadButton"),
  dropZone: document.querySelector("#dropZone"),
  fileInput: document.querySelector("#fileInput"),
  imageCanvas: document.querySelector("#imageCanvas"),
  inputHint: document.querySelector("#inputHint"),
  insightPanel: document.querySelector("#insightPanel"),
  modeMetric: document.querySelector("#modeMetric"),
  predictionCode: document.querySelector("#predictionCode"),
  previewFrame: document.querySelector(".preview-frame"),
  probabilityList: document.querySelector("#probabilityList"),
  resetButton: document.querySelector("#resetButton"),
  resultsHeading: document.querySelector("#results-heading"),
  reviewMetric: document.querySelector("#reviewMetric"),
  riskBadge: document.querySelector("#riskBadge"),
  runtimeDetail: document.querySelector("#runtimeDetail"),
  runtimePill: document.querySelector("#runtimePill"),
  startCameraButton: document.querySelector("#startCameraButton"),
  stopCameraButton: document.querySelector("#stopCameraButton"),
  topClassLabel: document.querySelector("#topClassLabel"),
  uploadPanel: document.querySelector("#uploadPanel"),
};

const canvasContext = els.imageCanvas.getContext("2d", { willReadFrequently: true });

init();

async function init() {
  clearCanvas();
  renderEmptyProbabilities();
  bindEvents();
  setControls();
  loadIcons();
  state.apiReady = true;
  setRuntimeStatus(
    "ready",
    "Standby",
    "Inference uses the Vercel LiteRT API when available."
  );
  setControls();
}

function bindEvents() {
  document.querySelectorAll("[data-mode]").forEach((button) => {
    button.addEventListener("click", () => setMode(button.dataset.mode));
  });

  els.fileInput.addEventListener("change", (event) => {
    const [file] = event.target.files;
    if (file) {
      handleImageFile(file);
    }
  });

  els.dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    els.dropZone.classList.add("dragging");
  });

  els.dropZone.addEventListener("dragleave", () => {
    els.dropZone.classList.remove("dragging");
  });

  els.dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    els.dropZone.classList.remove("dragging");
    const [file] = event.dataTransfer.files;
    if (file && file.type.startsWith("image/")) {
      handleImageFile(file);
    }
  });

  els.analyzeButton.addEventListener("click", analyzeCurrentImage);
  els.startCameraButton.addEventListener("click", startCamera);
  els.captureButton.addEventListener("click", captureCameraFrame);
  els.stopCameraButton.addEventListener("click", stopCamera);
  els.downloadButton.addEventListener("click", downloadReport);
  els.resetButton.addEventListener("click", resetReview);
}

function loadScript(src) {
  const existing = document.querySelector(`script[src="${src}"]`);
  if (existing) {
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = src;
    script.async = true;
    script.onload = resolve;
    script.onerror = () => reject(new Error(`Failed to load ${src}`));
    document.head.appendChild(script);
  });
}

async function loadIcons() {
  try {
    await loadScript(
      `https://cdn.jsdelivr.net/npm/lucide@${LUCIDE_VERSION}/dist/umd/lucide.min.js`
    );
    if (window.lucide) {
      window.lucide.createIcons();
    }
  } catch (error) {
    console.warn("Lucide icons could not load.", error);
  }
}

function setMode(mode) {
  state.activeMode = mode;
  const isCamera = mode === "camera";
  els.uploadPanel.hidden = isCamera;
  els.cameraPanel.hidden = !isCamera;

  document.querySelectorAll("[data-mode]").forEach((button) => {
    const active = button.dataset.mode === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });

  if (!isCamera) {
    stopCamera();
  }

  setControls();
}

async function handleImageFile(file) {
  const url = URL.createObjectURL(file);
  const image = new Image();

  image.onload = async () => {
    drawSourceToCanvas(image);
    URL.revokeObjectURL(url);
    state.hasImage = true;
    setControls();
    await analyzeCurrentImage();
  };

  image.onerror = () => {
    URL.revokeObjectURL(url);
    els.inputHint.textContent = "Could not read that image file.";
  };

  image.src = url;
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    els.inputHint.textContent = "Camera access is not available in this browser.";
    return;
  }

  try {
    state.cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    });
    els.cameraPreview.srcObject = state.cameraStream;
    await els.cameraPreview.play();
  } catch (error) {
    els.inputHint.textContent = "Camera permission was blocked or unavailable.";
    console.error(error);
  } finally {
    setControls();
  }
}

async function captureCameraFrame() {
  if (!els.cameraPreview.videoWidth) {
    return;
  }

  drawSourceToCanvas(els.cameraPreview);
  state.hasImage = true;
  setControls();
  await analyzeCurrentImage();
}

function stopCamera() {
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach((track) => track.stop());
  }
  state.cameraStream = null;
  els.cameraPreview.srcObject = null;
  setControls();
}

async function analyzeCurrentImage() {
  if (!state.hasImage || state.isBusy) {
    return;
  }

  state.isBusy = true;
  setControls();
  els.inputHint.textContent = "Analyzing image...";

  try {
    const result = state.apiReady
      ? await runServerInference()
      : buildDemoPrediction();
    state.lastResult = result;
    renderResult(result);
    els.inputHint.textContent = state.demoMode
      ? "Demo scores are placeholders because live runtime is unavailable."
      : "Analysis complete.";
  } catch (error) {
      state.demoMode = true;
    const result = buildDemoPrediction();
    state.lastResult = result;
    renderResult(result);
    setRuntimeStatus(
      "error",
      "Demo",
      "Inference API failed. Placeholder scores are shown instead."
    );
    els.inputHint.textContent = "Inference API failed. Showing demo scores.";
    console.error(error);
  } finally {
    state.isBusy = false;
    setControls();
  }
}

async function runServerInference() {
  const image = els.imageCanvas.toDataURL("image/jpeg", 0.92);
  const response = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image }),
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `Inference API returned ${response.status}`);
  }

  return buildResultFromApi(payload);
}

function buildDemoPrediction() {
  const pixels = canvasContext.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE).data;
  let checksum = 0;
  for (let index = 0; index < pixels.length; index += 37) {
    checksum = (checksum + pixels[index]) % 997;
  }

  const predictedIndex = checksum % CLASS_NAMES.length;
  const values = CLASS_NAMES.map((_, index) => {
    const distance = Math.abs(predictedIndex - index) + 1;
    return 1 / distance;
  });
  return buildResult(normalize(values), "Demo only");
}

function buildResultFromApi(payload) {
  const ranked = payload.all_probs.map(([className, score]) => ({
    className,
    score: Number(score),
    info: CLASS_INFO[className],
  }));
  const top = ranked[0];

  return {
    predictedClass: top.className,
    fullName: top.info.label,
    confidence: top.score,
    risk: top.info.risk,
    description: top.info.description,
    recommendation: top.info.recommendation,
    ranked,
    modeLabel: "Vercel API",
    timestamp: new Date(),
  };
}

function buildResult(values, modeLabel) {
  const normalized = normalize(values);
  const ranked = normalized
    .map((score, index) => ({
      className: CLASS_NAMES[index],
      score,
      info: CLASS_INFO[CLASS_NAMES[index]],
    }))
    .sort((a, b) => b.score - a.score);

  const top = ranked[0];
  return {
    predictedClass: top.className,
    fullName: top.info.label,
    confidence: top.score,
    risk: top.info.risk,
    description: top.info.description,
    recommendation: top.info.recommendation,
    ranked,
    modeLabel,
    timestamp: new Date(),
  };
}

function normalize(values) {
  const safeValues = values.map((value) => (Number.isFinite(value) ? Math.max(value, 0) : 0));
  const total = safeValues.reduce((sum, value) => sum + value, 0);
  if (total <= 0) {
    return safeValues.map(() => 1 / safeValues.length);
  }
  return safeValues.map((value) => value / total);
}

function renderResult(result) {
  const isHigh = result.risk === "High";
  els.resultsHeading.textContent = result.fullName;
  els.predictionCode.textContent = `${result.predictedClass.toUpperCase()} class output`;
  els.confidenceMetric.textContent = formatPercent(result.confidence);
  els.reviewMetric.textContent = isHigh ? "High" : "Lower";
  els.modeMetric.textContent = result.modeLabel;
  els.topClassLabel.textContent = result.predictedClass.toUpperCase();

  els.riskBadge.textContent = isHigh ? "High review" : "Lower review";
  els.riskBadge.className = `risk-badge ${isHigh ? "risk-high" : "risk-low"}`;

  els.insightPanel.className = `insight-panel ${isHigh ? "high" : "low"}`;
  els.insightPanel.innerHTML = `
    <span data-lucide="${isHigh ? "shield-alert" : "shield-check"}" aria-hidden="true"></span>
    <p><strong>${escapeHtml(result.description)}</strong> ${escapeHtml(result.recommendation)}</p>
  `;

  renderProbabilities(result.ranked);
  if (window.lucide) {
    window.lucide.createIcons();
  }
}

function renderProbabilities(items) {
  els.probabilityList.replaceChildren(
    ...items.map((item) => {
      const row = document.createElement("div");
      const riskClass = item.info.risk === "High" ? "high" : "low";
      row.className = "probability-item";
      row.innerHTML = `
        <div class="probability-meta">
          <span><strong>${escapeHtml(item.className.toUpperCase())}</strong> ${escapeHtml(item.info.label)}</span>
          <span>${formatPercent(item.score)}</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill ${riskClass}" style="width: ${Math.round(item.score * 100)}%"></div>
        </div>
      `;
      return row;
    })
  );
}

function renderEmptyProbabilities() {
  const items = CLASS_NAMES.map((className) => ({
    className,
    score: 0,
    info: CLASS_INFO[className],
  }));
  renderProbabilities(items);
}

function resetReview() {
  stopCamera();
  clearCanvas();
  state.hasImage = false;
  state.lastResult = null;
  els.fileInput.value = "";
  els.resultsHeading.textContent = "Awaiting image";
  els.predictionCode.textContent = "Model output will appear here.";
  els.confidenceMetric.textContent = "--";
  els.reviewMetric.textContent = "--";
  els.modeMetric.textContent = state.demoMode ? "Demo only" : "Vercel API";
  els.topClassLabel.textContent = "None";
  els.riskBadge.textContent = "Pending";
  els.riskBadge.className = "risk-badge neutral";
  els.insightPanel.className = "insight-panel";
  els.insightPanel.innerHTML = `
    <span data-lucide="shield-alert" aria-hidden="true"></span>
    <p>Results are informational only. Use the score as a prompt for clinical review, not as a diagnosis.</p>
  `;
  els.inputHint.textContent = "Choose an image to enable analysis.";
  renderEmptyProbabilities();
  setControls();
  if (window.lucide) {
    window.lucide.createIcons();
  }
}

function clearCanvas() {
  canvasContext.clearRect(0, 0, IMAGE_SIZE, IMAGE_SIZE);
  canvasContext.fillStyle = "#101722";
  canvasContext.fillRect(0, 0, IMAGE_SIZE, IMAGE_SIZE);
  els.previewFrame.classList.remove("has-image");
}

function drawSourceToCanvas(source) {
  const width = source.videoWidth || source.naturalWidth || source.width;
  const height = source.videoHeight || source.naturalHeight || source.height;
  const side = Math.min(width, height);
  const sx = Math.max(0, (width - side) / 2);
  const sy = Math.max(0, (height - side) / 2);

  canvasContext.clearRect(0, 0, IMAGE_SIZE, IMAGE_SIZE);
  canvasContext.drawImage(source, sx, sy, side, side, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
  els.previewFrame.classList.add("has-image");
}

function downloadReport() {
  if (!state.lastResult) {
    return;
  }

  const report = buildReport(state.lastResult);
  const blob = new Blob([report], { type: "text/markdown" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "dermo-scope-report.md";
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function buildReport(result) {
  const rows = result.ranked
    .map(
      (item) =>
        `| ${item.className.toUpperCase()} | ${item.info.label} | ${formatPercent(item.score)} | ${item.info.risk} |`
    )
    .join("\n");

  return `# Dermo-Scope Lite Report

Generated: ${result.timestamp.toLocaleString()}

## Top prediction

- Class: ${result.predictedClass.toUpperCase()}
- Name: ${result.fullName}
- Confidence: ${formatPercent(result.confidence)}
- Review level: ${result.risk}
- Mode: ${result.modeLabel}

## Recommendation

${result.recommendation}

## All class probabilities

| Class | Name | Probability | Review |
|---|---|---:|---|
${rows}

## Disclaimer

This report is educational only and is not a medical diagnosis. Consult a qualified clinician for any skin concern.
`;
}

function setControls() {
  const hasCamera = Boolean(state.cameraStream);
  els.analyzeButton.disabled = !state.hasImage || state.isBusy;
  els.downloadButton.disabled = !state.lastResult;
  els.captureButton.disabled = !hasCamera || state.isBusy;
  els.startCameraButton.disabled = hasCamera;
  els.stopCameraButton.disabled = !hasCamera;
  els.modeMetric.textContent = state.demoMode ? "Demo only" : "Vercel API";
}

function setRuntimeStatus(kind, label, detail) {
  els.runtimePill.textContent = label;
  els.runtimePill.className = `status-pill status-${kind}`;
  els.runtimeDetail.textContent = detail;
}

function formatPercent(value) {
  return `${Math.round(value * 1000) / 10}%`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
