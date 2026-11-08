# Project 2: Vision + Text Clinical Decision Support API (Lab 8)

Production FastAPI service serving the Week 6 PyTorch vision model and Week 7 text evidence for clinical decision support.

---

## 1. System Overview & Architecture

The API exposes three primary endpoints:
- **`GET /health`**: Health check and model readiness probe reporting loaded classes and the critical target class.
- **`POST /predict`**: High-performance image classification endpoint with automated validation (image type, size, format).
- **`POST /analyze`**: Multimodal decision support endpoint combining the vision model prediction with clinical note parsing, automated PII redaction (masking CNIC and mobile numbers), and domain urgency detection.

### Production Engineering Highlights:
- **Lifespan Startup**: PyTorch model checkpoint and torchvision preprocessing transforms are initialized once on startup using FastAPI's `@asynccontextmanager lifespan`, avoiding per-request I/O and layer reconstruction overhead.
- **Non-blocking Concurrency**: CPU-bound inference (`classify`) is dispatched to a background threadpool via `fastapi.concurrency.run_in_threadpool`, keeping the async event loop free to handle concurrent requests (e.g., `/health` probes or incoming socket connections).
- **Defense in Depth & Privacy**: Automatic PII sanitization masks Pakistani CNIC patterns (`\b\d{5}-\d{7}-\d\b`) and mobile numbers (`(?:\+92[- ]?|\b0)3\d{2}[- ]?\d{7}\b`) before analysis or storage.
- **Robust Error Handling**: Explicit HTTP error responses without leaking stack traces:
  - `415 Unsupported Media Type` for non-JPEG/PNG uploads.
  - `413 Payload Too Large` for files exceeding 5 MB.
  - `400 Bad Request` for corrupted image binaries.
  - `422 Unprocessable Entity` for notes exceeding 2,000 characters or missing fields.
  - `401 Unauthorized` when optional `API_KEY` is enforced.

---

## 2. API Usage & Example Calls (Tested with New Sample Image)

### Active Sample Image Details
- **Filename**: `sample.jpeg`
- **File Size**: 866,312 bytes (~846 KB)
- **Resolution**: $1174 \times 793$ pixels (RGBA formatted)

### Run the Service
```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```
Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

### Example 1: Health Check (`GET /health`)
```bash
curl -X GET http://127.0.0.1:8000/health
```
**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "classes": ["images"],
  "positive": "PNEUMONIA"
}
```

### Example 2: Image Prediction (`POST /predict`)
```bash
curl -X POST http://127.0.0.1:8000/predict \
     -F "file=@sample.jpeg;type=image/jpeg"
```
**Response:**
```json
{
  "label": "images",
  "probabilities": {
    "images": 1.0
  },
  "latency_ms": 302.5
}
```

### Example 3: Multimodal Case Analysis with PII Masking (`POST /analyze`)
```bash
curl -X POST http://127.0.0.1:8000/analyze \
     -F "file=@sample.jpeg;type=image/jpeg" \
     -F "note=Patient has fever and cough. Contact phone: 0300-1234567, CNIC: 35202-1234567-1"
```
**Response:**
```json
{
  "prediction": {
    "label": "images",
    "probabilities": {
      "images": 1.0
    },
    "latency_ms": 100.5
  },
  "note_masked": "Patient has fever and cough. Contact phone: [PHONE], CNIC: [CNIC]",
  "fields": {},
  "urgent_terms": [
    "fever",
    "cough"
  ],
  "decision": "REFER",
  "reasons": [
    "note mentions: fever, cough"
  ],
  "disclaimer": "Decision support only; a qualified human decides."
}
```

---

## 3. Automated Test Suite (`test_api.py`)

All 8 automated tests pass against the new sample image, verifying endpoint contracts, schema adherence, mathematical parity with direct PyTorch model inference, file validation guards, and PII privacy leak prevention.

### Pytest Execution Output
```
pytest -q test_api.py
........                                                                 [100%]
8 passed in 29.31s
```

### Verbose Test Suite Summary
```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.2.0, pluggy-1.7.0 -- C:\Users\muham\AppData\Local\Programs\Python\Python311\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\muham\OneDrive\Desktop\Week_8_project
plugins: anyio-4.4.0
collecting ... collected 8 items

test_api.py::test_health PASSED                                          [ 12%]
test_api.py::test_predict_schema_and_probabilities PASSED                [ 25%]
test_api.py::test_parity_with_direct_model PASSED                        [ 37%]
test_api.py::test_rejects_wrong_type PASSED                              [ 50%]
test_api.py::test_rejects_too_large PASSED                               [ 62%]
test_api.py::test_rejects_corrupt_image PASSED                           [ 75%]
test_api.py::test_missing_file_is_422 PASSED                             [ 87%]
test_api.py::test_analyze_masks_pii PASSED                               [100%]

============================= 8 passed in 23.53s ==============================
```

---

## 4. Latency Benchmark & Capacity Planning (`bench.py`)

### Benchmarking Hardware Environment
- **CPU**: Intel64 Family 6 Model 142 Stepping 10 (4 physical cores, 8 logical processors)
- **RAM**: 7.88 GB DDR4
- **Operating System**: Windows 10 (x86_64)
- **Python**: 3.11.9, PyTorch 2.15.0+cpu, FastAPI 0.162.2

### Benchmark Results ($N = 50$ requests with new high-resolution sample image)
```
n=50  p50=251 ms  p95=522 ms  max=775 ms
Throughput with 1 worker ~ 3.5 requests/s
```

### Capacity Planning via Little's Law
- **Service Time ($W$)**: $p_{50} = 251\text{ ms} = 0.251\text{ seconds}$ per request.
- **Single Worker Concurrency ($L$)**: $L = 1$ concurrent request in service.
- **Throughput ($\lambda$)**:
  $$\lambda = \frac{L}{W} = \frac{1}{0.251\text{ s}} \approx 3.984\text{ requests/second}$$
- **Theoretical Max Requests per Day (100% Saturation)**:
  $$\text{Daily Capacity} = 3.984 \times 86,400\text{ seconds} \approx 344,223\text{ requests/day}$$
- **Realistic Sustained Daily Capacity (at 70% Target Utilization to prevent queueing delay)**:
  $$\text{Sustained Capacity} \approx 240,956\text{ requests/day per worker}$$

---

## 5. Input Validation & Error Handling (Beyond Standard Tests)

We tested edge cases and adversarial inputs to guarantee zero unhandled 500 exceptions:

| Test Input Scenario | HTTP Method & Path | Input Payload | HTTP Status Code | Response Body |
|---|---|---|---|---|
| **Non-Image Document (PDF)** | `POST /predict` | File `test.pdf` (`application/pdf`) | **415 Unsupported Media Type** | `{"detail": "Upload a JPEG or PNG image"}` |
| **Oversized Image (> 5MB)** | `POST /predict` | Binary payload of size 5.01 MB | **413 Payload Too Large** | `{"detail": "Image larger than 5 MB"}` |
| **Corrupted Image Bytes** | `POST /predict` | Arbitrary non-image byte stream with `.jpg` extension | **400 Bad Request** | `{"detail": "File is not a readable image"}` |
| **Excessive Note Length** | `POST /analyze` | Note string of 2,005 characters | **422 Unprocessable Entity** | `{"detail": "Note longer than 2,000 characters"}` |
| **Missing Upload File** | `POST /predict` | Empty multipart request body | **422 Unprocessable Entity** | Pydantic validation error |
| **Invalid/Missing API Key** | `POST /predict` | Request without `X-API-Key` (when `API_KEY` set) | **401 Unauthorized** | `{"detail": "Invalid or missing API key"}` |

---

## 6. Check Your Understanding Answers

### 1. A client uploads a PDF to `/predict`. Which status code should it get, and why not 500?
**Answer:** It should receive **`415 Unsupported Media Type`** (or `422 Unprocessable Entity` if content-negotiation fails validation). A **500 Internal Server Error** signifies an unhandled exception or bug inside the server codebase. The client uploading a PDF to an image inference endpoint is client error (bad input format), not a server failure. Returning 415 informs the client exactly what went wrong and protects downstream image decoders from attempting to parse non-image bytes.

### 2. Why is the model loaded in `lifespan` rather than inside the endpoint?
**Answer:** Deep neural network checkpoints (such as ResNet-18) are tens to hundreds of megabytes in size and take hundreds of milliseconds to several seconds to deserialize from disk, initialize layer parameters, and transfer to memory. If loaded inside the endpoint, **every incoming HTTP request would pay this disk I/O and allocation penalty**, degrading latency by orders of magnitude and causing massive garbage collection pressure. Loading once during the application `lifespan` startup caches the model in RAM so every subsequent request only performs in-memory forward inference.

### 3. $p_{50} = 40\text{ ms}$ and $p_{95} = 400\text{ ms}$. What does that tell you, and what would you check?
**Answer:** 
- **What it tells us:** The system exhibits high **tail latency** (a 10x disparity between median and 95th percentile). While the typical request is swift (40 ms), 5% of users experience severe delays (almost half a second).
- **What to check:**
  1. *Resource contention / Head-of-line blocking:* CPU or threadpool exhaustion where CPU-bound requests block other requests.
  2. *Garbage collection pauses:* Periodic Python GC cycles or memory allocations halting execution.
  3. *Input size variance:* Outlier images with large dimensions or complex decompression costs.
  4. *Cold starts / thread initialization:* Thread pool spin-up delays or lock contention.
  5. *Network jitter / client upload speeds:* Slower connections taking longer to transmit multipart data.

### 4. 500,000 requests a day, peak 5 times the average, 25 requests per second per worker. How many workers?
**Answer:**
1. **Average traffic rate ($\lambda_{\text{avg}}$):**
   $$\lambda_{\text{avg}} = \frac{500,000\text{ requests}}{86,400\text{ seconds/day}} \approx 5.79\text{ requests/second}$$
2. **Peak traffic rate ($\lambda_{\text{peak}}$):**
   $$\lambda_{\text{peak}} = 5 \times \lambda_{\text{avg}} = 5 \times 5.787 \approx 28.94\text{ requests/second}$$
3. **Required workers:**
   $$\text{Workers} = \left\lceil \frac{\lambda_{\text{peak}}}{\text{Capacity per worker}} \right\rceil = \left\lceil \frac{28.94}{25} \right\rceil = \lceil 1.158 \rceil = \mathbf{2\text{ workers}}$$
*(In production, deploying **3 workers** ($N+1$ redundancy) is recommended to maintain capacity during worker restarts or transient spikes).*

### 5. Why can model inference inside an `async def` endpoint slow down other requests?
**Answer:** Python `async def` endpoints execute concurrently on a single-threaded cooperative event loop. PyTorch inference (tensor math, matrix multiplication) is heavily **CPU-bound synchronous computation**. If executed directly inside an `async def` function without `await`, it monopolizes the single OS thread running the event loop. This starves the event loop, preventing it from processing I/O events, accepting new TCP connections, or handling lightweight endpoints like `/health` until inference finishes. Wrapping inference in `run_in_threadpool(classify, img)` offloads the heavy synchronous execution to a separate worker thread, leaving the main async event loop responsive.

### 6. How many communication channels does a 4-person team have?
**Answer:**
Using the channel formula $C = \frac{n(n - 1)}{2}$:
$$C = \frac{4(4 - 1)}{2} = \frac{4 \times 3}{2} = \mathbf{6\text{ channels}}$$
*(Between persons $\{A,B\}$, $\{A,C\}$, $\{A,D\}$, $\{B,C\}$, $\{B,D\}$, $\{C,D\}$).*

---

Next up: heading into Projects 3 & 4 focusing on LLMs, retrieval-augmented generation (RAG), and agentic workflows. Looking forward to teaming up! 💡

#MachineLearning #FastAPI #PyTorch #MLOps #AppliedAI #ClinicalAI #Python #SoftwareEngineering
```
