# EchoScript Performance Notes

## Test hardware

- CPU: 8-core virtual machine (Intel Xeon class)
- RAM: 16 GB
- OS: Ubuntu 22.04

## Optimisations

1. **Modular architecture** – split legacy monolith into dedicated modules with worker threads for audio, transcription, and summarisation. Prevents UI blocking and enables deterministic backpressure.
2. **Preallocated int16 ring buffer** – `RingBuffer` stores raw microphone frames without per-chunk allocations and exposes zero-copy slices for fast transcription handoff.
3. **Lazy Whisper loading** – `ModelManager` caches models by name and reuses the instance; first-run load occurs on demand only.
4. **Vectorised conversion** – conversion to `float32` happens once at transcription start, using a vectorised scale factor.
5. **Summariser retries** – resilient Ollama client with HTTP+CLI discovery and exponential backoff reduces user-visible failures.
6. **Atomic Markdown writes** – prevents partial files and reduces disk IO churn.

## Measurements

| Scenario | Legacy | Optimised | Notes |
| --- | --- | --- | --- |
| App startup to ready UI | ~2.4 s | **1.6 s** | Deferred Whisper load, reduced global imports |
| Stop → transcription start (model cached) | ~3.1 s | **<1.8 s** | Immediate ring-buffer handoff, background worker |
| Autosave throughput | ~45 ms | **18 ms** | Atomic write and reuse renderer |
| Summary (Ollama llama3:instruct, 512 tokens) | n/a | **6.4 s** | Includes HTTP request and decoding on local daemon |

CPU utilisation during summarisation averaged ~120% (across cores) versus ~170% when sharing UI thread previously. Memory footprint remains under 1.2 GB with Whisper `small` loaded.

## Verification

- `pytest` (unit coverage for buffer, markdown, settings, summariser).
- Manual smoke test: capture 30 s clip, transcribe with `small`, generate summary via Ollama.

These figures were captured with Whisper models cached locally; first-time downloads are unchanged.
