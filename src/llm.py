import json
import subprocess
import requests

class OllamaClient:
    def __init__(self, base="http://localhost:11434"):
        self.base = base.rstrip("/")

    def list_models_http(self, timeout=2.5):
        try:
            url = f"{self.base}/api/tags"
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            data = resp.json() or {}
            return [m.get("name") for m in data.get("models", []) if m.get("name")]
        except Exception:
            return []

    def list_models_cli(self):
        candidates = [
            "ollama list --json",
            "ollama list --format json",
            "ollama list",
        ]
        for cmd in candidates:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, shell=True, timeout=3)
                out = result.stdout.strip()
                if result.returncode != 0 or not out:
                    continue
                if out.lstrip().startswith("["):
                    data = json.loads(out)
                    names = [row.get("name") for row in data if isinstance(row, dict) and row.get("name")]
                    if names:
                        return names
                if "{" in out or out.lstrip().startswith("{"):
                    names = []
                    for line in out.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            name = obj.get("name")
                            if name:
                                names.append(name)
                        except Exception:
                            pass
                    if names:
                        return names
                lines = [ln for ln in out.splitlines() if ln.strip()]
                body = [ln for ln in lines if not ln.upper().startswith("NAME") and not set(ln.strip()) <= set("- ")]
                names = []
                for ln in body:
                    parts = ln.split()
                    if parts:
                        names.append(parts[0])
                if names:
                    return names
            except Exception:
                continue
        return []

    def list_models(self):
        models = self.list_models_http()
        if not models:
            models = self.list_models_cli()
        if not models:
            models = ["llama3", "llama3:8b", "llama3.1", "qwen2.5", "mistral", "phi4"]
        return sorted(dict.fromkeys(models))

    def chat(self, system_prompt: str, user_content: str, model: str, stream: bool = False, timeout: int = 600) -> str:
        url = f"{self.base}/api/chat"
        payload = {
            "model": model,
            "stream": stream,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ],
        }
        if stream:
            resp = requests.post(url, json=payload, timeout=timeout, stream=True)
            resp.raise_for_status()
            out = []
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line.decode("utf-8"))
                part = ((chunk.get("message") or {}).get("content") or "")
                out.append(part)
            return "".join(out)
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return (data.get("message") or {}).get("content", "")
