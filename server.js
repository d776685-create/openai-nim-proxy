import express from "express";
import cors from "cors";
import axios from "axios";

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

// ---- NIM config ----
const NIM_API_BASE =
  process.env.NIM_API_BASE || "https://integrate.api.nvidia.com/v1";
const NIM_API_KEY = process.env.NIM_API_KEY;

// ---- model mapping (safe) ----
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'deepseek-ai/deepseek-v3.2',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'moonshotai/kimi-k2.5',
  'claude-3-opus': 'z-ai/glm4.7',
  'claude-3-sonnet': 'qwen/qwen3.5-397b-a17b',
  'gemini-pro': 'z-ai/glm5'
};

const FALLBACK_MODEL = "meta/llama-3.1-8b-instruct";

// ---- health ----
app.get("/health", (_, res) => {
  res.json({ status: "ok" });
});

// ---- models (OpenAI compatible) ----
app.get("/v1/models", (_, res) => {
  res.json({
    object: "list",
    data: Object.keys(MODEL_MAPPING).map(id => ({
      id,
      object: "model",
      owned_by: "nvidia-nim-proxy"
    }))
  });
});

// ---- chat completions ----
app.post("/v1/chat/completions", async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    const nimModel = MODEL_MAPPING[model];

    const nimRequest = {
      model: nimModel,
      messages,
      temperature: temperature ?? 0.7,
      max_tokens: max_tokens ?? 32000,
      stream: Boolean(stream)
    };

    const nimResponse = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          "Content-Type": "application/json",
          "Accept": stream ? "text/event-stream" : "application/json"
        },
        responseType: stream ? "stream" : "json",
        proxy: false
      }
    );

    // ---- STREAMING: PASS THROUGH ----
    if (stream) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");
      res.flushHeaders();

      // CRITICAL: raw passthrough (no parsing)
      nimResponse.data.pipe(res);

      nimResponse.data.on("error", () => res.end());
      return;
    }

    // ---- NON-STREAM ----
    res.json({
      id: `chatcmpl-${Date.now()}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model,
      choices: nimResponse.data.choices,
      usage: nimResponse.data.usage ?? {}
    });
  } catch (err) {
    console.error("Proxy error:", err?.response?.data || err.message);
    res.status(500).json({
      error: { message: "Upstream NIM error" }
    });
  }
});

// ---- Express 5 catch-all ----
app.use((req, res) => {
  res.status(404).json({ error: { message: "Not found" } });
});

app.listen(PORT, () => {
  console.log(`🚀 Proxy running on port ${PORT}`);
});
