import express from "express";
import cors from "cors";
import axios from "axios";

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));


// ---- NIM Config ----
const NIM_API_BASE = process.env.NIM_API_BASE || "https://integrate.api.nvidia.com/v1";
const NIM_API_KEY = process.env.NIM_API_KEY;

// ---- Model Mapping ----
const MODEL_MAPPING = {
  "claude-opus-4": "z-ai/glm-5.1",              // Best agentic execution loop
  "claude-sonnet-4": "deepseek-ai/deepseek-v4-pro", // King of pure algorithmic logic
  "claude-3-7-sonnet": "minimaxai/minimax-m3",    // Massive context ingestion
  "claude-3-5-sonnet": "deepseek-ai/deepseek-v4-pro",
  "gpt-4o": "moonshotai/kimi-k2.6",
  "gpt-4": "deepseek-ai/deepseek-v3.2",
  "gpt-4-turbo": "z-ai/glm4.7"
};

const FALLBACK_MODEL = "meta/llama-3.1-8b-instruct";

// ---- Exponential Backoff Retry Interceptor ----
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function fetchWithRetry(url, data, config, retries = 3, backoff = 2000) {
  try {
    return await axios.post(url, data, config);
  } catch (err) {
    // If upstream NIM returns an HTTP 429 (Rate Limit Exhausted) and we have retries left
    if (err.response?.status === 429 && retries > 0) {
      console.warn(`⚠️ NVIDIA Rate limit hit. Retrying in ${backoff}ms... (${retries} retries left)`);
      await delay(backoff);
      // Double the waiting time for the next retry attempt
      return fetchWithRetry(url, data, config, retries - 1, backoff * 2);
    }
    throw err;
  }
}

// ---- Health Check ----
app.get("/health", (_, res) => {
  res.json({ status: "ok" });
});

// ---- Models Endpoint (OpenAI-compatible) ----
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

// ---- Chat Completions Handler ----
app.post("/v1/chat/completions", async (req, res) => {
  try {
    // 1. Destructure incoming parameters including tools
    const { model, messages, temperature, max_tokens, stream, tools, tool_choice } = req.body;

    const nimModel = MODEL_MAPPING[model] || FALLBACK_MODEL;

    const nimRequest = {
      model: nimModel,
      messages,
      temperature: temperature ?? 0.7,
      max_tokens: max_tokens ?? 32000,
      stream: Boolean(stream),
    };

    // 2. Conditionally pass downstream tool calls directly to NVIDIA
    if (tools) nimRequest.tools = tools;
    if (tool_choice) nimRequest.tool_choice = tool_choice;

    // 3. Inject model-specific chat configurations
    if (nimModel.includes('deepseek') || nimModel.includes('kimi')) {
        nimRequest.chat_template_kwargs = {
            "thinking": true,
            "reasoning_effort": 1
        };
    } else if (nimModel.includes('glm') || nimModel.includes('qwen') || nimModel.includes('nemotron')) {
        nimRequest.chat_template_kwargs = {
            enable_thinking: true
        };
    }

    // 4. Send request using the retry interceptor wrapper
    const nimResponse = await fetchWithRetry(
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

    // ---- STREAMING PIPELINE ----
    if (stream) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");
      res.flushHeaders();

      // Raw pass-through seamlessly pipes text or tool stream chunks to OpenCode
      nimResponse.data.pipe(res);

      nimResponse.data.on("error", () => res.end());
      return;
    }

    // ---- NON-STREAMING RESPONSE ----
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
      error: { message: "Upstream NIM error occurred" }
    });
  }
});

// ---- Catch-all router ----
app.use((req, res) => {
  res.status(404).json({ error: { message: "Endpoint configuration not found" } });
});

app.listen(PORT, () => {
  console.log(`🚀 Proxy cleanly executing on port ${PORT}`);
});
