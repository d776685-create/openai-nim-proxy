import express from "express";
import cors from "cors";
import axios from "axios";

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));


// ---- NIM config ----
const NIM_API_BASE =
  process.env.NIM_API_BASE || "https://integrate.api.nvidia.com/v1";
const NIM_API_KEY = process.env.NIM_API_KEY;

// ---- model mapping (safe) ----
const MODEL_MAPPING = {
  "glm-5.2": "z-ai/glm-5.2",
  "gpt-4": "deepseek-ai/deepseek-v4-pro",
  "minimax-m3": "minimaxai/minimax-m3",
  "kimi-2.6": "moonshotai/kimi-k2.6",
  "deepseek-v4-flash": "deepseek-ai/deepseek-v4-flash",
  "glm4.7": "z-ai/glm4.7",
  "nemotron3": "nvidia/nemotron-3-ultra-550b-a55b"
};

const FALLBACK_MODEL = "z-ai/glm-5.1";

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
    // 1. Extract tools and tool_choice from incoming payload
    const { model, messages, temperature, max_tokens, stream, tools, tool_choice } = req.body;

    // Use fallback safely if client passes a model outside the mapping list
    const nimModel = MODEL_MAPPING[model] || FALLBACK_MODEL;

    const nimRequest = {
      model: nimModel,
      messages,
      temperature: temperature ?? 0.7,
      max_tokens: max_tokens ?? 32000,
      stream: Boolean(stream),
    };

    // 2. Forward tool parameters conditionally if provided by the client
    if (tools) nimRequest.tools = tools;
    if (tool_choice) nimRequest.tool_choice = tool_choice;

    // Conditionally add chat_template_kwargs based on model type
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

      // CRITICAL: Raw passthrough works perfectly for tool-calling stream chunks too!
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
      // MiniMax's tool_calls array will naturally sit inside this choices response block
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
