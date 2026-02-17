// server.js — OpenAI → NVIDIA NIM Proxy (FINAL)

import express from "express";
import cors from "cors";
import axios from "axios";

const app = express();
const PORT = process.env.PORT || 3000;

// -------------------- Middleware --------------------
app.use(cors());
app.use(express.json());

// -------------------- NIM Config --------------------
const NIM_API_BASE =
  process.env.NIM_API_BASE || "https://integrate.api.nvidia.com/v1";
const NIM_API_KEY = process.env.NIM_API_KEY;

// -------------------- Feature Toggles --------------------
const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = false;

// -------------------- Model Mapping (SAFE) --------------------
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'deepseek-ai/deepseek-v3.2',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'moonshotai/kimi-k2.5',
  'claude-3-opus': 'z-ai/glm4.7',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'z-ai/glm5'
};

const FALLBACK_MODEL = "meta/llama-3.1-8b-instruct";

// -------------------- Health --------------------
app.get("/health", (_, res) => {
  res.json({
    status: "ok",
    service: "OpenAI → NVIDIA NIM Proxy",
    reasoning: SHOW_REASONING,
    thinking: ENABLE_THINKING_MODE
  });
});

// -------------------- Models --------------------
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

// -------------------- Chat Completions --------------------
app.post("/v1/chat/completions", async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    const nimModel = MODEL_MAPPING[model] || FALLBACK_MODEL;

    const nimRequest = {
      model: nimModel,
      messages,
      temperature: temperature ?? 0.6,
      max_tokens: max_tokens ?? 4096,
      stream: Boolean(stream),
      ...(ENABLE_THINKING_MODE && {
        extra_body: { chat_template_kwargs: { thinking: true } }
      })
    };

    const nimResponse = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          "Content-Type": "application/json"
        },
        responseType: stream ? "stream" : "json",
        proxy: false
      }
    );

    // ---------------- STREAMING ----------------
    if (stream) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");
      res.flushHeaders();

      let buffer = "";

      nimResponse.data.on("data", chunk => {
        buffer += chunk.toString();
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          if (line.includes("[DONE]")) {
            res.write("data: [DONE]\n\n");
            return;
          }

          try {
            const parsed = JSON.parse(line.slice(5));
            const delta = parsed.choices?.[0]?.delta ?? {};

            let content = delta.content ?? "";
            const reasoning = delta.reasoning_content ?? "";

            if (SHOW_REASONING && reasoning) {
              content = `<think>${reasoning}</think>\n\n${content}`;
            }

            parsed.choices[0].delta = { content };
            res.write(`data: ${JSON.stringify(parsed)}\n\n`);
          } catch {
            // ignore malformed chunks
          }
        }
      });

      nimResponse.data.on("end", () => res.end());
      nimResponse.data.on("error", () => res.end());
      return;
    }

    // ---------------- NON-STREAM ----------------
    const openaiResponse = {
      id: `chatcmpl-${Date.now()}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model,
      choices: nimResponse.data.choices.map(c => ({
        index: c.index,
        message: {
          role: "assistant",
          content: SHOW_REASONING && c.message?.reasoning_content
            ? `<think>${c.message.reasoning_content}</think>\n\n${c.message.content}`
            : c.message.content
        },
        finish_reason: c.finish_reason
      })),
      usage: nimResponse.data.usage ?? {}
    };

    res.json(openaiResponse);
  } catch (err) {
    console.error("Proxy error:", err?.response?.data || err.message);
    res.status(500).json({
      error: {
        message: "Upstream NIM error",
        type: "proxy_error"
      }
    });
  }
});

// -------------------- 404 --------------------
app.all("*", (_, res) => {
  res.status(404).json({ error: { message: "Not found" } });
});

// -------------------- Start --------------------
app.listen(PORT, () => {
  console.log(`🚀 Proxy running on port ${PORT}`);
});
