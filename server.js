import express from "express";
import cors from "cors";
import axios from "axios";

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));

const NIM_API_BASE = process.env.NIM_API_BASE || "https://integrate.api.nvidia.com/v1";
const NIM_API_KEY = process.env.NIM_API_KEY;

const MODEL_MAPPING = {
  "glm-5.1": "z-ai/glm-5.1",
  "deepseek-v4-pro": "deepseek-ai/deepseek-v4-pro",
  "minimax-m3": "minimaxai/minimax-m3",
  "kimi-2.6": "moonshotai/kimi-k2.6",
  "deepseek-v4-flash": "deepseek-ai/deepseek-v4-flash",
  "glm4.7": "z-ai/glm4.7"
};

const FALLBACK_MODEL = "meta/llama-3.1-8b-instruct";

// ---- 🔒 LIGHTWEIGHT TEXT-STREAM LOCK ----
// Only locks heavy output generations, allowing directory lookups to run instantly
class StreamLock {
  constructor() {
    this.activeLock = Promise.resolve();
  }
  acquire(operation) {
    return new Promise((resolve, reject) => {
      this.activeLock = this.activeLock
        .then(() => operation())
        .then(resolve)
        .catch(reject);
    });
  }
}
const textStreamLock = new StreamLock();

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function fetchWithRetry(url, data, config, retries = 3, backoff = 1000) {
  try {
    return await axios.post(url, data, config);
  } catch (err) {
    const status = err.response?.status;
    if ((status === 429 || status === 503) && retries > 0) {
      await delay(backoff);
      return fetchWithRetry(url, data, config, retries - 1, backoff * 1.5);
    }
    throw err;
  }
}

function sanitizeHistoryForNIM(messages) {
  if (!Array.isArray(messages)) return [];
  return messages.map(msg => {
    const cleanMsg = { role: msg.role };
    if (Array.isArray(msg.content)) {
      cleanMsg.content = msg.content.map(c => typeof c === 'string' ? c : (c?.text || '')).join('\n').trim();
    } else {
      cleanMsg.content = typeof msg.content === 'string' ? msg.content : "";
    }
    if (msg.tool_calls && Array.isArray(msg.tool_calls)) {
      cleanMsg.tool_calls = msg.tool_calls.map(tc => ({
        id: tc.id || `call_${Math.random().toString(36).substr(2, 9)}`,
        type: "function",
        function: {
          name: tc.function?.name || "",
          arguments: typeof tc.function?.arguments === 'object' ? JSON.stringify(tc.function.arguments) : (tc.function?.arguments || "{}")
        }
      }));
    }
    if (msg.tool_call_id) {
      cleanMsg.tool_call_id = msg.tool_call_id;
      cleanMsg.name = msg.name || "tool_response";
    }
    return cleanMsg;
  });
}

app.get("/health", (_, res) => res.json({ status: "ok" }));

app.get("/v1/models", (_, res) => {
  res.json({
    object: "list",
    data: Object.keys(MODEL_MAPPING).map(id => ({ id, object: "model", owned_by: "nvidia-nim-proxy" }))
  });
});

// ---- Optimized Base Chat Completions Handler ----
app.post("/v1/chat/completions", async (req, res) => {
  const { model, messages, temperature, max_tokens, stream, tools, tool_choice } = req.body;
  
  // ⚡ HIGH-SPEED SPEED RUN ROUTE: 
  // If OpenCode is doing a rapid background check or metadata call (non-streaming tools check), 
  // completely bypass the queue lock and execute it instantly in parallel!
  const isParallelSafe = !stream && (!tools || tools.length === 0);

  const executeRequest = async () => {
    try {
      const nimModel = MODEL_MAPPING[model] || FALLBACK_MODEL;
      const safeMaxTokens = max_tokens && max_tokens <= 4096 ? max_tokens : 4096;

      const nimRequest = {
        model: nimModel,
        messages: sanitizeHistoryForNIM(messages),
        temperature: temperature ?? 0.4,
        max_tokens: safeMaxTokens,
        stream: Boolean(stream),
      };

      if (tools && Array.isArray(tools) && tools.length > 0) {
        nimRequest.tools = tools;
        if (tool_choice) nimRequest.tool_choice = tool_choice;
      }

      if (!nimRequest.tools) {
        if (nimModel.includes('deepseek') || nimModel.includes('kimi')) {
          nimRequest.chat_template_kwargs = { "thinking": true, "reasoning_effort": 1 };
        } else if (nimModel.includes('glm')) {
          nimRequest.chat_template_kwargs = { "enable_thinking": true };
        }
      }

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

      if (stream) {
        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");
        res.flushHeaders();

        await new Promise((resolveStream, rejectStream) => {
          nimResponse.data.pipe(res);
          nimResponse.data.on("end", () => resolveStream());
          nimResponse.data.on("error", (err) => {
            res.end();
            rejectStream(err);
          });
        });
        return;
      }

      return res.json({
        id: `chatcmpl-${Date.now()}`,
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model,
        choices: nimResponse.data.choices, 
        usage: nimResponse.data.usage || {}
      });

    } catch (err) {
      const statusCode = err.response?.status;
      const errorData = err.response?.data;

      console.error(`❌ Proxy Router Failure [HTTP ${statusCode || 'NET_ERR'}]:`, errorData || err.message);

      if (statusCode === 402 || (statusCode === 403 && JSON.stringify(errorData).includes("quota"))) {
        if (!res.headersSent) {
          return res.status(402).json({ error: { message: "NVIDIA Developer Program credits exhausted." } });
        }
      }

      if ((statusCode === 500 || statusCode === 400 || statusCode === 429) && req.body.model !== "claude-3-7-sonnet") {
        console.warn(`🔄 Route failure. Switching backend line execution to MiniMax-M3...`);
        try {
          req.body.model = "claude-3-7-sonnet";
          const fallbackResponse = await axios.post(`http://localhost:${PORT}/v1/chat/completions`, req.body);
          if (!res.headersSent) return res.json(fallbackResponse.data);
        } catch (fallbackErr) {
          console.error("🚨 Critical Crash: All resilient model fallback lines exhausted.");
        }
      }

      if (!res.headersSent) {
        res.status(500).json({ error: { message: "Upstream pipeline execution failure" } });
      }
    }
  };

  // ---- Smart Traffic Distribution Routing ----
  if (isParallelSafe) {
    // If it's a structural call, fire it parallel instantly!
    return executeRequest();
  } else {
    // If it's a heavy text/tool generation stream, drop it in the line lock
    return textStreamLock.acquire(executeRequest).catch((err) => {
      if (!res.headersSent) res.status(500).json({ error: { message: "Pipeline error" } });
    });
  }
});

app.use((req, res) => res.status(404).json({ error: { message: "Endpoint signature missing" } }));

app.listen(PORT, () => {
  console.log(`🚀 Optimized High-Speed Resilient Proxy active on port ${PORT}`);
});
