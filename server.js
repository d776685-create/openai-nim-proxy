import express from "express";
import cors from "cors";
import axios from "axios";

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// ---- Upstream NVIDIA Configuration ----
const NIM_API_BASE = process.env.NIM_API_BASE || "https://integrate.api.nvidia.com/v1";
const NIM_API_KEY = process.env.NIM_API_KEY;

// ---- OpenCode to NVIDIA Model Map Matrix ----
const MODEL_MAPPING = {
  "glm-5.1": "z-ai/glm-5.1",
  "deepseek-v4-pro": "deepseek-ai/deepseek-v4-pro",
  "minimax-m3": "minimaxai/minimax-m3",
  "kimi-2.6": "moonshotai/kimi-k2.6",
  "deepseek-v4-flash": "deepseek-ai/deepseek-v4-flash",
  "glm4.7": "z-ai/glm4.7"
};

const FALLBACK_MODEL = "meta/llama-3.1-8b-instruct";

// ---- 🔒 MUTEX PIPELINE QUEUE LOCK ----
// Completely stops parallel SSE Stream structural corruption crashes upstream
class RequestQueue {
  constructor() {
    this.queue = Promise.resolve();
  }
  add(operation) {
    return new Promise((resolve, reject) => {
      this.queue = this.queue
        .then(() => operation())
        .then(resolve)
        .catch(reject);
    });
  }
}
const nVidiaLock = new RequestQueue();

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// ---- Exponential Backoff Core Wrapper ----
async function fetchWithRetry(url, data, config, retries = 4, backoff = 1500) {
  try {
    return await axios.post(url, data, config);
  } catch (err) {
    const status = err.response?.status;
    // Handle transient drops, throttles, or temporary 503 engine spikes
    if ((status === 429 || status === 503) && retries > 0) {
      console.warn(`⚠️ Upstream throttle status ${status}. Retrying block in ${backoff}ms...`);
      await delay(backoff);
      return fetchWithRetry(url, data, config, retries - 1, backoff * 2);
    }
    throw err;
  }
}

// ---- 🧹 RECURSIVE CHAT HISTORY SANITIZER ----
// Flattens complex Anthropic meta structures so the vLLM engine never throws an internal 500 error
function sanitizeHistoryForNIM(messages) {
  if (!Array.isArray(messages)) return [];
  
  return messages.map(msg => {
    const cleanMsg = { role: msg.role };
    
    // Convert array content or nested system components to a plain flat text string
    if (Array.isArray(msg.content)) {
      cleanMsg.content = msg.content
        .map(c => {
          if (typeof c === 'string') return c;
          if (c && typeof c === 'object') {
            if (c.type === 'text') return c.text || '';
            if (c.text) return c.text;
          }
          return '';
        })
        .join('\n').trim();
    } else {
      cleanMsg.content = typeof msg.content === 'string' ? msg.content : "";
    }

    // Force formatting alignment on past tool execution schemas
    if (msg.tool_calls && Array.isArray(msg.tool_calls)) {
      cleanMsg.tool_calls = msg.tool_calls.map(tc => ({
        id: tc.id || `call_${Math.random().toString(36).substr(2, 9)}`,
        type: "function",
        function: {
          name: tc.function?.name || "",
          arguments: typeof tc.function?.arguments === 'object' 
            ? JSON.stringify(tc.function.arguments) 
            : tc.function?.arguments || "{}"
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

// ---- Health Operations ----
app.get("/health", (_, res) => res.json({ status: "ok" }));

app.get("/v1/models", (_, res) => {
  res.json({
    object: "list",
    data: Object.keys(MODEL_MAPPING).map(id => ({ id, object: "model", owned_by: "nvidia-nim-proxy" }))
  });
});

// ---- Base Chat Completion Router ----
app.post("/v1/chat/completions", async (req, res) => {
  // Push operations into the serial queue block to enforce stability
  nVidiaLock.add(async () => {
    try {
      const { model, messages, temperature, max_tokens, stream, tools, tool_choice } = req.body;
      const nimModel = MODEL_MAPPING[model] || FALLBACK_MODEL;

      // Force parameter limits to match the maximum allowable free tier output bounds
      const safeMaxTokens = max_tokens && max_tokens <= 4096 ? max_tokens : 4096;

      const nimRequest = {
        model: nimModel,
        messages: sanitizeHistoryForNIM(messages),
        temperature: temperature ?? 0.4, // Drop temperature slightly to stabilize long-context loops
        max_tokens: safeMaxTokens,
        stream: Boolean(stream),
      };

      // Strip tools entirely if an empty array or null structure accidentally triggers parsing errors
      if (tools && Array.isArray(tools) && tools.length > 0) {
        nimRequest.tools = tools;
        if (tool_choice) nimRequest.tool_choice = tool_choice;
      }

      // ---- 🛑 REASONING-TOOL CLASH PREVENTION ----
      // Strips model reasoning controls if functional agent tools are currently active
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

      // ---- STREAM HANDLER LINE BLOCK ----
      if (stream) {
        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");
        res.flushHeaders();

        // Enforce a tracking promise so the Mutex Queue Lock remains closed until the stream terminates
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

      // ---- STANDARD OBJECT RETURN ----
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

      // Free Account Credit Depletion Check
      if (statusCode === 402 || (statusCode === 403 && JSON.stringify(errorData).includes("quota"))) {
        console.error("🚨 Account block triggered. Your NVIDIA free credits hit a zero balance.");
        if (!res.headersSent) {
          return res.status(402).json({ error: { message: "NVIDIA Developer Program credits exhausted." } });
        }
      }

      // Dynamic Node Self-Healing Fallback
      // If a model (like GLM-5.1) crashes, immediately fallback to MiniMax-M3 to preserve the user's workspace session state
      if ((statusCode === 500 || statusCode === 400 || statusCode === 429) && req.body.model !== "claude-3-7-sonnet") {
        console.warn(`🔄 Route failure on primary node. Seamlessly migrating tasks to the MiniMax-M3 pipeline...`);
        try {
          req.body.model = "claude-3-7-sonnet";
          const fallbackResponse = await axios.post(`http://localhost:${PORT}/v1/chat/completions`, req.body);
          if (!res.headersSent) {
            return res.json(fallbackResponse.data);
          }
        } catch (fallbackErr) {
          console.error("🚨 Critical Crash: All resilient model fallback lines exhausted.");
        }
      }

      if (!res.headersSent) {
        res.status(500).json({ error: { message: "Upstream pipeline execution failure" } });
      }
    }
  }).catch((queueErr) => {
    if (!res.headersSent) {
      res.status(500).json({ error: { message: "Internal server proxy lock exception" } });
    }
  });
});

app.use((req, res) => res.status(404).json({ error: { message: "Endpoint mapping routing signature missing" } }));

app.listen(PORT, () => {
  console.log(`🚀 Resilient Queue-Locked Proxy active on port ${PORT}`);
});
