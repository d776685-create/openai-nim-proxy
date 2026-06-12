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
  "claude-opus-4": "z-ai/glm-5.1",
  "claude-sonnet-4": "deepseek-ai/deepseek-v4-pro",
  "claude-3-7-sonnet": "minimaxai/minimax-m3",
  "claude-3-5-sonnet": "deepseek-ai/deepseek-v4-pro",
  "gpt-4o": "moonshotai/kimi-k2.6",
  "gpt-4": "deepseek-ai/deepseek-v3.2",
  "gpt-4-turbo": "z-ai/glm4.7"
};

const FALLBACK_MODEL = "meta/llama-3.1-8b-instruct";

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function fetchWithRetry(url, data, config, retries = 3, backoff = 1500) {
  try {
    return await axios.post(url, data, config);
  } catch (err) {
    if (err.response?.status === 429 && retries > 0) {
      console.warn(`⚠️ NVIDIA Rate limit hit. Retrying in ${backoff}ms...`);
      await delay(backoff);
      return fetchWithRetry(url, data, config, retries - 1, backoff * 2);
    }
    throw err;
  }
}

// ---- CRITICAL FIX: CHAT HISTORY SANITIZER ----
// Normalizes historical tool metadata to prevent upstream vLLM engine crashes
function sanitizeHistoryForNIM(messages) {
  return messages.map(msg => {
    const cleanMsg = { role: msg.role };
    
    // Ensure text content is a flat string to avoid tokenization parsing drops
    if (Array.isArray(msg.content)) {
      cleanMsg.content = msg.content
        .map(c => (c.type === 'text' ? c.text : ''))
        .join('\n');
    } else {
      cleanMsg.content = msg.content || "";
    }

    // Flatten tool call definitions if they exist in the history
    if (msg.tool_calls) {
      cleanMsg.tool_calls = msg.tool_calls.map(tc => ({
        id: tc.id,
        type: "function",
        function: {
          name: tc.function.name,
          arguments: typeof tc.function.arguments === 'object' 
            ? JSON.stringify(tc.function.arguments) 
            : tc.function.arguments
        }
      }));
    }

    if (msg.tool_call_id) {
      cleanMsg.tool_call_id = msg.tool_call_id;
      cleanMsg.name = msg.name;
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

app.post("/v1/chat/completions", async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream, tools, tool_choice } = req.body;
    const nimModel = MODEL_MAPPING[model] || FALLBACK_MODEL;

    // Standardize input bounds to mitigate token cutoff exceptions
    const safeMaxTokens = max_tokens && max_tokens <= 4096 ? max_tokens : 4096;

    const nimRequest = {
      model: nimModel,
      messages: sanitizeHistoryForNIM(messages), // Pass through our cleaning block
      temperature: temperature ?? 0.5,
      max_tokens: safeMaxTokens,
      stream: Boolean(stream),
    };

    if (tools && tools.length > 0) {
      nimRequest.tools = tools;
      if (tool_choice) nimRequest.tool_choice = tool_choice;
    }

    // Suppress experimental template formatting properties if tools are active
    if (!tools || tools.length === 0) {
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

      nimResponse.data.pipe(res);
      nimResponse.data.on("error", () => res.end());
      return;
    }

    res.json({
      id: `chatcmpl-${Date.now()}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model,
      choices: nimResponse.data.choices, 
      usage: nimResponse.data.usage ?? {}
    });

  } catch (err) {
    const statusCode = err.response?.status;
    const errorData = err.response?.data;

    console.error(`❌ Proxy Error [HTTP ${statusCode || 'NET_ERR'}]:`, errorData || err.message);

    // Auto-swap fallback: If long history triggers a backend failure on your active model,
    // clear the thinking template and immediately throw it to MiniMax-M3 to maintain session persistence.
    if ((statusCode === 500 || statusCode === 400 || statusCode === 429) && req.body.model !== "claude-3-7-sonnet") {
      console.warn(`🔄 Session degradation detected on primary node. Seamlessly routing context to MiniMax-M3 backend...`);
      try {
        req.body.model = "claude-3-7-sonnet";
        const fallbackResponse = await axios.post(`http://localhost:${PORT}/v1/chat/completions`, req.body);
        return res.json(fallbackResponse.data);
      } catch (fallbackErr) {
        console.error("🚨 Critical Error: Session fallback failed.");
      }
    }

    res.status(500).json({ error: { message: "Upstream NIM error occurred" } });
  }
});

app.use((req, res) => res.status(404).json({ error: { message: "Endpoint not found" } }));

app.listen(PORT, () => {
  console.log(`🚀 Proxy cleanly executing on port ${PORT}`);
});
