import express from "express";
import cors from "cors";
import axios from "axios";

const app = express();

app.use(cors());

app.use(express.json({
  limit: "100mb"
}));

app.use(express.urlencoded({
  extended: true,
  limit: "100mb"
}));

const PORT =
  process.env.PORT || 3000;

const NIM_API_BASE =
  process.env.NIM_API_BASE ||
  "https://integrate.api.nvidia.com/v1";

const NIM_API_KEY =
  process.env.NIM_API_KEY;

const MODEL_MAPPING = {
  "claude-sonnet-4":
    "deepseek-ai/deepseek-v4-pro",

  "claude-haiku-4":
    "deepseek-ai/deepseek-v4-flash",

  "claude-opus-4":
    "z-ai/glm-5.1"
};

const DEFAULT_MODEL =
  "deepseek-ai/deepseek-v4-pro";

function anthropicToOpenAI(
  messages = []
) {
  return messages.map(msg => {
    if (
      !Array.isArray(msg.content)
    ) {
      return msg;
    }

    const text =
      msg.content
        .filter(
          x =>
            x.type === "text"
        )
        .map(x => x.text)
        .join("\n");

    return {
      role: msg.role,
      content: text
    };
  });
}

function getMappedModel(
  model
) {
  return (
    MODEL_MAPPING[model] ||
    DEFAULT_MODEL
  );
}

function buildThinkingConfig(
  model
) {
  if (
    model.includes(
      "deepseek"
    ) ||
    model.includes("kimi")
  ) {
    return {
      chat_template_kwargs: {
        thinking: true,
        reasoning_effort: 1
      }
    };
  }

  if (
    model.includes("glm") ||
    model.includes("qwen") ||
    model.includes(
      "nemotron"
    )
  ) {
    return {
      chat_template_kwargs: {
        enable_thinking: true
      }
    };
  }

  return {};
}

app.get(
  "/health",
  (_, res) => {
    res.json({
      status: "ok"
    });
  }
);

app.get(
  "/v1/models",
  (_, res) => {
    res.json({
      data: [
        {
          type: "model",
          id: "claude-sonnet-4",
          display_name:
            "Claude Sonnet 4"
        },
        {
          type: "model",
          id: "claude-haiku-4",
          display_name:
            "Claude Haiku 4"
        },
        {
          type: "model",
          id: "claude-opus-4",
          display_name:
            "Claude Opus 4"
        }
      ],
      has_more: false
    });
  }
);

app.post(
  "/v1/messages/count_tokens",
  (req, res) => {
    const text =
      JSON.stringify(
        req.body.messages || []
      );

    const estimated =
      Math.ceil(
        text.length / 4
      );

    res.json({
      input_tokens:
        estimated
    });
  }
);

app.post(
  "/v1/messages",
  async (req, res) => {
    try {
      const {
        model,
        messages,
        max_tokens,
        temperature,
        stream
      } = req.body;

      const mappedModel =
        getMappedModel(
          model
        );

      const nimRequest = {
        model: mappedModel,
        messages:
          anthropicToOpenAI(
            messages
          ),
        max_tokens:
          max_tokens ||
          32000,
        temperature:
          temperature ??
          0.7,
        stream:
          Boolean(stream),
        ...buildThinkingConfig(
          mappedModel
        )
      };

      const response =
        await axios.post(
          `${NIM_API_BASE}/chat/completions`,
          nimRequest,
          {
            headers: {
              Authorization:
                `Bearer ${NIM_API_KEY}`,
              "Content-Type":
                "application/json",
              Accept: stream
                ? "text/event-stream"
                : "application/json"
            },
            responseType:
              stream
                ? "stream"
                : "json",
            proxy: false
          }
        );

      if (stream) {
        res.setHeader(
          "Content-Type",
          "text/event-stream"
        );

        res.setHeader(
          "Cache-Control",
          "no-cache"
        );

        res.setHeader(
          "Connection",
          "keep-alive"
        );

        res.flushHeaders();

        res.write(
          `event: message_start\n`
        );

        res.write(
          `data: ${JSON.stringify(
            {
              type:
                "message_start",
              message: {
                id:
                  "msg_" +
                  Date.now(),
                type:
                  "message",
                role:
                  "assistant",
                model
              }
            }
          )}\n\n`
        );

        response.data.on(
          "data",
          chunk => {
            const raw =
              chunk.toString();

            const lines =
              raw
                .split("\n")
                .filter(
                  l =>
                    l.startsWith(
                      "data:"
                    )
                );

            for (const line of lines) {
              const payload =
                line.replace(
                  "data:",
                  ""
                );

              if (
                payload.trim() ===
                "[DONE]"
              ) {
                continue;
              }

              try {
                const json =
                  JSON.parse(
                    payload
                  );

                const delta =
                  json.choices?.[0]
                    ?.delta
                    ?.content;

                if (
                  !delta
                )
                  continue;

                res.write(
                  `event: content_block_delta\n`
                );

                res.write(
                  `data: ${JSON.stringify(
                    {
                      type:
                        "content_block_delta",
                      index: 0,
                      delta: {
                        type:
                          "text_delta",
                        text: delta
                      }
                    }
                  )}\n\n`
                );
              } catch {}
            }
          }
        );

        response.data.on(
          "end",
          () => {
            res.write(
              `event: message_stop\n`
            );

            res.write(
              `data: {}\n\n`
            );

            res.end();
          }
        );

        response.data.on(
          "error",
          () => {
            res.end();
          }
        );

        return;
      }

      const text =
        response.data
          ?.choices?.[0]
          ?.message
          ?.content ||
        "";

      res.json({
        id:
          "msg_" +
          Date.now(),

        type: "message",

        role:
          "assistant",

        model,

        content: [
          {
            type: "text",
            text
          }
        ],

        stop_reason:
          "end_turn",

        stop_sequence:
          null,

        usage: {
          input_tokens:
            response.data
              ?.usage
              ?.prompt_tokens ||
            0,

          output_tokens:
            response.data
              ?.usage
              ?.completion_tokens ||
            0
        }
      });
    } catch (error) {
      console.error(
        error?.response?.data ||
          error.message
      );

      res
        .status(500)
        .json({
          type:
            "error",

          error: {
            type:
              "api_error",

            message:
              "Upstream NVIDIA error"
          }
        });
    }
  }
);

app.use(
  (req, res) => {
    res
      .status(404)
      .json({
        error: {
          message:
            "Not found"
        }
      });
  }
);

app.listen(
  PORT,
  () => {
    console.log(
      `Gateway listening on ${PORT}`
    );
  }
