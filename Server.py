from gevent import monkey
monkey.patch_all()

from gevent import sleep

from flask import Flask, request, jsonify, Response, stream_with_context
import requests
from requests.adapters import HTTPAdapter
import os
import json
import re
import time
from collections import OrderedDict
from exa_py import Exa

app = Flask(__name__)

# 🔐 Use environment variable (recommended)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

EXA_API_KEY = os.getenv("EXA_API_KEY")
exa = Exa(EXA_API_KEY)

MAX_CACHE_SIZE = 100

# ✅ IMPROVED SESSION WITH CONNECTION POOLING
session = requests.Session()

adapter = HTTPAdapter(
    pool_connections=100,
    pool_maxsize=100
)

session.mount("http://", adapter)
session.mount("https://", adapter)

# ✅ IMPROVED FIFO CACHE
search_cache = OrderedDict()

# ✅ SYSTEM PROMPT
SYSTEM_PROMPT = """
You are Neuronix AI.

Rules:
- Be accurate and helpful.
- Format code properly using markdown.
- Keep answers clean and readable.
- Use web search results only if relevant.
- Ignore irrelevant web results.
- Never hallucinate facts.
- For coding answers, give properly formatted code blocks.
"""

ALLOWED_MODELS = {
    "mistral-small": "mistralai/mistral-small-2603",
    "mistral-medium": "mistralai/mistral-medium-3",
    "llama3": "meta-llama/llama-3-8b-instruct",
    "openai": "openai/gpt-3.5-turbo",
    "gpt4o-mini": "openai/gpt-4o-mini",
    "claude-haiku": "anthropic/claude-3-haiku",
    "claude-sonnet": "anthropic/claude-3-haiku",
    "deepseek-chat": "deepseek/deepseek-chat",
    "deepseek-coder": "deepseek/deepseek-chat-v3.1",
    "deepseek-reasoner": "deepseek/deepseek-r1",
    "gemma2-9b": "google/gemma-2-27b-it",
    "qwen-7b": "qwen/qwen-2.5-7b-instruct",
    "qwen-coder": "qwen/qwen-2.5-coder-32b-instruct",
    "qwq-32b": "qwen/qwen3.6-27b",
}

SEARCH_KEYWORDS = [
    # realtime
    "latest",
    "today",
    "live",
    "breaking",
    "recent",

    # news/events
    "news",
    "who won",

    # dynamic data
    "weather",
    "score",
    "stock price",
    "price of",
    "share price",

    # releases
    "release date",

    # sports
    "match result",
    "live score",

    # finance
    "market cap",

    # explicit search intent
    "search web",
    "search online"
]

# ✅ IMPROVED SMART SEARCH DETECTION
def needs_search_keywords(message):

    message = message.lower().strip()

    # ✅ Search keywords
    keyword_match = any(
        re.search(rf"\b{re.escape(keyword)}\b", message)
        for keyword in SEARCH_KEYWORDS
    )

    return keyword_match

def ai_needs_search(message):

    try:

        classifier_prompt = f"""
You are a search detection classifier.

Your task:
Determine whether answering the user's message requires real-time web search.

Examples requiring search:
- latest news
- current weather
- live scores
- stock prices
- recent events
- today's updates
- release dates
- online information

Examples NOT requiring search:
- explanations
- coding help
- math
- physics
- storytelling
- grammar
- historical facts
- general knowledge

Reply ONLY with:
YES
or
NO

User message:
{message}
"""

        response = session.post(

            "https://openrouter.ai/api/v1/chat/completions",

            headers={

                "Authorization":
                    f"Bearer {OPENROUTER_API_KEY}",

                "Content-Type":
                    "application/json",

                "HTTP-Referer":
                    "https://neuronix-backend-g7oq.onrender.com",

                "X-Title":
                    "Neuronix AI",
            },

            json={

                # FAST + CHEAP classifier model
                "model": "google/gemma-2-27b-it",

                "messages": [
                    {
                        "role": "user",
                        "content": classifier_prompt
                    }
                ],

                "temperature": 0,

                "max_tokens": 5
            },

            timeout=(5, 10)
        )

        if response.status_code != 200:
            return False

        data = response.json()

        reply = (
            data["choices"][0]["message"]["content"]
            .strip()
            .upper()
        )
        print("CLASSIFIER RAW:", reply)
        return reply.startswith("YES")

    except Exception as e:

        print("CLASSIFIER ERROR:", str(e))

        return False

# ✅ IMPROVED SEARCH FUNCTION
def search_web(query):

    # ✅ CACHE CHECK
    if query in search_cache:

        print("⚡ USING CACHED SEARCH")

        cached_result = search_cache.pop(query)
        search_cache[query] = cached_result

        return cached_result

    try:

        start_time = time.time()

        result = exa.search_and_contents(

            query,

            type="auto",

            num_results=5,

            text=True
        )

        print(
            f"⚡ Exa search completed in "
            f"{time.time() - start_time:.2f}s"
        )

        formatted_results = []

        for item in result.results:

            title = item.title or ""

            content = item.text or ""

            url = item.url or ""

            # ✅ CLEAN CONTENT
            content = re.sub(
                r"\s+",
                " ",
                content
            ).strip()

            # ✅ LIMIT SIZE
            content = content[:700]

            if not title and not content:
                continue

            formatted_results.append(
                f"""
SOURCE {len(formatted_results) + 1}

Title:
{title}

Snippet:
{content}

URL:
{url}
"""
            )

        final_result = "\n\n".join(formatted_results)

        # ✅ FIFO CACHE LIMIT
        if len(search_cache) >= MAX_CACHE_SIZE:
            search_cache.popitem(last=False)

        # ✅ SAVE CACHE
        search_cache[query] = final_result

        return final_result

    except Exception as e:

        print("EXA SEARCH ERROR:", str(e))

        return ""

@app.route("/chat", methods=["POST"])
def chat():

    try:

        # ✅ Safe JSON parsing
        data = request.get_json(silent=True)

        if not data or "message" not in data:
            return jsonify({
                "reply": "Invalid request"
            }), 400

        # ✅ Clean input
        user_message = data.get(
            "message",
            ""
        ).strip()

        model_id = data.get("model")

        if not user_message:
            return jsonify({
                "reply": "Empty message"
            }), 400

        if not model_id:
            return jsonify({
                "reply": "Model not provided"
            }), 400

        if model_id not in ALLOWED_MODELS:
            return jsonify({
                "reply": "Invalid model selected"
            }), 400

        model_to_use = ALLOWED_MODELS[model_id]

        # ✅ API key check
        if (
            not OPENROUTER_API_KEY or
            OPENROUTER_API_KEY == "YOUR_KEY"
        ):
            return jsonify({
                "reply": "Server not configured properly"
            }), 500

        # ✅ MESSAGES PAYLOAD
        messages_payload = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

        # ✅ SMART SEARCH SYSTEM

        should_search = False

        # 🔥 Layer 1 — Fast keyword detection
        if needs_search_keywords(user_message):

            print("⚡ KEYWORD SEARCH DETECTED")

            should_search = True


        elif len(user_message.split()) <= 1:
            should_search = False

        # 🔥 Layer 2 — AI classifier
        else:

            # ✅ Skip classifier for obvious static prompts
            static_keywords = [
                "explain",
                "code",
                "python",
                "java",
                "math",
                "physics",
                "chemistry",
                "essay",
                "story",
                "derive",
                "formula"
            ]

            is_static = any(
                keyword in user_message.lower()
                for keyword in static_keywords
            )

            if not is_static:

                print("🧠 RUNNING AI CLASSIFIER")

                should_search = ai_needs_search(user_message)

                print("CLASSIFIER RESULT:", should_search)

            else:

                print("⚡ STATIC PROMPT — SKIPPING CLASSIFIER")

        if should_search:

            print("🔎 SEARCHING WEB...")

            search_results = search_web(user_message)

            # ✅ SEARCH FAILED
            if not search_results:

                enhanced_prompt = f"""
User Question:
{user_message}

No reliable web results were found.

Instructions:
- Answer carefully.
- Avoid hallucinating facts.
- If uncertain, clearly say so.
"""

            else:

                enhanced_prompt = f"""
User Question:
{user_message}

Web Search Results:
{search_results}

Instructions:
- Use the search results only if relevant.
- Ignore irrelevant search results.
- Give a clean and accurate answer.
- Format properly using markdown.
- Do not mention that web search was used unless necessary.
"""

            messages_payload.append({
                "role": "user",
                "content": enhanced_prompt
            })

        else:

            messages_payload.append({
                "role": "user",
                "content": user_message
            })

        # 🔥 Call OpenRouter with streaming
        response = session.post(
            "https://openrouter.ai/api/v1/chat/completions",

            headers={

                "Authorization":
                    f"Bearer {OPENROUTER_API_KEY}",

                "Content-Type":
                    "application/json",

                "HTTP-Referer":
                    "https://neuronix-backend-g7oq.onrender.com",

                "X-Title":
                    "Neuronix AI",

                "Accept":
                    "text/event-stream",

                "User-Agent":
                    "NeuronixAI/1.0"
            },

            json={

                "model": model_to_use,

                "messages": messages_payload,

                "stream": True,

                # ✅ BETTER RESPONSE QUALITY
                "temperature": 0.7,

                # ✅ SAFETY LIMIT
                "max_tokens": 2000
            },

            stream=True,

            # ✅ BETTER TIMEOUTS
            timeout=(10, 60)
        )

        # ✅ Handle API errors early
        if response.status_code == 429:
            print("RATE LIMIT:", response.text)

            return jsonify({
                "reply":
                    "Model is busy or rate-limited. Please try another model."
            }), 429

        if response.status_code != 200:

            print("API ERROR:", response.text)

            try:
                error_data = response.json()

                error_message = (
                    error_data
                    .get("error", {})
                    .get("message", "Unknown AI error")
                )

            except Exception:
                error_message = "Unknown AI error"

            return jsonify({
                "reply": f"AI Error: {error_message}"
            }), response.status_code

        # 🔥 STREAMING GENERATOR
        @stream_with_context
        def generate():

            try:

                # ✅ ULTRA SMOOTH STREAMING
                for line in response.iter_lines(
                    decode_unicode=True,
                    chunk_size=1
                ):

                    if line:

                        if line.startswith("data: "):

                            data_str = line[6:].strip()

                            if data_str == "[DONE]":
                                break

                            try:

                                data_json = json.loads(
                                    data_str
                                )

                                delta = data_json.get(
                                    "choices",
                                    [{}]
                                )[0].get(
                                    "delta",
                                    {}
                                )

                                if "content" in delta:

                                    # ✅ STREAM TOKEN
                                    yield delta["content"]

                                    # ✅ ALLOW OTHER GREENLETS
                                    sleep(0)

                            except (
                                json.JSONDecodeError,
                                KeyError,
                                TypeError
                            ):
                                continue

            except Exception as stream_error:

                print(
                    "STREAM ERROR:",
                    str(stream_error)
                )

                yield (
                    "\n\n⚠️ Connection interrupted "
                    "while streaming response."
                )

            finally:

                response.close()

        # 🔥 RETURN STREAM
        return Response(

            generate(),

            content_type=
                "text/plain; charset=utf-8",

            headers={

                "Cache-Control":
                    "no-cache",

                "X-Accel-Buffering":
                    "no",

                "Connection":
                    "keep-alive"
            }
        )

    except requests.exceptions.Timeout:

        return jsonify({
            "reply":
                "AI request timed out"
        }), 504

    except requests.exceptions.RequestException:

        return jsonify({
            "reply":
                "Network error connecting to AI"
        }), 503

    except Exception as server_error:

        print(
            "SERVER ERROR:",
            str(server_error)
        )

        return jsonify({
            "reply":
                "Server error occurred"
        }), 500

if __name__ == "__main__":

    # ✅ Local development only
    app.run(
        host="0.0.0.0",
        port=5000,

        # ✅ Disable in production
        debug=False,

        threaded=True
    )
