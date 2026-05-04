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
from tavily import TavilyClient

app = Flask(__name__)

# 🔐 Use environment variable (recommended)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily = TavilyClient(api_key=TAVILY_API_KEY)

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
    "mistral-small": "mistralai/mistral-small-24b-instruct-2501",
    "mistral-medium": "mistralai/mixtral-8x7b-instruct",
    "llama3": "meta-llama/llama-3-8b-instruct",
    "openai": "openai/gpt-3.5-turbo",
    "gpt4o-mini": "openai/gpt-4o-mini",
    "claude-haiku": "anthropic/claude-3-haiku",
    "claude-sonnet": "anthropic/claude-3-sonnet",
    "deepseek-chat": "deepseek/deepseek-chat",
    "deepseek-coder": "deepseek/deepseek-coder",
    "deepseek-reasoner": "deepseek/deepseek-r1",
    "gemma2-9b": "google/gemma-2-9b-it",
    "qwen-7b": "qwen/qwen-2.5-7b-instruct",
    "qwen-coder": "qwen/qwen-2.5-coder-32b-instruct",
    "qwq-32b": "qwen/qwq-32b-preview",
}

SEARCH_KEYWORDS = [
    "latest",
    "today",
    "news",
    "current",
    "weather",
    "score",
    "price",
    "update",
    "recent",
    "who won",
    "search",
    "internet",
    "online",
    "live",
    "stock",
    "release date",
    "breaking",
]

STATIC_KNOWLEDGE_KEYWORDS = [
    "explain",
    "meaning",
    "definition",
    "formula",
    "theory",
    "derive",
]

# ✅ IMPROVED SMART SEARCH DETECTION
def needs_search(message):

    message = message.lower().strip()

    # ✅ Ignore very short messages
    if len(message) < 8:
        return False

    # ✅ Static knowledge bypass
    if any(
        keyword in message
        for keyword in STATIC_KNOWLEDGE_KEYWORDS
    ):
        return False

    # ✅ Search keywords
    keyword_match = any(
        keyword in message
        for keyword in SEARCH_KEYWORDS
    )

    # ✅ Dynamic patterns
    dynamic_patterns = [
        r"\bwho won\b",
        r"\btoday\b",
        r"\blatest\b",
        r"\brecent\b",
        r"\bcurrent\b",
        r"\blive\b",
        r"\bnews\b",
        r"\bprice\b",
        r"\bweather\b",
        r"\bscore\b",
        r"\bstock\b",
        r"\brelease date\b",
    ]

    pattern_match = any(
        re.search(pattern, message)
        for pattern in dynamic_patterns
    )

    return keyword_match or pattern_match

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

        result = tavily.search(
            query=query,
            search_depth="basic",
            max_results=5,
            timeout=10
        )

        print(
            f"⚡ Search completed in "
            f"{time.time() - start_time:.2f}s"
        )

        formatted_results = []

        for item in result.get("results", []):

            title = item.get("title", "")

            content = item.get("content", "")

            # ✅ CLEAN CONTENT
            content = re.sub(
                r"\s+",
                " ",
                content
            ).strip()

            # ✅ LIMIT SIZE
            content = content[:500]

            url = item.get("url", "")

            # ✅ SKIP EMPTY RESULTS
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

        print("SEARCH ERROR:", str(e))

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

        # ✅ SMART SEARCH
        if needs_search(user_message):

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
                    "http://localhost",

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
                "max_tokens": 4000
            },

            stream=True,

            # ✅ BETTER TIMEOUTS
            timeout=(10, 60)
        )

        # ✅ Handle API errors early
        if response.status_code == 429:

            return jsonify({
                "reply":
                    "AI service is busy, try again later"
            }), 429

        if response.status_code != 200:

            print("API ERROR:", response.text)

            return jsonify({
                "reply":
                    "Error from AI service"
            }), 500

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