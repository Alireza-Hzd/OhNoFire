"""
demo_client.py (OpenRouter / Free Models Version)

Connects to mcp_server.py over stdio, exposes its tool schema to a free LLM 
via OpenRouter (OpenAI-compatible client), executes the model's requested 
tool call against CDSE, and streams the grounded result back for a final answer.

Requires:
    OPENROUTER_API_KEY set in environment
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY set (for CDSE S3 access)
"""
import os
import pyproj

# Dynamically set PROJ_LIB to pyproj's active data directory
os.environ["PROJ_LIB"] = pyproj.datadir.get_data_dir()
os.environ["PROJ_DATA"] = pyproj.datadir.get_data_dir()

import asyncio
import json
import os
import sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
import logging

logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
# Hard-mute botocore and network loggers
for logger_name in ["botocore", "botocore.credentials", "urllib3", "s3transfer"]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.ERROR)
    logger.disabled = True
    logger.propagate = False
os.environ["GDAL_HTTP_MAX_RETRY"] = "10"
os.environ["GDAL_HTTP_RETRY_DELAY"] = "3"
# Free model choices on OpenRouter:
# - "openrouter/free": Auto-routes to an available free model that supports function calling
# - "minimax/minimax-m2.5:free": MiniMax free model
# - "google/gemini-2.0-flash-exp:free": Gemini 2.0 Flash experimental (free)
# - "meta-llama/llama-3.3-70b-instruct:free": Llama 3.3 70B (free)

# 1. Validate API Key
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError(
        "OPENROUTER_API_KEY environment variable is missing. "
        "Run `export OPENROUTER_API_KEY='sk-or-v1-...'` in terminal."
    )

# 2. Resolve script directory so server path is always correct
SCRIPT_DIR = Path(__file__).parent.resolve()
SERVER_SCRIPT = SCRIPT_DIR / "mcp_server.py"

server_params = StdioServerParameters(
    command=sys.executable,
    args=[str(SERVER_SCRIPT)],
    env=os.environ.copy(),  # Passes AWS and OpenRouter credentials to subprocess
)

OPENROUTER_MODEL = "openrouter/free"

QUESTION = (
    "There was a wildfire near Terzigno, on the Mount Somma side of Vesuvius "
    "National Park, Italy, active 2025-08-07 to 2025-08-18. Using a bounding box "
    "of roughly 14.425,40.792 to 14.510,40.850, and pre-fire imagery from "
    "2025-05-09 to 2025-08-06 and post-fire imagery from 2025-08-19 to "
    "2025-11-16, how many hectares burned in total, and which areas should be "
    "prioritized for erosion-control work before the rainy season?"
    "Provide the final analysis in English."
)

async def mcp_tools_to_openai_schema(session: ClientSession) -> list[dict]:
    """Convert MCP tool schemas into OpenAI-compatible function schemas."""
    listed = await session.list_tools()
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.input_schema,  # Uses snake_case input_schema
            },
        }
        for t in listed.tools
    ]

async def run_demo(question: str) -> None:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await mcp_tools_to_openai_schema(session)
            messages = [{"role": "user", "content": question}]

            response = client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=messages,
                tools=tools,
            )

            msg = response.choices[0].message

            # Handle tool calls if requested by the model
            if msg.tool_calls:
                messages.append(msg)
                for tool_call in msg.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)
                    print(f"[model called tool] {fn_name}({fn_args})")

                    tool_result = await session.call_tool(fn_name, fn_args)

                    result_text = "\n".join(
                        c.text for c in tool_result.content if hasattr(c, "text")
                    )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text,
                    })

                # Fetch final answer from model using tool outputs
                final_response = client.chat.completions.create(
                    model=OPENROUTER_MODEL,
                    messages=messages,
                )
                print("\n=== Grounded answer ===")
                print(final_response.choices[0].message.content)
            else:
                print(msg.content)

if __name__ == "__main__":
    asyncio.run(run_demo(QUESTION))