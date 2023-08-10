# References
# https://huggingface.co/spaces/matthoffner/wizardcoder-ggml/blob/main/main.py
# https://github.com/abacaj/replit-3B-inference/blob/main/inference.py
import json
import os
from typing import Callable, List, Dict, Any, Generator
from functools import partial

import fastapi
import uvicorn
from fastapi import HTTPException, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from anyio import create_memory_object_stream
from anyio.to_thread import run_sync
from ctransformers import AutoModelForCausalLM
from pydantic import BaseModel
import argparse

from dataclasses import dataclass, asdict

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    required=False,
    default="abacaj/Replit-v2-CodeInstruct-3B-ggml",
    help="Path to model file or name of Huggingface model. (Default: abacaj/Replit-v2-CodeInstruct-3B-ggml)",
)
parser.add_argument(
    "-t",
    "--model_type",
    required=False,
    default="replit",
    help="Type of model (e.g. replit, starcoder. Default: replit)",
)
parser.add_argument(
    "-p",
    "--port",
    required=False,
    default=8000,
    help="Port to run the inference server in (Default: 8000)",
)
parser.add_argument(
    "-d",
    "--debug",
    required=False,
    default=False,
    help="Whether to log the prompts and responses. (Default: False)",
)

args = parser.parse_args()

# "./models/teknium-Replit-v2-CodeInstruct-3B-ggml-q4_1.bin",
# model_file="WizardCoder-15B-1.0.ggmlv3.q5_0.bin",
# "TheBloke/WizardCoder-15B-1.0-GGML",
model_path = args.model
# model_type="starcoder",
# model_type="replit",
model_type = args.model_type
debug = args.debug
port = args.port


llm = AutoModelForCausalLM.from_pretrained(
    model_path,
    model_type=model_type,
    threads=8,
)
app = fastapi.FastAPI(title="🪄LLM Inference server💫")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
# @todo: this is for replit, need to find out equivalent for WizardCoder
EOS_TOKEN = "<|endoftext|>"


class Message(BaseModel):
    role: str
    content: str


@dataclass
class GenerationConfig:
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    max_new_tokens: int
    seed: int
    reset: bool
    stream: bool
    threads: int
    stop: list[str]


def format_prompt(messages: List[Message]):
    """
    Formats a List of Message objects into an Alpaca style Instruction-Response
    prompt string that can be passed to the model for generation
    """
    text = ""
    prev = ""
    for message in messages[1:]:
        role = message["role"]
        content = message["content"]
        if role == ROLE_USER:
            # If the messages of user are consecutive, no need to add another "Instruction block"
            if prev == role:
                text += "\n\n" + content
            else:
                text += f"""

### Instruction:
{content}
"""
        else:
            # If the messages of user are consecutive, no need to add another "Instruction block"
            if prev == role:
                text += "\n\n" + content
            else:
                text += f"""

### Response:
{content}
"""
        # set current role as previous role for the next iteration
        prev = role
    # At the end, add a "Response" block so that the model knows to respond
    text += """
### Response:
"""
    return text


def generate(
    llm: AutoModelForCausalLM,
    generation_config: GenerationConfig,
    messages: List[Message],
):
    """run model inference, will return a Generator if streaming is true"""

    formatted_prompt = format_prompt(
        messages,
    )
    if debug:
        print("formatted_prompt being passed to LLM", formatted_prompt)
    return llm(
        formatted_prompt,
        **asdict(generation_config),
    )


@app.get("/")
async def index():
    html_content = """
    <html>
        <head>
        </head>
        <body style="font-family:system-ui">
            It works!
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@app.post("/v1/completions")
async def completion(request: Request, response_mode=None):
    payload = await request.json()
    if debug:
        print("plain completion payload", payload)
    messages = payload.get("messages", [])
    generation_config = GenerationConfig(
        temperature=payload.get("temperature", 0.2),
        top_k=payload.get("top_k", 50),
        top_p=payload.get("top_p", 0.9),
        repetition_penalty=payload.get("repetition_penalty", 1.0),
        max_new_tokens=payload.get("max_tokens", 512),  # adjust as needed
        seed=payload.get("seed", 42),
        reset=True,  # reset history (cache)
        stream=payload.get("stream", True),  # streaming per word/token
        threads=int(os.cpu_count() / 6),  # adjust for your CPU
        stop=[EOS_TOKEN],
    )
    response = generate(llm, generation_config, messages)
    return response


@app.post("/v1/chat/completions")
async def chat(request: Request):
    payload = await request.json()
    if debug:
        print("chat completion api payload", payload)
    messages = payload.get("messages", [])
    generation_config = GenerationConfig(
        temperature=payload.get("temperature", 0.2),
        top_k=payload.get("top_k", 50),
        top_p=payload.get("top_p", 0.9),
        repetition_penalty=payload.get("repetition_penalty", 1.0),
        max_new_tokens=payload.get("max_tokens", 512),  # adjust as needed
        seed=payload.get("seed", 42),
        reset=True,  # reset history (cache)
        stream=payload.get("stream", True),  # streaming per word/token
        threads=int(os.cpu_count() / 6),  # adjust for your CPU
        stop=[EOS_TOKEN],
    )
    return EventSourceResponse(stream_response(llm, generation_config, messages))


async def stream_response(llm, generation_config, messages):
    try:
        iterator: Generator = generate(llm, generation_config, messages)
        for chat_chunk in iterator:
            response = {
                "choices": [
                    {
                        "delta": {"content": chat_chunk},
                        "finish_reason": "stop"
                        if chat_chunk == EOS_TOKEN
                        else "unknown",
                    }
                ]
            }
            yield dict(data=json.dumps(response))
        yield dict(data="[DONE]")
    except Exception as e:
        print(f"Exception in event publisher: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port)
