# Replit Inference Server

This is a simple inference server to load ggml versions of Replit models (ex: https://huggingface.co/teknium/Replit-v2-CodeInstruct-3B and its ggml version: https://huggingface.co/abacaj/Replit-v2-CodeInstruct-3B-ggml) and use them as OpenAI compatible REST API servers. This is so that they can be used as backend model for continue.dev VS Code extension

### References:
- https://github.com/abacaj/replit-3B-inference/blob/main/inference.py (Using `ctransformers` for inference of replit models)
- https://github.com/marella/ctransformers/issues/26#issuecomment-1601052575 (Using `ctransformers` to wrap them around APIs)
- https://huggingface.co/spaces/matthoffner/wizardcoder-ggml/blob/main/main.py (Using `ctransformers` to wrap them around APIs)
- https://github.com/continuedev/continue/blob/main/continuedev/src/continuedev/libs/llm/ggml.py (The endpoints used by `GGML` class in continue.dev extension)

### Setting up
- Install `miniconda3` (to create a virtual environment). Or you could use `virtualenv` or `poetry` as well
- Create an environment
- `pip install -r requirements.txt`
- `python app.py` or use `python app.py --help` to see available flags
- Install `continue.dev` from VS Code extensions
- Use instructions from https://continue.dev/docs/customization#local-models-with-ggml to configure `continue` to use ggml models. However, instead of using their "5 minute quickstart" server, use this server instead
