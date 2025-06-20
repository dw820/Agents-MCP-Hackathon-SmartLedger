
import modal

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.7.2",
        "huggingface_hub[hf_transfer]==0.26.2",
        "flashinfer-python==0.2.0.post2",  # pinning, very unstable
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.5",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
)

# In its 0.7 release, vLLM added a new version of its backend infrastructure,
# the [V1 Engine](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html).
# Using this new engine can lead to some [impressive speedups](https://github.com/modal-labs/modal-examples/pull/1064),
# but as of version 0.7.2 the new engine does not support all inference engine features
# (including important performance optimizations like
# [speculative decoding](https://docs.vllm.ai/en/v0.7.2/features/spec_decode.html)).

# The features we use in this demo are supported, so we turn the engine on by setting an environment variable
# on the Modal Image.

vllm_image = vllm_image.env({"VLLM_USE_V1": "1"})

# ## Download the model weights

# We'll be running a pretrained foundation model -- Meta's LLaMA 3.1 8B
# in the Instruct variant that's trained to chat and follow instructions,
# quantized to 4-bit by [Neural Magic](https://neuralmagic.com/) and uploaded to Hugging Face.

# You can read more about the `w4a16` "Machete" weight layout and kernels
# [here](https://neuralmagic.com/blog/introducing-machete-a-mixed-input-gemm-kernel-optimized-for-nvidia-hopper-gpus/).

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
# MODEL_REVISION = ""

# Although vLLM will download weights on-demand, we want to cache them if possible. We'll use [Modal Volumes](https://modal.com/docs/guide/volumes),
# which act as a "shared disk" that all Modal Functions can access, for our cache.

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


# ## Build a vLLM engine and serve it

# The function below spawns a vLLM instance listening at port 8000, serving requests to our model. vLLM will authenticate requests
# using the API key we provide it.

# We wrap it in the [`@modal.web_server` decorator](https://modal.com/docs/guide/webhooks#non-asgi-web-servers)
# to connect it to the Internet.

app = modal.App("qwen2.5-vl-7b-instruct")

N_GPU = 1  # tip: for best results, first upgrade to more powerful GPUs, and only then increase GPU count
API_KEY = "super-secret-key"  # api key, for auth. for production use, replace with a modal.Secret

MINUTES = 60  # seconds

VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    scaledown_window=15 * MINUTES,  # how long should we stay up with no requests?
    timeout=10 * MINUTES,  # how long should we wait for container start?
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(
    max_inputs=100
)  # how many requests can one replica handle? tune carefully!
@modal.web_server(port=VLLM_PORT, startup_timeout=5 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        # "--revision",
        # MODEL_REVISION,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--api-key",
        API_KEY,
    ]

    subprocess.Popen(" ".join(cmd), shell=True)


# ## Deploy the server

# To deploy the API on Modal, just run
# ```bash
# modal deploy modal/llama_inference.py
# ```

# This will create a new app on Modal, build the container image for it if it hasn't been built yet,
# and deploy the app.