FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"

RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home && \
    apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg

USER camenduru

RUN pip install -q opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod \
    xformers==0.0.25 torchsde==0.2.6 einops==0.8.0 diffusers==0.28.0 transformers==4.41.2 accelerate==0.30.1 matplotlib==3.9.1 insightface \
    onnx onnxruntime-gpu onnxruntime accelerate timm SentencePiece git+https://github.com/XPixelGroup/BasicSR ftfy einops facexlib fire && \
    GIT_LFS_SKIP_SMUDGE=1 git clone -b tost https://github.com/camenduru/PuLID-FLUX-hf /content/PuLID-FLUX && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/PuLID/resolve/main/antelopev2/1k3d68.onnx -d /content/PuLID-FLUX/models/antelopev2 -o 1k3d68.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/PuLID/resolve/main/antelopev2/2d106det.onnx -d /content/PuLID-FLUX/models/antelopev2 -o 2d106det.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/PuLID/resolve/main/antelopev2/genderage.onnx -d /content/PuLID-FLUX/models/antelopev2 -o genderage.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/PuLID/resolve/main/antelopev2/glintr100.onnx -d /content/PuLID-FLUX/models/antelopev2 -o glintr100.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/PuLID/resolve/main/antelopev2/scrfd_10g_bnkps.onnx -d /content/PuLID-FLUX/models/antelopev2 -o scrfd_10g_bnkps.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/PuLID/raw/main/clip-vit-large-patch14/config.json -d /content/PuLID-FLUX/models/clip-vit-large-patch14 -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/PuLID/raw/main/clip-vit-large-patch14/merges.txt -d /content/PuLID-FLUX/models/clip-vit-large-patch14 -o merges.txt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/PuLID/resolve/main/clip-vit-large-patch14/model.safetensors -d /content/PuLID-FLUX/models/clip-vit-large-patch14 -o model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/PuLID/raw/main/clip-vit-large-patch14/special_tokens_map.json -d /content/PuLID-FLUX/models/clip-vit-large-patch14 -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/PuLID/raw/main/clip-vit-large-patch14/tokenizer.json -d /content/PuLID-FLUX/models/clip-vit-large-patch14 -o tokenizer.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/PuLID/raw/main/clip-vit-large-patch14/tokenizer_config.json -d /content/PuLID-FLUX/models/clip-vit-large-patch14 -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/PuLID/raw/main/clip-vit-large-patch14/vocab.json -d /content/PuLID-FLUX/models/clip-vit-large-patch14 -o vocab.json  && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/PuLID/raw/main/xflux_text_encoders/added_tokens.json -d /content/PuLID-FLUX/models/xflux_text_encoders -o added_tokens.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/PuLID/raw/main/xflux_text_encoders/config.json -d /content/PuLID-FLUX/models/xflux_text_encoders -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/PuLID/resolve/main/xflux_text_encoders/model-00001-of-00002.safetensors -d /content/PuLID-FLUX/models/xflux_text_encoders -o model-00001-of-00002.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/PuLID/resolve/main/xflux_text_encoders/model-00002-of-00002.safetensors -d /content/PuLID-FLUX/models/xflux_text_encoders -o model-00002-of-00002.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/PuLID/raw/main/xflux_text_encoders/model.safetensors.index.json -d /content/PuLID-FLUX/models/xflux_text_encoders -o model.safetensors.index.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/PuLID/raw/main/xflux_text_encoders/special_tokens_map.json -d /content/PuLID-FLUX/models/xflux_text_encoders -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/PuLID/resolve/main/xflux_text_encoders/spiece.model -d /content/PuLID-FLUX/models/xflux_text_encoders -o spiece.model && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/PuLID/raw/main/xflux_text_encoders/tokenizer_config.json -d /content/PuLID-FLUX/models/xflux_text_encoders -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/ae.sft -d /content/PuLID-FLUX/models -o ae.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/flux1-dev.sft -d /content/PuLID-FLUX/models -o flux1-dev.safetensors && \
    # aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/flux1-dev-fp8.safetensors -d /content/PuLID-FLUX/models -o flux1-dev.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/PuLID/resolve/main/pulid_flux_v0.9.0.safetensors -d /content/PuLID-FLUX/models -o pulid_flux_v0.9.0.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/PuLID/resolve/main/bpe_simple_vocab_16e6.txt.gz -d /content/PuLID-FLUX/eva_clip -o bpe_simple_vocab_16e6.txt.gz

COPY ./worker_runpod.py /content/PuLID-FLUX/worker_runpod.py
WORKDIR /content/PuLID-FLUX
CMD python worker_runpod.py