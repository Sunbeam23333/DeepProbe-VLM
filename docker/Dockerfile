# Known documented CUDA 13.0.1 NVIDIA container, not a claim of local GPU validation.
# Pin the pulled image digest in your run record after verifying the host driver.
FROM nvcr.io/nvidia/pytorch:25.09-py3
WORKDIR /workspace/DeepProbe-VLM
COPY requirements/train.txt requirements/train.txt
RUN python -m pip install -r requirements/train.txt
COPY . .
RUN python -m pip install --no-deps -e .
ENV TOKENIZERS_PARALLELISM=false
CMD ["bash"]
