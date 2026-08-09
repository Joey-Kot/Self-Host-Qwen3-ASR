import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    y = x @ x
    print("FP16 CUDA smoke test:", y.shape, y.dtype)
