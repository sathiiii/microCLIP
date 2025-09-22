import torch

while True:
    size = torch.randint(10000, 20000, (1,)).item()
    dummy_tensor = torch.rand((size, 10000)).cuda()
    dummy_tensor = dummy_tensor @ dummy_tensor.T
    dummy_tensor = dummy_tensor / torch.mean(dummy_tensor)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
