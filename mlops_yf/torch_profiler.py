'''
Example usage of PyTorch profiler.
'''
import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

model = models.resnet18()
# model = models.resnet34()
inputs = torch.randn(5, 3, 224, 224)

# Profile multiple iterations
with profile(
    activities=[ProfilerActivity.CPU], record_shapes=True,
    on_trace_ready=tensorboard_trace_handler('./logs/resnet18'),
) as prof:
    for i in range(10):
        model(inputs)
        prof.step()

# Print to terminal
print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=10))
