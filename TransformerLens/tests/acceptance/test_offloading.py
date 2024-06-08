from transformer_lens import HookedTransformer
import torch

MODEL = "solu-2l"

def time_diff(func):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    func()

    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event)

@torch.no_grad()
def test_offload_params_after():
    model = HookedTransformer.from_pretrained(MODEL, device="cuda")
    allocated_before = torch.cuda.memory_allocated(0)
    model.offload_params_after("blocks.0.hook_resid_post", torch.tensor([[0]], device="cuda"))
    allocated_after = torch.cuda.memory_allocated(0)
    assert allocated_after < allocated_before * 0.55

@torch.no_grad()
def test_run_with_cache_until():
    model = HookedTransformer.from_pretrained(MODEL, device="cuda")
    def forward():
        model.run_with_cache("Hello, world!", names_filter=["blocks.0.hook_resid_post"])
    forward_time = time_diff(forward)
    def forward_until():
        model.run_with_cache_until("Hello, world!", names_filter=["blocks.0.hook_resid_post"])
    forward_fake_time = time_diff(forward_until)
    assert forward_fake_time < forward_time * 0.7
        