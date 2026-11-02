"""Synchronized wall clock; measured CUDA peaks are separate from state bytes."""
import importlib.metadata
import platform
import time
import torch


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def environment(device):
    packages = {}
    for name in ("torch", "numpy", "transformers", "datasets", "huggingface-hub", "datasketch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu_details = None
    if torch.device(device).type == "cuda":
        props = torch.cuda.get_device_properties(device)
        gpu_details = dict(total_memory_bytes=props.total_memory, compute_capability=[props.major, props.minor],
                           multiprocessors=props.multi_processor_count)
    return dict(python=platform.python_version(), platform=platform.platform(), packages=packages,
                device=str(device), gpu=torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None,
                gpu_details=gpu_details, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())


class Profiler:
    def __init__(self, device, warmup_steps=5):
        self.device, self.warmup = device, warmup_steps
        self.cuda = torch.device(device).type == "cuda"
        self.full_peak = self.full_reserved = 0
        self.profile_seconds = 0.0
        self.profile_tokens = self.profile_steps = 0

    def capture_peak(self):
        if self.cuda:
            self.full_peak = max(self.full_peak, torch.cuda.max_memory_allocated(self.device))
            self.full_reserved = max(self.full_reserved, torch.cuda.max_memory_reserved(self.device))

    def begin(self):
        sync(self.device)
        self.capture_peak()
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        return time.perf_counter()

    def end(self, started, step, processed_tokens):
        sync(self.device)
        seconds = time.perf_counter() - started
        self.capture_peak()
        if step > self.warmup:
            self.profile_seconds += seconds
            self.profile_tokens += processed_tokens
            self.profile_steps += 1
        return seconds

    def report(self):
        return dict(peak_gpu_allocated_bytes=self.full_peak if self.cuda else None,
                    peak_gpu_reserved_bytes=self.full_reserved if self.cuda else None,
                    profiled_steps=self.profile_steps, profiled_processed_tokens=self.profile_tokens,
                    profile_seconds=self.profile_seconds,
                    iteration_ms=1000 * self.profile_seconds / self.profile_steps if self.profile_steps else None,
                    throughput_processed_tokens_s=self.profile_tokens / self.profile_seconds if self.profile_seconds else None)
