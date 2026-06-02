import gc
import contextlib

import torch


def clear_gpu_memory() -> None:
    """Best-effort release of training-side GPU resources.

    Synchronizes pending CUDA work, drops Python-side references via
    ``gc.collect()``, then asks the allocator to release cached blocks and
    IPC buffers. The CUDA driver context itself stays resident in the process.
    """
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception as exc:
            print(
                '[WARN][gpu::clear_gpu_memory] CUDA sync failed after prior '
                'GPU work; context may be poisoned:',
                exc,
            )
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


@contextlib.contextmanager
def force_autograd_on(tag: str):
    """Guarantee autograd is active inside the context body.

    GS training depends on ``total_loss.backward()``. A previous serial stage
    in the same process may leave the current thread in inference mode or with
    grad disabled, so this context explicitly exits inference mode and enables
    grad before running training code. Both managers are idempotent when the
    thread is already in the normal training state.
    """
    inference_cm = None
    if hasattr(torch, 'inference_mode'):
        try:
            inference_cm = torch.inference_mode(False)
        except TypeError:
            inference_cm = None
    enter_ctx: contextlib.ExitStack = contextlib.ExitStack()
    with enter_ctx:
        if inference_cm is not None:
            enter_ctx.enter_context(inference_cm)
        enter_ctx.enter_context(torch.enable_grad())
        yield
