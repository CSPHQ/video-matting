import os
from onnxruntime import (
    GraphOptimizationLevel, InferenceSession,
    SessionOptions, get_available_providers
)

NUM_THREADS = min(4, os.cpu_count())


def create_model_for_provider(
    model_path: str,
    provider: str = 'auto',
    num_threads=None
) -> InferenceSession:
    if provider == 'auto':
        if 'CUDAExecutionProvider' in get_available_providers():
            provider = 'CUDAExecutionProvider'
        else:
            provider = 'CPUExecutionProvider'
        print('model provider', provider)
    assert provider in get_available_providers(), \
        f"provider {provider} not found, {get_available_providers()}"

    # Few properties that might have an impact on performances (provided by MS)
    options = SessionOptions()
    if num_threads is not None:
        options.intra_op_num_threads = int(num_threads)
    else:
        options.intra_op_num_threads = int(os.environ.get('NUM_THREADS', NUM_THREADS))
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    # Load the model as a graph and prepare the CPU backend
    session = InferenceSession(model_path, options, providers=[provider])
    session.disable_fallback()
    return session
