"""vLLM general plugin for DeepSeek-V4-Flash-Vision-Exp."""

__all__ = ["register"]


def register() -> None:
    from vllm.model_executor.models.registry import ModelRegistry

    from .hooks import install_arch_convertor, install_encoding_hooks

    install_arch_convertor()
    install_encoding_hooks()
    ModelRegistry.register_model(
        "DeepseekV4ForConditionalGeneration",
        "dsv4_vision.vl_model:DeepseekV4ForConditionalGeneration",
    )
