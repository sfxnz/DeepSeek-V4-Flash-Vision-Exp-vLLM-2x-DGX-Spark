"""vLLM general plugin for DeepSeek-V4-Flash-Vision-Exp."""

__all__ = ["register"]


def register() -> None:
    from vllm.model_executor.models.registry import ModelRegistry

    ModelRegistry.register_model(
        "DeepseekV4ForCausalLM",
        "dsv4_vision.model:DeepseekV4VisionForCausalLM",
    )
    ModelRegistry.register_model(
        "DeepseekV4VisionForCausalLM",
        "dsv4_vision.model:DeepseekV4VisionForCausalLM",
    )
