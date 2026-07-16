from types import SimpleNamespace

import torch
from torch import nn

from seq_core.gptq import (
    _prepare_replay_call,
    gptq_quantize_model_sequential,
    gptq_quantize_weight,
)


class _DummyCache:
    def update(self, *args, **kwargs):
        return None

    def get_seq_length(self):
        return 0


class _Tokenizer:
    def __call__(self, prompt, **kwargs):
        sample = int(prompt)
        return {"input_ids": torch.tensor([[sample, sample + 1]], dtype=torch.long)}


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)
        self.calls = []
        with torch.no_grad():
            self.proj.weight.copy_(torch.tensor([[1.0, 0.25], [-0.5, 0.75]]))

    def forward(
        self,
        hidden_states,
        *,
        sample_tag,
        past_key_values=None,
        use_cache=True,
    ):
        self.calls.append((int(sample_tag), past_key_values, bool(use_cache)))
        return (self.proj(hidden_states),)


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block(), _Block()])


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self.model = _Backbone()

    def forward(self, input_ids, use_cache=None):
        enabled = self.config.use_cache if use_cache is None else use_cache
        cache = _DummyCache() if enabled else None
        values = input_ids.to(torch.float32)
        hidden_states = torch.stack((values, values.square()), dim=-1)
        sample_tag = input_ids[0, 0]
        for block in self.model.layers:
            hidden_states = block(
                hidden_states,
                sample_tag=sample_tag,
                past_key_values=cache,
                use_cache=enabled,
            )[0]
        return (hidden_states,)


def test_prepare_replay_call_removes_cache_objects_and_cache_kwargs():
    cache = _DummyCache()
    tensor = torch.tensor([1.0], requires_grad=True)

    args, kwargs = _prepare_replay_call(
        (tensor, cache),
        {
            "past_key_values": cache,
            "use_cache": True,
            "nested": {"cache": cache, "tensor": tensor},
        },
        torch.device("cpu"),
    )

    assert args[1] is None
    assert not args[0].requires_grad
    assert "past_key_values" not in kwargs
    assert kwargs["use_cache"] is False
    assert kwargs["nested"]["cache"] is None
    assert kwargs["nested"]["tensor"].device.type == "cpu"


def test_weight_quantizer_preserves_hessian_by_default():
    weight = torch.tensor([[1.0, -0.25], [0.5, 0.75]])
    hessian = torch.tensor([[2.0, 0.25], [0.25, 1.0]])
    original = hessian.clone()

    quantized = gptq_quantize_weight(weight, hessian, bits=4, group_size=2)

    assert quantized.shape == weight.shape
    assert torch.isfinite(quantized).all()
    assert torch.equal(hessian, original)


def test_sequential_gptq_replays_each_samples_metadata_without_a_kv_cache():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _Model().to(device)
    device = next(model.parameters()).device

    result = gptq_quantize_model_sequential(
        model,
        _Tokenizer(),
        ["1", "3"],
        bits=4,
        group_size=2,
        seq_len=2,
        device=str(device),
        max_prompts=None,
        out_dtype=torch.float32,
    )

    assert model.config.use_cache is True
    assert set(result) == {"model.layers.0.proj", "model.layers.1.proj"}
    for block in model.model.layers:
        assert next(block.parameters()).device == device
        # One Hessian pass and one propagation pass, preserving each sample tag.
        assert [tag for tag, _, _ in block.calls] == [1, 3, 1, 3]
        assert all(cache is None for _, cache, _ in block.calls)
        assert all(use_cache is False for _, _, use_cache in block.calls)
