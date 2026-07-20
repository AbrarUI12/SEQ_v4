import torch

from seq_core.channel_protect import ChannelProtectedLinear, materialize_channel_protection


class _FakeQuantModule(torch.nn.Module):
    """Mimics an HQQ-style backend module: no usable ``.weight`` (del_orig),
    dense weight recoverable only via the backend's ``dequantize_weight``."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("_wdq", torch.round(weight * 8) / 8)  # cheap fake-quant
        self.register_buffer("weight", torch.empty(0))  # HQQ leaves an empty placeholder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self._wdq.t())


class _FakeBackend:
    name = "fake"

    def quantize_linear(self, layer, bits, *, device="cpu", compute_dtype=torch.float32,
                        group_size=None, **kwargs):
        return _FakeQuantModule(layer.weight.detach().to(compute_dtype))

    def dequantize_weight(self, module):
        return module._wdq.detach()


def test_materialized_channel_protection_backend_path_is_not_empty():
    """Regression: HQQ-style q_full has an empty .weight; export must dequantize
    through the backend instead of saving a Size([0]) tensor."""
    torch.manual_seed(11)
    weight = torch.randn(5, 7, dtype=torch.float32)
    module = ChannelProtectedLinear(
        weight, None, [2, 3], backend=_FakeBackend(), base_bits=4,
        device="cpu", compute_dtype=torch.float32, group_size=None,
    )
    model = torch.nn.Sequential(module)
    inputs = torch.randn(4, 7, dtype=torch.float32)
    expected = model(inputs)
    assert materialize_channel_protection(model) == 1
    dense = model[0]
    assert isinstance(dense, torch.nn.Linear)
    assert tuple(dense.weight.shape) == (5, 7)  # not Size([0])
    assert dense.weight.numel() == 35
    torch.testing.assert_close(model(inputs), expected)


def test_materialized_channel_protection_preserves_output():
    torch.manual_seed(7)
    weight = torch.randn(5, 7, dtype=torch.float32)
    base = weight + 0.1 * torch.randn_like(weight)
    bias = torch.randn(5, dtype=torch.float32)
    module = ChannelProtectedLinear(
        weight,
        bias,
        [1, 4],
        backend=None,
        base_bits=4,
        device="cpu",
        compute_dtype=torch.float32,
        precomputed_base=base,
    )
    model = torch.nn.Sequential(module)
    inputs = torch.randn(3, 7, dtype=torch.float32)

    expected = model(inputs)
    assert materialize_channel_protection(model) == 1
    assert isinstance(model[0], torch.nn.Linear)
    torch.testing.assert_close(model(inputs), expected)


def test_materialized_channel_protection_is_idempotent():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2))
    assert materialize_channel_protection(model) == 0
