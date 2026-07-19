import torch

from seq_core.channel_protect import ChannelProtectedLinear, materialize_channel_protection


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
