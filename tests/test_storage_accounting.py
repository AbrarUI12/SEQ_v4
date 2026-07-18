from seq_core.storage_accounting import account_layer, account_storage


def test_uniform_4bit_with_group_metadata_exceeds_four_bits():
    report = account_layer(128, 64, 4)
    assert report["actual_model_bits_per_parameter"] > 4.0


def test_fp16_correction_is_additive_to_full_base():
    base = account_layer(128, 64, 4)
    corrected = account_layer(128, 64, 4, protected_fp16=4)
    assert corrected["dense_quantized_weight_bytes"] == base["dense_quantized_weight_bytes"]
    assert corrected["fp16_residual_bytes"] == 4 * 64 * 2
    assert corrected["actual_model_bits_per_parameter"] > base["actual_model_bits_per_parameter"]


def test_mixed_int8_fp16_and_serialized_size():
    report = account_layer(256, 32, 4, protected_fp16=2, protected_int8=4)
    assert report["fp16_residual_bytes"] == 2 * 32 * 2
    assert report["int8_tier_bytes"] == 4 * 32
    measured = account_storage(quantized_values=100, quantized_bits=4,
                               parameter_count=100, serialized_checkpoint_bytes=80)
    assert measured["serialized_checkpoint_bits_per_parameter"] == 6.4
    assert measured["actual_total_bytes"] == 80
