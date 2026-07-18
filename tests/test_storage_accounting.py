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


def test_weight_only_axis_excludes_fp16_embeddings():
    # A 4-bit base with large FP16 embeddings/lm_head: the weight-only axis stays
    # ~4 bits, while the full-model average is inflated to ~7 — the exact artifact
    # that charged the GPTQ base 7.90 instead of ~4.0 in COMPARISON.md.
    qv = 973_000_000          # quantized linear weights (Llama-3.2-1B decoder linears)
    emb = 262_000_000         # FP16 (tied) embedding / lm_head params
    groups = qv // 64
    rep = account_storage(
        quantized_values=qv, quantized_bits=4,
        scale_values=groups, zero_point_values=groups,
        embedding_values=emb, parameter_count=qv + emb,
    )
    w = rep["actual_weight_bits_per_param"]
    m = rep["actual_model_bits_per_parameter"]
    assert 4.0 <= w < 5.0, w          # weight-only ~= base bits + small scale overhead
    assert m > 6.0, m                 # full-model inflated by FP16 embeddings
    assert m > w + 1.5                # the mis-plot the fix removes


def test_weight_only_pure_base_is_base_bits():
    rep = account_storage(quantized_values=1000, quantized_bits=4, parameter_count=1000)
    assert abs(rep["actual_weight_bits_per_param"] - 4.0) < 1e-6


def test_build_comparison_recomputes_weight_bits_from_byte_breakdown():
    # Regenerating the table must recover ~4 bits from the saved byte breakdown
    # even for OLD runs that stored only the inflated full-model number — this is
    # what makes regeneration a CPU-only build_comparison re-run.
    from analysis.build_comparison import _weight_bits_from_storage

    qv = 800_000_000
    groups = qv // 128
    rep = account_storage(
        quantized_values=qv, quantized_bits=4,
        scale_values=groups, zero_point_values=groups,
        embedding_values=200_000_000, parameter_count=qv + 200_000_000,
    )
    got_new = _weight_bits_from_storage(rep, 4)                  # new field present
    old = {k: v for k, v in rep.items() if k != "actual_weight_bits_per_param"}
    got_old = _weight_bits_from_storage(old, 4)                  # byte-recompute path
    assert got_new is not None and 4.0 <= got_new < 5.0, got_new
    assert got_old is not None and 4.0 <= got_old < 5.0, got_old
    assert got_old < rep["actual_model_bits_per_parameter"] - 1.5
    # protection raises the weight axis but nowhere near the full-model number
    prot = account_storage(quantized_values=qv, quantized_bits=4,
                           scale_values=groups, zero_point_values=groups,
                           fp16_residual_values=qv // 20, channel_index_values=qv // 20 // 2048,
                           embedding_values=200_000_000, parameter_count=qv + 200_000_000)
    assert _weight_bits_from_storage(prot, 4) > got_new
