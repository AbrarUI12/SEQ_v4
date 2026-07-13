#!/usr/bin/env python3
"""Pure-stdlib tests for seq_core.channel_utils (runnable without torch)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seq_core.channel_utils import layer_effective_bits, select_protected_channels  # noqa: E402

FAILS = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    (print("ok  :", msg) if cond else (FAILS.append(msg), print("FAIL:", msg)))


# --- select_protected_channels ---
s = [0.1, 0.9, 0.5, 0.3, 0.8]  # top-2 => indices 1(0.9),4(0.8)
check(select_protected_channels(s, 0.0) == [], "k=0 -> none")
check(select_protected_channels(s, 0.4) == [1, 4], "top-40% picks two highest, sorted")
check(select_protected_channels(s, 1.0) == [0, 1, 2, 3, 4], "k=1.0 -> all")
check(select_protected_channels([], 0.5) == [], "empty scores -> none")
# ceil rounding: 10% of 5 = 0.5 -> ceil -> 1 channel (the max, idx 1)
check(select_protected_channels(s, 0.1) == [1], "ceil rounding picks 1")
# non-finite never selected
sn = [float("nan"), 1.0, float("inf"), 2.0]
check(select_protected_channels(sn, 0.5) == [1, 3], "non-finite excluded")

# --- layer_effective_bits ---
# 100 in-features, protect 10 at 16-bit, rest 90 at 4-bit -> (90*4+10*16)/100 = 5.2
check(abs(layer_effective_bits(100, 10, 4, 16) - 5.2) < 1e-9, "eff bits column split 5.2")
check(layer_effective_bits(100, 0, 4) == 4.0, "no protection -> base bits")
check(layer_effective_bits(100, 100, 4, 16) == 16.0, "all protected -> 16")
check(layer_effective_bits(0, 0, 3) == 3.0, "zero in-features guard")
# protect at 8-bit instead of 16: 90*3 + 10*8 = 350 /100 = 3.5
check(abs(layer_effective_bits(100, 10, 3, 8) - 3.5) < 1e-9, "eff bits 8-bit protect 3.5")
# clamp over-count
check(layer_effective_bits(50, 999, 4, 16) == 16.0, "num_protected clamped to in_features")

print("\n%d checks, %d failures" % (CHECKS, len(FAILS)))
sys.exit(1 if FAILS else 0)
