#!/usr/bin/env python3
"""Pure-stdlib tests for seq_core.channel_utils (runnable without torch)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seq_core.channel_utils import (  # noqa: E402
    assign_tiers,
    bucket_by_rank,
    combine_scores,
    greedy_bit_alloc_by_value,
    layer_effective_bits,
    normalize_minmax,
    packed_storage_bits,
    parse_tiers,
    select_protected_channels,
)

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

# --- bucket_by_rank ---
sc = [0.1, 0.9, 0.5, 0.3, 0.8, 0.2]  # rank desc: 1(.9),4(.8),2(.5),3(.3),5(.2),0(.1)
bk = bucket_by_rank(sc, 3)
check(len(bk) == 3, "bucket count")
check(bk[0] == [1, 4], "bucket 0 = top-2 by score (sorted idx)")
check(bk[1] == [2, 3], "bucket 1 = middle")
check(bk[2] == [0, 5], "bucket 2 = bottom")
check(sorted(i for b in bk for i in b) == [0, 1, 2, 3, 4, 5], "buckets partition all indices")
# uneven split: 7 into 3 -> sizes 3,2,2
bk2 = bucket_by_rank(list(range(7)), 3)
check([len(b) for b in bk2] == [3, 2, 2], "uneven split remainder to front")
check(bucket_by_rank([], 4) == [], "empty -> no buckets")

# --- normalize_minmax ---
check(normalize_minmax([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0], "minmax scales to [0,1]")
check(normalize_minmax([3.0, 3.0]) == [0.5, 0.5], "constant -> 0.5")
check(normalize_minmax([]) == [], "empty -> empty")
nn2 = normalize_minmax([1.0, float("nan"), 3.0])  # lo=1,hi=3 -> [0, 0, 1]
check(nn2[0] == 0.0 and nn2[2] == 1.0 and nn2[1] == 0.0, "non-finite -> 0")

# --- combine_scores ---
# mul favors channels high in BOTH; A=[0,1] B=[1,0] -> norm same -> product [0,0]
check(combine_scores([[0.0, 1.0], [1.0, 0.0]], "mul") == [0.0, 0.0], "mul: no channel high in both")
# A=[0,1,2] B=[0,1,2] -> norm [0,.5,1] each -> product [0,.25,1]
check(combine_scores([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], "mul") == [0.0, 0.25, 1.0], "mul agreeing signals")
# add: [0,.5,1]+[0,.5,1] = [0,1,2]
check(combine_scores([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], "add") == [0.0, 1.0, 2.0], "add agreeing signals")
# ranking: a channel high in one and mid in other beats one high in one and low in other (mul)
comp = combine_scores([[10.0, 10.0, 0.0], [0.0, 5.0, 10.0]], "mul")  # norm A=[1,1,0] B=[0,.5,1]
check(comp[1] > comp[0] and comp[1] > comp[2], "mul rewards jointly-high channel")
check(len(combine_scores([[1.0, 2.0, 3.0], [1.0, 2.0]], "mul")) == 2, "combine truncates to min length")

# --- packed_storage_bits (honest actual bits) ---
# no protection, no base scales: pure base bits
check(packed_storage_bits(2048, 2048, 4, 0, count_base_scales=False) == 4.0, "packed: base only = base_bits")
# nominal effective bits ignores overhead; packed >= nominal
nom = layer_effective_bits(2048, 205, 4, 16)  # ~10% protected -> ~5.2 nominal
pk = packed_storage_bits(2048, 2048, 4, 205, group_size=64, count_base_scales=False)
check(pk > nom, "packed (with index overhead) exceeds nominal effective bits")
# base scales add real overhead
pk_scales = packed_storage_bits(2048, 2048, 4, 0, group_size=64)
check(pk_scales > 4.0, "base group scales/zeros add overhead over nominal 4-bit")
# more protected channels -> more bits
check(packed_storage_bits(2048, 2048, 4, 400, count_base_scales=False)
      > packed_storage_bits(2048, 2048, 4, 100, count_base_scales=False), "more protected -> more bits")

# --- parse_tiers / assign_tiers ---
check(parse_tiers("16:0.02,8:0.08") == [(16, 0.02), (8, 0.08)], "parse tiers sorted bits desc")
check(parse_tiers("8:0.1,16:0.05") == [(16, 0.05), (8, 0.1)], "parse tiers reorders desc")
# scores: rank desc = idx 4(.9? ) ... use clear scores
sc_t = [0.1, 0.9, 0.5, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6, 0.05]  # 10 channels
# top 20% (2) -> 16bit = idx 1(.9),4(.8); next 30% (3) -> 8bit = 6(.7),8(.6),2(.5)
at = assign_tiers(sc_t, [(16, 0.2), (8, 0.3)])
check(at.get(16) == [1, 4], "tier16 = top-2 by score")
check(at.get(8) == [2, 6, 8], "tier8 = next-3 by score (sorted idx)")
# tiers disjoint and within range
allidx = at.get(16, []) + at.get(8, [])
check(len(allidx) == len(set(allidx)) == 5, "tiers disjoint, 5 protected")
check(assign_tiers([], [(16, 0.5)]) == {}, "empty scores -> no tiers")

# --- greedy_bit_alloc_by_value (error-per-byte tier allocation) ---
# 2 channels, tiers [4,16]; ch0 big distortion reduction, ch1 small. index_bits=0.
# upgrade cost 16-4=12/channel; budget=6 -> total budget=12 -> exactly one upgrade.
D2 = [[10.0, 0.0], [1.0, 0.0]]
check(greedy_bit_alloc_by_value(D2, [4, 16], 6.0) == [1, 0], "budget for one upgrade picks highest-value channel")
check(greedy_bit_alloc_by_value(D2, [4, 16], 0.0) == [0, 0], "zero budget -> all base tier")
check(greedy_bit_alloc_by_value(D2, [4, 16], 100.0) == [1, 1], "ample budget -> all upgraded")
# no benefit -> never upgrade even with budget
check(greedy_bit_alloc_by_value([[5.0, 5.0]], [4, 16], 100.0) == [0], "no distortion reduction -> stay base")
# 3 tiers [4,8,16], single channel: tight budget takes the high-value 4->8 step,
# not the low-value 8->16 step; larger budget reaches tier 2.
D3 = [[10.0, 3.0, 0.0]]  # 4->8 reduces 7 @ cost 4 (v=1.75); 8->16 reduces 3 @ cost 8 (v=0.375)
check(greedy_bit_alloc_by_value(D3, [4, 8, 16], 4.0) == [1], "tight budget takes best-value single step (tier 8)")
check(greedy_bit_alloc_by_value(D3, [4, 8, 16], 12.0) == [2], "full budget climbs to tier 16")
# value-greedy across channels: given budget for one step, upgrade the better ratio
Dv = [[8.0, 0.0], [20.0, 0.0]]  # both cost 12; ch1 bigger reduction -> upgrade ch1 first
check(greedy_bit_alloc_by_value(Dv, [4, 16], 6.0) == [0, 1], "value-greedy upgrades better benefit/cost first")
check(greedy_bit_alloc_by_value([], [4, 16], 5.0) == [], "empty channels -> empty alloc")
check(greedy_bit_alloc_by_value([[1.0]], [4], 5.0) == [0], "single tier -> no upgrade possible")

print("\n%d checks, %d failures" % (CHECKS, len(FAILS)))
sys.exit(1 if FAILS else 0)
