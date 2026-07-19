#!/usr/bin/env python3
"""Pure-Python validation of the asymmetric affine quantization math used by GPTQ.

Cannot import seq_core.gptq here (needs torch), so this replicates the exact
_find_params / _quantize_affine formulas and checks the properties GPTQ relies
on — chiefly that the zero value maps back to zero (correct zero-point), values
stay in range, and quantization is monotonic. A conceptual error (e.g. a
zero-point sign flip) would fail here.
"""
import sys


def find_params(row, maxq):
    xmax = max(0.0, max(row))
    xmin = min(0.0, min(row))
    scale = (xmax - xmin) / maxq
    if scale == 0:
        scale = 1.0
    zero = round(-xmin / scale)
    return scale, zero


def q(w, scale, zero, maxq):
    qi = min(max(round(w / scale) + zero, 0), maxq)
    return scale * (qi - zero)


FAILS = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    (print("ok  :", msg) if cond else (FAILS.append(msg), print("FAIL:", msg)))


maxq = 3.0  # 2-bit
row = [-2.0, 0.0, 6.0]
scale, zero = find_params(row, maxq)
deq = [q(w, scale, zero, maxq) for w in row]
check(abs(scale - 8.0 / 3.0) < 1e-9, "scale = (xmax-xmin)/maxq")
check(zero == 1, "zero-point rounds -xmin/scale")
check(abs(q(0.0, scale, zero, maxq)) < 1e-9, "zero value maps back to zero (asymmetric)")
check(deq[0] <= deq[1] <= deq[2], "monotonic reconstruction")
check(all(abs(d) <= max(abs(min(row)), abs(max(row))) + scale for d in deq), "reconstruction bounded")

# all-equal row -> scale guarded to 1, no divide-by-zero
s2, z2 = find_params([3.0, 3.0, 3.0], maxq)
check(s2 == (3.0 - 0.0) / maxq, "positive constant row uses xmax with xmin=0")

# 4-bit finer grid reconstructs better than 2-bit (sanity of maxq scaling)
r = [-1.0, -0.3, 0.2, 0.9]
s4, z4 = find_params(r, 15.0)
s2b, z2b = find_params(r, 3.0)
err4 = sum(abs(w - q(w, s4, z4, 15.0)) for w in r)
err2 = sum(abs(w - q(w, s2b, z2b, 3.0)) for w in r)
check(err4 < err2, "4-bit reconstructs better than 2-bit")

print("\n%d checks, %d failures" % (CHECKS, len(FAILS)))
if __name__ == "__main__":
    sys.exit(1 if FAILS else 0)
