"""Helpers shared by the spreg regimes oracle generators (dev-only)."""

import numpy as np
from spreg.regimes import wald_test


def flat(a):
    return np.asarray(a, dtype=float).flatten().tolist()


def pair(t):
    return [float(t[0]), float(t[1])]


def corrected_chow(reg, x_flags, y_flags):
    """Chow tests with the restriction matrix on the real coefficient order (x regime blocks, x
    globals, yend regime blocks, yend globals), tested with spreg's own `wald_test`. The betas
    drop the trailing lambda, as the covariance matrix has no entry for it."""
    nr = int(reg.nr)
    kxr, kyr = sum(x_flags), sum(y_flags)
    kxf, kyf = len(x_flags) - kxr, len(y_flags) - kyr
    total = nr * (kxr + kyr) + kxf + kyf
    betas = np.asarray(reg.betas, dtype=float).reshape(-1, 1)[:total]
    vm = np.asarray(reg.vm, dtype=float)[:total, :total]  # Het / Hom: lambda is in vm as well
    rows_by_var = []
    for base, stride, count in ((0, kxr, kxr), (nr * kxr + kxf, kyr, kyr)):
        for v in range(count):
            block = np.zeros((nr - 1, total))
            for j in range(nr - 1):
                block[j, base + v] = 1.0
                block[j, base + v + (j + 1) * stride] = -1.0
            rows_by_var.append(block)
    regi = np.array([wald_test(betas, r, np.zeros((r.shape[0], 1)), vm) for r in rows_by_var])
    r_all = np.vstack(rows_by_var)
    joint = wald_test(betas, r_all, np.zeros((r_all.shape[0], 1)), vm)
    return [float(joint[0]), float(joint[1])], regi.tolist()
