"""CPU learning example: validate online attention formulas, not an NPU kernel test.

Run from the repository root: python examples/concepts/verify_online_softmax.py
Uses Python double precision and the standard library only.
"""

import math
import random


def reference(scores, values):
    m = max(scores)
    if m == -math.inf:
        raise ValueError("An all-masked row needs an explicit output convention.")
    weights = [math.exp(x - m) for x in scores]
    total = math.fsum(weights)
    output = [
        math.fsum(w * v[d] for w, v in zip(weights, values)) / total
        for d in range(len(values[0]))
    ]
    return m, total, output


def online(scores, values, block_size):
    m, total = -math.inf, 0.0
    numerator = [0.0] * len(values[0])
    trace = []
    for start in range(0, len(scores), block_size):
        xs = scores[start : start + block_size]
        vs = values[start : start + block_size]
        block_max = max(xs)
        # A completely masked block contributes nothing. Avoid -inf - (-inf).
        if block_max == -math.inf:
            continue
        new_m = max(m, block_max)
        alpha = math.exp(m - new_m)
        weights = [math.exp(x - new_m) for x in xs]
        total = alpha * total + math.fsum(weights)
        numerator = [
            alpha * old + math.fsum(w * v[d] for w, v in zip(weights, vs))
            for d, old in enumerate(numerator)
        ]
        m = new_m
        trace.append((m, alpha, total, numerator[:]))
    if total == 0.0:
        raise ValueError("An all-masked row needs an explicit output convention.")
    return m, total, [x / total for x in numerator], trace


def main():
    scores = [1.0, 2.0, 3.0, 4.0]
    values = [[10.0], [20.0], [30.0], [40.0]]
    m, total, output, trace = online(scores, values, 2)
    for i, (mi, alpha, li, ui) in enumerate(trace, 1):
        print(f"block {i}: m={mi:.9f}, alpha={alpha:.9f}, l={li:.9f}, u={ui[0]:.9f}")
    probabilities = [math.exp(x - m) / total for x in scores]
    print("P:", ", ".join(f"{p:.9f}" for p in probabilities))
    print(f"online O={output[0]:.9f}, reference O={reference(scores, values)[2][0]:.9f}")
    wrong_u = trace[0][3][0] + math.fsum(
        math.exp(x - m) * v[0] for x, v in zip(scores[2:], values[2:])
    )
    print(f"missing numerator rescale: O={wrong_u / total:.9f}")

    rng = random.Random(20260910)
    rows = [
        scores,
        [4.0, 3.0, 2.0, 1.0],
        [2.0] * 9,
        [1000.0, 999.0, -1000.0, 1001.0],
        [-1000.0, -1001.0, -999.0],
        [-math.inf, -math.inf, 1.0, -math.inf, 2.0],
    ]
    rows += [[rng.uniform(-30, 30) for _ in range(rng.randint(1, 65))] for _ in range(100)]
    cases, max_error = 0, 0.0
    for row in rows:
        vs = [[rng.uniform(-10, 10) for _ in range(3)] for _ in row]
        expected_m, expected_l, expected_o = reference(row, vs)
        for size in sorted({1, 2, 3, 7, len(row)}):
            actual_m, actual_l, actual_o, _ = online(row, vs, size)
            assert actual_m == expected_m
            assert math.isclose(actual_l, expected_l, rel_tol=1e-12, abs_tol=1e-12)
            for actual, expected in zip(actual_o, expected_o):
                error = abs(actual - expected)
                max_error = max(max_error, error)
                assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
            cases += 1
    print(f"validated {cases} partitions; max absolute output error={max_error:.3e}")

    x = 1.23
    for scale in (0.1, 0.01, 0.001):
        raw_q = round(x / scale)
        q = max(-127, min(127, raw_q))
        reconstructed = q * scale
        print(f"INT8 s={scale:g}: q={q}, x_hat={reconstructed:.3f}, error={abs(x-reconstructed):.3f}")


if __name__ == "__main__":
    main()

