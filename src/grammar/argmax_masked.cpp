// src/grammar/argmax_masked.cpp - see argmax.hpp.
#include "strata/grammar/argmax.hpp"

#include <cmath>

namespace strata::grammar {

int32_t argmax_masked(const float* logits, const uint8_t* mask, int n) {
    int32_t best = -1;
    float best_v = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (!mask[i]) continue;
        const float v = logits[i];
        if (!std::isfinite(v)) continue;
        if (best < 0 || v > best_v) { best = i; best_v = v; }
    }
    return best;
}

}  // namespace strata::grammar
