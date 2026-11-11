// include/strata/grammar/argmax.hpp - the masked argmax the grammar Guard resamples a window row with.
#pragma once

#include <cstdint>

namespace strata::grammar {

/// The id of the largest finite logit among the entries mask[id] != 0, ties to the smallest id; -1 when no entry
/// survives the mask. Non-finite logits (a NaN row's poison) never win. The Guard's resample runs this on the host
/// over one row read by Verifier::copy_logits - a window-resampling violation is rare, and CPU and every backend
/// agree on the answer bitwise because there is exactly one comparison rule here.
int32_t argmax_masked(const float* logits, const uint8_t* mask, int n);

}  // namespace strata::grammar
