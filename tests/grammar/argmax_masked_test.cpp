// tests/grammar/argmax_masked_test.cpp - the masked argmax comparison rule: largest finite logit, smallest id
// on ties, -1 on an empty mask; NaN/inf never win even when they are the largest.
#include "strata/grammar/argmax.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

using namespace strata::grammar;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

int main() {
    std::vector<float> v = {1.0f, 5.0f, 3.0f, 5.0f, -2.0f};
    std::vector<uint8_t> m(v.size(), 1);
    CHECK(argmax_masked(v.data(), m.data(), (int) v.size()) == 1);   // tie 5.0 at 1 and 3: the smaller id
    m[1] = 0;
    CHECK(argmax_masked(v.data(), m.data(), (int) v.size()) == 3);
    m[3] = 0;
    CHECK(argmax_masked(v.data(), m.data(), (int) v.size()) == 2);
    std::fill(m.begin(), m.end(), 0);
    CHECK(argmax_masked(v.data(), m.data(), (int) v.size()) == -1);  // nothing survives
    std::vector<float> p = {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(), 2.0f};
    std::vector<uint8_t> pm(p.size(), 1);
    CHECK(argmax_masked(p.data(), pm.data(), (int) p.size()) == 2);  // inf and NaN never win
    CHECK(argmax_masked(p.data(), pm.data(), 0) == -1);
    std::printf("argmax_masked_test: %s\n", g_fail ? "FAILED" : "ok");
    return g_fail;
}
