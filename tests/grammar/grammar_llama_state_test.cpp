// tests/program/grammar_llama_state_test.cpp - llama.cpp's tests/test-llama-grammar.cpp against Basalt's grammar
// state. The rules are the same hand-built element lists; the checks are the same facts: the initial stacks (compared
// as a set, element by element) and which candidate characters each stack rejects (llama.cpp checks them one stack at
// a time; Basalt exposes the union, so the check is that a character is accepted exactly when some stack accepts it).
// Data copied from llama.cpp (MIT, see THIRD_PARTY_LICENSES).
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "strata/grammar/grammar.hpp"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace strata::grammar;

enum llama_gretype {
    LLAMA_GRETYPE_END = 0,
    LLAMA_GRETYPE_ALT = 1,
    LLAMA_GRETYPE_RULE_REF = 2,
    LLAMA_GRETYPE_CHAR = 3,
    LLAMA_GRETYPE_CHAR_NOT = 4,
    LLAMA_GRETYPE_CHAR_RNG_UPPER = 5,
    LLAMA_GRETYPE_CHAR_ALT = 6,
    LLAMA_GRETYPE_CHAR_ANY = 7,
};
struct llama_grammar_element {
    llama_gretype type;
    uint32_t value;
};

int main()
{
    int n_checks = 0;
    std::vector<std::pair<std::string, uint32_t>> expected = {
        {"expr", 2},
        {"expr_6", 6},
        {"expr_7", 7},
        {"ident", 8},
        {"ident_10", 10},
        {"num", 9},
        {"num_11", 11},
        {"root", 0},
        {"root_1", 1},
        {"root_5", 5},
        {"term", 4},
        {"ws", 3},
        {"ws_12", 12},
    };

    std::vector<std::vector<llama_grammar_element>> expected_rules = {
        {{LLAMA_GRETYPE_RULE_REF, 5}, {LLAMA_GRETYPE_END, 0}},
        {
            {LLAMA_GRETYPE_RULE_REF, 2},
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_RULE_REF, 4},
            {LLAMA_GRETYPE_CHAR, 10},
            {LLAMA_GRETYPE_END, 0},
        },
        {{LLAMA_GRETYPE_RULE_REF, 4}, {LLAMA_GRETYPE_RULE_REF, 7}, {LLAMA_GRETYPE_END, 0}},
        {{LLAMA_GRETYPE_RULE_REF, 12}, {LLAMA_GRETYPE_END, 0}},
        {
            {LLAMA_GRETYPE_RULE_REF, 8},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_RULE_REF, 9},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_CHAR, 40},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_RULE_REF, 2},
            {LLAMA_GRETYPE_CHAR, 41},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_END, 0},
        },
        {{LLAMA_GRETYPE_RULE_REF, 1}, {LLAMA_GRETYPE_RULE_REF, 5}, {LLAMA_GRETYPE_ALT, 0}, {LLAMA_GRETYPE_RULE_REF, 1}, {LLAMA_GRETYPE_END, 0}},
        {
            {LLAMA_GRETYPE_CHAR, 45},
            {LLAMA_GRETYPE_CHAR_ALT, 43},
            {LLAMA_GRETYPE_CHAR_ALT, 42},
            {LLAMA_GRETYPE_CHAR_ALT, 47},
            {LLAMA_GRETYPE_RULE_REF, 4},
            {LLAMA_GRETYPE_END, 0},
        },
        {{LLAMA_GRETYPE_RULE_REF, 6}, {LLAMA_GRETYPE_RULE_REF, 7}, {LLAMA_GRETYPE_ALT, 0}, {LLAMA_GRETYPE_END, 0}},
        {
            {LLAMA_GRETYPE_CHAR, 97},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 122},
            {LLAMA_GRETYPE_RULE_REF, 10},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_END, 0},
        },
        {{LLAMA_GRETYPE_RULE_REF, 11}, {LLAMA_GRETYPE_RULE_REF, 3}, {LLAMA_GRETYPE_END, 0}},
        {
            {LLAMA_GRETYPE_CHAR, 97},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 122},
            {LLAMA_GRETYPE_CHAR_ALT, 48},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 57},
            {LLAMA_GRETYPE_CHAR_ALT, 95},
            {LLAMA_GRETYPE_RULE_REF, 10},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_END, 0},
        },
        {
            {LLAMA_GRETYPE_CHAR, 48},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 57},
            {LLAMA_GRETYPE_RULE_REF, 11},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_CHAR, 48},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 57},
            {LLAMA_GRETYPE_END, 0},
        },
        {
            {LLAMA_GRETYPE_CHAR, 32},
            {LLAMA_GRETYPE_CHAR_ALT, 9},
            {LLAMA_GRETYPE_CHAR_ALT, 10},
            {LLAMA_GRETYPE_RULE_REF, 12},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_END, 0},
        },
    };

    std::vector<std::vector<llama_grammar_element>> expected_stacks = {
        {
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_CHAR, 40},
        },
        {
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_CHAR, 48},
        },
        {
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_CHAR, 48},
        },
        {
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_CHAR, 97},
        },
        {
            {LLAMA_GRETYPE_RULE_REF, 5},
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_CHAR, 40},
        },
        {
            {LLAMA_GRETYPE_RULE_REF, 5},
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_CHAR, 48},
        },
        {
            {LLAMA_GRETYPE_RULE_REF, 5},
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_CHAR, 48},
        },
        {
            {LLAMA_GRETYPE_RULE_REF, 5},
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_CHAR, 97},
        }};

    std::vector<std::vector<std::pair<uint32_t, uint16_t>>> expected_reject = {
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {11, 48},
            {12, 49},
            {13, 50},
            {14, 51},
            {15, 52},
            {16, 53},
            {17, 54},
            {18, 55},
            {19, 56},
            {20, 57},
            {21, 58},
            {22, 59},
            {23, 60},
        },
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {3, 40},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {21, 58},
            {22, 59},
            {23, 60},
        },
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {3, 40},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {21, 58},
            {22, 59},
            {23, 60},
        },
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {3, 40},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {11, 48},
            {12, 49},
            {13, 50},
            {14, 51},
            {15, 52},
            {16, 53},
            {17, 54},
            {18, 55},
            {19, 56},
            {20, 57},
            {21, 58},
            {22, 59},
        },
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {11, 48},
            {12, 49},
            {13, 50},
            {14, 51},
            {15, 52},
            {16, 53},
            {17, 54},
            {18, 55},
            {19, 56},
            {20, 57},
            {21, 58},
            {22, 59},
            {23, 60},
        },
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {3, 40},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {21, 58},
            {22, 59},
            {23, 60},
        },
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {3, 40},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {21, 58},
            {22, 59},
            {23, 60},
        },
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {3, 40},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {11, 48},
            {12, 49},
            {13, 50},
            {14, 51},
            {15, 52},
            {16, 53},
            {17, 54},
            {18, 55},
            {19, 56},
            {20, 57},
            {21, 58},
            {22, 59},
        },
    };


    std::vector<std::vector<Element>> rules;
    for (const auto & rule : expected_rules) {
        rules.emplace_back();
        for (const auto & e : rule) rules.back().push_back({(ElemType) (int) e.type, e.value});
    }
    std::string err;
    uint32_t root = 0;
    for (const auto & p : expected) if (p.first == "root") root = p.second;
    auto grammar = Grammar::from_rules(rules, root, err);
    if (!grammar) {
        std::fprintf(stderr, "Failed to initialize grammar: %s\n", err.c_str());
        return 1;
    }
    GrammarState state(grammar);


    // the initial stacks, as a multiset of element lists
    using Flat = std::vector<std::pair<int, uint32_t>>;
    std::vector<Flat> want, got;
    for (const auto & st : expected_stacks) {
        want.emplace_back();
        for (const auto & e : st) want.back().push_back({(int) e.type, e.value});
    }
    for (const auto & st : state.stack_elements()) {
        got.emplace_back();
        for (const auto & e : st) got.back().push_back({(int) e.type, e.value});
    }
    std::sort(want.begin(), want.end());
    std::sort(got.begin(), got.end());
    assert(want.size() == got.size());
    assert(want == got);
    ++n_checks;

    // candidate characters 37..59: accepted iff some stack does not reject them
    std::vector<std::set<uint32_t>> rejects_per_stack;
    for (const auto & rej : expected_reject) {
        rejects_per_stack.emplace_back();
        for (const auto & r : rej) rejects_per_stack.back().insert(r.second);
    }
    assert(rejects_per_stack.size() == expected_stacks.size());
    for (uint32_t cp = 37; cp < 60; ++cp) {
        bool some_stack_accepts = false;
        for (const auto & rs : rejects_per_stack) {
            if (!rs.count(cp)) some_stack_accepts = true;
        }
        bool allowed = state.allows(std::string(1, (char) cp));
        if (allowed != some_stack_accepts) std::fprintf(stderr, "candidate %u: expected %d, got %d\n", cp, (int) some_stack_accepts, (int) allowed);
        assert(allowed == some_stack_accepts);
        ++n_checks;
    }

    std::printf("grammar_llama_state_test: %d checks passed\n", n_checks);
    return 0;
}
