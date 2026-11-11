// include/strata/program/grammar/guard.hpp - one request's grammar, checked against the tokens a verify window produced.
//
// vocab_from_pack builds token id -> bytes from the pack's tokenizer files. A Guard holds the request's GrammarState and,
// after each window, scans the sampled tokens in order: the first one the grammar does not allow is reported, the caller
// resamples that row under mask_bias() and hands the new token to take(). Nothing here knows a GPU or an OS.
#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "strata/grammar/grammar.hpp"

namespace strata::grammar {

struct PackVocab {
    std::shared_ptr<const Vocab> vocab;
    int32_t think_open = -1;    ///< the token that opens a thinking block, or -1
    int32_t think_close = -1;   ///< the token that closes it, or -1
};

/// `vocab_json` is tokenizer/vocab.json ({"token text": id}, byte-level BPE spelling), `types_json` is
/// tokenizer/token_type.json (one type per id: 3 control, 4 user-defined added token, 5 unused, 6 byte). Ids from
/// 0 to n_vocab - 1; an id without an entry has no text. `end_ids` are the tokens that finish generation.
bool vocab_from_pack(std::string_view vocab_json, std::string_view types_json, size_t n_vocab,
                     const std::vector<int32_t>& end_ids, PackVocab& out, std::string& err);

class Guard {
public:
    Guard(PackVocab vocab) : pv_(std::move(vocab)) {}

    /// Starts a request. `grammar_json` is the request's "grammar" object. `prompt` ends where generation starts: an
    /// open thinking block there defers the grammar until the block is closed. False with `err` on an unusable grammar.
    /// A request whose grammar is null (tool_choice "none") leaves the guard off.
    bool begin(const std::string& grammar_json, const int64_t* prompt, size_t n_prompt, std::string& err);
    bool on() const { return on_; }

    /// Feeds out[0..rows) in order and returns the first row the grammar does not allow, or -1. The state has then
    /// advanced over every row before it.
    int scan(const int32_t* out, int rows);
    /// After a miss from scan(): bias[id] = user_bias[id] where the grammar allows id, else -inf. `user_bias` may be empty.
    void mask_bias(const std::vector<float>& user_bias, std::vector<float>& bias);
    /// The token drawn under mask_bias(). False (and the guard turns itself off) if it is not allowed after all.
    bool take(int32_t token);

    int64_t windows() const { return windows_; }
    int64_t misses() const { return misses_; }
    double ms() const { return (double) ns_ / 1e6; }

private:
    // 1: fed, 0: not allowed
    bool feed(int32_t token);

    PackVocab pv_;
    GrammarState st_;
    std::vector<uint8_t> mask_;
    bool on_ = false;
    bool thinking_ = false;
    int64_t windows_ = 0, misses_ = 0, ns_ = 0;
};

}  // namespace strata::grammar
