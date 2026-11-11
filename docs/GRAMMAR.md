# Grammar-constrained decoding (opt-in)

`src/grammar` is a port of the Basalt grammar engine (MIT; see `src/grammar/ATTRIBUTION.md`): a GBNF parser, a
JSON-Schema-to-GBNF converter and a live parse state that masks the vocabulary per token. Strata enforces it
**post-window** - the Guard rule: the first token a verify window emitted that leaves the language truncates the
window at its row, and that row's argmax is taken under the grammar's mask instead (`Verifier::copy_logits` +
`argmax_masked`, host side). Windows without a violation - the common case - never leave the graph: same bits,
same speed, every backend. A grammar that dead-ends (no token is allowed at all) ends the request with what was
already committed.

## Command line

```
--grammar FILE          GBNF text; the answers are inside the language from the first generated token
--grammar-schema FILE   JSON Schema, converted with json_schema_to_gbnf() and then enforced as GBNF
```

The two are mutually exclusive; the vocabulary comes from `tokenizer.ggml.tokens` of the first native shard.
Not with `--batch` (its commit path is not wired): asking for a grammar with a batch is a start-up refusal, not
a silent bypass.

## serve protocol

Between requests (never beside a running one), one line sets the grammar for all following requests; `GRAM -`
takes it back off. Both forms carry the text as one line of base64 - GBNF and JSON Schema contain newlines,
which the line protocol cannot:

```
GRAM <base64 GBNF>          GBNF text
GRAMS <base64 JSON Schema>  converted, then enforced as GBNF
GRAM -                      back to no grammar (or to the --grammar file if one was given at start-up)
```

A bad argument answers `ERR grammar: <why>` and changes nothing; a good one is silent, like `BCO`. The grammar
recompiles only when the argument differs from the previous one, so a client may repeat it before every request.

## Limits (the phase this landed in)

* solo windows only: the merged prompt window, `STRATA_FOLLOW` continuations and every `--batch` slot bypass
  enforcement (the second two by design at start-up, merged windows carry no fresh sample).
* the resample is an argmax over the mask - on a temperature request it constrains *where*, not the shape of
  the distribution beyond that.
* the tests live in `tests/grammar` (`grammar_test` ... `grammar_llama_state_test`, `argmax_masked_test`).
