# grammar enforcement A/B (GB10, unsloth UD-IQ3_XXS pack, --spec 4 MTP)

generate loop, greedy, 64 new tokens, same prompt; `--grammar-schema schema.json`
(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}`)
vs no grammar. Wall seconds include model load (~45 s), so the delta is within run noise:
the post-window rule costs one row copy + masked argmax only on a violating row - and this
schema's JSON run violates in almost every window, yet shows no slowdown.

| run | tokens | wall |
|---|---|---|
| no grammar | 64 (+prompt echo 2) | 61.6 s |
| --grammar-schema | 64 (+prompt echo 2) | 59.0 s |

Functional checks the same day: generate `--grammar yesno.gbnf` forces `yes` (88 68 82) where the
free run emits EOS first; serve `GRAM <b64>` forces, `GRAM -` releases (docs/GRAMMAR.md).
Grammar unit tests 8/8 green (`grammar_test` ... `argmax_masked_test`).
