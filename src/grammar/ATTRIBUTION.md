# src/grammar - port of the Basalt grammar engine

Ported 1:1 (namespace/include renames only) from https://github.com/jesdga95/basalt
(main@101401ac49a0a9999e27859caa9b12b300139ced, 2026-11-10), MIT licensed.
Copyright (c) the Basalt authors; Basalt itself credits Strata (Niko1221/Strata) as its origin.

Mapping: src/program/grammar -> src/grammar, include/basalt/program/grammar -> include/strata/grammar,
include/basalt/program/json_scan.hpp and include/basalt/log.hpp -> include/strata/grammar/ (kept beside their
only user, guard.cpp), namespace basalt::program::grammar -> strata::grammar. Tests: basalt
tests/program/grammar_*.cpp -> tests/grammar/.

To re-sync with upstream basalt: diff the original paths against this directory and re-apply the mapping above.
