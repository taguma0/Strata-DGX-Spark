#!/bin/sh
set -e
cd "$(dirname "$0")"

# 0.2.16 layout: setup.py and the installed engine live in terminal/ (the repo's engine/ is the engine's sources)
export PATH=/usr/local/cuda/bin:"$(dirname "$0")"/terminal/.venv/bin:$PATH
rm -f terminal/engine/strata terminal/engine/strata-vision
rm -rf terminal/build terminal/build-vision
(cd terminal && .venv/bin/python -c "import setup; setup.build_engine(setup.gpu_info(), 'gpu', True, setup.get_llama_cpp(), toolkit=13)")

# stdbuf log wrapper (rebuilt here because build_engine replaces the engine/ dir)
# The engine's stdout is piped to serve/server.py; libc buffering loses abort
# messages. server.py prepends --serve to the config args, so the config "exe"
# must point at this wrapper, not at /usr/bin/stdbuf directly.
cat > terminal/engine/strata-stdbuf <<'EOF'
#!/bin/bash
exec stdbuf -o0 -e0 "$(dirname "$0")/strata" "$@"
EOF
chmod +x terminal/engine/strata-stdbuf
echo "wrote terminal/engine/strata-stdbuf"

# Shape engines (strata_control/shape_engine.py): Inkling-2-Mini (and Hy4) run on their own
# executables, NOT the main strata. The run configs' exe points at
# terminal/engine-inkling2/strata-inkling; the signed feed has no linux-arm64 shape package,
# so GB10 must compile them here (setup.py build_shape_engine -> CMake target strata-<shape>
# from engine/, installs to terminal/engine-<shape>/ + BUILD.json). Incremental by design:
# the cmake cache in engine/build-<shape> survives; do NOT wipe it here.
# Default shapes: inkling2. Extend per run:  ./build.sh inkling2 hy4
for shape in ${*:-inkling2}; do
  (cd terminal && .venv/bin/python -c "import setup; setup.build_shape_engine('$shape', setup.gpu_info(), False, 13, True)")
done
