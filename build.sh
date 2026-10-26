#!/bin/sh
set -e
cd "$(dirname "$0")"

export PATH=/usr/local/cuda/bin:"$(dirname "$0")"/.venv/bin:$PATH
rm -f engine/strata
rm -rf build build-vision
.venv/bin/python -c "import setup; setup.build_engine(setup.gpu_info(), 'gpu', True, setup.get_llama_cpp(), toolkit=13)"

# stdbuf log wrapper (rebuilt here because build_engine replaces the engine/ dir)
# The engine's stdout is piped to serve/server.py; libc buffering loses abort
# messages. server.py prepends --serve to the config args, so the config "exe"
# must point at this wrapper, not at /usr/bin/stdbuf directly.
cat > engine/strata-stdbuf <<'EOF'
#!/bin/bash
exec stdbuf -o0 -e0 "$(dirname "$0")/strata" "$@"
EOF
chmod +x engine/strata-stdbuf
echo "wrote engine/strata-stdbuf"
