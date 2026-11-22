#!/bin/bash
cd /home/taguma/Strata-DGX-Spark || exit 1
RES=bench/results/2026-11-22-df2-profile
echo "=== stop prod $(date +%H:%M:%S)"
pkill -f "[s]erve/server.py" 2>/dev/null; sleep 6; pkill -9 -f "[t]erminal/engine/strata" 2>/dev/null; sleep 3
echo "=== start profile arm $(date +%H:%M:%S)"
( STRATA_DF2_PROFILE=1 terminal/.venv/bin/python terminal/serve/server.py --engine strata --config terminal/strata-profile.json --host :: --port 8000 > $RES/server3.log 2>&1 & )
python3 - <<PY
import json,urllib.request,time
def req(i):
    body={"model":"strata","max_tokens":256,"temperature":0,"reasoning_effort":"none","messages":[{"role":"user","content":"Write a detailed story about a robot painter. pass %d"%i}]}
    r=urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r,timeout=600))
i=0;tries=0
while i<4:
    try:
        d=req(i); tm=d["timings"]
        print("pass %d: decode %.1f tok/s"%(i,tm["predicted_per_second"]),flush=True); i+=1
    except Exception as e:
        tries+=1
        if tries>40: raise
        print("retry:",repr(e)[:60],flush=True); time.sleep(15)
PY
sleep 3
echo "=== graceful: stop the SERVER (it stops the engine in turn) $(date +%H:%M:%S)"
pkill -f "[s]trata-profile.json" 2>/dev/null
for i in $(seq 1 45); do pgrep -f "[t]erminal/engine/strata --serve" >/dev/null || break; sleep 2; done
sleep 2; pkill -9 -f "[t]erminal/engine/strata" 2>/dev/null; sleep 1
cp terminal/strata-profile.log $RES/engine-run3.log 2>/dev/null
echo "=== profile dump:"; grep -i "dflash2 profile" $RES/engine-run3.log server3.log || tail -12 server3.log
echo "=== relaunch prod"
setsid nohup terminal/.venv/bin/python terminal/serve/server.py --engine strata --config terminal/strata-unsloth-ud-iq3_xxs.json --host :: --port 8000 >>terminal/server-restart.log 2>&1 &
sleep 12
curl -s -o /dev/null -w "prod_relaunch_http=%{http_code}\n" http://127.0.0.1:8000/v1/models
echo P23D_WINDOW_DONE
