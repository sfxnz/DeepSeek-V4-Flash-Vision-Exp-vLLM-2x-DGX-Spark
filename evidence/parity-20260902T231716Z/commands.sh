# parity-deepseek Wed Sep  2 11:17:16 PM UTC 2026
unit=parity-deepseek lane=pair ttl=2h
HEAD=dc6de7802c8d5b69701423ca6254355d014f3c0b branch=agent/kit
image rebuilt in evidence/rebuild-20260902T224800Z
kit/doctor.sh . | tee $EV/doctor-preboot.txt
VALIDATE_ONLY=1 ./run.sh | tee $EV/validate.txt
./run.sh 2>&1 | tee $EV/run.log   # defaults, no env overrides
kit/doctor.sh . | tee $EV/doctor.txt
docker inspect ... | tee $EV/image-check.txt
kit/probes/run-all.sh . $EV
python3 kit/bench_decode.py --recipe . --phase both --out $EV
python3 kit/bench_compare.py evidence/rebench-20260902T202935Z/bench.json $EV/bench.json | tee $EV/parity.txt
./stop.sh 2>&1 | tee $EV/stop.txt
