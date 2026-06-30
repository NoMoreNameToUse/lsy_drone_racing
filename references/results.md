(lsy_drone_racing:gpu) spacezhang@spacezhang-ubuntu:~/Desktop/University/ADR_Practica/lsy_drone_racing$ python lsy_drone_racing/control/controllers/utility/sim/evaluate_seeds.py --config level3.toml
W0630 12:08:07.164376 2656054 cuda_executor.cc:1802] GPU interconnect information not available: INTERNAL: NVML doesn't support extracting fabric info or NVLink is not used by the device.
W0630 12:08:07.169921 2656007 cuda_executor.cc:1802] GPU interconnect information not available: INTERNAL: NVML doesn't support extracting fabric info or NVLink is not used by the device.

simplified barebone 

/home/spacezhang/Desktop/University/ADR_Practica/lsy_drone_racing/.pixi/envs/gpu/lib/python3.13/site-packages/jax/_src/abstract_arrays.py:135: RuntimeWarning: overflow encountered in cast
  return literals.TypedNdArray(np.asarray(x, dtype), weak_type=False)
[ 1/36] seed           4  PASS     t= 8.16s  vmax=2.23  -> finished
[ 2/36] seed          10  PASS     t= 9.46s  vmax=2.17  -> finished
[ 3/36] seed   537525082  PASS     t=10.18s  vmax=1.87  -> finished
[ 4/36] seed   548046238  PASS     t= 9.38s  vmax=1.97  -> finished
[ 5/36] seed   437125220  PASS     t= 9.50s  vmax=1.90  -> finished
[ 6/36] seed   334327978  PASS     t=13.68s  vmax=1.99  -> finished
[ 7/36] seed  1898163101  PASS     t= 8.54s  vmax=2.25  -> finished
[ 8/36] seed   380878779  PASS     t= 7.60s  vmax=2.03  -> finished
[ 9/36] seed   368113776  PASS     t=11.58s  vmax=2.01  -> finished
[10/36] seed  1522539070  PASS     t=11.44s  vmax=1.82  -> finished
[11/36] seed   394335597  PASS     t= 7.14s  vmax=2.19  -> finished
[12/36] seed  2070717517  PASS     t=12.62s  vmax=2.02  -> finished
[13/36] seed  1756798599  PASS     t=11.12s  vmax=2.36  -> finished
[14/36] seed   601977100  PASS     t= 8.04s  vmax=2.05  -> finished
[15/36] seed   447542283  PASS     t=12.78s  vmax=1.92  -> finished
[16/36] seed  1943230833  FAIL@g0  t= 1.86s  vmax=2.07  -> collision near gate 0 (d=0.11 m) while targeting gate 0
[17/36] seed  1008680058  FAIL@g1  t= 2.94s  vmax=2.08  -> collision near gate 0 (d=0.14 m) while targeting gate 1
[18/36] seed   992857546  FAIL@g1  t= 2.08s  vmax=2.65  -> collision near gate 0 (d=0.16 m) while targeting gate 1
[19/36] seed  1674664546  FAIL@g1  t= 2.40s  vmax=1.96  -> collision near obstacle 0 (d=0.13 m) while targeting gate 1
[20/36] seed   223246164  FAIL@g0  t= 2.10s  vmax=2.63  -> collision near gate 0 (d=0.44 m) while targeting gate 0
[21/36] seed  1243516087  FAIL@g3  t= 8.68s  vmax=2.28  -> collision near gate 2 (d=0.10 m) while targeting gate 3
[22/36] seed   168155228  FAIL@g3  t= 6.80s  vmax=2.00  -> collision near gate 2 (d=0.10 m) while targeting gate 3
[23/36] seed  1015453627  FAIL@g1  t= 3.36s  vmax=2.15  -> collision near obstacle 1 (d=0.13 m) while targeting gate 1
[24/36] seed   502743744  FAIL@g0  t= 0.76s  vmax=1.67  -> collision near obstacle 0 (d=0.17 m) while targeting gate 0
[25/36] seed  1749845046  FAIL@g3  t= 9.46s  vmax=1.94  -> collision near obstacle 3 (d=0.23 m) while targeting gate 3
[26/36] seed  1609583425  FAIL@g3  t=11.96s  vmax=2.13  -> controller ended trajectory early at gate 3 (never reached the finish)
[27/36] seed   826175110  FAIL@g1  t= 1.86s  vmax=2.52  -> collision near gate 0 (d=0.23 m) while targeting gate 1
[28/36] seed  1450051549  FAIL@g2  t= 4.70s  vmax=1.66  -> collision near gate 1 (d=0.10 m) while targeting gate 2
[29/36] seed   780252868  FAIL@g1  t= 2.68s  vmax=2.11  -> collision near obstacle 0 (d=0.15 m) while targeting gate 1
[30/36] seed  1345132680  FAIL@g1  t= 2.58s  vmax=2.17  -> collision near gate 0 (d=0.23 m) while targeting gate 1
[31/36] seed   350327268  FAIL@g2  t= 4.68s  vmax=2.44  -> collision near gate 1 (d=0.13 m) while targeting gate 2
[32/36] seed   673638594  FAIL@g1  t= 1.84s  vmax=2.62  -> collision near gate 0 (d=0.18 m) while targeting gate 1
[33/36] seed   196241771  FAIL@g2  t= 3.70s  vmax=2.08  -> collision near gate 1 (d=0.15 m) while targeting gate 2
[34/36] seed   191876941  FAIL@g0  t= 1.18s  vmax=2.00  -> collision near obstacle 0 (d=0.58 m) while targeting gate 0
[35/36] seed   848575542  FAIL@g0  t= 1.02s  vmax=2.15  -> collision near obstacle 0 (d=0.42 m) while targeting gate 0
[36/36] seed    72996710  FAIL@g2  t= 4.72s  vmax=2.28  -> collision near gate 1 (d=0.19 m) while targeting gate 2

==============================================================================
SEED EVALUATION REPORT
config=level3.toml  seeds=36  generated=2026-06-30T12:08:47
==============================================================================
Total passed     : 15 / 36  (41.7%)
Median speed     : 0.948 m/s (passed runs)   | all runs: 0.964 m/s
Median finish    : 9.5 s (passed runs)
Fastest / slowest finish: 7.14 s / 13.68 s

Not-passed gate histogram (gate -> #fails): {0: 5, 1: 8, 2: 4, 3: 4}
Failure categories: {'collision': 20, 'early': 1}
** 8 of 21 failures were on INFEASIBLE tracks (obstacle clipped a gate) -- not a controller bug. **

------------------------------------------------------------------------------
FAILED RUNS
------------------------------------------------------------------------------
seed   191876941  gate 0/4  t= 1.18s  vmax=2.00  reason: collision near obstacle 0 (d=0.58 m) while targeting gate 0
seed   223246164  gate 0/4  t= 2.10s  vmax=2.63  reason: collision near gate 0 (d=0.44 m) while targeting gate 0  [INFEASIBLE TRACK]
seed   502743744  gate 0/4  t= 0.76s  vmax=1.67  reason: collision near obstacle 0 (d=0.17 m) while targeting gate 0
seed   848575542  gate 0/4  t= 1.02s  vmax=2.15  reason: collision near obstacle 0 (d=0.42 m) while targeting gate 0
seed  1943230833  gate 0/4  t= 1.86s  vmax=2.07  reason: collision near gate 0 (d=0.11 m) while targeting gate 0
seed   673638594  gate 1/4  t= 1.84s  vmax=2.62  reason: collision near gate 0 (d=0.18 m) while targeting gate 1
seed   780252868  gate 1/4  t= 2.68s  vmax=2.11  reason: collision near obstacle 0 (d=0.15 m) while targeting gate 1
seed   826175110  gate 1/4  t= 1.86s  vmax=2.52  reason: collision near gate 0 (d=0.23 m) while targeting gate 1  [INFEASIBLE TRACK]
seed   992857546  gate 1/4  t= 2.08s  vmax=2.65  reason: collision near gate 0 (d=0.16 m) while targeting gate 1
seed  1008680058  gate 1/4  t= 2.94s  vmax=2.08  reason: collision near gate 0 (d=0.14 m) while targeting gate 1  [INFEASIBLE TRACK]
seed  1015453627  gate 1/4  t= 3.36s  vmax=2.15  reason: collision near obstacle 1 (d=0.13 m) while targeting gate 1
seed  1345132680  gate 1/4  t= 2.58s  vmax=2.17  reason: collision near gate 0 (d=0.23 m) while targeting gate 1  [INFEASIBLE TRACK]
seed  1674664546  gate 1/4  t= 2.40s  vmax=1.96  reason: collision near obstacle 0 (d=0.13 m) while targeting gate 1
seed    72996710  gate 2/4  t= 4.72s  vmax=2.28  reason: collision near gate 1 (d=0.19 m) while targeting gate 2  [INFEASIBLE TRACK]
seed   196241771  gate 2/4  t= 3.70s  vmax=2.08  reason: collision near gate 1 (d=0.15 m) while targeting gate 2  [INFEASIBLE TRACK]
seed   350327268  gate 2/4  t= 4.68s  vmax=2.44  reason: collision near gate 1 (d=0.13 m) while targeting gate 2
seed  1450051549  gate 2/4  t= 4.70s  vmax=1.66  reason: collision near gate 1 (d=0.10 m) while targeting gate 2  [INFEASIBLE TRACK]
seed   168155228  gate 3/4  t= 6.80s  vmax=2.00  reason: collision near gate 2 (d=0.10 m) while targeting gate 3
seed  1243516087  gate 3/4  t= 8.68s  vmax=2.28  reason: collision near gate 2 (d=0.10 m) while targeting gate 3  [INFEASIBLE TRACK]
seed  1609583425  gate 3/4  t=11.96s  vmax=2.13  reason: controller ended trajectory early at gate 3 (never reached the finish)
seed  1749845046  gate 3/4  t= 9.46s  vmax=1.94  reason: collision near obstacle 3 (d=0.23 m) while targeting gate 3

------------------------------------------------------------------------------
PASSED RUNS (seed: time s @ avg m/s, vmax)
------------------------------------------------------------------------------
seed   394335597  t= 7.14s  avg=0.98 m/s  vmax=2.19
seed   380878779  t= 7.60s  avg=0.96 m/s  vmax=2.03
seed   601977100  t= 8.04s  avg=0.96 m/s  vmax=2.05
seed           4  t= 8.16s  avg=1.00 m/s  vmax=2.23
seed  1898163101  t= 8.54s  avg=0.94 m/s  vmax=2.25
seed   548046238  t= 9.38s  avg=0.93 m/s  vmax=1.97
seed          10  t= 9.46s  avg=0.92 m/s  vmax=2.17
seed   437125220  t= 9.50s  avg=0.93 m/s  vmax=1.90
seed   537525082  t=10.18s  avg=0.94 m/s  vmax=1.87
seed  1756798599  t=11.12s  avg=1.05 m/s  vmax=2.36
seed  1522539070  t=11.44s  avg=0.95 m/s  vmax=1.82
seed   368113776  t=11.58s  avg=0.97 m/s  vmax=2.01
seed  2070717517  t=12.62s  avg=0.89 m/s  vmax=2.02
seed   447542283  t=12.78s  avg=0.95 m/s  vmax=1.92
seed   334327978  t=13.68s  avg=0.92 m/s  vmax=1.99
==============================================================================
Wall-clock eval time: 32.0 s for 36 seeds


improved baseline 

Saved report to /home/spacezhang/Desktop/University/ADR_Practica/lsy_drone_racing/debug_outputs/seed_eval_report.txt and .json
(lsy_drone_racing:gpu) spacezhang@spacezhang-ubuntu:~/Desktop/University/ADR_Practica/lsy_drone_racing$ python lsy_drone_racing/control/controllers/utility/sim/evaluate_seeds.py --config level3.toml
W0630 12:11:23.947856 2663508 cuda_executor.cc:1802] GPU interconnect information not available: INTERNAL: NVML doesn't support extracting fabric info or NVLink is not used by the device.
W0630 12:11:23.950529 2663449 cuda_executor.cc:1802] GPU interconnect information not available: INTERNAL: NVML doesn't support extracting fabric info or NVLink is not used by the device.
/home/spacezhang/Desktop/University/ADR_Practica/lsy_drone_racing/.pixi/envs/gpu/lib/python3.13/site-packages/jax/_src/abstract_arrays.py:135: RuntimeWarning: overflow encountered in cast
  return literals.TypedNdArray(np.asarray(x, dtype), weak_type=False)
[ 1/36] seed           4  PASS     t= 9.46s  vmax=2.54  -> finished
[ 2/36] seed          10  PASS     t= 9.92s  vmax=2.17  -> finished
[ 3/36] seed   537525082  PASS     t=10.26s  vmax=1.88  -> finished
[ 4/36] seed   548046238  PASS     t= 9.54s  vmax=1.97  -> finished
[ 5/36] seed   437125220  FAIL@g0  t= 1.22s  vmax=1.63  -> collision near gate 0 (d=0.19 m) while targeting gate 0
[ 6/36] seed   334327978  PASS     t=13.78s  vmax=2.05  -> finished
[ 7/36] seed  1898163101  PASS     t= 8.66s  vmax=2.25  -> finished
[ 8/36] seed   380878779  PASS     t= 8.12s  vmax=1.67  -> finished
[ 9/36] seed   368113776  PASS     t=11.60s  vmax=2.00  -> finished
[10/36] seed  1522539070  PASS     t=11.78s  vmax=1.97  -> finished
[11/36] seed   394335597  PASS     t= 7.16s  vmax=2.19  -> finished
[12/36] seed  2070717517  PASS     t=12.32s  vmax=2.02  -> finished
[13/36] seed  1756798599  PASS     t=11.48s  vmax=1.83  -> finished
[14/36] seed   601977100  PASS     t= 8.06s  vmax=2.06  -> finished
[15/36] seed   447542283  PASS     t=12.48s  vmax=2.04  -> finished
[16/36] seed  1943230833  PASS     t=12.06s  vmax=2.06  -> finished
[17/36] seed  1008680058  FAIL@g1  t= 4.08s  vmax=2.09  -> collision near obstacle 1 (d=0.12 m) while targeting gate 1
[18/36] seed   992857546  FAIL@g2  t= 6.24s  vmax=2.65  -> collision near gate 1 (d=0.12 m) while targeting gate 2
[19/36] seed  1674664546  PASS     t= 9.78s  vmax=2.15  -> finished
[20/36] seed   223246164  FAIL@g0  t= 2.10s  vmax=2.63  -> collision near gate 0 (d=0.44 m) while targeting gate 0
[21/36] seed  1243516087  PASS     t=10.58s  vmax=2.28  -> finished
[22/36] seed   168155228  FAIL@g3  t= 9.62s  vmax=2.00  -> collision near gate 3 (d=0.16 m) while targeting gate 3
[23/36] seed  1015453627  FAIL@g1  t= 3.84s  vmax=2.14  -> collision near obstacle 1 (d=0.11 m) while targeting gate 1
[24/36] seed   502743744  FAIL@g0  t= 0.74s  vmax=1.68  -> collision near obstacle 0 (d=0.17 m) while targeting gate 0
[25/36] seed  1749845046  FAIL@g3  t= 9.32s  vmax=1.95  -> collision near obstacle 3 (d=0.20 m) while targeting gate 3
[26/36] seed  1609583425  FAIL@g3  t= 9.76s  vmax=2.13  -> collision near obstacle 3 (d=0.26 m) while targeting gate 3
[27/36] seed   826175110  FAIL@g1  t= 5.52s  vmax=2.52  -> collision near obstacle 1 (d=0.11 m) while targeting gate 1
[28/36] seed  1450051549  FAIL@g2  t= 5.22s  vmax=1.70  -> collision near gate 1 (d=0.10 m) while targeting gate 2
[29/36] seed   780252868  PASS     t= 8.80s  vmax=2.46  -> finished
[30/36] seed  1345132680  FAIL@g1  t= 4.26s  vmax=2.16  -> collision near gate 1 (d=0.07 m) while targeting gate 1
[31/36] seed   350327268  PASS     t=12.00s  vmax=2.45  -> finished
[32/36] seed   673638594  FAIL@g1  t= 1.82s  vmax=2.62  -> collision near gate 0 (d=0.17 m) while targeting gate 1
[33/36] seed   196241771  PASS     t= 9.12s  vmax=2.55  -> finished
[34/36] seed   191876941  FAIL@g3  t= 9.92s  vmax=2.09  -> controller ended trajectory early at gate 3 (never reached the finish)
[35/36] seed   848575542  FAIL@g0  t= 1.54s  vmax=2.16  -> collision near obstacle 0 (d=0.35 m) while targeting gate 0
[36/36] seed    72996710  FAIL@g3  t= 9.46s  vmax=2.28  -> collision near gate 3 (d=0.21 m) while targeting gate 3

==============================================================================
SEED EVALUATION REPORT
config=level3.toml  seeds=36  generated=2026-06-30T12:12:22
==============================================================================
Total passed     : 20 / 36  (55.6%)
Median speed     : 0.936 m/s (passed runs)   | all runs: 0.937 m/s
Median finish    : 10.09 s (passed runs)
Fastest / slowest finish: 7.16 s / 13.78 s

Not-passed gate histogram (gate -> #fails): {0: 4, 1: 5, 2: 2, 3: 5}
Failure categories: {'collision': 15, 'early': 1}
** 7 of 16 failures were on INFEASIBLE tracks (obstacle clipped a gate) -- not a controller bug. **

------------------------------------------------------------------------------
FAILED RUNS
------------------------------------------------------------------------------
seed   223246164  gate 0/4  t= 2.10s  vmax=2.63  reason: collision near gate 0 (d=0.44 m) while targeting gate 0  [INFEASIBLE TRACK]
seed   437125220  gate 0/4  t= 1.22s  vmax=1.63  reason: collision near gate 0 (d=0.19 m) while targeting gate 0  [INFEASIBLE TRACK]
seed   502743744  gate 0/4  t= 0.74s  vmax=1.68  reason: collision near obstacle 0 (d=0.17 m) while targeting gate 0
seed   848575542  gate 0/4  t= 1.54s  vmax=2.16  reason: collision near obstacle 0 (d=0.35 m) while targeting gate 0
seed   673638594  gate 1/4  t= 1.82s  vmax=2.62  reason: collision near gate 0 (d=0.17 m) while targeting gate 1
seed   826175110  gate 1/4  t= 5.52s  vmax=2.52  reason: collision near obstacle 1 (d=0.11 m) while targeting gate 1  [INFEASIBLE TRACK]
seed  1008680058  gate 1/4  t= 4.08s  vmax=2.09  reason: collision near obstacle 1 (d=0.12 m) while targeting gate 1  [INFEASIBLE TRACK]
seed  1015453627  gate 1/4  t= 3.84s  vmax=2.14  reason: collision near obstacle 1 (d=0.11 m) while targeting gate 1
seed  1345132680  gate 1/4  t= 4.26s  vmax=2.16  reason: collision near gate 1 (d=0.07 m) while targeting gate 1  [INFEASIBLE TRACK]
seed   992857546  gate 2/4  t= 6.24s  vmax=2.65  reason: collision near gate 1 (d=0.12 m) while targeting gate 2
seed  1450051549  gate 2/4  t= 5.22s  vmax=1.70  reason: collision near gate 1 (d=0.10 m) while targeting gate 2  [INFEASIBLE TRACK]
seed    72996710  gate 3/4  t= 9.46s  vmax=2.28  reason: collision near gate 3 (d=0.21 m) while targeting gate 3  [INFEASIBLE TRACK]
seed   168155228  gate 3/4  t= 9.62s  vmax=2.00  reason: collision near gate 3 (d=0.16 m) while targeting gate 3
seed   191876941  gate 3/4  t= 9.92s  vmax=2.09  reason: controller ended trajectory early at gate 3 (never reached the finish)
seed  1609583425  gate 3/4  t= 9.76s  vmax=2.13  reason: collision near obstacle 3 (d=0.26 m) while targeting gate 3
seed  1749845046  gate 3/4  t= 9.32s  vmax=1.95  reason: collision near obstacle 3 (d=0.20 m) while targeting gate 3

------------------------------------------------------------------------------
PASSED RUNS (seed: time s @ avg m/s, vmax)
------------------------------------------------------------------------------
seed   394335597  t= 7.16s  avg=0.99 m/s  vmax=2.19
seed   601977100  t= 8.06s  avg=0.96 m/s  vmax=2.06
seed   380878779  t= 8.12s  avg=0.89 m/s  vmax=1.67
seed  1898163101  t= 8.66s  avg=0.93 m/s  vmax=2.25
seed   780252868  t= 8.80s  avg=1.01 m/s  vmax=2.46
seed   196241771  t= 9.12s  avg=0.94 m/s  vmax=2.55
seed           4  t= 9.46s  avg=0.93 m/s  vmax=2.54
seed   548046238  t= 9.54s  avg=0.92 m/s  vmax=1.97
seed  1674664546  t= 9.78s  avg=0.97 m/s  vmax=2.15
seed          10  t= 9.92s  avg=0.90 m/s  vmax=2.17
seed   537525082  t=10.26s  avg=0.94 m/s  vmax=1.88
seed  1243516087  t=10.58s  avg=0.97 m/s  vmax=2.28
seed  1756798599  t=11.48s  avg=0.89 m/s  vmax=1.83
seed   368113776  t=11.60s  avg=0.97 m/s  vmax=2.00
seed  1522539070  t=11.78s  avg=0.94 m/s  vmax=1.97
seed   350327268  t=12.00s  avg=1.00 m/s  vmax=2.45
seed  1943230833  t=12.06s  avg=0.92 m/s  vmax=2.06
seed  2070717517  t=12.32s  avg=0.90 m/s  vmax=2.02
seed   447542283  t=12.48s  avg=0.98 m/s  vmax=2.04
seed   334327978  t=13.78s  avg=0.91 m/s  vmax=2.05
==============================================================================
Wall-clock eval time: 50.1 s for 36 seeds

Saved report to /home/spacezhang/Desktop/University/ADR_Practica/lsy_drone_racing/debug_outputs/seed_eval_report.txt and .json
(lsy_drone_racing:gpu) spacezhang@spacezhang-ubuntu:~/Desktop/University/ADR_Practica/lsy_drone_racing$ 