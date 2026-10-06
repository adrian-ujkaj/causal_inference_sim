# GNSS/INS navigation: design and validation

## Filters

| | 15-state ESKF (`Control/ESKF.py`, default) | 6-state filter (`Control/EKF.py`, baseline) |
|---|---|---|
| State | position, velocity, attitude, accelerometer bias, gyro bias | position, velocity |
| Attitude | estimated (error-state on SO(3), quaternion nominal state) | **true attitude given** by the simulator |
| IMU biases | estimated | absorbed by process-noise margin |
| GNSS update | position + velocity, per-measurement R from the receiver's reported accuracy, NIS gate (χ², p = 0.001) with bounded consecutive rejections | same |

Select with `filter.type: eskf | kf6` in `config.yaml`. With
`filter.attitude_source: filter`, the inner attitude loop flies on the ESKF
attitude instead of the true one.

Implementation notes that matter for consistency:

- **IMU as increments.** The simulated IMU delivers the exact rotation of the
  interval and the velocity increment projected with the mid-interval attitude,
  like a real strapdown IMU. Projecting with the end-of-interval attitude would
  create a 0.12 m/s² error at 2 rad/s, the size of a real accelerometer bias.
- **Predict, then update, then publish the corrected state** at each control step.
- **Yaw observability.** There is no magnetometer: yaw is only observable during
  horizontal accelerations and drifts with the gyro bias in hover.

Reference: J. Solà, *Quaternion kinematics for the error-state Kalman filter*, 2017.

## Validation method

`analysis/filter_validation.py` runs Monte-Carlo campaigns of full closed-loop
PyBullet flights (4 drones, generated city, wind) and computes, per drone:

- RMSE of position and velocity, compared with the GNSS error at measurement time;
- share of samples outside the reported 3σ envelope (expected ≈ 0.3 %);
- **NEES** e<sup>T</sup>P<sup>-1</sup>e (needs ground truth; expected value = state
  dimension): above → over-confident filter, below → too conservative;
- **NIS** y<sup>T</sup>S<sup>-1</sup>y on the GNSS innovation (expected 6; available
  in real flight, hence the basis of integrity monitoring).

χ² bounds are printed for reference but are optimistic (successive samples are
correlated); the verdict uses a ±25 % tolerance around the expected value.

## Results (24 flights × 30 s, 4 drones)

| Filter | Successful flights | Position RMSE | Gain vs GNSS | NEES pos-vel (expected 6) | NIS (expected 6) | Outside 3σ |
|---|---|---|---|---|---|---|
| ESKF, 15 states | 96/96 | 0.038–0.040 m | 4.4–4.6× | 5.80–6.11 | 5.95–6.03 | 0.19–0.33 % |
| 6-state filter, drones 0, 1, 3 | 96/96 | 0.039–0.040 m | 4.4× | 4.84–4.96 | 5.53–5.58 | 0.33–0.53 % |
| 6-state filter, **drone_2** (biased accelerometer) | | **0.126 m** | 1.4× | **55.9** | **33.6** | **7.3 %** |

ESKF 15-state NEES (attitude and biases included, expected 15): 12.3–14.4.

drone_2 carries a 0.5 m/s² accelerometer offset (`accel_noise_mean` in
`config.yaml`). The 6-state filter cannot estimate it, even with the true
attitude: it is 3 times less accurate and strongly over-confident (NEES 55.9 instead of 6). The ESKF estimates the bias and stays consistent.
ESKF attitude error: 0.3–0.7° in roll/pitch, 1–2° in yaw (no magnetometer).

## GNSS outage (offline replay, 40 Monte-Carlo runs, 10 s outage)

`analysis/gnss_outage.py` replays a recorded trajectory, regenerates the sensor
measurements 40 times, and runs **both filters on exactly the same measurements**
(a fair comparison: in closed loop, each filter would fly a different trajectory).

| IMU | Filter | Drift at end of outage (median / 95th pct.) | Inside 3σ during outage | Reconvergence |
|---|---|---|---|---|
| navigation-grade MEMS (`mems_nav`) | ESKF | 2.9 m / 4.2 m | 100 % | 0.46 s |
| navigation-grade MEMS (`mems_nav`) | 6-state, **true attitude given** | 4.4 m / 7.0 m | 94.2 % | 0.71 s |
| noisy IMU of `config.yaml` + biases | ESKF | 6.6 m / 11.0 m | 99.9 % | 0.51 s |
| noisy IMU of `config.yaml` + biases | 6-state, **true attitude given** | 5.9 m / 9.6 m | 82.8 % | 0.95 s |

With a realistic MEMS, the ESKF drifts less because it learned the biases before
the outage. With the very noisy gyro of `config.yaml` (about 40 times a consumer
MEMS) it drifts a little more than the 6-state filter, which receives the true
attitude for free. In both cases **the 6-state filter leaves its own envelope: it
is wrong without knowing it**, whereas the ESKF remains consistent (integrity).

![GNSS outage](images/gnss_outage_mems.png)

## Reproduce

```bash
python analysis/filter_validation.py --runs 24 --tmax 30                 # ESKF
python analysis/filter_validation.py --runs 24 --tmax 30 --filter kf6    # 6-state filter
python analysis/gnss_outage.py --truth runs/mc/s1/drone_1_truth.csv --outage 12 22 --runs 40 --imu-preset mems_nav
python analysis/plot_uav_log.py logs/drone_1.csv                          # one drone: trajectory and errors
```

`--jam`, `--outage` and `--agent-override` inject GNSS jamming, outages or any
per-drone configuration change into a campaign.
