# Experiment results: how to read and interpret each plot

This repository generates plots mainly from:

- `analysis/plot_uav_log.py` (single-UAV localization / filtering)
- `analysis/causal_analysis.py` (swarm failure detection + causal inference)

---

## Conventions used in `causal_analysis.py`

### 1) Matrix orientation (very important)
Most matrices/heatmaps are labeled **“receiver i, sender j”**.

- **Row = receiver i** (the drone being affected)
- **Column = sender j** (the drone doing the affecting)
- So a bright cell at (i, j) means **j → i** is strong.

Diagonals (i == j) are usually not meaningful (self-edges are excluded in NRI), but they may still appear as zeros.

### 2) What “events” mean in this repo
Events are **heuristic labels** computed from logs, not oracle ground truth:

- **collision:** min inter-drone distance < `collision_dist`
- **formation_loss:** deviation from initial formation offsets > `formation_thresh`
- **gnss_degradation:** `gnss_error_mag` > per-drone quantile threshold
- **wind_loss:** `wind_mag` > per-drone quantile threshold **and** tracking error is increasing (positive slope)
- **suboptimal_traj:** tracking error > per-drone quantile threshold for at least `min_persist_steps`

Because they’re heuristic, interpret all “causal” outputs as evidence/diagnostics, not proof.

---

## Plots from `analysis/plot_uav_log.py`

### A) Trajectory XY (ground truth vs GPS vs EKF)
**What it shows**
- The 2D path in the XY plane.
- Ground truth is a continuous line.
- GPS (if enabled) is scattered points with noise/outliers.
- EKF (if enabled) is a smoothed estimate line.

**How to interpret**
- **EKF close to ground truth** → filter is tracking well.
- **EKF smoother than GPS** is expected (it fuses dynamics + measurements).
- **EKF drifting away** may indicate bias in the motion model, wrong noise tuning, or long measurement dropouts.
- **GPS points far from ground truth** quantify measurement noise/outliers.

**What you can conclude**
- Whether the EKF improves localization compared to raw GPS, and where it fails (sharp turns, high acceleration, bad GNSS, etc.).

### B) Position error norm vs time
**What it shows**
- Magnitude of position error ‖p_est − p_true‖ over time for GPS and/or EKF.

**How to interpret**
- Lower curve = better accuracy.
- **EKF curve below GPS curve** → filtering adds value.
- **Spikes** often correspond to sudden maneuvers, GNSS degradation, or filter re-convergence after outliers.
- **Persistent offset** suggests systematic bias (model mismatch or mis-calibrated sensor noise).

---

## Plots from `analysis/causal_analysis.py`

### 1) `nri_training_loss.png` — NRI training loss (MSE)
**What it shows**
- Training mean-squared error for next-step prediction during NRI training.

**How to interpret**
- A **downward trend then plateau** is healthy.
- **No decrease** → model/optimizer settings may be wrong, or data insufficient.
- **Highly noisy curve** can mean batch instability, too high learning rate, or too-short sequences.

**What you can conclude**
- Whether the NRI model is learning a predictive dynamics model (prerequisite for meaningful inferred edges).

---

### 2) `nri_edge_probs_heatmap.png` — Interaction probability heatmap
**What it shows**
- Matrix of inferred interaction probabilities in [0, 1].
- Entry (i, j) ≈ probability that **drone j influences drone i**.

**How to interpret**
- Bright off-diagonal cells identify likely directed influences **j → i**.
- If one column is bright across many rows, that “sender” is a hub that affects many drones.
- If a row is bright across many columns, that “receiver” is influenced by many others.
- Approximately symmetric bright spots (i, j) and (j, i) suggest bidirectional coupling.

**What you can conclude**
- The *structure* of inferred coupling in the swarm (who affects whom), but not yet the *type* of influence.

---

### 3) `nri_signed_influence_heatmap.png` — Signed influence heatmap (heuristic)
**What it shows**
- Same adjacency, but multiplied by a **sign** derived from observed motion:
  - **Positive**: receiver accelerates away from sender (repulsive)
  - **Negative**: receiver accelerates toward sender (attractive)
- Magnitude still scales with edge probability.

**How to interpret**
- Large positive value at (i, j): strong evidence of **avoidance/repulsion** from j affecting i.
- Large negative value at (i, j): strong evidence of **attraction/cohesion** influence j → i.
- Near zero: either weak edge probability or ambiguous sign in the data.

**What you can conclude**
- A qualitative view of whether coupling looks “repulsive” (collision avoidance) or “attractive” (formation keeping), but it’s a heuristic—validate with domain knowledge.

---

### 4) `nri_inferred_graph.png` — Directed inferred influence graph
**What it shows**
- A thresholded directed graph derived from `edge_probs`.
- An arrow **j → i** appears when edge_prob(i, j) ≥ `graph_thresh`.
- Edge thickness/label reflect the probability weight.

**How to interpret**
- Focus on:
  - **Hubs** (many outgoing edges): potential leaders or strong influencers.
  - **Vulnerable nodes** (many incoming edges): strongly affected by others.
  - **Communities/clusters**: subgroups that move/behave in a coupled way.
- If the plot shows “no edges above threshold”, lower `graph_thresh` or check if training converged.

**What you can conclude**
- A readable summary of the swarm interaction topology implied by NRI.

---

### 5) `nri_edge_type_<k>_heatmap.png` — Edge-type probability heatmaps
**What it shows**
- For each latent edge type k, a matrix of P(type=k for j→i).
- Convention used here: **type 0 = “no edge”**.

**How to interpret**
- Types k>0 are *latent modes* of interaction; they are not automatically “repulsion/attraction” unless you calibrate them.
- Use these heatmaps to see if:
  - Different subsets of edges prefer different types
  - The model is confident (one type dominates) vs uncertain (diffuse probabilities)

**What you can conclude**
- Whether the swarm interactions appear to have multiple distinct regimes (e.g., strong vs weak coupling), even if the semantic meaning of each type needs extra analysis.

---

### 6) `granger_<event>_heatmap.png` — Granger evidence: exogenous factors → event
**What it shows**
- A 2×N matrix per event type:
  - rows: `wind_mag`, `gnss_error_mag`
  - columns: drones
- Score is computed as **1 − (minimum p-value over lags)**, clipped to [0, 1].

**How to interpret**
- Values near **1**: strong statistical evidence that the factor helps predict the event (Granger “causes”).
- Values near **0**: little evidence.
- If wind shows high evidence for *many* event types, it may be a shared driver (confounder) rather than a direct cause for each event.

**What you can conclude (carefully)**
- Whether wind/GNSS changes tend to *precede* a given failure label for each drone.
- Caveats: Granger is linear and sensitive to non-stationarity and autocorrelation; interpret as “predictive precedence,” not definitive causality.

---

### 7) `avg_root_cause_<event>.png` — Average root-cause probabilities per event
**What it shows**
- Heatmap of **cause groups × drones**.
- Cause groups are:
  - `wind` (wind_mag)
  - `gnss` (gnss_error_mag)
  - `interaction` (repulsive force, pairwise distance, interaction pressure)
  - `formation_tracking` (tracking error and formation error)
- The values are normalized **within each event instance** using a softmax over positive contributions.

**How to interpret**
- A brighter cell means that, *when that event happened for that drone*, that group tended to provide a larger share of the model’s positive evidence.
- It’s best read as **“relative attribution among groups”**, not an absolute probability that a group truly caused the failure.

**What you can conclude**
- For each event type, which category of factors most consistently explains positive instances (per drone).

---

### 8) `event_probabilities_<drone>.png` — Predicted failure probabilities over time
**What it shows**
- For each drone, time series of predicted probabilities P(event) for up to 5 event types.

**How to interpret**
- Spikes indicate moments the model believes a failure is likely.
- A good early-warning signal rises **before** a failure label activates (you need to compare against the raw event labels to confirm).
- Frequent spikes with no corresponding events indicate false positives or overly sensitive features.

**What you can conclude**
- Whether the features + simple model can produce actionable risk signals, and which event types dominate risk for each drone.

---

### 9) `impact_<event>_heatmap.png` — Systemic impact / cascading failures
**What it shows**
- For each event type, an N×N matrix:
  - Rows: trigger drone i
  - Columns: affected drone j
- Cell value:
  - impact[i→j] = P(generic failure of j within horizon | event of i at t) − P(generic failure of j)

“Generic failure” is the OR of all failure labels.

**How to interpret**
- Positive values: the trigger event on i tends to **increase** the chance that j fails soon after (possible cascade).
- Near zero: little measurable systemic propagation.
- Negative values can happen due to sampling noise or because the event occurs in “easy” conditions; treat with caution.

**What you can conclude**
- Which drones’ failures are most likely to propagate, and which drones are most susceptible, given the chosen horizon.

---

## Practical guidance for writing the “Results” section
A clean way to narrate results is:

1. **Show NRI convergence** (loss curve) → confirm the model learns predictable dynamics.
2. **Present the coupling structure** (edge-prob heatmap + graph) → highlight strongest edges/hubs.
3. **Connect failures to exogenous drivers** (Granger heatmaps) → show where wind/GNSS precede failures.
4. **Explain failures** (root-cause heatmaps + per-drone probability series) → which factor groups explain which failures and when.
5. **Discuss propagation** (impact heatmaps) → evidence for cascade vs isolated failures.

If you want, you can paste your actual plots (or the `summary_report.json`) and I can help write *plot-specific* interpretations (e.g., “Drone_2 is a hub influencing Drone_3/4…”), rather than the general reading guide above.
