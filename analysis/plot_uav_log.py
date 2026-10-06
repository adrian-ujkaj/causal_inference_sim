#!/usr/bin/env python3
import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def load_log_csv(path):
    """
    Charge le CSV de log produit par UAV et renvoie un dict de np.array.
    Les valeurs vides sont converties en NaN.
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise RuntimeError(f"Fichier vide ou sans header: {path}")

        data = {name: [] for name in fieldnames}

        for row in reader:
            for name in fieldnames:
                val = row.get(name, "")
                if val is None or val == "":
                    data[name].append(np.nan)
                else:
                    try:
                        data[name].append(float(val))
                    except ValueError:
                        data[name].append(np.nan)

    for k in data:
        data[k] = np.asarray(data[k], dtype=float)

    return data


def plot_log(path):
    data = load_log_csv(path)
    title = os.path.basename(path)

    # Colonnes du journal de uav.py
    alias = {
        "t": "time",
        "x_true": "gt_x",
        "y_true": "gt_y",
        "z_true": "gt_z",
        "x_gps": "meas_x",
        "y_gps": "meas_y",
        "z_gps": "meas_z",
        "x_ekf": "ekf_x",
        "y_ekf": "ekf_y",
        "z_ekf": "ekf_z",
    }
    for old, new in alias.items():
        if old not in data and new in data:
            data[old] = data[new]

    t = data["t"]

    # Vérité terrain
    x_true = data["x_true"]
    y_true = data["y_true"]
    z_true = data["z_true"]

    # GPS (peut être NaN si désactivé)
    x_gps = data.get("x_gps", np.full_like(t, np.nan))
    y_gps = data.get("y_gps", np.full_like(t, np.nan))
    z_gps = data.get("z_gps", np.full_like(t, np.nan))

    # EKF (peut être NaN si désactivé)
    x_ekf = data.get("x_ekf", np.full_like(t, np.nan))
    y_ekf = data.get("y_ekf", np.full_like(t, np.nan))
    z_ekf = data.get("z_ekf", np.full_like(t, np.nan))

    # ---------------- Trajectoire XY ----------------
    plt.figure(figsize=(6, 6))
    plt.plot(x_true, y_true, label="Vérité terrain", linewidth=2)

    if np.isfinite(x_gps).any():
        plt.scatter(x_gps, y_gps, s=4, alpha=0.5, label="GPS", marker="x")
    if np.isfinite(x_ekf).any():
        plt.plot(x_ekf, y_ekf, linestyle="--", label="EKF")

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"Trajectoire XY - {title}")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    # ---------------- Erreur de position (norme) ----------------
    # Erreur GPS
    err_gps = np.full_like(t, np.nan)
    if np.isfinite(x_gps).any():
        mask_gps = np.isfinite(x_gps) & np.isfinite(x_true)
        dx = x_gps - x_true
        dy = y_gps - y_true
        dz = z_gps - z_true
        err_norm = np.sqrt(dx**2 + dy**2 + dz**2)
        err_gps[mask_gps] = err_norm[mask_gps]

    # Erreur EKF
    err_ekf = np.full_like(t, np.nan)
    if np.isfinite(x_ekf).any():
        mask_ekf = np.isfinite(x_ekf) & np.isfinite(x_true)
        dx = x_ekf - x_true
        dy = y_ekf - y_true
        dz = z_ekf - z_true
        err_norm = np.sqrt(dx**2 + dy**2 + dz**2)
        err_ekf[mask_ekf] = err_norm[mask_ekf]

    plt.figure(figsize=(8, 4))
    if np.isfinite(err_gps).any():
        plt.plot(t, err_gps, label="Erreur GPS", linewidth=1.5)
    if np.isfinite(err_ekf).any():
        plt.plot(t, err_ekf, label="Erreur EKF", linewidth=1.5)

    plt.xlabel("Temps [s]")
    plt.ylabel("||erreur position|| [m]")
    plt.title(f"Erreur de position - {title}")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage : python analysis/plot_uav_log.py logs/drone_0.csv")
        sys.exit(1)

    log_path = sys.argv[1]
    if not os.path.isfile(log_path):
        print(f"Fichier introuvable : {log_path}")
        sys.exit(1)

    plot_log(log_path)


if __name__ == "__main__":
    main()
