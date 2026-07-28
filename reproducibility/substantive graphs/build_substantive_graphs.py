"""Build substantive, thesis-ready visual explanations.

The figures in this directory are intentionally split into:

* conceptual figures that make the argument and estimands explicit; and
* empirical/descriptive figures backed by frozen project outputs.

Every figure is exported as SVG for Typst and PNG for quick inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
import numpy as np
import pandas as pd
import patsy
import requests
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
FUNCTIONAL_EFFECTS = (
    ROOT
    / "outputs/test_speeches/empirical_regression_audit/functional_form_effects.csv"
)
JAB_HYBRID_RANK_PANEL = (
    ROOT
    / "outputs/test_speeches/plda_jab_extended/robustness_samples/hybrid_extended/data/rank_salience_panel.csv"
)

INK = "#25313A"
MUTED = "#64737D"
GRID = "#D9E0E4"
PAPER = "#FFFFFF"
PALE = "#F2F5F6"
BLUE = "#2C6E9B"
ORANGE = "#C96B32"
GREEN = "#39856B"
PURPLE = "#76558F"
RED = "#B24A62"
GOLD = "#B18428"

REGION_COLORS = {
    "Central-East Europe": PURPLE,
    "Nordics": BLUE,
    "West Europe": GREEN,
    "South Europe": ORANGE,
    "Separate case": RED,
}

COUNTRIES = pd.DataFrame(
    [
        ("AT", "AUT", "Austria", "West Europe", 383_090, 45_234, "1996-01-15", "2022-10-12", 13.3, 47.6),
        ("BE", "BEL", "Belgium", "West Europe", 616_502, 119_380, "2014-06-19", "2022-07-13", 4.7, 50.8),
        ("CZ", "CZE", "Czech Republic", "Central-East Europe", 269_469, 9_996, "2013-11-25", "2023-07-26", 15.4, 49.8),
        ("DK", "DNK", "Denmark", "Nordics", 190_641, 12_939, "2014-10-07", "2022-06-07", 9.3, 56.1),
        ("EE", "EST", "Estonia", "Central-East Europe", 228_418, 13_116, "2011-04-04", "2022-06-17", 25.5, 58.6),
        ("ES", "ESP", "Spain", "South Europe", 146_848, 59_890, "2015-01-20", "2023-02-23", -3.7, 40.4),
        ("FI", "FIN", "Finland", "Nordics", 302_662, 14_059, "2015-05-05", "2022-01-28", 26.0, 64.8),
        ("GB", "GBR", "Great Britain", "Separate case", 687_566, 29_894, "2017-01-09", "2021-12-16", -3.3, 54.4),
        ("GR", "GRC", "Greece", "South Europe", 931_143, 11_780, "2015-02-06", "2022-02-01", 22.2, 39.1),
        ("IT", "ITA", "Italy", "South Europe", 201_330, 12_345, "2013-03-15", "2022-09-20", 12.5, 42.6),
        ("LV", "LVA", "Latvia", "Central-East Europe", 184_858, 1_029, "2014-11-04", "2022-10-27", 24.6, 57.0),
        ("NL", "NLD", "Netherlands", "West Europe", 655_389, 70_275, "2014-04-16", "2022-07-06", 5.5, 52.2),
        ("NO", "NOR", "Norway", "Nordics", 1_195_867, 85_127, "1998-10-01", "2022-05-12", 10.0, 62.0),
        ("PL", "POL", "Poland", "Central-East Europe", 505_075, 13_041, "2015-11-12", "2022-06-30", 19.2, 52.0),
        ("PT", "PRT", "Portugal", "South Europe", 510_282, 55_236, "2015-01-07", "2024-03-13", -8.0, 39.7),
        ("SE", "SWE", "Sweden", "Nordics", 529_071, 9_371, "2015-09-15", "2022-05-17", 15.2, 62.0),
    ],
    columns=[
        "code",
        "iso3",
        "country",
        "region",
        "speech_rows",
        "manifesto_quasi",
        "start",
        "end",
        "label_lon",
        "label_lat",
    ],
)
COUNTRIES["start"] = pd.to_datetime(COUNTRIES["start"])
COUNTRIES["end"] = pd.to_datetime(COUNTRIES["end"])


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "svg.fonttype": "none",
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def clean_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def node(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    title: str,
    body: str = "",
    *,
    color: str = BLUE,
    title_size: float = 9.5,
    body_size: float = 8.2,
) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.15,
        edgecolor=color,
        facecolor=color + "18",
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * (0.62 if body else 0.5),
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        weight="bold",
        color=INK,
    )
    if body:
        ax.text(
            x + w / 2,
            y + h * 0.28,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            color=MUTED,
            linespacing=1.25,
        )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    connectionstyle: str = "arc3",
    linewidth: float = 1.15,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=linewidth,
            color=color,
            connectionstyle=connectionstyle,
        )
    )


def figure_research_design() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    node(ax, (0.03, 0.78), (0.27, 0.15), "Manifesto Project", "Programmatic baseline\nparty-election quasi-sentences", color=PURPLE)
    node(ax, (0.365, 0.78), (0.27, 0.15), "ParlaMint", "Parliamentary behaviour\nspeech segments and dates", color=BLUE)
    node(ax, (0.70, 0.78), (0.27, 0.15), "ParlGov", "Political context\nelections, cabinets, seat share", color=GREEN)

    node(
        ax,
        (0.25, 0.55),
        (0.50, 0.13),
        "Party–time linkage",
        "map party identifiers • select latest prior manifesto • locate party in electoral cycle",
        color=GOLD,
    )
    for x in [0.165, 0.50, 0.835]:
        arrow(ax, (x, 0.78), (0.50, 0.68))

    node(
        ax,
        (0.05, 0.27),
        (0.40, 0.17),
        "Salience alignment",
        "Eight topic shares\nDoes speech attention follow manifesto attention?",
        color=BLUE,
    )
    node(
        ax,
        (0.55, 0.27),
        (0.40, 0.17),
        "Substantive inconsistency",
        "Two issue domains\nAre retrieved speech and manifesto claims hard to reconcile?",
        color=ORANGE,
    )
    arrow(ax, (0.43, 0.55), (0.25, 0.44), color=BLUE)
    arrow(ax, (0.57, 0.55), (0.75, 0.44), color=ORANGE)

    node(
        ax,
        (0.05, 0.05),
        (0.40, 0.12),
        "Outcome 1",
        "party–month–topic speech salience / alignment",
        color=BLUE,
        title_size=9,
        body_size=8,
    )
    node(
        ax,
        (0.55, 0.05),
        (0.40, 0.12),
        "Outcome 2",
        "party–topic–month inconsistent-pair share",
        color=ORANGE,
        title_size=9,
        body_size=8,
    )
    arrow(ax, (0.25, 0.27), (0.25, 0.17), color=BLUE)
    arrow(ax, (0.75, 0.27), (0.75, 0.17), color=ORANGE)
    ax.text(
        0.50,
        0.005,
        "Both outcomes are evaluated against electoral-cycle position, government status, and their interaction.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=MUTED,
    )
    save(fig, "01-research-design")


def figure_conceptual_space() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fills = [
        ((0, 1), ORANGE, "Same priorities,\nconflicting claims", "Position reversal"),
        ((1, 1), GREEN, "Same priorities,\ncompatible claims", "Full coherence"),
        ((0, 0), RED, "Priorities and\nclaims diverge", "Dual incoherence"),
        ((1, 0), BLUE, "Changed priorities,\nno conflicting claim", "Agenda shift"),
    ]
    for (x, y), color, main, short in fills:
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.03, y + 0.03),
                0.94,
                0.94,
                boxstyle="round,pad=0.015,rounding_size=0.025",
                linewidth=1.2,
                edgecolor=color,
                facecolor=color + "18",
            )
        )
        ax.text(x + 0.5, y + 0.61, main, ha="center", va="center", fontsize=12, weight="bold")
        ax.text(x + 0.5, y + 0.28, short, ha="center", va="center", fontsize=9.5, color=MUTED)

    ax.annotate(
        "",
        xy=(2.03, -0.08),
        xytext=(-0.02, -0.08),
        arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.2),
        annotation_clip=False,
    )
    ax.annotate(
        "",
        xy=(-0.10, 2.04),
        xytext=(-0.10, -0.02),
        arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.2),
        annotation_clip=False,
    )
    ax.text(1.0, -0.24, "Substantive consistency", ha="center", va="center", fontsize=10.5, weight="bold")
    ax.text(-0.28, 1.0, "Salience alignment", ha="center", va="center", fontsize=10.5, weight="bold", rotation=90)
    ax.text(0, -0.15, "low", ha="left", va="center", fontsize=8.5, color=MUTED)
    ax.text(2, -0.15, "high", ha="right", va="center", fontsize=8.5, color=MUTED)
    ax.text(-0.17, 0, "low", ha="center", va="bottom", fontsize=8.5, color=MUTED, rotation=90)
    ax.text(-0.17, 2, "high", ha="center", va="top", fontsize=8.5, color=MUTED, rotation=90)
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.15, top=0.98)
    save(fig, "02-coherence-conceptual-space")


def figure_hypotheses() -> None:
    x = np.linspace(0, 1, 200)
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.8))
    panels = [
        (
            axes[0],
            0.40 + 0.30 * x,
            0.28 + 0.27 * x,
            "Salience alignment",
            "H1a: alignment increases",
            "H2a: opposition higher",
            (0.70, 0.75),
        ),
        (
            axes[1],
            0.42 - 0.22 * x,
            0.58 - 0.25 * x,
            "Substantive inconsistency",
            "H1b: inconsistency decreases",
            "H2b: opposition lower",
            (0.67, 0.24),
        ),
    ]
    for ax, opposition, government, ylabel, h1, h2, ann_xy in panels:
        ax.plot(x, opposition, color=BLUE, linewidth=2.4, label="Opposition")
        ax.plot(x, government, color=ORANGE, linewidth=2.4, label="Government")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1], ["Post-election", "Mid-cycle", "Pre-election"])
        ax.set_yticks([])
        ax.set_xlabel("Position in the electoral cycle")
        ax.set_ylabel(ylabel)
        clean_axis(ax)
        ax.annotate(
            h1,
            xy=ann_xy,
            xycoords="axes fraction",
            xytext=(0.34, 0.91),
            textcoords="axes fraction",
            fontsize=8.5,
            color=INK,
            arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=1),
        )
        idx = 100
        ax.annotate(
            "",
            xy=(x[idx], government[idx]),
            xytext=(x[idx], opposition[idx]),
            arrowprops=dict(arrowstyle="<->", color=PURPLE, linewidth=1.35),
        )
        ax.text(
            0.52,
            (government[idx] + opposition[idx]) / 2,
            h2,
            fontsize=8.1,
            color=PURPLE,
            va="center",
            ha="left",
            path_effects=[patheffects.withStroke(linewidth=3, foreground=PAPER)],
        )
    axes[0].legend(frameon=False, loc="lower right")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.2, top=0.96, wspace=0.28)
    save(fig, "03-hypotheses-over-cycle")


def laea(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon0 = np.deg2rad(12.0)
    lat0 = np.deg2rad(52.0)
    lam = np.deg2rad(lon)
    phi = np.deg2rad(lat)
    denom = 1 + np.sin(lat0) * np.sin(phi) + np.cos(lat0) * np.cos(phi) * np.cos(lam - lon0)
    denom = np.clip(denom, 1e-9, None)
    k = np.sqrt(2 / denom)
    x = k * np.cos(phi) * np.sin(lam - lon0)
    y = k * (np.cos(lat0) * np.sin(phi) - np.sin(lat0) * np.cos(phi) * np.cos(lam - lon0))
    return x, y


def iter_polygons(geometry: dict) -> list[list[list[float]]]:
    if geometry["type"] == "Polygon":
        return geometry["coordinates"]
    if geometry["type"] == "MultiPolygon":
        return [ring for polygon in geometry["coordinates"] for ring in polygon]
    return []


def feature_iso3(properties: dict) -> str:
    for key in ["ADM0_A3", "ISO_A3", "SOV_A3", "GU_A3"]:
        value = properties.get(key)
        if value and value != "-99":
            return value
    return ""


def figure_country_map() -> None:
    url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    features = response.json()["features"]
    region_by_iso = COUNTRIES.set_index("iso3")["region"].to_dict()

    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    for feature in features:
        props = feature["properties"]
        if props.get("CONTINENT") != "Europe" and feature_iso3(props) not in region_by_iso:
            continue
        iso3 = feature_iso3(props)
        color = REGION_COLORS.get(region_by_iso.get(iso3, ""), "#D7DEE2")
        edge = PAPER if iso3 in region_by_iso else "#B8C3C9"
        zorder = 2 if iso3 in region_by_iso else 1
        for ring in iter_polygons(feature["geometry"]):
            arr = np.asarray(ring)
            if arr.ndim != 2 or arr.shape[0] < 3:
                continue
            px, py = laea(arr[:, 0], arr[:, 1])
            ax.add_patch(
                Polygon(
                    np.column_stack([px, py]),
                    closed=True,
                    facecolor=color,
                    edgecolor=edge,
                    linewidth=0.7,
                    zorder=zorder,
                )
            )

    lx, ly = laea(
        COUNTRIES["label_lon"].to_numpy(),
        COUNTRIES["label_lat"].to_numpy(),
    )
    for x, y, code in zip(lx, ly, COUNTRIES["code"]):
        ax.text(
            x,
            y,
            code,
            ha="center",
            va="center",
            fontsize=7.2,
            weight="bold",
            color=PAPER,
            zorder=3,
            path_effects=[patheffects.withStroke(linewidth=1.7, foreground=INK)],
        )

    corner_lon = np.array([-13, 33, -13, 33])
    corner_lat = np.array([34, 34, 71, 71])
    cx, cy = laea(corner_lon, corner_lat)
    ax.set_xlim(cx.min() - 0.08, cx.max() + 0.08)
    ax.set_ylim(cy.min() - 0.08, cy.max() + 0.08)
    ax.set_aspect("equal")
    ax.axis("off")
    handles = [
        Line2D([], [], marker="s", linestyle="none", markersize=9, color=color, label=region)
        for region, color in REGION_COLORS.items()
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=3,
        fontsize=8,
    )
    save(fig, "04-country-sample-map")


def figure_country_coverage() -> None:
    data = COUNTRIES.sort_values(["start", "country"], ascending=[True, True]).reset_index(drop=True)
    y = np.arange(len(data))
    colors = [REGION_COLORS[r] for r in data["region"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 5.5), sharey=True, gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    for ypos, start, end, color in zip(y, data["start"], data["end"], colors):
        ax.plot([start, end], [ypos, ypos], color=color, linewidth=4.2, solid_capstyle="round")
        ax.scatter([start, end], [ypos, ypos], s=18, color=color, zorder=3)
    ax.set_yticks(y, data["country"])
    ax.invert_yaxis()
    ax.set_xlim(pd.Timestamp("1995-01-01"), pd.Timestamp("2025-01-01"))
    ax.set_xlabel("Parliamentary speech coverage")
    ax.set_title("A. Temporal reach", loc="left", fontsize=10, weight="bold")
    clean_axis(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0)

    ax = axes[1]
    speech = data["speech_rows"].to_numpy()
    manifesto = data["manifesto_quasi"].to_numpy()
    for ypos, left, right, color in zip(y, manifesto, speech, colors):
        ax.plot([left, right], [ypos, ypos], color=GRID, linewidth=2.4, zorder=1)
        ax.scatter(right, ypos, s=38, color=color, marker="o", zorder=3)
        ax.scatter(left, ypos, s=30, color=INK, marker="D", zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("Records per country (log scale)")
    ax.set_title("B. Corpus scale", loc="left", fontsize=10, weight="bold")
    clean_axis(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", color=BLUE, label="Speech rows"),
            Line2D([], [], marker="D", linestyle="none", color=INK, label="Manifesto quasi-sentences"),
        ],
        frameon=False,
        loc="lower right",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.12, top=0.92, wspace=0.16)
    save(fig, "05-country-coverage-and-scale")


def figure_measurement_scope() -> None:
    topics = [
        "Macroeconomics",
        "GAL–TAN",
        "Economics",
        "Welfare & human development",
        "Institutions, rights & law",
        "Foreign & security",
        "Environment, land & energy",
        "Infrastructure & technology",
    ]
    cols = [
        "Manifesto\ntopic shares",
        "Speech\ntopic shares",
        "Salience\nalignment",
        "Substantive\ninconsistency",
    ]
    matrix = np.ones((8, 4), dtype=int)
    matrix[2:, 3] = 0
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.set_xlim(-0.55, 4)
    ax.set_ylim(-0.7, 8)
    ax.axis("off")
    for row, topic in enumerate(topics):
        y = 7.5 - row
        ax.text(-0.05, y, topic, ha="right", va="center", fontsize=9)
        ax.plot([0.05, 3.85], [y - 0.48, y - 0.48], color=GRID, linewidth=0.6)
        for col in range(4):
            active = bool(matrix[row, col])
            color = [PURPLE, BLUE, GREEN, ORANGE][col] if active else GRID
            ax.scatter(
                col + 0.5,
                y,
                s=240 if active else 110,
                color=color if active else PAPER,
                edgecolor=color,
                linewidth=1.2,
                marker="o",
            )
            ax.text(
                col + 0.5,
                y,
                "✓" if active else "—",
                ha="center",
                va="center",
                fontsize=10,
                color=PAPER if active else MUTED,
                weight="bold",
            )
    for col, label in enumerate(cols):
        ax.text(col + 0.5, 7.95, label, ha="center", va="bottom", fontsize=9, weight="bold")
    ax.add_patch(
        FancyBboxPatch(
            (3.06, 5.55),
            0.88,
            1.9,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            linewidth=1.2,
            edgecolor=ORANGE,
            facecolor="none",
        )
    )
    ax.text(
        3.5,
        -0.38,
        "The salience leg spans all eight substantive topics; the classifier is deployed only in two domains.",
        ha="center",
        va="center",
        fontsize=8.5,
        color=MUTED,
    )
    save(fig, "06-measurement-scope")


def figure_sample_construction() -> None:
    fig, ax = plt.subplots(figsize=(10.6, 5.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    node(
        ax,
        (0.22, 0.82),
        (0.56, 0.12),
        "Linked comparative corpus",
        "16 countries • 401 manifestos • 562,712 manifesto quasi-sentences • 7,538,211 speech rows",
        color=GOLD,
    )
    arrow(ax, (0.39, 0.82), (0.25, 0.72), color=BLUE)
    arrow(ax, (0.61, 0.82), (0.75, 0.72), color=ORANGE)

    ax.text(0.25, 0.75, "SALIENCE LEG", ha="center", va="bottom", fontsize=10, weight="bold", color=BLUE)
    ax.text(0.75, 0.75, "INCONSISTENCY LEG", ha="center", va="bottom", fontsize=10, weight="bold", color=ORANGE)

    left = [
        ("Topic inference", "Eight broad substantive topics"),
        ("Party–month profiles", "manifesto share ↔ speech share"),
        ("Analytical panel", "95,048 country–party–month–topic rows\n15 countries in frozen PLDA panel"),
    ]
    right = [
        ("Domain restriction", "3,249,128 speech segments\nMacroeconomics + GAL–TAN"),
        ("Hybrid retrieval", "Top 3 candidates per segment\n9,658,591 classified pairs"),
        ("Silver-label sample", "8,000 pairs × 3 LLM judgments\n7,423 majority-consensus examples"),
        ("Deployment panel", "local classifier over retrieved population\n19,798 party–topic–month rows"),
    ]
    y_left = [0.57, 0.35, 0.10]
    y_right = [0.60, 0.41, 0.22, 0.03]
    for i, ((title, body), y) in enumerate(zip(left, y_left)):
        node(ax, (0.06, y), (0.38, 0.13), title, body, color=BLUE, title_size=9, body_size=7.8)
        if i:
            arrow(ax, (0.25, y + 0.18), (0.25, y + 0.13), color=BLUE)
    for i, ((title, body), y) in enumerate(zip(right, y_right)):
        node(ax, (0.56, y), (0.38, 0.13), title, body, color=ORANGE, title_size=9, body_size=7.7)
        if i:
            arrow(ax, (0.75, y + 0.18), (0.75, y + 0.13), color=ORANGE)
    save(fig, "07-sample-construction")


def figure_classification_cost() -> None:
    retrieved = 9_658_591
    values = np.array([retrieved, 3 * retrieved, 3 * 8_000], dtype=float)
    labels = [
        "Manual coding of\nretrieved population",
        "Three hosted LLMs on\nretrieved population",
        "Chosen silver-label stage\n(3 models × 8,000 pairs)",
    ]
    colors = [RED, PURPLE, GREEN]
    fig, ax = plt.subplots(figsize=(8.7, 4.3))
    y = np.arange(3)[::-1]
    ax.barh(y, values, color=colors, height=0.55)
    ax.set_yticks(y, labels)
    ax.set_xscale("log")
    ax.set_xlim(10_000, 60_000_000)
    ax.set_xlabel("External judgments required (log scale)")
    clean_axis(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0)
    for ypos, value in zip(y, values):
        ax.text(
            value * 1.08,
            ypos,
            f"{value:,.0f}",
            va="center",
            ha="left",
            fontsize=9,
            weight="bold",
        )
    reduction = 100 * (1 - values[2] / values[1])
    ax.text(
        0.99,
        0.03,
        f"{reduction:.2f}% fewer hosted judgments than full three-model deployment\n"
        "The fine-tuned local classifier then labels the full retrieved population.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.7,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.31, right=0.96, bottom=0.18, top=0.96)
    save(fig, "08-classification-cost")


def figure_trust_chain() -> None:
    fig, ax = plt.subplots(figsize=(11.0, 3.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = [
        ("1", "Generic NLI\ndiagnostic", "36.53% overall agreement\n6.74% for contradictions", RED),
        ("2", "Three-model\nconsensus", "8,000 sampled pairs\n92.79% retained", PURPLE),
        ("3", "Leakage-resistant\ntest", "40 unseen manifestos\n1,524 held-out pairs", BLUE),
        ("4", "Task-specific\nclassifier", "Macro-F1 0.670\nInconsistent F1 0.505", GREEN),
        ("5", "Blind human\naudit", "192 primary + 48 double-coded\ncoding pending", GOLD),
    ]
    xs = np.linspace(0.02, 0.79, len(steps))
    w = 0.19
    for i, ((num, title, body, color), x) in enumerate(zip(steps, xs)):
        node(ax, (x, 0.27), (w, 0.46), f"{num}. {title}", body, color=color, title_size=9.3, body_size=7.9)
        if i < len(steps) - 1:
            arrow(ax, (x + w, 0.50), (xs[i + 1], 0.50), color=MUTED)
    ax.text(
        0.5,
        0.10,
        "The chain establishes adaptation to the silver labels; the final human-validity claim remains open until blind coding is completed.",
        ha="center",
        va="center",
        fontsize=8.8,
        color=MUTED,
    )
    save(fig, "09-measurement-trust-chain")


def figure_classifier_performance() -> None:
    metrics = ["Accuracy", "Macro-F1", "Weighted-F1", "Inconsistent F1"]
    tuned = np.array([0.7585, 0.6701, 0.7600, 0.5049])
    generic = np.array([0.2277, 0.2223, 0.2064, 0.2611])
    x = np.arange(len(metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    b1 = ax.bar(x - width / 2, generic, width, color=GRID, edgecolor=MUTED, label="Generic NLI")
    b2 = ax.bar(x + width / 2, tuned, width, color=GREEN, label="Task-specific classifier")
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Held-out score")
    clean_axis(ax)
    ax.legend(frameon=False, loc="upper left")
    for bars in [b1, b2]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=8.3,
            )
    ax.text(
        0.99,
        0.96,
        "Document-disjoint test: 1,524 pairs from 40 unseen manifestos",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.3,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.15, top=0.96)
    save(fig, "10-classifier-performance")


def figure_human_validation_design() -> None:
    countries = COUNTRIES["code"].tolist()
    labels = ["Consistent", "Unrelated", "Inconsistent"]
    fig, ax = plt.subplots(figsize=(7.7, 6.0))
    ax.set_xlim(-0.8, 3)
    ax.set_ylim(-1.2, 16.9)
    ax.axis("off")
    for i, code in enumerate(countries):
        y = 15.5 - i
        ax.text(-0.12, y, code, ha="right", va="center", fontsize=8.5, weight="bold")
        for j, label in enumerate(labels):
            color = [GREEN, BLUE, ORANGE][j]
            ax.add_patch(
                FancyBboxPatch(
                    (j + 0.08, y - 0.38),
                    0.84,
                    0.76,
                    boxstyle="round,pad=0.01,rounding_size=0.035",
                    facecolor=color + "18",
                    edgecolor=color,
                    linewidth=0.85,
                )
            )
            ax.text(j + 0.5, y + 0.09, "4 primary", ha="center", va="center", fontsize=7.8, weight="bold")
            ax.text(j + 0.5, y - 0.17, "+ 1 second", ha="center", va="center", fontsize=7.1, color=MUTED)
    for j, label in enumerate(labels):
        ax.text(j + 0.5, 16.35, label, ha="center", va="bottom", fontsize=9.2, weight="bold")
    ax.text(
        1.5,
        -0.45,
        "Balanced blind design: 16 countries × 3 predicted classes × 4 primary cases = 192.\n"
        "One case per country–class cell is independently double-coded (48).",
        ha="center",
        va="center",
        fontsize=8.5,
        color=MUTED,
    )
    ax.text(
        1.5,
        -1.02,
        "Design complete; human coding pending",
        ha="center",
        va="center",
        fontsize=9,
        weight="bold",
        color=GOLD,
    )
    save(fig, "11-human-validation-design")


def figure_fixed_effects() -> None:
    fig, ax = plt.subplots(figsize=(10.3, 5.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    levels = [
        (0.04, 0.73, 0.20, 0.15, "Countries", "15–16 political systems", PURPLE),
        (0.28, 0.59, 0.20, 0.15, "Country–parties", "repeated party histories", BLUE),
        (0.52, 0.45, 0.20, 0.15, "Party–months", "cycle position + status", GREEN),
        (0.76, 0.31, 0.20, 0.15, "Topic outcomes", "salience or inconsistency", ORANGE),
    ]
    for i, (x, y, w, h, title, body, color) in enumerate(levels):
        node(ax, (x, y), (w, h), title, body, color=color)
        if i:
            prev = levels[i - 1]
            arrow(ax, (prev[0] + prev[2], prev[1] + prev[3] / 2), (x, y + h / 2), color=MUTED)

    effects = [
        (
            (0.04, 0.16),
            "Country-party fixed effects",
            "Remove stable differences in ideology, language,\ninstitutions, and long-run rhetoric.",
            BLUE,
        ),
        (
            (0.52, 0.16),
            "Year fixed effects",
            "Remove shocks shared in a calendar year:\ncommon events, macro conditions, and drift.",
            PURPLE,
        ),
        (
            (0.04, 0.01),
            "Topic fixed effects",
            "Remove average differences in topic frequency\nand classification difficulty.",
            GREEN,
        ),
        (
            (0.52, 0.01),
            "Target comparison",
            "Estimate within-party cycle change, then compare\ngovernment and opposition trajectories.",
            ORANGE,
        ),
    ]
    for xy, title, body, color in effects:
        node(
            ax,
            xy,
            (0.44, 0.115),
            title,
            body,
            color=color,
            title_size=8.8,
            body_size=7.6,
        )

    save(fig, "12-fixed-effects-identification")


def figure_incoherence_mechanisms() -> None:
    rows = [
        ("Direct policy reversal", "Party advocates the opposite of an earlier commitment.", "Substantive inconsistency", "Programmatic U-turn"),
        ("Coalition compromise", "Manifesto position is moderated to sustain a governing bargain.", "Government-linked divergence", "Constraint of office"),
        ("External shock or learning", "New events or information alter what is feasible or desirable.", "Time-specific break", "Adaptive change"),
        ("Agenda reprioritization", "Speech attention moves to new issues without an opposing claim.", "Low salience alignment", "Not a contradiction"),
        ("Strategic repositioning", "A party changes emphasis or stance as the next election approaches.", "Cycle-linked divergence", "Electoral strategy"),
        ("Scope or qualification mismatch", "Claims differ in target, time horizon, or condition.", "Apparent textual conflict", "Measurement risk"),
    ]
    headers = ["Mechanism", "Observable process", "Expected empirical signal", "Interpretation"]
    widths = [0.21, 0.36, 0.23, 0.20]
    x_edges = np.cumsum([0] + widths)
    fig, ax = plt.subplots(figsize=(11.2, 5.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    top = 0.95
    header_h = 0.11
    row_h = 0.135
    for j, header in enumerate(headers):
        ax.add_patch(
            FancyBboxPatch(
                (x_edges[j] + 0.004, top - header_h),
                widths[j] - 0.008,
                header_h - 0.008,
                boxstyle="round,pad=0.004,rounding_size=0.01",
                facecolor=INK,
                edgecolor=INK,
            )
        )
        ax.text(
            x_edges[j] + widths[j] / 2,
            top - header_h / 2,
            header,
            ha="center",
            va="center",
            fontsize=8.8,
            color=PAPER,
            weight="bold",
        )
    row_colors = [RED, ORANGE, PURPLE, BLUE, GOLD, MUTED]
    for i, row in enumerate(rows):
        y_top = top - header_h - i * row_h
        if i % 2 == 0:
            ax.add_patch(
                FancyBboxPatch(
                    (0.004, y_top - row_h + 0.004),
                    0.992,
                    row_h - 0.008,
                    boxstyle="round,pad=0.002,rounding_size=0.005",
                    facecolor=PALE,
                    edgecolor="none",
                )
            )
        for j, value in enumerate(row):
            if j == 0:
                ax.scatter(x_edges[j] + 0.025, y_top - row_h / 2, s=45, color=row_colors[i])
                x = x_edges[j] + 0.045
                ha = "left"
                weight = "bold"
            else:
                x = x_edges[j] + 0.014
                ha = "left"
                weight = "normal"
            wrap = [22, 40, 25, 21][j]
            ax.text(
                x,
                y_top - row_h / 2,
                fill(value, wrap),
                ha=ha,
                va="center",
                fontsize=7.9,
                weight=weight,
                color=INK if j != 3 else MUTED,
                linespacing=1.2,
            )
        ax.plot([0.01, 0.99], [y_top - row_h, y_top - row_h], color=GRID, linewidth=0.7)
    ax.text(
        0.01,
        0.015,
        "The classifier captures textual relations, not the political cause; mechanisms must be interpreted using timing and government status.",
        ha="left",
        va="bottom",
        fontsize=8.3,
        color=MUTED,
    )
    save(fig, "13-incoherence-mechanisms")


def figure_nonlinear_profile() -> None:
    effects = pd.read_csv(FUNCTIONAL_EFFECTS)
    rows = effects.loc[
        effects["family"].eq("new_macro")
        & effects["functional_form"].eq("restricted cubic spline")
    ].copy()
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    for trajectory, label, color in [
        ("opposition_relative_midcycle", "Opposition", BLUE),
        ("government_relative_midcycle", "Government", ORANGE),
    ]:
        part = rows.loc[rows["trajectory"].eq(trajectory)].sort_values("proximity")
        x = part["proximity"].to_numpy()
        estimate = part["estimate"].to_numpy() * 100
        low = part["ci_95_low"].to_numpy() * 100
        high = part["ci_95_high"].to_numpy() * 100
        ax.fill_between(x, low, high, color=color, alpha=0.14, linewidth=0)
        ax.plot(x, estimate, color=color, linewidth=2.2, label=label)
    ax.axhline(0, color=INK, linewidth=0.9)
    ax.axvline(0.5, color=GRID, linewidth=0.9)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["Post-election", "Q2", "Mid-cycle", "Q4", "Pre-election"])
    ax.set_ylabel("Fitted change in inconsistency\nrelative to mid-cycle (percentage points)")
    ax.set_xlabel("Position in the electoral cycle")
    clean_axis(ax)
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.99,
        0.04,
        "Shaded bands: 95% country-level intervals\nFlexible specification: shape is not forced to be linear",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.2,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.17, top=0.97)
    save(fig, "14-nonlinear-cycle-profile")


def figure_jab_nonlinear_cycle() -> None:
    data = pd.read_csv(JAB_HYBRID_RANK_PANEL)
    data = data.loc[
        data["rank_valid"].fillna(False).astype(bool)
        & data["spearman_rho"].notna()
        & data["electoral_proximity"].notna()
        & data["party_in_government"].notna()
    ].copy()
    data["weight"] = pd.to_numeric(
        data["speech_volume_words"], errors="coerce"
    ).fillna(0.0)
    data["party_in_government"] = pd.to_numeric(
        data["party_in_government"], errors="coerce"
    )
    data = data.loc[
        data["weight"] > 0
        & data["party_in_government"].isin([0, 1])
    ].copy()

    design = patsy.dmatrix(
        "cr(electoral_proximity, df=4) * C(party_in_government)",
        data=data,
        return_type="dataframe",
    )
    design_info = design.design_info
    fit = sm.WLS(
        data["spearman_rho"].to_numpy(),
        design,
        weights=data["weight"].to_numpy(),
    ).fit(cov_type="HC1")

    grid = np.linspace(0, 1, 201)
    covariance = fit.cov_params().to_numpy()

    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    for gov_status, label, color in [
        (0, "Opposition", BLUE),
        (1, "Government", ORANGE),
    ]:
        pred = pd.DataFrame(
            {
                "electoral_proximity": grid,
                "party_in_government": gov_status,
            }
        )
        pred_mid = pd.DataFrame(
            {
                "electoral_proximity": [0.5],
                "party_in_government": [gov_status],
            }
        )
        design_grid = patsy.build_design_matrices(
            [design_info], pred, return_type="dataframe"
        )[0]
        design_mid = patsy.build_design_matrices(
            [design_info], pred_mid, return_type="dataframe"
        )[0]
        diff = design_grid.to_numpy() - design_mid.to_numpy()[0]
        estimate = diff @ fit.params.to_numpy()
        variance = np.einsum("ij,jk,ik->i", diff, covariance, diff)
        se = np.sqrt(np.clip(variance, 0, None))
        ax.fill_between(
            grid,
            estimate - 1.96 * se,
            estimate + 1.96 * se,
            color=color,
            alpha=0.14,
            linewidth=0,
        )
        ax.plot(
            grid,
            estimate,
            color=color,
            linewidth=2.2,
            label=label,
        )
    ax.axhline(0, color=INK, linewidth=0.9)
    ax.axvline(0.5, color=GRID, linewidth=0.9)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["Post-election", "Q2", "Mid-cycle", "Q4", "Pre-election"])
    ax.set_ylabel("Fitted change in Spearman alignment\nrelative to mid-cycle")
    ax.set_xlabel("Position in the electoral cycle")
    clean_axis(ax)
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.99,
        0.04,
        "Hybrid extended panel: ParlaMint plus pre-ParlaMint JAB months\n"
        "Restricted cubic spline with HC1 intervals; 13,650 party-months",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.2,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.17, top=0.97)
    save(fig, "15-jab-nonlinear-cycle-profile")


def main() -> None:
    configure()
    builders = [
        figure_research_design,
        figure_conceptual_space,
        figure_hypotheses,
        figure_country_map,
        figure_country_coverage,
        figure_measurement_scope,
        figure_sample_construction,
        figure_classification_cost,
        figure_trust_chain,
        figure_classifier_performance,
        figure_human_validation_design,
        figure_fixed_effects,
        figure_incoherence_mechanisms,
        figure_nonlinear_profile,
        figure_jab_nonlinear_cycle,
    ]
    for builder in builders:
        builder()
        print(f"built {builder.__name__}")


if __name__ == "__main__":
    main()
