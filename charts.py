"""
All 11 analysis charts recreated as interactive Plotly figures.
Dark theme with vibrant accent colors, no emojis.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


# -- Theme constants --
BG = "#0E1117"
PAPER = "#0E1117"
TEXT = "#FAFAFA"
GRID = "#2A2A3E"

# Vibrant color palette
INDIGO = "#7C6BF0"
TEAL = "#2DD4A8"
CORAL = "#FF6B6B"
BLUE = "#4DA8FF"
AMBER = "#FFB84D"
PINK = "#FF6EB4"
CYAN = "#00D9FF"
LIME = "#A8E06C"

POSITIVE = "#2DD4A8"
NEGATIVE = "#FF6B6B"

LAYOUT_DEFAULTS = dict(
    paper_bgcolor=PAPER,
    plot_bgcolor=BG,
    font=dict(color=TEXT, family="Inter, Arial, sans-serif"),
    title_font=dict(size=18, color=TEXT),
    margin=dict(l=60, r=30, t=60, b=60),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
)


def _apply_theme(fig):
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


# ============================================================
# CHART 1: Average & Median Price by District
# ============================================================
def chart_price_by_district(df):
    district_stats = (
        df[~df["district"].isin(["unknown", "Other"])]
        .groupby("district")
        .agg(
            count=("unified_price", "size"),
            mean_price=("unified_price", "mean"),
            median_price=("unified_price", "median"),
        )
        .reset_index()
    )
    top = district_stats.nlargest(16, "count").sort_values("median_price")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=top["district"],
            x=top["mean_price"],
            name="Mean Price",
            orientation="h",
            marker_color=INDIGO,
            text=[f"{v/1e6:.1f}M" for v in top["mean_price"]],
            textposition="outside",
            textfont=dict(color=TEXT),
        )
    )
    fig.add_trace(
        go.Bar(
            y=top["district"],
            x=top["median_price"],
            name="Median Price",
            orientation="h",
            marker_color=TEAL,
            text=[f"{v/1e6:.1f}M" for v in top["median_price"]],
            textposition="outside",
            textfont=dict(color=TEXT),
        )
    )
    fig.update_layout(
        title="Average vs Median Price by District",
        xaxis_title="Price (EGP)",
        yaxis_title="",
        barmode="group",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return _apply_theme(fig)


# ============================================================
# CHART 2: Price-per-sqm by District
# ============================================================
def chart_ppsm_by_district(df):
    ppsm = (
        df[~df["district"].isin(["unknown", "Other"])]
        .groupby("district")["price_per_sqm"]
        .median()
        .sort_values()
        .reset_index()
    )
    ppsm.columns = ["district", "median_ppsm"]

    fig = go.Figure(
        go.Bar(
            y=ppsm["district"],
            x=ppsm["median_ppsm"],
            orientation="h",
            marker=dict(
                color=ppsm["median_ppsm"],
                colorscale=[[0, TEAL], [0.5, BLUE], [1, CORAL]],
                colorbar=dict(title="EGP/sqm"),
            ),
            text=[f"{v:,.0f}" for v in ppsm["median_ppsm"]],
            textposition="outside",
            textfont=dict(color=TEXT),
        )
    )
    fig.update_layout(
        title="Median Price per Square Meter by District",
        xaxis_title="Price per sqm (EGP)",
        yaxis_title="",
        height=550,
    )
    return _apply_theme(fig)


# ============================================================
# CHART 3: Opportunity Matrix (Bubble Chart)
# ============================================================
def chart_opportunity_matrix(df):
    district_stats = (
        df[~df["district"].isin(["unknown", "Other"])]
        .groupby("district")
        .agg(
            count=("unified_price", "size"),
            median_ppsm=("price_per_sqm", "median"),
            mean_roi=("estimated_roi_percent", "mean"),
        )
        .reset_index()
    )
    district_stats = district_stats[district_stats["count"] >= 5]

    fig = go.Figure(
        go.Scatter(
            x=district_stats["median_ppsm"],
            y=district_stats["mean_roi"],
            mode="markers+text",
            text=district_stats["district"],
            textposition="top center",
            textfont=dict(size=10, color=TEXT),
            marker=dict(
                size=np.sqrt(district_stats["count"]) * 4,
                color=district_stats["mean_roi"],
                colorscale=[[0, CORAL], [0.5, AMBER], [1, TEAL]],
                colorbar=dict(title="ROI %"),
                line=dict(width=1, color=TEXT),
                opacity=0.85,
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Price/sqm: %{x:,.0f} EGP<br>"
                "ROI: %{y:.1f}%<br>"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Investment Opportunity Matrix",
        xaxis_title="Median Price per sqm (EGP)",
        yaxis_title="Estimated ROI (%)",
        height=550,
    )
    return _apply_theme(fig)


# ============================================================
# CHART 4: Forest Plot - Amenity Price Premium
# ============================================================
def chart_amenity_forest_plot(df):
    amenities = ["elevator", "security", "balcony", "pool", "garden", "parking"]
    results = []

    for amenity in amenities:
        col = f"has_{amenity}"
        if col not in df.columns:
            continue
        with_a = df[df[col] == 1]["unified_price"]
        without_a = df[df[col] == 0]["unified_price"]

        if len(with_a) < 5 or len(without_a) < 5:
            continue

        median_with = with_a.median()
        median_without = without_a.median()
        premium = (median_with - median_without) / median_without * 100

        try:
            stat_result = stats.mannwhitneyu(with_a, without_a, alternative="two-sided")
            p_val = stat_result.pvalue
        except Exception:
            p_val = 1.0

        results.append(
            {
                "feature": amenity.capitalize(),
                "premium_pct": premium,
                "p_value": p_val,
                "significant": p_val < 0.05,
                "q25": with_a.quantile(0.25),
                "q75": with_a.quantile(0.75),
                "median_with": median_with,
                "count_with": len(with_a),
                "count_without": len(without_a),
            }
        )

    rdf = pd.DataFrame(results).sort_values("premium_pct")

    fig = go.Figure()

    colors_map = [TEAL, BLUE, INDIGO, AMBER, PINK, CORAL]
    for idx, (_, row) in enumerate(rdf.iterrows()):
        color = TEAL if row["premium_pct"] >= 0 else CORAL
        fig.add_trace(
            go.Bar(
                y=[row["feature"]],
                x=[row["premium_pct"]],
                orientation="h",
                marker_color=color,
                showlegend=False,
                text=f"{row['premium_pct']:+.1f}%",
                textposition="outside",
                textfont=dict(color=TEXT),
                hovertemplate=(
                    f"<b>{row['feature']}</b><br>"
                    f"Premium: {row['premium_pct']:+.1f}%<br>"
                    f"p-value: {row['p_value']:.4f}<br>"
                    f"With: {row['count_with']:,} listings<br>"
                    f"Without: {row['count_without']:,} listings<extra></extra>"
                ),
            )
        )

    fig.add_vline(x=0, line_dash="dash", line_color="#555", line_width=1)
    fig.update_layout(
        title="Amenity Price Premium (Forest Plot)",
        xaxis_title="Median Price Premium (%)",
        yaxis_title="",
        height=400,
    )
    return _apply_theme(fig)


# ============================================================
# CHART 5: Premium Bar Chart - Amenity Ranking
# ============================================================
def chart_amenity_premium_rank(df):
    amenities = ["elevator", "security", "balcony", "pool", "garden", "parking"]
    results = []

    for amenity in amenities:
        col = f"has_{amenity}"
        if col not in df.columns:
            continue
        with_a = df[df[col] == 1]["unified_price"]
        without_a = df[df[col] == 0]["unified_price"]

        if len(with_a) < 5 or len(without_a) < 5:
            continue

        median_with = with_a.median()
        median_without = without_a.median()
        premium = (median_with - median_without) / median_without * 100

        results.append({"feature": amenity.capitalize(), "premium_pct": premium})

    rdf = pd.DataFrame(results).sort_values("premium_pct", ascending=True)

    colors = [TEAL if v >= 0 else CORAL for v in rdf["premium_pct"]]

    fig = go.Figure(
        go.Bar(
            y=rdf["feature"],
            x=rdf["premium_pct"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.1f}%" for v in rdf["premium_pct"]],
            textposition="outside",
            textfont=dict(color=TEXT),
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#555", line_width=1)
    fig.update_layout(
        title="Price Premium from Each Amenity (vs. listings without it)",
        xaxis_title="Median Price Premium (%)",
        yaxis_title="",
        height=400,
    )
    return _apply_theme(fig)


# ============================================================
# CHART 6: Correlation Heatmap
# ============================================================
def chart_correlation_heatmap(df):
    numeric_cols = [
        "unified_price",
        "unified_area",
        "price_per_sqm",
        "unified_rooms",
        "unified_bathrooms",
        "estimated_roi_percent",
        "has_elevator",
        "has_security",
        "has_balcony",
        "has_pool",
        "has_garden",
        "has_parking",
    ]
    existing = [c for c in numeric_cols if c in df.columns]
    corr = df[existing].corr()

    labels = [c.replace("unified_", "").replace("has_", "").replace("_", " ").title() for c in existing]

    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            colorscale=[
                [0, CORAL],
                [0.25, "#2A1A3E"],
                [0.5, "#1A1A2E"],
                [0.75, "#0F3E3A"],
                [1, TEAL],
            ],
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=9, color=TEXT),
            colorbar=dict(title="Corr"),
        )
    )
    fig.update_layout(
        title="Feature Correlation Matrix",
        height=600,
        width=700,
        xaxis=dict(tickangle=45),
    )
    return _apply_theme(fig)


# ============================================================
# CHART 7: Price Distribution Histogram
# ============================================================
def chart_price_distribution(df):
    prices = df["unified_price"] / 1e6

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=prices,
            nbinsx=60,
            marker_color=INDIGO,
            marker_line_color="#3A2F8E",
            marker_line_width=0.5,
            opacity=0.85,
            name="Distribution",
        )
    )

    mean_val = prices.mean()
    median_val = prices.median()

    fig.add_vline(x=mean_val, line_dash="dash", line_color=CORAL, line_width=2,
                  annotation_text=f"Mean: {mean_val:.1f}M", annotation_position="top right",
                  annotation_font_color=CORAL)
    fig.add_vline(x=median_val, line_dash="solid", line_color=TEAL, line_width=2,
                  annotation_text=f"Median: {median_val:.1f}M", annotation_position="top left",
                  annotation_font_color=TEAL)

    fig.update_layout(
        title="Property Price Distribution",
        xaxis_title="Price (Millions EGP)",
        yaxis_title="Count",
        height=450,
    )
    return _apply_theme(fig)


# ============================================================
# CHART 8: Price by Rooms (Box Plot)
# ============================================================
def chart_price_by_rooms(df):
    filtered = df[(df["unified_rooms"] >= 1) & (df["unified_rooms"] <= 5)].copy()

    room_colors = {1: BLUE, 2: TEAL, 3: INDIGO, 4: AMBER, 5: CORAL}

    fig = go.Figure()
    for rooms in sorted(filtered["unified_rooms"].unique()):
        subset = filtered[filtered["unified_rooms"] == rooms]
        fig.add_trace(
            go.Box(
                y=subset["unified_price"] / 1e6,
                name=f"{int(rooms)} Rooms",
                marker_color=room_colors.get(int(rooms), INDIGO),
                line_color=room_colors.get(int(rooms), INDIGO),
                boxmean=True,
            )
        )

    fig.update_layout(
        title="Price Distribution by Number of Rooms",
        xaxis_title="Rooms",
        yaxis_title="Price (Millions EGP)",
        height=500,
        showlegend=False,
    )
    return _apply_theme(fig)


# ============================================================
# CHART 9: ROI by District (Violin Plot)
# ============================================================
def chart_roi_by_district(df):
    filtered = df[~df["district"].isin(["unknown", "Other"])].copy()
    top_districts = (
        filtered.groupby("district").size().nlargest(12).index.tolist()
    )
    filtered = filtered[filtered["district"].isin(top_districts)]

    palette = [INDIGO, TEAL, CORAL, BLUE, AMBER, PINK, CYAN, LIME,
               "#B39DDB", "#80CBC4", "#FFAB91", "#81D4FA"]

    fig = go.Figure()
    for i, district in enumerate(sorted(top_districts)):
        subset = filtered[filtered["district"] == district]
        c = palette[i % len(palette)]
        fig.add_trace(
            go.Violin(
                x=subset["estimated_roi_percent"],
                name=district,
                orientation="h",
                line_color=c,
                fillcolor=c,
                opacity=0.5,
                meanline_visible=True,
            )
        )

    fig.update_layout(
        title="ROI Distribution by District",
        xaxis_title="Estimated ROI (%)",
        yaxis_title="",
        height=600,
        showlegend=False,
    )
    return _apply_theme(fig)


# ============================================================
# CHART 10: Risk / Consistency Map
# ============================================================
def chart_risk_consistency_map(df):
    district_stats = (
        df[~df["district"].isin(["unknown", "Other"])]
        .groupby("district")
        .agg(
            count=("unified_price", "size"),
            mean_price=("unified_price", "mean"),
            median_price=("unified_price", "median"),
            std_price=("unified_price", "std"),
        )
        .reset_index()
    )
    district_stats = district_stats[district_stats["count"] >= 5]
    district_stats["cv"] = district_stats["std_price"] / district_stats["mean_price"]

    fig = go.Figure(
        go.Scatter(
            x=district_stats["median_price"] / 1e6,
            y=district_stats["cv"],
            mode="markers+text",
            text=district_stats["district"],
            textposition="top center",
            textfont=dict(size=10, color=TEXT),
            marker=dict(
                size=np.sqrt(district_stats["count"]) * 3,
                color=district_stats["cv"],
                colorscale=[[0, TEAL], [0.5, AMBER], [1, CORAL]],
                colorbar=dict(title="CV"),
                line=dict(width=1, color=TEXT),
                opacity=0.85,
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Median Price: %{x:.1f}M EGP<br>"
                "CV: %{y:.2f}<br>"
                "<extra></extra>"
            ),
        )
    )

    mean_cv = district_stats["cv"].mean()
    mean_price = district_stats["median_price"].mean() / 1e6

    fig.add_hline(y=mean_cv, line_dash="dash", line_color="#555", line_width=1,
                  annotation_text="Avg CV", annotation_position="bottom right")
    fig.add_vline(x=mean_price, line_dash="dash", line_color="#555", line_width=1,
                  annotation_text="Avg Price", annotation_position="top left")

    fig.update_layout(
        title="Price Consistency Map (Risk Assessment)",
        xaxis_title="Median Price (Millions EGP)",
        yaxis_title="Coefficient of Variation (CV)",
        height=550,
    )
    return _apply_theme(fig)


# ============================================================
# CHART 11: Model Performance Comparison
# ============================================================
def chart_model_comparison():
    # Hardcoded from notebook results
    models = ["Linear Regression", "Random Forest"]
    r2_scores = [0.5165, 0.9128]
    mae_scores = [2807612, 877923]
    rmse_scores = [5956745, 2527615]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["R-squared (higher = better)", "MAE (lower = better)", "RMSE (lower = better)"],
    )

    fig.add_trace(
        go.Bar(x=models, y=r2_scores, marker_color=[CORAL, TEAL],
               text=[f"{v:.4f}" for v in r2_scores], textposition="outside",
               textfont=dict(color=TEXT), showlegend=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=models, y=[v / 1e6 for v in mae_scores], marker_color=[CORAL, TEAL],
               text=[f"{v/1e6:.2f}M" for v in mae_scores], textposition="outside",
               textfont=dict(color=TEXT), showlegend=False),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(x=models, y=[v / 1e6 for v in rmse_scores], marker_color=[CORAL, TEAL],
               text=[f"{v/1e6:.2f}M" for v in rmse_scores], textposition="outside",
               textfont=dict(color=TEXT), showlegend=False),
        row=1, col=3,
    )

    fig.update_layout(
        title="Model Performance Comparison",
        height=400,
    )
    for i in range(1, 4):
        fig.update_xaxes(gridcolor=GRID, row=1, col=i)
        fig.update_yaxes(gridcolor=GRID, row=1, col=i)

    return _apply_theme(fig)
