"""
Shared chart components for PD-Aware.

Provides reusable Plotly and Streamlit chart functions for
model metrics, feature importance, SHAP explanations, etc.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

logger = logging.getLogger(__name__)

# Color palette
COLORS = {
    "primary": "#2962A0",
    "secondary": "#3B82F6",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "info": "#6366F1",
    "light_bg": "#F8FAFC",
    "text": "#334155",
    "muted": "#94A3B8",
    "positive": "#EF4444",  # Red for increases risk
    "negative": "#10B981",  # Green for decreases risk
}


def model_comparison_chart(comparison_data: List[Dict]) -> go.Figure:
    """Create a grouped bar chart comparing model metrics."""
    if not comparison_data:
        return go.Figure()

    models = [d["name"] for d in comparison_data]
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

    fig = go.Figure()

    colors = [COLORS["primary"], COLORS["secondary"], COLORS["success"],
              COLORS["warning"], COLORS["info"]]

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [d.get(metric, 0) for d in comparison_data]
        fig.add_trace(go.Bar(
            name=label,
            x=models,
            y=values,
            marker_color=colors[i],
            text=[f"{v:.3f}" for v in values],
            textposition="auto",
        ))

    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Score",
        yaxis_range=[0, 1.05],
        barmode="group",
        template="plotly_white",
        height=450,
        font=dict(family="Inter, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    return fig


def confusion_matrix_chart(cm: List[List[int]], model_name: str = "Model") -> go.Figure:
    """Create a confusion matrix heatmap."""
    labels = ["Healthy (0)", "Parkinson's (1)"]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=[[str(v) for v in row] for row in cm],
        texttemplate="%{text}",
        textfont={"size": 20},
        colorscale="Blues",
        showscale=False,
    ))

    fig.update_layout(
        title=f"Confusion Matrix — {model_name}",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template="plotly_white",
        height=350,
        width=400,
        font=dict(family="Inter, sans-serif"),
    )

    return fig


def roc_curve_chart(roc_data: Dict, model_name: str = "Model") -> go.Figure:
    """Create an ROC curve plot."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=roc_data.get("fpr", []),
        y=roc_data.get("tpr", []),
        mode="lines",
        name=f"{model_name}",
        line=dict(color=COLORS["primary"], width=2),
    ))

    # Diagonal reference
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Classifier",
        line=dict(color=COLORS["muted"], width=1, dash="dash"),
    ))

    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        height=400,
        font=dict(family="Inter, sans-serif"),
        legend=dict(x=0.6, y=0.1),
    )

    return fig


def feature_importance_chart(importances: List[Dict], top_n: int = 15) -> go.Figure:
    """Create a horizontal bar chart of feature importances."""
    top = importances[:top_n]
    top.reverse()  # For bottom-to-top display

    fig = go.Figure(go.Bar(
        x=[d["importance"] for d in top],
        y=[d.get("description", d["feature"])[:35] for d in top],
        orientation="h",
        marker_color=COLORS["secondary"],
        text=[f'{d["importance"]:.3f}' for d in top],
        textposition="outside",
    ))

    fig.update_layout(
        title="Global Feature Importance",
        xaxis_title="Importance Score",
        template="plotly_white",
        height=max(350, top_n * 28),
        font=dict(family="Inter, sans-serif"),
        margin=dict(l=200),
    )

    return fig


def shap_waterfall_chart(contributions: List[Dict], top_n: int = 10) -> go.Figure:
    """Create a SHAP-style waterfall chart for a single prediction."""
    top = contributions[:top_n]

    colors = [
        COLORS["positive"] if c["shap_value"] > 0 else COLORS["negative"]
        for c in top
    ]

    fig = go.Figure(go.Bar(
        x=[c["shap_value"] for c in top],
        y=[c.get("description", c["feature"])[:35] for c in top],
        orientation="h",
        marker_color=colors,
        text=[f'{c["shap_value"]:+.3f}' for c in top],
        textposition="outside",
    ))

    fig.update_layout(
        title="Feature Contributions to This Prediction",
        xaxis_title="SHAP Value (impact on model output)",
        template="plotly_white",
        height=max(350, top_n * 35),
        font=dict(family="Inter, sans-serif"),
        margin=dict(l=220),
        annotations=[
            dict(
                text="← Decreases risk score | Increases risk score →",
                xref="paper", yref="paper",
                x=0.5, y=-0.12,
                showarrow=False,
                font=dict(size=10, color=COLORS["muted"]),
            )
        ]
    )

    return fig


def risk_gauge_chart(score: float, category: str) -> go.Figure:
    """Create a gauge chart for risk visualization."""
    # Map category to color
    if category == "healthy":
        color = COLORS["success"]
        title = "Model Assessment: Lower Risk Pattern"
    else:
        color = COLORS["warning"]
        title = "Model Assessment: Higher Risk Pattern"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        number=dict(suffix="%", font=dict(size=36)),
        title=dict(text=title, font=dict(size=14)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1),
            bar=dict(color=color),
            bgcolor="white",
            borderwidth=2,
            bordercolor="gray",
            steps=[
                dict(range=[0, 30], color="#ECFDF5"),
                dict(range=[30, 60], color="#FEF9C3"),
                dict(range=[60, 100], color="#FEF2F2"),
            ],
        ),
    ))

    fig.update_layout(
        height=280,
        template="plotly_white",
        font=dict(family="Inter, sans-serif"),
        annotations=[
            dict(
                text="This score is the model's internal confidence, not a clinical probability.",
                xref="paper", yref="paper",
                x=0.5, y=-0.1,
                showarrow=False,
                font=dict(size=9, color=COLORS["muted"]),
            )
        ]
    )

    return fig


def metric_card(label: str, value, description: str = "", delta: str = ""):
    """Display a metric card using Streamlit's metric component."""
    st.metric(
        label=label,
        value=value,
        delta=delta if delta else None,
        help=description,
    )
