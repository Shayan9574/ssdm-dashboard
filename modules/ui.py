"""
ui.py

Checkpoint E design system: theme injection, the persistent context
strip, tier cards, quadrant and probability charts (Plotly), and small
shared components. Tier colors are identical everywhere: red Tier 1,
amber Tier 2, blue Tier 3, slate beyond.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

C = {"bg": "#0e1526", "panel": "#141d33", "panel2": "#1a2441",
     "line": "#25304f", "text": "#e8edf7", "muted": "#93a0bd",
     "teal": "#2dd4bf", "amber": "#fbbf24", "red": "#f87171",
     "blue": "#60a5fa", "green": "#34d399", "slate": "#5b6a8c"}
TIER_COLORS = {1: C["red"], 2: C["amber"], 3: C["blue"], 4: C["slate"]}
QUADRANT_COLORS = {"Structural priority": C["green"],
                   "Conditionally critical": C["amber"],
                   "Stable low priority": C["blue"],
                   "Latent risk": C["red"]}
SHORT = {"Influenza": "Influenza", "COVID-19": "COVID-19",
         "Respiratory Syncytial Virus (RSV)": "RSV",
         "Meningococcal Disease (Meningitis)": "Meningococcal"}

CSS = f"""
<style>
.stApp {{ background: {C['bg']}; }}
section[data-testid="stSidebar"] {{ background: {C['panel']};
  border-right: 1px solid {C['line']}; }}
h1, h2, h3 {{ color: {C['text']} !important; }}
.ctx-strip {{ display:flex; gap:9px; flex-wrap:wrap; align-items:center;
  background:{C['panel']}; border:1px solid {C['line']}; border-radius:12px;
  padding:9px 14px; margin-bottom:6px; font-size:13px; }}
.ctx-chip {{ background:{C['panel2']}; border:1px solid {C['line']};
  border-radius:999px; padding:3px 12px; color:{C['muted']}; }}
.ctx-chip b {{ color:{C['text']}; font-weight:600; }}
.ctx-chip.ok b {{ color:{C['green']}; }} .ctx-chip.ev b {{ color:{C['amber']}; }}
.tier-card {{ display:flex; align-items:center; gap:14px; padding:12px 14px;
  border-radius:12px; background:{C['panel']}; border:1px solid {C['line']};
  margin-bottom:9px; }}
.tier-badge {{ min-width:56px; text-align:center; font-weight:700;
  border-radius:9px; padding:8px 0; font-size:13px; }}
.tier-nm {{ font-weight:600; color:{C['text']}; font-size:15px; }}
.tier-why {{ color:{C['muted']}; font-size:12.5px; margin-top:2px; }}
.tag {{ font-size:11px; border-radius:6px; padding:2px 8px; margin-left:8px; }}
.sig {{ display:flex; gap:10px; align-items:flex-start; border-radius:10px;
  padding:10px 13px; font-size:13px; margin-bottom:8px; color:{C['text']};
  background:rgba(251,191,36,.08); border:1px solid rgba(251,191,36,.25); }}
.sig.warn {{ background:rgba(248,113,113,.08);
  border-color:rgba(248,113,113,.3); }}
div[data-testid="stMetricValue"] {{ color:{C['teal']}; }}
</style>"""


def apply_theme():
    st.markdown(CSS, unsafe_allow_html=True)


def context_strip(chips):
    """chips: list of (label, value, css_class or '')"""
    html = '<div class="ctx-strip">' + "".join(
        f'<span class="ctx-chip {cls}">{lab} <b>{val}</b></span>'
        for lab, val, cls in chips) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def tier_card(tier, name, tags, why, is_target=False):
    color = TIER_COLORS.get(tier, C["slate"])
    tag_html = "".join(
        f'<span class="tag" style="background:rgba(255,255,255,.06);'
        f'color:{QUADRANT_COLORS.get(t, C["teal"])}">{t}</span>' for t in tags)
    if is_target:
        tag_html += (f'<span class="tag" style="background:{C["panel2"]};'
                     f'color:{C["teal"]}">Target state</span>')
    st.markdown(
        f'<div class="tier-card"><span class="tier-badge" '
        f'style="background:{color}22;color:{color}">Tier {tier}</span>'
        f'<div><div class="tier-nm">{name}{tag_html}</div>'
        f'<div class="tier-why">{why}</div></div></div>',
        unsafe_allow_html=True)


def signal(text, warn=False):
    st.markdown(f'<div class="sig{" warn" if warn else ""}>'
                f'<span>&#9679;</span><div>{text}</div></div>',
                unsafe_allow_html=True)


def _base_layout(fig, height=330):
    fig.update_layout(
        height=height, paper_bgcolor=C["panel"], plot_bgcolor=C["panel2"],
        font=dict(color=C["text"], size=13), margin=dict(l=10, r=10, t=34, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor=C["line"], zerolinecolor=C["line"])
    fig.update_yaxes(gridcolor=C["line"], zerolinecolor=C["line"])
    return fig


def quadrant_chart(quad: pd.DataFrame, theta: float, title=""):
    n_a = len(quad)
    cut = (n_a + 1) / 2.0
    fig = go.Figure()
    fig.add_vline(x=cut, line=dict(color=C["line"], dash="dash"))
    fig.add_hline(y=theta, line=dict(color=C["line"], dash="dash"))
    for _, r in quad.iterrows():
        col = QUADRANT_COLORS[r["Quadrant"]]
        fig.add_trace(go.Scatter(
            x=[r["R (expected rank)"]], y=[r["SI (stability)"]],
            mode="markers+text", text=[SHORT.get(r["Disease"], r["Disease"])],
            textposition="bottom center", textfont=dict(color=C["text"]),
            marker=dict(size=16, color=col, line=dict(color=C["bg"], width=2)),
            name=r["Quadrant"], showlegend=False,
            hovertemplate=(f"{r['Disease']}<br>R = %{{x:.3f}}<br>"
                           f"SI = %{{y:.3f}}<br>{r['Quadrant']}<extra></extra>")))
    fig.update_xaxes(title="Expected rank R (lower = higher priority)",
                     autorange="reversed")
    fig.update_yaxes(title=f"Stability SI (theta = {theta:g})",
                     range=[-0.03, 1.03])
    fig.update_layout(title=title)
    return _base_layout(fig)


def prob_bars(prob: pd.DataFrame):
    d = prob.dropna(subset=["p_s"]).sort_values("p_s", ascending=True)
    fig = go.Figure(go.Bar(
        x=d["p_s"], y=d["Scenario ID"] + "  " + d["Name"].str.slice(0, 28),
        orientation="h", marker_color=C["teal"],
        hovertemplate="p = %{x:.4f}<extra></extra>"))
    fig.update_xaxes(title="Normalized probability p_s")
    return _base_layout(fig, height=300)


def tier_heatmap(tm: pd.DataFrame, title="Tiers across decision environments"):
    z = tm.values
    fig = go.Figure(go.Heatmap(
        z=z, x=list(tm.columns),
        y=[SHORT.get(i, i) for i in tm.index],
        colorscale=[[0, C["red"]], [0.5, C["amber"]], [1, C["blue"]]],
        zmin=1, zmax=max(3, z.max()), showscale=False,
        text=z, texttemplate="%{text}",
        hovertemplate="%{y} under %{x}: Tier %{z}<extra></extra>"))
    fig.update_layout(title=title)
    return _base_layout(fig, height=260)
