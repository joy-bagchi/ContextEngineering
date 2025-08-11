# auto_chart.py
import pandas as pd
import numpy as np
import altair as alt

def _coerce_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            # try parsing ISO strings
            try:
                parsed = pd.to_datetime(out[c], errors="raise", utc=False)
                # accept only if many values are valid datetimes
                valid_ratio = parsed.notna().mean()
                if valid_ratio >= 0.7:
                    out[c] = parsed
            except Exception:
                pass
    return out

def _roles(df: pd.DataFrame):
    dt_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
    num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    # categorical: objects or small cardinality numerics
    cat_cols = []
    for c in df.columns:
        if c in dt_cols or c in num_cols:
            continue
        if df[c].dtype == object:
            cat_cols.append(c)
        else:
            # treat low-cardinality non-numeric as categorical
            if df[c].nunique(dropna=True) <= 50:
                cat_cols.append(c)
    # also treat any column (incl. numeric) with very low cardinality as categorical candidate
    for c in num_cols:
        if df[c].nunique(dropna=True) <= 12:
            cat_cols.append(c)
    # de-dup, preserve order
    cat_cols = list(dict.fromkeys(cat_cols))
    return dt_cols, num_cols, cat_cols

def auto_chart(df_in: pd.DataFrame):
    """
    Returns (chart, description) where chart is an Altair chart or None.
    """
    if df_in is None or df_in.empty:
        return None, "No data."

    df = _coerce_datetimes(df_in)
    dt_cols, num_cols, cat_cols = _roles(df)

    # 1) Time series: datetime + numeric
    if dt_cols and num_cols:
        x = dt_cols[0]
        y = num_cols[0]
        # Aggregate if duplicate timestamps
        agg = df.groupby(x, as_index=False)[y].sum()
        chart = (
            alt.Chart(agg)
            .mark_line(point=True)
            .encode(x=alt.X(x, title=str(x)), y=alt.Y(y, title=str(y)), tooltip=[x, y])
            .properties(height=360)
        )
        return chart, f"Line chart of {y} over {x}"

    # 2) Categorical + numeric(s)
    if cat_cols and num_cols:
        cat = cat_cols[0]
        if len(num_cols) == 1:
            y = num_cols[0]
            # Aggregate identical categories
            agg = df.groupby(cat, as_index=False)[y].sum()
            chart = (
                alt.Chart(agg)
                .mark_bar()
                .encode(x=alt.X(cat, sort='-y', title=str(cat)),
                        y=alt.Y(y, title=str(y)),
                        tooltip=[cat, y])
                .properties(height=360)
            )
            return chart, f"Bar chart of {y} by {cat}"
        else:
            # Melt multiple numerics into long form for grouped bars
            long = df.melt(id_vars=[cat], value_vars=num_cols, var_name="metric", value_name="value")
            agg = long.groupby([cat, "metric"], as_index=False)["value"].sum()
            chart = (
                alt.Chart(agg)
                .mark_bar()
                .encode(x=alt.X(cat, sort='-y', title=str(cat)),
                        y=alt.Y("value", title="value"),
                        color=alt.Color("metric", legend=alt.Legend(title="Metric")),
                        tooltip=[cat, "metric", "value"])
                .properties(height=360)
            )
            return chart, f"Grouped bar of metrics by {cat}"

    # 3) Scatter: two numeric columns
    if len(num_cols) >= 2:
        x, y = num_cols[:2]
        chart = (
            alt.Chart(df)
            .mark_circle(size=60)
            .encode(x=alt.X(x, title=str(x)), y=alt.Y(y, title=str(y)),
                    tooltip=[x, y] + [c for c in df.columns if c not in (x, y)][:3])
            .properties(height=360)
        )
        return chart, f"Scatter plot of {y} vs {x}"

    # 4) Single numeric → histogram
    if len(num_cols) == 1:
        v = num_cols[0]
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(alt.X(v, bin=alt.Bin(maxbins=30), title=str(v)),
                    alt.Y('count()', title='count'))
            .properties(height=360)
        )
        return chart, f"Histogram of {v}"

    # Fallback
    return None, "Table view (no obvious chart type)."
