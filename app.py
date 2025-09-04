
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

st.set_page_config(page_title="需要データ 可視化ツール", layout="wide")

st.title("需要データ 可視化ツール (日本トムソン)")

st.markdown("""
- 入力: Excelファイル（各地区がシート）  
- 時刻別30分データを自動検出して可視化  
- kWh→kW換算（30分値は×2）をオプションで切替  
""")

# -------- Utility --------
TIME_PATTERN = re.compile(r"^\d{1,2}:\d{2}$")

def find_time_header_row(df: pd.DataFrame):
    """時刻ヘッダー行（0:00, 0:30,...が多く含まれる行）を推定して返す"""
    for ridx in range(len(df)):
        row = df.iloc[ridx]
        time_like = sum(1 for v in row if isinstance(v, str) and TIME_PATTERN.match(v.strip()))
        if time_like >= 20:
            return ridx
    return None

def detect_day_column(df: pd.DataFrame, start_row: int, end_row: int):
    """0列目または1列目など、日(1..31)が入っていそうな列を探索"""
    candidate_cols = [0,1]
    best_col, best_hits = None, -1
    for c in candidate_cols:
        hits = 0
        for r in range(start_row, end_row):
            v = df.iloc[r, c]
            try:
                iv = int(v)
                if 1 <= iv <= 31:
                    hits += 1
            except Exception:
                pass
        if hits > best_hits:
            best_hits = hits
            best_col = c
    return best_col

def extract_time_series_block(df: pd.DataFrame):
    """シートから時刻列とデータ本体（日×時刻の2次元配列）、日列、月セグメント情報を抽出"""
    time_header_row = find_time_header_row(df)
    if time_header_row is None:
        raise ValueError("時刻ヘッダー行が見つかりませんでした。シートの形式をご確認ください。")
    row = df.iloc[time_header_row]
    time_cols = [i for i, v in enumerate(row) if isinstance(v, str) and TIME_PATTERN.match(v.strip())]
    if not time_cols:
        raise ValueError("時刻形式の列が検出できませんでした。")
    first_time_col = min(time_cols)

    data_rows = []
    for ridx in range(time_header_row+1, len(df)):
        vals = df.iloc[ridx, first_time_col:]
        numeric_count = np.sum(pd.to_numeric(vals, errors="coerce").notna())
        if numeric_count >= 24:
            data_rows.append(ridx)

    data = df.iloc[data_rows, first_time_col:].apply(pd.to_numeric, errors="coerce")
    time_labels = df.iloc[time_header_row, first_time_col:]
    day_col = detect_day_column(df, min(data_rows), max(data_rows)+1)
    day_series = None
    if day_col is not None:
        day_series = df.iloc[data_rows, day_col].reset_index(drop=True)

    ym_labels = []
    for ridx in range(0, time_header_row):
        v = df.iloc[ridx, 0]
        if isinstance(v, (int, float)):
            iv = int(v)
            if 200001 <= iv <= 209912:
                ym_labels.append(str(iv))
        elif isinstance(v, str) and v.isdigit() and (len(v)==6 or len(v)==8):
            ym_labels.append(v[:6])
    ym_labels = [y for y in ym_labels if len(y)==6]

    segments = []
    if day_series is not None:
        start = 0
        for i in range(1, len(day_series)):
            try:
                prev_d = int(day_series.iloc[i-1])
                cur_d  = int(day_series.iloc[i])
            except Exception:
                continue
            if cur_d == 1 and prev_d != 1:
                segments.append((start, i))
                start = i
        segments.append((start, len(day_series)))

    month_map = {}
    for idx, (s, e) in enumerate(segments):
        month_label = ym_labels[idx] if idx < len(ym_labels) else f"Month{idx+1:02d}"
        month_map[month_label] = (s, e)

    return {
        "time_labels": time_labels.reset_index(drop=True),
        "data": data.reset_index(drop=True),
        "day_series": day_series,
        "month_segments": month_map
    }

def plot_curves(time_labels, curves, labels=None, title=""):
    plt.figure(figsize=(11,4.5))
    for i, y in enumerate(curves):
        if labels and i < len(labels):
            plt.plot(time_labels, y, marker="o", label=labels[i])
        else:
            plt.plot(time_labels, y, alpha=0.35)
    plt.title(title)
    plt.xlabel("時刻")
    plt.ylabel("値")
    plt.xticks(rotation=45)
    plt.grid(True)
    if labels:
        plt.legend()
    st.pyplot(plt.gcf())
    plt.close()

uploaded = st.file_uploader("Excelファイルを選択（例：日本トムソン様_使用量データ.xlsx）", type=["xlsx"])

if uploaded is None:
    st.info("左上の「Browse files」からExcelをアップロードしてください。")
    st.stop()

xls = pd.ExcelFile(uploaded)
sheets = [s for s in xls.sheet_names if s not in ("需要場所リスト", )]
sheet = st.sidebar.selectbox("地区（シート）を選択", sheets)

convert_to_kw = st.sidebar.checkbox("kWh→kW換算（30分値 ×2）", value=True)

df = pd.read_excel(uploaded, sheet_name=sheet, header=None)
parsed = extract_time_series_block(df)
time_labels = parsed["time_labels"]
data = parsed["data"]
day_series = parsed["day_series"]
month_segments = parsed["month_segments"]

if convert_to_kw:
    data_vis = data * 2.0
    y_label = "使用量 [kW]"
else:
    data_vis = data
    y_label = "使用量 [kWh/30min]"

tab1, tab2, tab3 = st.tabs(["日単位（1日分）", "月単位（全日重ね）", "年単位（全日重ね）"])

with tab1:
    st.subheader("日単位（1日分）")
    if day_series is not None and len(month_segments)>0:
        month_key = st.selectbox("月を選択", list(month_segments.keys()))
        s, e = month_segments[month_key]
        days_in_month = pd.to_numeric(day_series.iloc[s:e], errors="coerce").dropna().astype(int).tolist()
        if len(days_in_month)==0:
            st.warning("この月セグメントに日データが見つかりません。")
        else:
            day_pick = st.selectbox("日を選択", days_in_month, index=0)
            row_idx = s + days_in_month.index(day_pick)
            curve = data_vis.iloc[row_idx]
            st.caption(f"{sheet} / {month_key} / {day_pick}日")
            plt.figure(figsize=(11,4.5))
            plt.plot(time_labels, curve, marker="o")
            plt.title(f"{sheet} {month_key}-{day_pick:02d} の需要カーブ")
            plt.xlabel("時刻")
            plt.ylabel(y_label)
            plt.xticks(rotation=45)
            plt.grid(True)
            st.pyplot(plt.gcf())
            plt.close()
    else:
        st.warning("日付列または月セグメントが見つからないため、日単位の選択はスキップします。")

with tab2:
    st.subheader("月単位（全日重ね）")
    if len(month_segments)>0:
        month_key = st.selectbox("月を選択（全日重ね）", list(month_segments.keys()), key="month_overlay")
        s, e = month_segments[month_key]
        curves = [data_vis.iloc[i] for i in range(s, e)]
        plot_curves(time_labels, curves, labels=None, title=f"{sheet} {month_key} 全日重ね")
    else:
        st.warning("月セグメントが検出できませんでした。")

with tab3:
    st.subheader("年単位（全日重ね）")
    curves = [data_vis.iloc[i] for i in range(len(data_vis))]
    plt.figure(figsize=(11,4.5))
    for y in curves:
        plt.plot(time_labels, y, alpha=0.15)
    plt.title(f"{sheet} 1年分 全日重ね")
    plt.xlabel("時刻")
    plt.ylabel(y_label)
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(plt.gcf())
    plt.close()
