
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import re
from datetime import datetime

st.set_page_config(page_title="需要データ 可視化ツール", layout="wide")
st.title("需要データ 可視化ツール (日本トムソン)")

st.markdown("""
- 入力: Excelファイル（各地区がシート）  
- 時刻別データを**自動検出**して可視化（30分/60分に対応）  
- kWh→kW換算は **時間分解能に応じて自動係数** を算出（48本→×2、24本→×1 など）  
""")

# -------- Utility --------
def _normalize_time_str(x):
    """文字列化 + 全角コロン置換して返す"""
    s = str(x).strip()
    s = s.replace('：', ':')
    return s

TIME_PATTERN = re.compile(r"^(?:[01]?\d|2[0-4]):[0-5]\d$")  # 0:00〜24:00 まで

def find_time_header_row(df: pd.DataFrame):
    """
    時刻ヘッダー行（0:00, 0:30,...が多く含まれる行）を推定して返す
    - 文字列/時刻/数値混在にも対応（全て文字列化して判定）
    """
    for ridx in range(len(df)):
        row = df.iloc[ridx]
        cnt = 0
        for v in row:
            s = _normalize_time_str(v)
            if TIME_PATTERN.match(s):
                cnt += 1
        if cnt >= 20:
            return ridx
    return None

def detect_time_block_start_col(df: pd.DataFrame, header_row: int):
    """ヘッダー行のうち、最初に時刻が現れる列インデックス"""
    row = df.iloc[header_row]
    for i, v in enumerate(row):
        s = _normalize_time_str(v)
        if TIME_PATTERN.match(s):
            return i
    return None

def detect_day_column(df: pd.DataFrame, start_row: int, end_row: int, search_cols: int = 6):
def detect_date_column(df: pd.DataFrame, start_row: int, end_row: int, search_cols: int = 8):
    """
    YYYYMMDD（8桁）形式の日付が多く入っている列を探す。戻りは列インデックス or None。
    """
    best_col, best_hits = None, -1
    for c in range(min(search_cols, df.shape[1])):
        hits = 0
        for r in range(start_row, end_row):
            v = df.iloc[r, c]
            if pd.isna(v): 
                continue
            s = str(v).strip()
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                try:
                    iv = int(v)
                    s = f"{iv:d}"
                except Exception:
                    pass
            if re.fullmatch(r"\d{8}", s):
                y, m, d = int(s[:4]), int(s[4:6]), int(s[6:])
                if 2000 <= y <= 2099 and 1 <= m <= 12 and 1 <= d <= 31:
                    hits += 1
        if hits > best_hits:
            best_hits = hits
            best_col = c
    return best_col if best_hits > 0 else None

def parse_date_series(df: pd.DataFrame, data_rows, search_cols: int = 8):
    """
    シートの左側列から YYYYMMDD を含む日付列を検出し、datetime.date の Series を返す。
    見つからない場合は None を返す。
    """
    if not data_rows:
        return None
    start_row, end_row = min(data_rows), max(data_rows)+1
    c = detect_date_column(df, start_row, end_row, search_cols=search_cols)
    if c is None:
        return None
    vals = []
    for r in data_rows:
        v = df.iloc[r, c]
        if pd.isna(v):
            vals.append(np.nan)
            continue
        s = str(v).strip()
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            try:
                iv = int(v)
                s = f"{iv:d}"
            except Exception:
                pass
        if re.fullmatch(r"\d{8}", s):
            try:
                y, m, d = int(s[:4]), int(s[4:6]), int(s[6:])
                vals.append(pd.Timestamp(year=y, month=m, day=d).date())
            except Exception:
                vals.append(np.nan)
        else:
            vals.append(np.nan)
    ser = pd.Series(vals)
    if ser.notna().sum() == 0:
        return None
    return ser

    """
    日(1..31)が入りそうな列を左から search_cols 列スキャンして最有力を返す
    """
    best_col, best_hits = None, -1
    for c in range(min(search_cols, df.shape[1])):
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

def parse_yearmonths_safe(df: pd.DataFrame, upto_row: int):
    """
    シート上部から YYYYMM を堅牢に抽出（NaN安全、正規表現）
    """
    ym_labels = []
    ym_re = re.compile(r'(20\d{2})(0[1-9]|1[0-2])')
    for ridx in range(0, upto_row):
        v = df.iloc[ridx, 0] if df.shape[1] > 0 else None
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        # 数値候補
        try:
            iv = int(v)
            if 200001 <= iv <= 209912:
                ym_labels.append(f"{iv:06d}")
                continue
        except Exception:
            pass
        # 文字列から抽出
        s = str(v)
        m = ym_re.search(s)
        if m:
            ym_labels.append(m.group(1) + m.group(2))
    # 重複除去しつつ順序維持
    seen = set()
    out = []
    for y in ym_labels:
        if y not in seen and len(y) == 6:
            out.append(y)
            seen.add(y)
    return out

def infer_interval_minutes(time_labels: pd.Series):
    """
    時刻ラベルから分解能（分）を推定（例：['0:00','0:30',...] -> 30）
    """
    # 正規化
    times = [_normalize_time_str(x) for x in list(time_labels)]
    # 最初の2つの差分で推定（失敗時は30分とする）
    def to_minutes(t):
        h, m = t.split(':')
        return int(h) * 60 + int(m)
    for i in range(len(times) - 1):
        a, b = times[i], times[i+1]
        if TIME_PATTERN.match(a) and TIME_PATTERN.match(b):
            try:
                return abs(to_minutes(b) - to_minutes(a)) or 30
            except Exception:
                continue
    # フォールバック：列数で推定
    n = len(times)
    if n in (48, 49):  # 49は24:00含む場合
        return 30
    if n in (24, 25):
        return 60
    return 30


def extract_time_series_block(df: pd.DataFrame):
    time_header_row = find_time_header_row(df)
    if time_header_row is None:
        raise ValueError("時刻ヘッダー行が見つかりませんでした。シートの形式をご確認ください。")

    first_time_col = detect_time_block_start_col(df, time_header_row)
    if first_time_col is None:
        raise ValueError("時刻形式の列が検出できませんでした。")

    data_rows = []
    for ridx in range(time_header_row + 1, len(df)):
        vals = df.iloc[ridx, first_time_col:]
        numeric_count = np.sum(pd.to_numeric(vals, errors="coerce").notna())
        if numeric_count >= 12:
            data_rows.append(ridx)
    if not data_rows:
        raise ValueError("時刻データ行が見つかりませんでした。")

    data = df.iloc[data_rows, first_time_col:].apply(pd.to_numeric, errors="coerce")
    time_labels = df.iloc[time_header_row, first_time_col:]
    time_labels = pd.Series([_normalize_time_str(x) for x in time_labels])

    date_series = parse_date_series(df, data_rows, search_cols=8)

    day_series = None
    if date_series is None:
        day_col = detect_day_column(df, min(data_rows), max(data_rows) + 1, search_cols=8)
        if day_col is not None:
            day_series = df.iloc[data_rows, day_col].reset_index(drop=True)

    ym_labels = parse_yearmonths_safe(df, time_header_row)

    month_map = {}
    if date_series is not None and date_series.notna().sum() > 0:
        ym_keys = date_series.dropna().apply(lambda d: f"{d.year:04d}{d.month:02d}")
        groups = {}
        for i in ym_keys.index:
            ymk = ym_keys.loc[i]
            groups.setdefault(ymk, []).append(i)
        for ymk, rows in groups.items():
            rows_sorted = sorted(rows)
            start = rows_sorted[0] - data_rows[0]
            end   = rows_sorted[-1] - data_rows[0] + 1
            month_map[ymk] = (start, end)
    else:
        segments = []
        if day_series is not None:
            start = 0
            for i in range(1, len(day_series)):
                try:
                    prev_d = int(day_series.iloc[i - 1])
                    cur_d = int(day_series.iloc[i])
                except Exception:
                    continue
                if cur_d == 1 and prev_d != 1:
                    segments.append((start, i))
                    start = i
            segments.append((start, len(day_series)))
        else:
            segments = [(0, len(data))]

        for idx, (s, e) in enumerate(segments):
            month_label = ym_labels[idx] if idx < len(ym_labels) else f"Month{idx+1:02d}"
            month_map[month_label] = (s, e)

    interval_min = infer_interval_minutes(time_labels)

    return {
        "time_labels": time_labels.reset_index(drop=True),
        "data": data.reset_index(drop=True),
        "date_series": date_series.reset_index(drop=True) if date_series is not None else None,
        "day_series": day_series,
        "month_segments": month_map,
        "interval_minutes": interval_min
    }
def plot_curves(time_labels, curves, labels=None, title="", y_label="値"):
    plt.figure(figsize=(11, 4.8))
    for i, y in enumerate(curves):
        if labels and i < len(labels):
            plt.plot(time_labels, y, marker="o", label=labels[i])
        else:
            plt.plot(time_labels, y, alpha=0.25)
    plt.title(title)
    plt.xlabel("時刻")
    plt.ylabel(y_label)
    plt.xticks(rotation=45)
    plt.grid(True)
    if labels:
        plt.legend()
    st.pyplot(plt.gcf())
    plt.close()

# -------- UI --------
uploaded = st.file_uploader("Excelファイルを選択（例：日本トムソン様_使用量データ.xlsx）", type=["xlsx"])

if uploaded is None:
    st.info("左上の「Browse files」からExcelをアップロードしてください。")
    st.stop()

xls = pd.ExcelFile(uploaded)
sheets = [s for s in xls.sheet_names if s not in ("需要場所リスト", )]
sheet = st.sidebar.selectbox("地区（シート）を選択", sheets)

df = pd.read_excel(uploaded, sheet_name=sheet, header=None)
parsed = extract_time_series_block(df)
time_labels = parsed["time_labels"]
data = parsed["data"]
date_series = parsed["date_series"]
day_series = parsed["day_series"]
month_segments = parsed["month_segments"]
interval_min = parsed["interval_minutes"]

# 換算（自動）
factor = 60.0 / interval_min  # 30分→2, 60分→1
convert_to_kw = st.sidebar.checkbox(f"kWh→kW換算（自動: ×{factor:.2f}）", value=True)

if convert_to_kw:
    data_vis = data * factor
    y_label = "使用量 [kW]"
else:
    data_vis = data
    y_label = f"使用量 [kWh/{int(interval_min)}min]"

tab1, tab3 = st.tabs(["日単位（1日分）", "年単位（全日重ね）"])

with tab1:
    st.subheader("日単位（1日分）")
    if (date_series is not None and len(month_segments) > 0):
        month_key = st.selectbox("月を選択", list(month_segments.keys()))
        s, e = month_segments[month_key]
        dates_in_seg = date_series.iloc[s:e].dropna().tolist()
        if len(dates_in_seg) == 0:
            st.warning("この月セグメントに日データが見つかりません。")
        else:
            options = list(range(len(dates_in_seg)))
            def _fmt(i): 
                d = dates_in_seg[i]
                return f"{d.strftime('%Y-%m-%d')}"
            idx_in_seg = st.selectbox("日を選択", options, index=0, format_func=_fmt)
            row_idx = s + idx_in_seg
            curve = data_vis.iloc[row_idx]
            st.caption(f"{sheet} / {month_key} / {dates_in_seg[idx_in_seg].strftime('%Y-%m-%d')}")
            plt.figure(figsize=(11, 4.8))
            plt.plot(time_labels, curve, marker="o")
            plt.title(f"{sheet} {dates_in_seg[idx_in_seg].strftime('%Y-%m-%d')} の需要カーブ")
            plt.xlabel("時刻")
            plt.ylabel(y_label)
            plt.xticks(rotation=45)
            plt.grid(True)
            st.pyplot(plt.gcf())
            plt.close()
    elif (day_series is not None and len(month_segments) > 0):
        month_key = st.selectbox("月を選択", list(month_segments.keys()))
        s, e = month_segments[month_key]
        days_in_month = pd.to_numeric(day_series.iloc[s:e], errors="coerce").dropna().astype(int).tolist()
        if len(days_in_month) == 0:
            st.warning("この月セグメントに日データが見つかりません。")
        else:
            idx_in_seg = st.selectbox("日を選択", list(range(len(days_in_month))), index=0, format_func=lambda i: f"{days_in_month[i]}日")
            row_idx = s + idx_in_seg
            curve = data_vis.iloc[row_idx]
            st.caption(f"{sheet} / {month_key} / {days_in_month[idx_in_seg]}日")
            plt.figure(figsize=(11, 4.8))
            plt.plot(time_labels, curve, marker="o")
            plt.title(f"{sheet} {month_key}-{days_in_month[idx_in_seg]:02d} の需要カーブ")
            plt.xlabel("時刻")
            plt.ylabel(y_label)
            plt.xticks(rotation=45)
            plt.grid(True)
            st.pyplot(plt.gcf())
            plt.close()
    else:
        st.warning("日付列が見つからないため、日単位の選択はスキップします。")

with tab3:
    st.subheader("年単位（全日重ね）")
    curves = [data_vis.iloc[i] for i in range(len(data_vis))]
    if len(curves) == 0:
        st.warning("年単位のデータが見つかりません。")
    else:
        plt.figure(figsize=(11, 4.8))
        for y in curves:
            plt.plot(time_labels, y, alpha=0.12)
        plt.title(f"{sheet} 1年分 全日重ね")
        plt.xlabel("時刻")
        plt.ylabel(y_label)
        plt.xticks(rotation=45)
        plt.grid(True)
        st.pyplot(plt.gcf())
        plt.close()
