import re
import pandas as pd

LATIN_PATTERN = r"[A-Za-zÁÉÍÓÚáéíóúÑñ]"

def is_number_like(val: str) -> bool:
    if pd.isna(val):
        return False
    s = str(val).strip()
    s = s.replace("$","").replace("€","").replace("ARS","").replace("USD","")
    # 1.234.567,89 -> 1234567.89
    if re.search(r"^\d{1,3}(\.\d{3})+,\d{2}$", s):
        s = s.replace(".","").replace(",",".")
    else:
        s = s.replace(",",".")
    try:
        float(s)
        return True
    except:
        return False

def to_float(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    s = s.replace("$","").replace("€","").replace("ARS","").replace("USD","")
    if re.search(r"^\d{1,3}(\.\d{3})+,\d{2}$", s):
        s = s.replace(".","").replace(",",".")
    else:
        s = s.replace(",",".")
    try:
        return float(s)
    except:
        return None

def first_data_row(df: pd.DataFrame, max_scan: int = 30) -> int:
    # Busca la primera fila con al menos una celda con letras y otra numérica
    for i in range(min(len(df), max_scan)):
        row = df.iloc[i].astype(str)
        text_like = (row.str.contains(LATIN_PATTERN, regex=True).sum()) >= 1
        num_like = row.map(is_number_like).sum() >= 1
        if text_like and num_like:
            return i
    return 0

def choose_columns_by_pattern(df: pd.DataFrame, min_text_len: int = 8):
    scores = {}
    for c in df.columns:
        ser = df[c].dropna().astype(str).head(200)
        if ser.empty:
            scores[c] = {"avg_len": 0, "alnum_short": 0, "num": 0, "text": 0, "uniq": 0}
            continue
        avg_len = ser.map(len).mean()
        alnum_short = ser.str.match(r"^[A-Za-z0-9\-_]{3,15}$").mean()
        num = ser.map(is_number_like).mean()
        text = ser.str.contains(LATIN_PATTERN, regex=True).mean()
        uniq = ser.nunique() / max(1, len(ser))
        scores[c] = dict(avg_len=avg_len, alnum_short=alnum_short, num=num, text=text, uniq=uniq)

    # codigo: alfanum corto + unicidad alta + texto moderado
    code_col = max(df.columns, key=lambda c: (scores[c]["alnum_short"], scores[c]["uniq"], -scores[c]["avg_len"]))
    # nombre: mucho texto y longitud
    name_col = max(df.columns, key=lambda c: (scores[c]["text"], scores[c]["avg_len"]))
    # precio: numérico predominante
    price_col = max(df.columns, key=lambda c: (scores[c]["num"], -abs(scores[c]["avg_len"] - 6)))

    # Si el nombre elegido es demasiado corto, buscar otra columna con mayor longitud promedio
    if scores[name_col]["avg_len"] < min_text_len:
        candidates = sorted(df.columns, key=lambda c: scores[c]["avg_len"], reverse=True)
        for cand in candidates:
            if scores[cand]["text"] > 0.5 and scores[cand]["avg_len"] >= min_text_len:
                name_col = cand
                break

    return {"codigo": code_col, "nombre": name_col, "precio": price_col}
