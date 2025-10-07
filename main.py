import os, re, io, json, yaml, pandas as pd
from typing import Dict, Any, List, Optional
# from fastmcp.server import MCP, Toolì
from fastapi import FastAPI, UploadFile, File, Form
from rules.detectors import is_number_like, to_float, first_data_row, choose_columns_by_pattern
import re, numpy as np

DEFAULT_RULES = {
    "code_header_hints": ["cod","código","codigo","sku","art"],
    "prefer_code_regex": [            # orden de prioridad (genéricos)
        r"^\d{5,7}-[A-Za-z]$",       # 900000-N
        r"^\d{5,8}$",                # 300580
        r"^[A-Za-z]{2,}\d{2,}$",     # ABC123
        r"^\d{3,}-[A-Za-z0-9]{1,3}$"
    ],
    "quantity_headers": ["pack","packs","x caja","cantidad","unid","und","unidades", "und x pack"],
    "qty_value_set": [1,5,10,20,25,40,50,100,200],
    "code_min_uniqueness": 0.25      # proporción mínima de unicidad para una buena col de código
}

def get_rules(profile: dict | None):
    r = {**DEFAULT_RULES}
    pp = (profile or {}).get("postprocess", {}) or {}
    # Permite override por proveedor (o extender listas)
    for k in ["code_header_hints","prefer_code_regex","quantity_headers","qty_value_set","code_min_uniqueness"]:
        if k in pp and pp[k] is not None:
            r[k] = pp[k]
    # listas → set donde aplica
    r["qty_value_set"] = set(r["qty_value_set"])
    return r

PRICE_HEADER_HINTS = [
    "precio","pvp","lista","p.u.","p.u","pu","unit","unitario",
    "usd","u$d","u$s","u$","dolar","dólar","precio usd","precio unitario",
    "usd lista pack"
]
NAME_HEADER_HINTS  = ["desc", "descrip", "producto", "detalle"]
CODE_HEADER_HINTS  = ["cod", "código", "sku", "art"]
QUANTITY_HEADER_HINTS = ["pack","packs","x caja","cantidad","cant","unid","unidades"]


def split_tables_by_blank_rows(df, min_cols=2, min_rows=3):
    """Parte una hoja en bloques (tablas) separadas por filas casi vacías."""
    df2 = df.copy()
    nn = df2.notna().sum(axis=1).tolist()
    blocks = []
    start = None
    for i, k in enumerate(nn + [0]):  # + [0] para cerrar el último bloque
        if k >= min_cols and start is None:
            start = i
        if (k < min_cols or i == len(nn)) and start is not None:
            end = i
            block = df.iloc[start:end].dropna(how="all").dropna(axis=1, how="all")
            if len(block) >= min_rows and block.shape[1] >= min_cols:
                blocks.append(block.reset_index(drop=True))
            start = None
    return blocks

def pick_price_by_relationship(df):
    """
    Si existen dos columnas de precio tales que: precio_caja ≈ precio_pack * packs_por_caja,
    se elige precio_pack (evita confundir '100' o '50' como precio).
    """
    num_cols = []
    for c in df.columns:
        s = df[c].dropna().astype(str)
        if s.empty: 
            continue
        # columna mayormente numérica
        if s.map(is_number_like).mean() >= 0.8:
            num_cols.append(c)

    if len(num_cols) < 2:
        return None

    # detectar posibles columnas de "packs_por_caja"
    qty_cols = []
    qty_values = {1,5,10,20,25,40,50,100,200}
    for c in df.columns:
        name = str(c).lower()
        if any(x in name for x in ["pack", "packs", "x caja", "und", "unid", "unidades"]):
            qty_cols.append(c)
        else:
            # por contenido (muchos enteros “típicos”)
            s = df[c].dropna().astype(str)
            try:
                vals = s.map(to_float).dropna().tolist()
                if vals:
                    ints = [v for v in vals if abs(v - round(v)) < 1e-9]
                    if len(ints) / len(vals) > 0.8:
                        top = set(sorted(ints)[:5])  # aprox
                        if any(int(round(v)) in qty_values for v in top):
                            qty_cols.append(c)
            except:
                pass

    # probar pares de precios vs cantidades
    for qc in qty_cols:
        q = df[qc].map(to_float)
        for a in num_cols:
            for b in num_cols:
                if a == b:
                    continue
                pa = df[a].map(to_float)
                pb = df[b].map(to_float)
                ok, total, matches = 0, 0, 0
                for x, y, z in zip(pa, pb, q):
                    if x is None or y is None or z is None or z == 0:
                        continue
                    total += 1
                    # y ≈ x * z (precio_caja ≈ precio_pack * packs)
                    if abs(y - (x * z)) <= max(0.05 * (x * z), 0.02):
                        matches += 1
                if total >= 6 and matches / total >= 0.6:
                    # 'a' es el precio por pack (o unitario), preferir 'a'
                    return a
    return None


def header_row_idx(block, max_scan=5):
    """Busca cabecera por palabras clave; si no encuentra, devuelve -1 (sin header)."""
    for i in range(min(max_scan, len(block))):
        row = block.iloc[i].astype(str).str.lower()
        rowtxt = " | ".join(row.tolist())
        if any(h in rowtxt for h in PRICE_HEADER_HINTS + NAME_HEADER_HINTS + CODE_HEADER_HINTS):
            return i
    return -1


def relabel_using_header(block, hdr_idx):
    """Si hay header real, úsalo; si no, dejá todas las filas como datos con nombres col_0.."""
    b = block.copy().reset_index(drop=True)
    if hdr_idx is None or hdr_idx < 0:
        b = b.dropna(how="all").dropna(axis=1, how="all")
        b.columns = [f"col_{i}" for i in range(len(b.columns))]
        return b
    # Header real
    b = b.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    hdr = b.iloc[hdr_idx].fillna("").astype(str).tolist()
    cols, used = [], set()
    for i, v in enumerate(hdr):
        name = v.strip() or f"col_{i}"
        base, k = name, 1
        while name in used:
            name = f"{base}_{k}"; k += 1
        used.add(name); cols.append(name)
    b.columns = cols
    b = b.iloc[hdr_idx+1:].reset_index(drop=True)
    return b


def column_scores_numeric(series: pd.Series):
    s = series.dropna().astype(str)
    if s.empty: 
        return dict(numeric_ratio=0, dec_ratio=0, six_digit_ratio=0, header_bonus=0)
    numeric_ratio = s.map(is_number_like).mean()
    dec_ratio = s.str.contains(r"[.,]\d{2}$").mean()
    six_digit_ratio = s.str.match(r"^\d{5,6}$").mean()  # castiga códigos internos 5-6 dígitos
    return dict(numeric_ratio=float(numeric_ratio), dec_ratio=float(dec_ratio), six_digit_ratio=float(six_digit_ratio))

def pick_price_col(df, profile: Optional[dict] = None):
    """
    Detecta la mejor columna de precio, considerando reglas del perfil YAML.
    """
    # --- Cargar reglas del perfil ---
    prefer_list = []
    bad_list = []
    if profile:
        prefer_list = [p.lower().strip() for p in (profile.get("postprocess", {}).get("prefer_price_headers", []) or [])]
        bad_list = [b.lower().strip() for b in (profile.get("postprocess", {}).get("bad_price_headers", []) or [])]

    best_col, best_score = None, -1e9
    for c in df.columns:
        s = df[c].dropna().astype(str)
        if s.empty:
            continue
        name = str(c).lower()
        score = 0.0

        # ❌ penalizar explícitamente si está en lista de "malos"
        if any(b in name for b in bad_list):
            score -= 10.0

        # ✅ preferencia explícita del perfil YAML
        for i, pref in enumerate(reversed(prefer_list), 1):
            if pref in name:
                score += 5.0 * i

        # BONUS por encabezado “de precio” (fijo)
        if any(h in name for h in PRICE_HEADER_HINTS):
            score += 3.0
        if "usd" in name or "u$s" in name or "u$d" in name:
            score += 2.5
        if "p.u" in name or "unit" in name or "unitario" in name:
            score += 1.5

        # PENALIZAR encabezados de cantidad
        if any(h in name for h in QUANTITY_HEADER_HINTS):
            score -= 3.5

        # Métricas numéricas
        stats = column_scores_numeric(df[c])
        score += 1.3 * stats["numeric_ratio"]
        score += 0.9 * stats["dec_ratio"]
        score -= 1.0 * stats["six_digit_ratio"]

        # Penalizar columnas casi constantes o con valores “típicos” de cantidad
        try:
            from collections import Counter
            vals_float = s.map(to_float).dropna().tolist()
            if len(vals_float) >= 10:
                mode, freq = Counter(vals_float).most_common(1)[0]
                mode_ratio = freq / len(vals_float)
                score -= 2.0 * mode_ratio
                qty_set = {1, 5, 10, 20, 25, 50, 100, 200}
                qty_ratio = sum(v in qty_set for v in vals_float) / len(vals_float)
                score -= 3.0 * qty_ratio
        except:
            pass

        # Penalizar secuencias tipo 300580, 300581 (códigos)
        try:
            import numpy as np
            ints = s.map(lambda x: re.sub(r"\D", "", x)).replace("", np.nan).dropna().astype(int).tolist()
            if len(ints) >= 8:
                diffs = np.diff(sorted(ints))
                consec_ratio = (diffs == 1).mean()
                score -= 1.5 * consec_ratio
        except:
            pass

        if score > best_score:
            best_col, best_score = c, score

    return best_col



def pick_code_col(df: pd.DataFrame, rules: dict):
    best = None
    best_score = -1e9

    code_hints = [str(x).lower() for x in rules["code_header_hints"]]
    qty_hdrs   = [str(x).lower() for x in rules["quantity_headers"]]
    qty_vals   = rules["qty_value_set"]
    regex_prio = rules["prefer_code_regex"]
    min_uniq   = float(rules["code_min_uniqueness"])

    for c in df.columns:
        s = df[c].dropna().astype(str)
        if s.empty:
            continue
        name = str(c).lower()

        # penalizar si el header “huele” a cantidad
        if any(h in name for h in qty_hdrs):
            header_qty_penalty = 1.0
        else:
            header_qty_penalty = 0.0

        # métricas
        n = len(s)
        uniq_ratio = s.nunique() / max(1, n)

        # ¿parece cantidad por valores?
        try:
            vals = s.map(to_float).dropna().tolist()
        except:
            vals = []
        qty_ratio_vals = 0.0
        if vals:
            qty_ratio_vals = sum(
                (v is not None and abs(v-round(v))<1e-9 and int(round(v)) in qty_vals)
                for v in vals
            ) / len(vals)

        # score por regex (usamos prioridad del perfil o defaults)
        regex_hit = 0.0
        for i, patt in enumerate(regex_prio[::-1], 1):  # último de la lista = más fuerte
            try:
                hit = s.str.match(patt, na=False).mean()
            except:
                hit = 0.0
            regex_hit += i * float(hit)

        # alfanum “corto”
        alnum_short = s.str.match(r"^[A-Za-z0-9\-_]{2,24}$", na=False).mean()

        # bonus por header que “huele” a código
        header_code_bonus = 1.0 if any(h in name for h in code_hints) else 0.0

        # puntaje final
        score  = 0.0
        score += 2.0*regex_hit
        score += 1.4*alnum_short
        score += 1.6*uniq_ratio
        score += 0.6*header_code_bonus
        score -= 3.0*qty_ratio_vals
        score -= 2.0*header_qty_penalty
        if uniq_ratio < min_uniq:
            score -= 1.5

        if score > best_score:
            best, best_score = c, score
    return best


def pick_name_col(df):
    # preferencia por encabezado
    header_hits = [c for c in df.columns if any(h in str(c).lower() for h in NAME_HEADER_HINTS)]
    if header_hits:
        return header_hits[0]
    # si no, columna con más texto
    best_col, best_score = None, -1e9
    for c in df.columns:
        s = df[c].dropna().astype(str)
        if s.empty: 
            continue
        text_ratio = s.str.contains(r"[A-Za-zÁÉÍÓÚáéíóúÑñ]").mean()
        avg_len = s.map(len).mean()
        score = 1.0*text_ratio + 0.01*avg_len
        if score > best_score:
            best_col, best_score = c, score
    return best_col


# ---- Lectores de archivos ----
def read_csv_auto(path: str, encoding: Optional[str]=None, sep: Optional[str]=None) -> pd.DataFrame:
    if encoding is None: encoding = "latin-1"
    if sep is None: sep = ";"
    return pd.read_csv(path, encoding=encoding, sep=sep, dtype=str)

def read_excel_auto(path: str, sheet: Optional[str|int]=None) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet, dtype=str)

def read_docx_tables(path: str) -> pd.DataFrame:
    # Convierte todas las tablas de un DOCX en un único DataFrame apilado
    from docx import Document
    doc = Document(path)
    frames = []
    for t in doc.tables:
        rows = []
        for r in t.rows:
            rows.append([cell.text.strip() for cell in r.cells])
        if rows:
            df = pd.DataFrame(rows)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    # Heurística: la primera fila suele ser encabezado
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(how="all", axis=1).dropna(how="all", axis=0)
    df.columns = [f"col_{i}" for i in range(len(df.columns))]
    return df

def read_pdf_tables(path: str) -> pd.DataFrame:
    # Extrae tablas con pdfplumber (mejor para PDF nativo). Une todas en una sola.
    import pdfplumber
    frames = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for tbl in tables or []:
                df = pd.DataFrame(tbl)
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    # Heurística: primera fila como encabezado si parece texto
    def head_as_header(df):
        df = df.dropna(how="all", axis=1).dropna(how="all", axis=0)
        df = df.reset_index(drop=True)
        if len(df) >= 2:
            df.columns = [c if c is not None else f"col_{i}" for i, c in enumerate(df.iloc[0].fillna("").tolist())]
            df = df.iloc[1:].reset_index(drop=True)
        return df
    frames = [head_as_header(d) for d in frames]
    return pd.concat(frames, ignore_index=True)

# ---- Perfiles (YAML) ----
def load_profiles(dir_path: str = "profiles") -> List[Dict[str, Any]]:
    profiles = []
    if not os.path.isdir(dir_path):
        return profiles
    for name in os.listdir(dir_path):
        if name.lower().endswith(".yaml") or name.lower().endswith(".yml"):
            with open(os.path.join(dir_path, name), "r", encoding="utf-8") as f:
                try:
                    data = yaml.safe_load(f) or {}
                    data["_file"] = name
                    profiles.append(data)
                except Exception:
                    pass
    return profiles

def match_profile(file_name: str, sheet_names: List[str], sample_text: str, profiles: List[Dict[str, Any]]):
    fn = file_name.lower()
    sn = " ".join([s.lower() for s in sheet_names or []])
    txt = (sample_text or "").lower()
    for p in profiles:
        m = p.get("match", {})
        hit = False
        # filename_patterns
        for patt in m.get("filename_patterns", []) or []:
            if patt.lower() in fn:
                hit = True
        # names
        for patt in m.get("names", []) or []:
            if patt.lower() in fn:
                hit = True
        # sheet_contains
        for patt in m.get("sheet_contains", []) or []:
            if patt.lower() in sn:
                hit = True
        # header_keywords
        for patt in m.get("header_keywords", []) or []:
            if patt.lower() in txt:
                hit = True
        if hit:
            return p
    return None

# ---- Normalización principal ----
def normalize_df(df: pd.DataFrame, mapping: Optional[Dict[str, Any]]=None, dsv_factor: Optional[float]=None):
    df = df.copy()
    # Remover filas/columnas totalmente vacías
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        return pd.DataFrame(columns=["codigo","nombre","precio"])

    # Detectar inicio real
    start = first_data_row(df)
    if start > 0:
        df = df.iloc[start:].reset_index(drop=True)

    # Si tenemos un mapping con sinónimos, elegir primera columna existente
    chosen = {}
    if mapping:
        for k, arr in mapping.items():
            if isinstance(arr, list):
                for cand in arr:
                    if cand in df.columns:
                        chosen[k] = cand
                        break
            else:
                if mapping[k] in df.columns:
                    chosen[k] = mapping[k]

    # Si faltan, usar detección por patrones
    if "codigo" not in chosen or "nombre" not in chosen or ("precio" not in chosen and "precio_base" not in chosen):
        guess = choose_columns_by_pattern(df)
        chosen.setdefault("codigo", guess.get("codigo"))
        chosen.setdefault("nombre", guess.get("nombre"))
        chosen.setdefault("precio", guess.get("precio"))

    # Determinar columna de precio final
    precio_col = chosen.get("precio") or chosen.get("precio_base")
    keep = [c for c in [chosen.get("codigo"), chosen.get("nombre"), precio_col] if c in df.columns]
    out = df[keep].rename(columns={
        chosen.get("codigo","codigo"): "codigo",
        chosen.get("nombre","nombre"): "nombre",
        precio_col: "precio"
    })
    # Parsear precios a float
    out["precio"] = out["precio"].map(to_float)

    # Aplicar DSV si corresponde
    if dsv_factor:
        out["precio"] = out["precio"].map(lambda x: round(x*(1+dsv_factor), 2) if x is not None else None)

    # Filtrado final
    out = out.dropna(subset=["nombre","precio"])
    # Trim strings
    out["codigo"] = out["codigo"].astype(str).str.strip()
    out["nombre"] = out["nombre"].astype(str).str.strip()
    out = out.drop_duplicates(subset=["codigo","nombre","precio"], keep="first").reset_index(drop=True)
    return out[["codigo","nombre","precio"]]

def process_file_to_df(path: str, profile: Optional[Dict[str, Any]]):
    """
    Abre un archivo (Excel, CSV, DOCX o PDF) y devuelve:
    - df: DataFrame con la hoja principal
    - sheet_names: lista de hojas (si aplica)
    - sample_text: texto de ejemplo (para detección de perfil)
    """
    import os

    ext = os.path.splitext(path)[1].lower()
    df = pd.DataFrame()
    sheet_names = []
    sample_text = ""

    # Tipo definido por el perfil (si existe)
    typ = profile.get("input", {}).get("type") if profile else "auto"

    try:
        # 🧩 CSV
        if typ == "csv" or (typ == "auto" and ext == ".csv"):
            enc = profile.get("input", {}).get("encoding") if profile else None
            sep = profile.get("input", {}).get("sep") if profile else None
            df = read_csv_auto(path, enc, sep)

        # 🧩 Excel (.xlsx / .xls)
        elif typ in ["xlsx", "auto"] and ext in [".xlsx", ".xls"]:
            sheet = profile.get("input", {}).get("sheet") if profile else None

            # Si no se especifica hoja → leer todas
            excel_obj = pd.read_excel(path, sheet_name=None, dtype=str)
            sheet_names = list(excel_obj.keys())

            # Heurística: elegir la hoja con más filas válidas
            best_sheet = None
            max_rows = 0
            for name, df_candidate in excel_obj.items():
                clean = df_candidate.dropna(how="all").dropna(axis=1, how="all")
                if len(clean) > max_rows:
                    best_sheet = name
                    max_rows = len(clean)

            # Si el perfil definió una hoja específica y existe, usar esa
            if sheet and sheet in excel_obj:
                df = excel_obj[sheet]
            elif best_sheet:
                df = excel_obj[best_sheet]
            else:
                df = list(excel_obj.values())[0]

        # 🧩 DOCX
        elif typ == "docx" or ext == ".docx":
            df = read_docx_tables(path)
            sample_text = " ".join(df.head(3).astype(str).fillna("").values.flatten().tolist())

        # 🧩 PDF
        elif typ == "pdf" or ext == ".pdf":
            df = read_pdf_tables(path)
            sample_text = " ".join(df.head(3).astype(str).fillna("").values.flatten().tolist())

        else:
            # Fallback: intentar Excel, luego CSV
            try:
                df = pd.read_excel(path, dtype=str)
            except Exception:
                df = pd.read_csv(path, dtype=str)
    except Exception as e:
        return {"error": f"Error al leer archivo: {e}"}

    # ⚙️ Si el DataFrame está vacío o es raro
    if isinstance(df, dict):
        # Si por algún motivo aún devuelve dict, elegir la primera hoja
        df = list(df.values())[0]

    if not isinstance(df, pd.DataFrame):
        return {"error": "El lector no devolvió un DataFrame válido."}

    return df, sheet_names, sample_text

def suggest_profile_stub(file_name: str, df: pd.DataFrame) -> Dict[str, Any]:
    # genera una propuesta basándose en las columnas “adivinadas”
    guess = choose_columns_by_pattern(df) if not df.empty else {}
    return {
        "match": {"filename_patterns": [file_name]},
        "input": {"type": "auto", "drop_empty_columns": True, "skip_rows_hint": 8},
        "mapping": {
            "codigo": [guess.get("codigo","")],
            "nombre": [guess.get("nombre","")],
            "precio": [guess.get("precio","")]
        },
        "detect": {"by_patterns": True},
        "transform": {"normalize_price": True},
        "output": {"columns": ["codigo","nombre","precio"]}
    }

# ---- Carga perfiles al inicio ----
PROFILES = load_profiles("profiles")


def leer_lista(file_path: str, proveedor: str = "") -> Any:
    """
    Procesa un archivo local; para Excel detecta y procesa múltiples tablas por hoja.
    """
    if not os.path.exists(file_path):
        return {"error": f"No existe el archivo: {file_path}"}

    # perfil (si existe)
    profile = None
    if proveedor:
        for p in PROFILES:
            names = (p.get("match", {}).get("names") or []) + (p.get("match", {}).get("filename_patterns") or [])
            if any(proveedor.lower() in n.lower() or n.lower() in proveedor.lower() for n in names):
                profile = p
                break

    # leer (puede traer varias hojas)
    df, sheets, sample_text = process_file_to_df(file_path, profile if profile else {"input": {"type": "auto"}})
    if not isinstance(df, pd.DataFrame) or df.dropna(how="all").dropna(axis=1, how="all").empty:
        return {"error": "No se pudo leer el archivo o no contiene datos válidos."}

    # ⚠️ NUEVO: cortar en múltiples tablas
    blocks = split_tables_by_blank_rows(df)
    if not blocks:
        blocks = [df]

    rows = []
    debug = []
    # factor DSV si el perfil lo define (no aplica acá pero mantenemos)
    dsv = None
    if profile and isinstance(profile.get("postprocess"), dict):
        try:
            dsv = float(profile["postprocess"].get("dsv_factor", 0))
        except Exception:
            dsv = None

    for idx, block in enumerate(blocks):
        # detectar cabecera dentro del bloque
        hidx = header_row_idx(block)
        tbl = relabel_using_header(block, hidx)

        mapping = profile.get("mapping") if profile else None
        if mapping:
            # buscamos por *contiene* (case-insensitive) y elegimos el mejor match
            def best_match(syns):
                if not syns:
                    return None
                syns = [str(x).lower() for x in syns]
                candidates = []
                for col in tbl.columns:
                    col_l = str(col).lower()
                    hit = any(s in col_l for s in syns)
                    if not hit:
                        continue
                    weight = 0.0
                    if any(k in col_l for k in PRICE_HEADER_HINTS):
                        weight += 2.0
                    if any(k in col_l for k in QUANTITY_HEADER_HINTS):
                        weight -= 2.5
                    if "usd" in col_l or "u$s" in col_l or "u$d" in col_l:
                        weight += 2.5
                    candidates.append((weight, col))
                if not candidates:
                    return None
                candidates.sort(reverse=True)
                return candidates[0][1]

            cand_code = best_match(mapping.get("codigo"))
            cand_name = best_match(mapping.get("nombre"))
            cand_prec = best_match(mapping.get("precio") or mapping.get("precio_base"))
        else:
            cand_code = cand_name = cand_prec = None

        # si faltan columnas, elegir por heurística
        # si faltan columnas, elegir por heurística
        rules = get_rules(profile)  # ← obtiene defaults o reglas del perfil

        # inicializar por si algo falla
        code_col = None
        name_col = None

        # detectar columnas candidatas según mapping o heurística
        code_col = cand_code or pick_code_col(tbl, rules)
        name_col = cand_name or pick_name_col(tbl)

        # ⚙️ Validación: asegurar que code_col exista
        if not code_col:
            # si no se detectó ninguna columna, intentar una genérica
            if "codigo" in tbl.columns:
                code_col = "codigo"
            elif "cod" in tbl.columns:
                code_col = "cod"
            else:
                # si sigue sin haber código, salta validación y continúa
                code_col = None

        # 🧩 Validación adicional: evitar que 'codigo' sea una columna de cantidad
        if code_col:
            s = tbl[code_col].dropna().astype(str)
            name = str(code_col).lower()
            header_qty = any(h in name for h in rules["quantity_headers"])

            try:
                vals = s.map(to_float).dropna().tolist()
            except Exception:
                vals = []

            qty_ratio_vals = 0.0
            if vals:
                qty_ratio_vals = sum(
                    (v is not None and abs(v - round(v)) < 1e-9 and int(round(v)) in rules["qty_value_set"])
                    for v in vals
                ) / len(vals)
            uniq_ratio = s.nunique() / max(1, len(s))

            if header_qty or qty_ratio_vals >= 0.5 or uniq_ratio < float(rules["code_min_uniqueness"]):
                candidates = []
                for c2 in tbl.columns:
                    if c2 == code_col:
                        continue
                    ss = tbl[c2].dropna().astype(str)
                    if ss.empty:
                        continue
                    n = len(ss)
                    u = ss.nunique() / max(1, n)
                    nm2 = str(c2).lower()

                    # medir coincidencia con patrones de código
                    reg = 0.0
                    for i, patt in enumerate(rules["prefer_code_regex"][::-1], 1):
                        try:
                            hit = ss.str.match(patt, na=False).mean()
                        except:
                            hit = 0.0
                        reg += i * float(hit)

                    alnum = ss.str.match(r"^[A-Za-z0-9\-_]{2,24}$", na=False).mean()
                    hdr_code = 1.0 if any(h in nm2 for h in rules["code_header_hints"]) else 0.0

                    # métricas de cantidad
                    try:
                        vv = ss.map(to_float).dropna().tolist()
                    except:
                        vv = []
                    qrat = 0.0
                    if vv:
                        qrat = sum(
                            (v is not None and abs(v - round(v)) < 1e-9 and int(round(v)) in rules["qty_value_set"])
                            for v in vv
                        ) / len(vv)
                    hdr_qty = 1.0 if any(h in nm2 for h in rules["quantity_headers"]) else 0.0

                    sc = 2.0 * reg + 1.4 * alnum + 1.6 * u + 0.6 * hdr_code - 3.0 * qrat - 2.0 * hdr_qty
                    if u < float(rules["code_min_uniqueness"]):
                        sc -= 1.5
                    candidates.append((sc, c2))

                if candidates:
                    candidates.sort(reverse=True)
                    code_col = candidates[0][1]

                        # 1) preferencia explícita por headers (del perfil), por orden
                        # 1) preferencia explícita (YAML) por orden
        price_col = None
        prefer_list = (profile or {}).get("postprocess", {}).get("prefer_price_headers", [])
        if prefer_list:
            for pref in prefer_list:
                hit = next((c for c in tbl.columns if pref in str(c).lower()), None)
                if hit:
                    price_col = hit
                    break
                
        # 1.b) si hay lista de “malos precios”, ignorar esas columnas
        bad_list = (profile or {}).get("postprocess", {}).get("bad_price_headers", [])
        if price_col and any(bad in str(price_col).lower() for bad in bad_list):
            price_col = None
            
        # 1.b) preferencia dura local: si existen ambos, siempre "usd x mt"
        if not price_col:
            mt = next((c for c in tbl.columns if "usd x mt" in str(c).lower()), None)
            rollo = next((c for c in tbl.columns if "usd x rollo" in str(c).lower()), None)
            if mt and rollo:
                price_col = mt

        # 2) relación pack↔caja (para evitar agarrar "100" como precio)
        if not price_col:
            rel_hit = pick_price_by_relationship(tbl)  # la función que te pasé antes
            if rel_hit:
                price_col = rel_hit

        # 3) mapping sugerido por YAML (best_match) o heurística general
        if not price_col:
            price_col = cand_prec or pick_price_col(tbl, profile)



        # construir salida del bloque
        sub = tbl[[c for c in [code_col, name_col, price_col] if c]].copy()
        sub.columns = ["codigo", "nombre", "precio"][:sub.shape[1]]

        # parseo de precio
        if "precio" in sub.columns:
            sub["precio"] = sub["precio"].map(to_float)

        # limpieza final del bloque
        if "nombre" in sub.columns:
            sub = sub.dropna(subset=["nombre", "precio"])
            sub["nombre"] = sub["nombre"].astype(str).str.strip()
        if "codigo" in sub.columns:
            sub["codigo"] = sub["codigo"].astype(str).str.strip()

        # aplicar DSV si hace falta
        if dsv and "precio" in sub.columns:
            sub["precio"] = sub["precio"].map(lambda x: round(x*(1+dsv), 2) if x is not None else None)

        # filtrar falsos positivos obvios (precio que parece código interno)
        sub = sub[~sub["precio"].astype(str).str.match(r"^\d{5,6}$", na=False)]

        rows.append(sub[["codigo","nombre","precio"]])

        debug.append({
            "tabla": idx+1,
            "shape": list(tbl.shape),
            "cols": list(tbl.columns),
            "mapping_usado": {"codigo": code_col, "nombre": name_col, "precio": price_col},
            "muestras": sub.head(3).to_dict(orient="records")
        })

    if not rows:
        return {"error": "No se detectaron tablas con (codigo/nombre/precio)."}

    out = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["codigo","nombre","precio"], keep="first")

    result = {"data": out.to_dict(orient="records"), "debug": debug}
    if profile:
        result["profile_used"] = profile.get("_file")
    else:
        result["profile_suggestion"] = suggest_profile_stub(os.path.basename(file_path), df)

    return result


def previsualizar_informe(file_path: str, proveedor: str = "") -> Any:
    """
    Devuelve diagnóstico: columnas, primeras filas, guess de mapping, primera fila útil.
    """
    if not os.path.exists(file_path):
        return {"error": f"No existe el archivo: {file_path}"}

    # cargar con heurística auto para preview
    df, sheets, sample_text = process_file_to_df(file_path, {"input":{"type":"auto"}})
    if df.empty:
        return {"error": "No se detectaron tablas."}

    start = first_data_row(df)
    guess = choose_columns_by_pattern(df)
    return {
        "file": os.path.basename(file_path),
        "shape": list(df.shape),
        "columns": list(df.columns),
        "first_data_row": int(start),
        "guess": guess,
        "sheets": sheets
    }


# ---- HTTP API ----
api = FastAPI(title="Agente Listas Proveedores")

@api.post("/parse")
async def http_parse(file: UploadFile = File(...), proveedor: str = Form(default="")):
    """
    Procesa un archivo Excel/CSV/PDF/DOCX y devuelve JSON normalizado.
    """
    # Guardar temporalmente el archivo
    tmp_path = f"tmp_{file.filename}"
    content = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(content)

    try:
        result = leer_lista(tmp_path, proveedor)
        # Si el resultado es un dict (como devuelve leer_lista), devolvelo directo
        if isinstance(result, dict):
            return result
        # Si el resultado fuera un DataFrame (no debería), convertirlo
        else:
            return {"data": result.to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e)}
    finally:
        # Limpieza del archivo temporal
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@api.post("/preview")
async def http_preview(file: UploadFile = File(...), proveedor: str = Form(default="")):
    content = await file.read()
    tmp_path = os.path.join("tmp_upload_" + file.filename)
    with open(tmp_path, "wb") as f:
        f.write(content)
    try:
        result = previsualizar_informe(tmp_path, proveedor)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return result

if __name__ == "__main__":
    app.run()
