# Agente MCP Multi-Proveedor (Listas de Precios)

Este proyecto expone **dos interfaces**:
1. **MCP (stdio)** con dos tools: `leer_lista` y `previsualizar_informe`.
2. **HTTP** (FastAPI) con endpoints `/parse` y `/preview`.

Devuelve siempre datos normalizados como:
```json
[{"codigo": "...", "nombre": "...", "precio": 123.45}]
```

## Requisitos
```bash
pip install -r requirements.txt
```

## Ejecutar como MCP (stdio)
```bash
python main.py
```
El binario expondrá los tools para un cliente MCP compatible.

## Ejecutar como servidor HTTP
```bash
uvicorn main:api --reload
```
- `POST /parse`  → procesa un archivo y devuelve JSON normalizado
- `POST /preview`→ diagnóstico de columnas/hojas y reglas usadas

### Ejemplos con `curl`
```bash
curl -F "file=@/ruta/a/archivo.xlsx" -F "proveedor=EUROTECH" http://127.0.0.1:8000/parse
curl -F "file=@/ruta/a/archivo.pdf" http://127.0.0.1:8000/preview
```

## Perfiles de proveedores
Agregá/ajustá YAMLs en `profiles/`. El agente intenta:
1) Match por nombre de archivo/keywords/hojas.
2) Mapeo por encabezados (sinónimos).
3) Detección por **patrones de contenido** (backup).

Si no reconoce un layout, propondrá un perfil nuevo (stub) para que lo guardes.

## Estructura
```
agente_mcp_listas/
├─ main.py
├─ requirements.txt
├─ profiles/
│  ├─ *.yaml
└─ rules/
   ├─ detectors.py
   └─ __init__.py
```
