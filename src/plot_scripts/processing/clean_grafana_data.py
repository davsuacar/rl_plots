#!/usr/bin/env python3
"""Limpia export CSV de Grafana (weather): columnas vacías, sufijos de unidad, renombres de termostatos."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = REPO_ROOT / (
    "data/paper/data/case_study/sim2real/weather-2026-04-20 10_48_28.csv"
)

# Prefijos en el nombre de columna (con _ final) para no tocar p. ej. thermostat_1_11_*.
THERMOSTAT_PREFIX_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("thermostat_2_5_", "bedroom_3_"),
    ("thermostat_2_4_", "bedroom_2_"),
    ("thermostat_2_3_", "bathroom_corridor_"),
    ("thermostat_2_2_", "bedroom_1_"),
    ("thermostat_2_1_", "bathroom_dressing_"),
    ("thermostat_1_4_", "bathroom_lobby_"),
    ("thermostat_1_1_", "living-kitchen_"),
)


def remove_degree_celsius_text(s: str) -> str:
    """Quita ºC, °C y variantes con espacio (Grafana / exportes mezclados)."""
    t = str(s)
    t = t.replace("\u00baC", "").replace("\u00b0C", "")
    t = re.sub(r"\s*[\u00b0\u00ba]\s*C\b", "", t, flags=re.IGNORECASE)
    return t.strip()


def rename_thermostat_column(col: str) -> str:
    new = col
    for old_prefix, new_prefix in THERMOSTAT_PREFIX_REPLACEMENTS:
        if new.startswith(old_prefix):
            new = new_prefix + new[len(old_prefix) :]
            break
    return remove_degree_celsius_text(new)


def _cell_empty(v: str) -> bool:
    return v.strip() == "" or v.strip().lower() in ("nan", "none", "null")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"CSV de entrada (por defecto: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="CSV de salida (por defecto: mismo nombre con sufijo _cleaned.csv)",
    )
    args = parser.parse_args()
    inp = args.input_csv.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"No existe el archivo: {inp}")
    out = args.output
    if out is None:
        out = inp.parent / f"{inp.stem}_cleaned{inp.suffix}"
    else:
        out = out.expanduser().resolve()

    with inp.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise SystemExit("CSV vacío")

    header = [rename_thermostat_column(remove_degree_celsius_text(h)) for h in rows[0]]
    body = rows[1:]

    cleaned_body: list[list[str]] = []
    for row in body:
        out_row: list[str] = []
        for cell in row:
            out_row.append(remove_degree_celsius_text(cell))
        # alinear longitud con cabecera
        while len(out_row) < len(header):
            out_row.append("")
        out_row = out_row[: len(header)]
        cleaned_body.append(out_row)

    # Columnas totalmente vacías (todas las filas de datos en blanco tras limpiar)
    if not cleaned_body:
        keep_idx = list(range(len(header)))
    else:
        keep_idx = [
            j
            for j in range(len(header))
            if not all(_cell_empty(r[j]) for r in cleaned_body)
        ]
    new_header = [header[j] for j in keep_idx]
    new_body = [[r[j] for j in keep_idx] for r in cleaned_body]

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(new_header)
        w.writerows(new_body)

    print(
        f"Escrito: {out} ({len(new_body)} filas, {len(new_header)} columnas; "
        f"eliminadas {len(header) - len(new_header)} columnas vacías)"
    )


if __name__ == "__main__":
    main()
