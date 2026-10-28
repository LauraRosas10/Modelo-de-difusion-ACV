import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import h5py
import nibabel as nib
import numpy as np
import pandas as pd


# ============================================================
# Columnas clínicas por defecto
# ============================================================

TARGET_COLUMNS = [
    "age",
    "gender",
    "hypertension",
    "diabetes",
    "dyslipidemia",
    "atrial_fibrillation",
    "previous_stroke",
    "time_to_er",
    "nihss_admission"
]


# ============================================================
# Utilidades
# ============================================================

def parse_csv_list(raw: str) -> List[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def extract_id(filename: str) -> Optional[str]:
    """
    Extrae el ID del sujeto con formato ACV-XXX.

    Ejemplos:
        ACV-019.nii.gz
        ncct-ACV-019.nii.gz
        dwi-ACV-019.nii.gz
    """
    match = re.search(r"(ACV-\d+)", filename, re.IGNORECASE)
    return match.group(1).upper() if match else None


def build_file_map(folder: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}

    if not folder or not os.path.exists(folder):
        return mapping

    for root, _, files in os.walk(folder):
        for name in files:

            if name.startswith("."):
                continue

            sid = extract_id(name)

            if sid is None:
                continue

            path = os.path.join(root, name)

            if sid not in mapping:
                mapping[sid] = path

    return mapping


def deduplicate_by_patient(
    df: pd.DataFrame,
    id_col: str
) -> pd.DataFrame:

    out = df.copy()

    out[id_col] = (
        out[id_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    out = out.dropna(subset=[id_col])

    out["nan_count"] = out.isna().sum(axis=1)

    out = (
        out
        .sort_values([id_col, "nan_count"])
        .drop_duplicates(
            subset=id_col,
            keep="first"
        )
    )

    out = out.drop(columns=["nan_count"])

    return out.reset_index(drop=True)


def infer_id_column(
    df: pd.DataFrame,
    explicit_id_col: Optional[str]
) -> str:

    if explicit_id_col:

        if explicit_id_col not in df.columns:
            raise ValueError(
                f"La columna de ID solicitada "
                f"'{explicit_id_col}' no existe en el CSV."
            )

        return explicit_id_col

    candidates = [
        "ID",
        "Subject",
        "PATNO",
        "Subject_ID",
        "patient_id",
        "ACV_ID"
    ]

    for col in candidates:

        if col in df.columns:
            return col

    # Buscar una columna que contenga ACV-XXX
    for col in df.columns:

        if (
            df[col]
            .astype(str)
            .str.contains(
                r"ACV-\d+",
                case=False,
                na=False
            )
            .any()
        ):
            return col

    raise ValueError(
        "No se pudo inferir la columna de ID "
        "(tipo ACV-XXX). Especifica --id-column."
    )


# ============================================================
# NUEVO:
# Leer exactamente el Fold solicitado desde el JSON
# ============================================================

def load_fold_from_json(
    json_path: str,
    fold_idx: int
) -> Tuple[List[str], List[str], List[str], Dict[str, str]]:

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"No existe el archivo JSON: {json_path}"
        )

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    if "folds" not in data:
        raise ValueError(
            "El JSON no contiene la clave 'folds'."
        )

    selected_fold = None

    for fold in data["folds"]:

        if fold.get("fold_idx") == fold_idx:
            selected_fold = fold
            break

    if selected_fold is None:
        available = [
            fold.get("fold_idx")
            for fold in data["folds"]
        ]

        raise ValueError(
            f"No se encontró el fold {fold_idx}. "
            f"Folds disponibles: {available}"
        )

    # --------------------------------------------------------
    # Extraer IDs
    # --------------------------------------------------------

    train_ids = [
        item["patient_id"].strip().upper()
        for item in selected_fold["train"]
    ]

    valid_ids = [
        item["patient_id"].strip().upper()
        for item in selected_fold["val"]
    ]

    test_ids = [
        item["patient_id"].strip().upper()
        for item in selected_fold["test"]
    ]

    # --------------------------------------------------------
    # Extraer labels del JSON
    # --------------------------------------------------------

    labels_by_id = {}

    for split_name in ["train", "val", "test"]:

        for item in selected_fold[split_name]:

            sid = item["patient_id"].strip().upper()

            label = str(item["label"]).strip().upper()

            if label in {"0", "1", "ACV"}:
                label = "ACV"
            else:
                raise ValueError(
                    f"Label desconocido para {sid}: {repr(item['label'])}"
                )

            labels_by_id[sid] = label

    # --------------------------------------------------------
    # Validaciones
    # --------------------------------------------------------

    train_set = set(train_ids)
    valid_set = set(valid_ids)
    test_set = set(test_ids)

    if train_set & valid_set:
        raise ValueError(
            "Hay pacientes repetidos entre TRAIN y VALID."
        )

    if train_set & test_set:
        raise ValueError(
            "Hay pacientes repetidos entre TRAIN y TEST."
        )

    if valid_set & test_set:
        raise ValueError(
            "Hay pacientes repetidos entre VALID y TEST."
        )

    all_ids = (
        train_ids
        + valid_ids
        + test_ids
    )

    if len(all_ids) != len(set(all_ids)):
        raise ValueError(
            "Existen IDs duplicados dentro del fold."
        )

    # --------------------------------------------------------
    # Mostrar información
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(f"USANDO FOLD {fold_idx} DEL JSON")
    print("=" * 60)

    print(f"Train : {len(train_ids)} pacientes")
    print(f"Valid : {len(valid_ids)} pacientes")
    print(f"Test  : {len(test_ids)} pacientes")
    print(f"Total : {len(all_ids)} pacientes")

    print()

    print("Distribución de clases:")

    for name, ids in [
        ("Train", train_ids),
        ("Valid", valid_ids),
        ("Test", test_ids)
    ]:

        counts = defaultdict(int)

        for sid in ids:
            counts[labels_by_id[sid]] += 1

        print(
            f"  {name}: "
            f"{dict(sorted(counts.items()))}"
        )

    print("=" * 60)
    print()

    return (
        train_ids,
        valid_ids,
        test_ids,
        labels_by_id
    )


# ============================================================
# Selección de variables clínicas
# ============================================================

def coerce_numeric(
    df: pd.DataFrame,
    columns: List[str]
) -> pd.DataFrame:

    out = df.copy()

    for col in columns:

        if col in out.columns:
            out[col] = pd.to_numeric(
                out[col],
                errors="coerce"
            )

    return out


def choose_tabular_columns(
    df: pd.DataFrame,
    explicit_keep: List[str],
    explicit_drop: List[str]
) -> List[str]:

    if explicit_keep:

        selected = [
            c for c in explicit_keep
            if c in df.columns
        ]

        missing = [
            c for c in explicit_keep
            if c not in df.columns
        ]

        if missing:

            print(
                "Advertencia: las siguientes columnas "
                f"no están en el CSV: {missing}"
            )

    else:

        selected = [
            c for c in TARGET_COLUMNS
            if c in df.columns
        ]

    selected = [
        c for c in selected
        if c not in set(explicit_drop)
    ]

    if not selected:

        raise RuntimeError(
            "No se encontraron columnas tabulares."
        )

    return selected


# ============================================================
# Crear H5
# ============================================================

def write_split_h5(
    output_path: str,
    ids: List[str],
    ncct_map: Dict[str, str],
    dwi_map: Dict[str, str],
    mask_map: Dict[str, str],
    labels_by_id: Dict[str, str],
    clinical_df: pd.DataFrame,
    id_col: str,
    tabular_columns: List[str],
    tabular_mean: np.ndarray,
    tabular_std: np.ndarray
) -> None:

    print()
    print(f"Creando: {output_path}")

    with h5py.File(output_path, "w") as hf:

        # ----------------------------------------------------
        # Estadísticas
        # ----------------------------------------------------

        stats = hf.create_group("stats")

        tab_group = stats.create_group("tabular")

        tab_group.create_dataset(
            "columns",
            data=np.array(
                tabular_columns,
                dtype="S"
            )
        )

        tab_group.create_dataset(
            "mean",
            data=tabular_mean.astype(np.float32)
        )

        tab_group.create_dataset(
            "stddev",
            data=tabular_std.astype(np.float32)
        )

        # ----------------------------------------------------
        # Pacientes
        # ----------------------------------------------------

        written = 0

        for sid in ids:

            row = clinical_df.loc[
                clinical_df[id_col] == sid
            ]

            if row.empty:

                print(
                    f"ADVERTENCIA: {sid} "
                    "no existe en el CSV."
                )

                continue

            row0 = row.iloc[0]

            ncct_path = ncct_map.get(sid)
            dwi_path = dwi_map.get(sid)
            mask_path = mask_map.get(sid)

            g = hf.create_group(sid)

            # ------------------------------------------------
            # NCCT
            # ------------------------------------------------

            if ncct_path and os.path.exists(ncct_path):

                ncct_data = (
                    nib.load(ncct_path)
                    .get_fdata()
                    .astype(np.float32)
                )

                g.create_dataset(
                    "NCCT/data",
                    data=ncct_data
                )

            # ------------------------------------------------
            # DWI
            # ------------------------------------------------

            if dwi_path and os.path.exists(dwi_path):

                dwi_data = (
                    nib.load(dwi_path)
                    .get_fdata()
                    .astype(np.float32)
                )

                g.create_dataset(
                    "DWI/data",
                    data=dwi_data
                )

            # ------------------------------------------------
            # MASK
            # ------------------------------------------------

            if mask_path and os.path.exists(mask_path):

                mask_data = (
                    nib.load(mask_path)
                    .get_fdata()
                    .astype(np.float32)
                )

                mask_bin = (
                    mask_data > 0
                ).astype(np.float32)

                g.create_dataset(
                    "MASK/data",
                    data=mask_bin
                )

            # ------------------------------------------------
            # Variables clínicas
            # ------------------------------------------------

            tabular = (
                row0[tabular_columns]
                .to_numpy(dtype=np.float32)
            )

            g.create_dataset(
                "tabular",
                data=tabular
            )

            # ------------------------------------------------
            # Label
            # ------------------------------------------------

            g.attrs["DX"] = labels_by_id.get(
                sid,
                "ACV"
            )

            g.attrs["ID"] = sid

            written += 1

        print(
            f"Pacientes escritos: {written}/{len(ids)}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Genera archivos HDF5 de ACV utilizando "
            "los IDs definidos en un fold de un JSON."
        )
    )

    # --------------------------------------------------------
    # Directorios
    # --------------------------------------------------------

    parser.add_argument(
        "--ncct-dir",
        default=None,
        help="Directorio con imágenes NCCT"
    )

    parser.add_argument(
        "--dwi-dir",
        default=None,
        help="Directorio con imágenes DWI"
    )

    parser.add_argument(
        "--mask-dir",
        default=None,
        help="Directorio con máscaras"
    )

    parser.add_argument(
        "--clinical-csv",
        required=True,
        help="Ruta al CSV clínico"
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directorio de salida"
    )

    # --------------------------------------------------------
    # NUEVO: JSON de folds
    # --------------------------------------------------------

    parser.add_argument(
        "--split-json",
        required=True,
        help="JSON que contiene los folds"
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Fold que se utilizará. Por defecto: 1"
    )

    # --------------------------------------------------------
    # ID
    # --------------------------------------------------------

    parser.add_argument(
        "--id-column",
        default=None,
        help=(
            "Columna del CSV que contiene "
            "el ID del paciente"
        )
    )

    # --------------------------------------------------------
    # Variables clínicas
    # --------------------------------------------------------

    parser.add_argument(
        "--tabular-columns",
        default="",
        help=(
            "Lista separada por comas de variables "
            "clínicas a utilizar."
        )
    )

    parser.add_argument(
        "--drop-columns",
        default="",
        help=(
            "Lista separada por comas de columnas "
            "a excluir."
        )
    )

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    parser.add_argument(
        "--target-label-col",
        default=None,
        help=(
            "Columna del CSV para la etiqueta "
            "si se desea utilizar."
        )
    )

    parser.add_argument(
        "--default-label",
        default="ACV",
        help="Etiqueta por defecto"
    )

    args = parser.parse_args()

    # ========================================================
    # Crear directorio
    # ========================================================

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    # ========================================================
    # Mapear imágenes
    # ========================================================

    print("Buscando imágenes...")

    ncct_map = (
        build_file_map(args.ncct_dir)
        if args.ncct_dir
        else {}
    )

    dwi_map = (
        build_file_map(args.dwi_dir)
        if args.dwi_dir
        else {}
    )

    mask_map = (
        build_file_map(args.mask_dir)
        if args.mask_dir
        else {}
    )

    print(
        f"NCCT encontrados: {len(ncct_map)}"
    )

    print(
        f"DWI encontrados: {len(dwi_map)}"
    )

    print(
        f"MASK encontrados: {len(mask_map)}"
    )

    # ========================================================
    # Leer CSV
    # ========================================================

    clinical_df = pd.read_csv(
        args.clinical_csv
    )

    id_col = infer_id_column(
        clinical_df,
        args.id_column
    )

    print(
        f"Columna de ID utilizada: {id_col}"
    )

    clinical_df[id_col] = (
        clinical_df[id_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    clinical_df = deduplicate_by_patient(
        clinical_df,
        id_col
    )

    # ========================================================
    # Leer Fold del JSON
    # ========================================================

    (
        train_ids,
        valid_ids,
        test_ids,
        json_labels
    ) = load_fold_from_json(
        args.split_json,
        args.fold
    )

    # ========================================================
    # Verificar que los IDs existen en imágenes/CSV
    # ========================================================

    available_image_ids = (
        set(ncct_map.keys())
        | set(dwi_map.keys())
    )

    clinical_ids = set(
        clinical_df[id_col].unique()
    )

    all_fold_ids = (
        set(train_ids)
        | set(valid_ids)
        | set(test_ids)
    )

    print()
    print("=" * 60)
    print("VERIFICACIÓN DE IDs")
    print("=" * 60)

    missing_csv = sorted(
        all_fold_ids - clinical_ids
    )

    missing_images = sorted(
        all_fold_ids - available_image_ids
    )

    print(
        f"IDs del fold: {len(all_fold_ids)}"
    )

    print(
        f"IDs faltantes en CSV: "
        f"{len(missing_csv)}"
    )

    print(
        f"IDs faltantes en imágenes: "
        f"{len(missing_images)}"
    )

    if missing_csv:

        print(
            "IDs faltantes en CSV:"
        )

        print(missing_csv)

    if missing_images:

        print(
            "IDs faltantes en NCCT/DWI:"
        )

        print(missing_images)

    print("=" * 60)

    # ========================================================
    # Filtrar CSV SOLO a pacientes del Fold
    # ========================================================

    clinical_df = clinical_df[
        clinical_df[id_col].isin(
            all_fold_ids
        )
    ].copy()

    if clinical_df.empty:

        raise RuntimeError(
            "No hubo coincidencias entre "
            "los IDs del JSON y el CSV."
        )

    # ========================================================
    # Seleccionar variables clínicas
    # ========================================================

    keep_cols = (
        parse_csv_list(
            args.tabular_columns
        )
        if args.tabular_columns
        else TARGET_COLUMNS
    )

    drop_cols = parse_csv_list(
        args.drop_columns
    )

    tabular_columns = choose_tabular_columns(
        clinical_df,
        keep_cols,
        drop_cols
    )

    print()
    print(
        f"Variables clínicas utilizadas "
        f"({len(tabular_columns)}):"
    )

    for i, col in enumerate(
        tabular_columns,
        start=1
    ):
        print(
            f"  {i}. {col}"
        )

    # ========================================================
    # Convertir a numérico
    # ========================================================

    clinical_df = coerce_numeric(
        clinical_df,
        tabular_columns
    )

    # ========================================================
    # Imputación
    # ========================================================

    for col in tabular_columns:

        col_mean = clinical_df[col].mean()

        fill_val = (
            col_mean
            if pd.notna(col_mean)
            else 0.0
        )

        clinical_df[col] = (
            clinical_df[col]
            .fillna(fill_val)
        )

    # ========================================================
    # Guardar CSV limpio
    # ========================================================

    clean_csv_path = os.path.join(
        args.output_dir,
        "clinical_clean.csv"
    )

    clinical_df.to_csv(
        clean_csv_path,
        index=False
    )

    print()
    print(
        f"CSV clínico limpio: "
        f"{clean_csv_path}"
    )

    # ========================================================
    # Estadísticas clínicas
    #
    # IMPORTANTE:
    # Se calculan SOLO con TRAIN
    # ========================================================

    train_rows = clinical_df[
        clinical_df[id_col].isin(
            train_ids
        )
    ]

    tabular_values = (
        train_rows[
            tabular_columns
        ]
        .to_numpy(dtype=np.float32)
    )

    if len(train_rows) > 0:

        tabular_mean = (
            tabular_values.mean(axis=0)
        )

        tabular_std = (
            tabular_values.std(axis=0)
        )

    else:

        tabular_mean = np.zeros(
            len(tabular_columns),
            dtype=np.float32
        )

        tabular_std = np.ones(
            len(tabular_columns),
            dtype=np.float32
        )

    tabular_std[
        tabular_std < 1e-8
    ] = 1.0

    # ========================================================
    # Crear TRAIN
    # ========================================================

    write_split_h5(
        os.path.join(
            args.output_dir,
            "train.h5"
        ),
        train_ids,
        ncct_map,
        dwi_map,
        mask_map,
        json_labels,
        clinical_df,
        id_col,
        tabular_columns,
        tabular_mean,
        tabular_std
    )

    # ========================================================
    # Crear VALID
    # ========================================================

    write_split_h5(
        os.path.join(
            args.output_dir,
            "valid.h5"
        ),
        valid_ids,
        ncct_map,
        dwi_map,
        mask_map,
        json_labels,
        clinical_df,
        id_col,
        tabular_columns,
        tabular_mean,
        tabular_std
    )

    # ========================================================
    # Crear TEST
    # ========================================================

    write_split_h5(
        os.path.join(
            args.output_dir,
            "test.h5"
        ),
        test_ids,
        ncct_map,
        dwi_map,
        mask_map,
        json_labels,
        clinical_df,
        id_col,
        tabular_columns,
        tabular_mean,
        tabular_std
    )

    # ========================================================
    # Manifest
    # ========================================================

    counts = defaultdict(int)

    for sid in all_fold_ids:

        counts[
            json_labels[sid]
        ] += 1

    split_manifest = {

        "source_json": args.split_json,

        "fold": args.fold,

        "class_counts": dict(
            counts
        ),

        "id_column": id_col,

        "tabular_columns": (
            tabular_columns
        ),

        "splits": {

            "train": train_ids,

            "valid": valid_ids,

            "test": test_ids

        }

    }

    manifest_path = os.path.join(
        args.output_dir,
        "split_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8"
    ) as handle:

        json.dump(
            split_manifest,
            handle,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # Resumen final
    # ========================================================

    print()
    print("=" * 60)
    print("PROCESO TERMINADO")
    print("=" * 60)

    print(
        f"Fold utilizado: {args.fold}"
    )

    print(
        f"Total sujetos: {len(all_fold_ids)}"
    )

    print(
        f"Train: {len(train_ids)}"
    )

    print(
        f"Valid: {len(valid_ids)}"
    )

    print(
        f"Test: {len(test_ids)}"
    )

    print(
        f"Variables clínicas: "
        f"{len(tabular_columns)}"
    )

    print()
    print(
        f"Manifest: {manifest_path}"
    )

    print()
    print(
        "Archivos H5 generados correctamente."
    )


if __name__ == "__main__":
    main()

