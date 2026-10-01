from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURACIÓN
# ============================================================

CANTIDAD_TOTAL = 1000

SEMILLA = 42
rng = np.random.default_rng(SEMILLA)


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ARCHIVO_RAW = BASE_DIR / "data" / "raw" / "DATASET_DTM.csv"

ARCHIVO_SALIDA = (
    BASE_DIR / "data" / "processed" / "DATASET_DTM_SINTETICO.csv"
)


# ============================================================
# COLUMNAS DE RECURSOS
# ============================================================

COLUMNAS_RECURSOS = [
    "VIAJES_NORMALES",
    "VIAJES_SOBREDIMENSIONADOS",
    "TRAILER",
    "TRACTOR VACTOR",
    "TRACTOR CISTERNA",
    "TRACTOR",
    "SEMIRREMOLQUE",
    "REMOLQUE",
    "PETROLERO",
    "MAQUINAS ESPECIALES",
    "MANLIFT",
    "MANIPULADOR",
    "LIVIANO",
    "GRUA",
    "GENERADOR",
    "DOLLY",
    "CISTERNA",
    "CARRETON",
    "CARGADORA",
    "BATEA",
    "AUTOELEVADOR",
    "ACCESORIO",
]


# ============================================================
# 1. CARGAR DATASET ORIGINAL
# ============================================================

print("\n==============================================")
print("CARGANDO DATASET ORIGINAL")
print("==============================================")

df = pd.read_csv(
    ARCHIVO_RAW,
    sep=";",
    encoding="utf-8",
    na_values=["?", ""]
)

df.columns = df.columns.str.strip()

print(f"Registros originales: {len(df)}")
print(f"Columnas originales: {len(df.columns)}")


# ============================================================
# 2. LIMPIAR LLS Y LLSADI
# ============================================================

def limpiar_lls(valor):
    """
    Convierte los LLS del CSV a enteros.

    Ejemplos:
        169.410 -> 169410
        174.997 -> 174997
        ?       -> vacío
    """

    if pd.isna(valor):
        return pd.NA

    valor = str(valor).strip()

    if valor in ["", "?"]:
        return pd.NA

    # El punto es separador de miles
    valor = valor.replace(".", "")

    try:
        return int(valor)
    except ValueError:
        return pd.NA


df["LLS"] = (
    df["LLS"]
    .apply(limpiar_lls)
    .astype("Int64")
)

df["LLSADI"] = (
    df["LLSADI"]
    .apply(limpiar_lls)
    .astype("Int64")
)


# ============================================================
# 3. VALIDAR LLS REALES
# ============================================================

# Si por algún error LLSADI es igual al LLS principal,
# anulamos el adicional.

mismo_lls = (
    df["LLS"].notna()
    & df["LLSADI"].notna()
    & (df["LLS"] == df["LLSADI"])
)

df.loc[mismo_lls, "LLSADI"] = pd.NA


# ============================================================
# 4. CONVERTIR FECHAS
# ============================================================

df["FECHA_INICIO"] = pd.to_datetime(
    df["FECHA_INICIO"],
    errors="coerce",
    dayfirst=True
)

df["FECHA_FIN"] = pd.to_datetime(
    df["FECHA_FIN"],
    errors="coerce",
    dayfirst=True
)


# ============================================================
# 5. CALCULAR DURACIÓN
# ============================================================

df["DURACION_DIAS"] = (
    df["FECHA_FIN"] - df["FECHA_INICIO"]
).dt.days

df.loc[
    df["DURACION_DIAS"] <= 0,
    "DURACION_DIAS"
] = np.nan


# ============================================================
# 6. LIMPIAR KM
# ============================================================

df["KM"] = (
    df["KM"]
    .astype(str)
    .str.replace(",", ".", regex=False)
)

df["KM"] = pd.to_numeric(
    df["KM"],
    errors="coerce"
)

df.loc[df["KM"] < 0, "KM"] = np.nan


# ============================================================
# 7. CONVERTIR RECURSOS A NUMÉRICOS
# ============================================================

for columna in COLUMNAS_RECURSOS:

    if columna in df.columns:

        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce"
        )

        df.loc[
            df[columna] < 0,
            columna
        ] = np.nan


# ============================================================
# 8. MARCAR REGISTROS REALES
# ============================================================

df["ORIGEN_REGISTRO"] = "REAL"


# ============================================================
# 9. IDENTIFICAR REGISTROS APTOS
# ============================================================

tiene_recurso = (
    df[COLUMNAS_RECURSOS]
    .notna()
    .any(axis=1)
)

df_base = df[
    df["KM"].notna()
    & df["DURACION_DIAS"].notna()
    & tiene_recurso
].copy()


print("\n==============================================")
print("CALIDAD DE DATOS")
print("==============================================")

print(f"Registros reales totales: {len(df)}")
print(f"Registros aptos como base: {len(df_base)}")


# ============================================================
# 10. PROPORCIÓN REAL DE LLS ADICIONAL
# ============================================================

# Calculamos qué porcentaje de registros reales
# tiene LLS adicional informado.

probabilidad_llsadi = df["LLSADI"].notna().mean()

print(
    f"Proporción de registros con LLS adicional: "
    f"{probabilidad_llsadi:.2%}"
)


# ============================================================
# 11. PREPARAR IDs
# ============================================================

# Guardamos TODOS los identificadores existentes.
# Esto incluye principales y adicionales.

ids_utilizados = set(
    df["LLS"]
    .dropna()
    .astype(int)
    .tolist()
)

ids_utilizados.update(
    df["LLSADI"]
    .dropna()
    .astype(int)
    .tolist()
)


# Buscamos el identificador máximo existente

if ids_utilizados:

    siguiente_lls = max(ids_utilizados) + 1

else:

    siguiente_lls = 1


def obtener_siguiente_lls():
    """
    Devuelve un LLS que todavía no fue utilizado.
    """

    global siguiente_lls

    while siguiente_lls in ids_utilizados:
        siguiente_lls += 1

    nuevo_id = siguiente_lls

    ids_utilizados.add(nuevo_id)

    siguiente_lls += 1

    return nuevo_id


# ============================================================
# 12. CANTIDAD DE SINTÉTICOS
# ============================================================

cantidad_sinteticos = max(
    0,
    CANTIDAD_TOTAL - len(df)
)


print("\n==============================================")
print("GENERACIÓN SINTÉTICA")
print("==============================================")

print(f"Objetivo total: {CANTIDAD_TOTAL}")
print(f"Registros reales: {len(df)}")
print(f"Registros sintéticos: {cantidad_sinteticos}")


# ============================================================
# 13. FUNCIÓN PARA VARIAR NÚMEROS
# ============================================================

def variar_numero(valor, variacion=0.10, entero=False):

    if pd.isna(valor):
        return np.nan

    valor = float(valor)

    if valor == 0:
        return 0

    desviacion = abs(
        valor * variacion
    )

    nuevo_valor = rng.normal(
        loc=valor,
        scale=desviacion
    )

    nuevo_valor = max(
        0,
        nuevo_valor
    )

    if entero:
        return int(round(nuevo_valor))

    return round(
        nuevo_valor,
        2
    )


# ============================================================
# 14. GENERAR REGISTROS SINTÉTICOS
# ============================================================

registros_sinteticos = []


if cantidad_sinteticos > 0:

    if df_base.empty:

        raise ValueError(
            "No existen registros reales suficientemente "
            "completos para generar datos sintéticos."
        )


    for _ in range(cantidad_sinteticos):

        # ====================================================
        # ELEGIR REGISTRO REAL BASE
        # ====================================================

        indice = rng.integers(
            0,
            len(df_base)
        )

        original = df_base.iloc[indice]

        nuevo = original.copy()


        # ====================================================
        # GENERAR LLS PRINCIPAL
        # ====================================================

        nuevo_lls = obtener_siguiente_lls()

        nuevo["LLS"] = nuevo_lls


        # ====================================================
        # GENERAR LLS ADICIONAL
        # ====================================================

        # El adicional NO es obligatorio.
        # Usamos la frecuencia observada en los datos reales.

        if rng.random() < probabilidad_llsadi:

            nuevo_llsadi = obtener_siguiente_lls()

            # Por construcción nunca será igual al principal
            nuevo["LLSADI"] = nuevo_llsadi

        else:

            nuevo["LLSADI"] = pd.NA


        # ====================================================
        # VARIAR DISTANCIA
        # ====================================================

        nuevo["KM"] = variar_numero(
            original["KM"],
            variacion=0.08
        )


        # ====================================================
        # VARIAR DURACIÓN
        # ====================================================

        nueva_duracion = variar_numero(
            original["DURACION_DIAS"],
            variacion=0.10,
            entero=True
        )

        nueva_duracion = max(
            1,
            nueva_duracion
        )

        nuevo["DURACION_DIAS"] = nueva_duracion


        # ====================================================
        # GENERAR FECHAS SINTÉTICAS
        # ====================================================

        desplazamiento = int(
            rng.integers(
                -180,
                181
            )
        )

        if pd.notna(original["FECHA_INICIO"]):

            nueva_fecha_inicio = (
                original["FECHA_INICIO"]
                + pd.Timedelta(
                    days=desplazamiento
                )
            )

            nueva_fecha_fin = (
                nueva_fecha_inicio
                + pd.Timedelta(
                    days=nueva_duracion
                )
            )

            nuevo["FECHA_INICIO"] = nueva_fecha_inicio

            nuevo["FECHA_FIN"] = nueva_fecha_fin


        # ====================================================
        # VARIAR RECURSOS
        # ====================================================

        for recurso in COLUMNAS_RECURSOS:

            valor_original = original[recurso]


            if pd.isna(valor_original):

                nuevo[recurso] = np.nan

                continue


            # Si el original es 0,
            # mantenemos 0.

            if valor_original == 0:

                nuevo[recurso] = 0

                continue


            nuevo[recurso] = variar_numero(
                valor_original,
                variacion=0.15,
                entero=True
            )


        # ====================================================
        # MARCAR ORIGEN
        # ====================================================

        nuevo["ORIGEN_REGISTRO"] = "SINTETICO"

        registros_sinteticos.append(
            nuevo
        )


# ============================================================
# 15. CREAR DATAFRAME SINTÉTICO
# ============================================================

if registros_sinteticos:

    df_sintetico = pd.DataFrame(
        registros_sinteticos
    )

else:

    df_sintetico = pd.DataFrame(
        columns=df.columns
    )


# ============================================================
# 16. UNIR REALES + SINTÉTICOS
# ============================================================

df_final = pd.concat(
    [
        df,
        df_sintetico
    ],
    ignore_index=True
)


# ============================================================
# 17. CORREGIR TIPOS
# ============================================================

df_final["LLS"] = (
    df_final["LLS"]
    .astype("Int64")
)

df_final["LLSADI"] = (
    df_final["LLSADI"]
    .astype("Int64")
)


df_final["KM"] = (
    df_final["KM"]
    .round(2)
)


df_final["DURACION_DIAS"] = (
    df_final["DURACION_DIAS"]
    .round()
    .astype("Int64")
)


for columna in COLUMNAS_RECURSOS:

    if columna in df_final.columns:

        df_final[columna] = (
            df_final[columna]
            .round()
            .astype("Int64")
        )


# ============================================================
# 18. VALIDACIONES DE LLS
# ============================================================

print("\n==============================================")
print("VALIDACIÓN DE LLS")
print("==============================================")


# ------------------------------------------------------------
# LLS PRINCIPALES VACÍOS
# ------------------------------------------------------------

lls_vacios = (
    df_final["LLS"]
    .isna()
    .sum()
)

print(
    f"LLS principales vacíos: {lls_vacios}"
)


# ------------------------------------------------------------
# LLS PRINCIPALES DUPLICADOS
# ------------------------------------------------------------

lls_duplicados = (
    df_final["LLS"]
    .dropna()
    .duplicated()
    .sum()
)

print(
    f"LLS principales duplicados: {lls_duplicados}"
)


# ------------------------------------------------------------
# LLSADI IGUAL AL LLS
# ------------------------------------------------------------

llsadi_igual = (
    (
        df_final["LLSADI"].notna()
        & (
            df_final["LLS"]
            == df_final["LLSADI"]
        )
    )
    .sum()
)

print(
    f"LLSADI iguales al LLS principal: {llsadi_igual}"
)


# ------------------------------------------------------------
# VERIFICAR IDs REPETIDOS ENTRE LLS Y LLSADI
# ------------------------------------------------------------

todos_ids = pd.concat(
    [
        df_final["LLS"],
        df_final["LLSADI"]
    ]
).dropna()

ids_duplicados_globales = (
    todos_ids
    .duplicated()
    .sum()
)

print(
    f"IDs repetidos entre LLS y LLSADI: "
    f"{ids_duplicados_globales}"
)


# ============================================================
# 19. CREAR CARPETA PROCESSED
# ============================================================

ARCHIVO_SALIDA.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 20. GUARDAR CSV
# ============================================================

df_final.to_csv(
    ARCHIVO_SALIDA,
    index=False,
    sep=";",
    encoding="utf-8-sig",
    date_format="%Y-%m-%d"
)


# ============================================================
# 21. RESUMEN FINAL
# ============================================================

cantidad_reales_final = (
    df_final["ORIGEN_REGISTRO"]
    == "REAL"
).sum()


cantidad_sinteticos_final = (
    df_final["ORIGEN_REGISTRO"]
    == "SINTETICO"
).sum()


print("\n==============================================")
print("PROCESO FINALIZADO")
print("==============================================")

print(
    f"Registros reales:     "
    f"{cantidad_reales_final}"
)

print(
    f"Registros sintéticos: "
    f"{cantidad_sinteticos_final}"
)

print(
    f"Registros totales:    "
    f"{len(df_final)}"
)


print("\nArchivo creado:")

print(
    ARCHIVO_SALIDA
)


print("\nPrimeros registros sintéticos:")

print(
    df_final[
        df_final["ORIGEN_REGISTRO"]
        == "SINTETICO"
    ][
        [
            "LLS",
            "LLSADI",
            "TIPO_DTM",
            "TIPO_MASTIL",
            "KM",
            "DURACION_DIAS",
            "TRACTOR",
            "SEMIRREMOLQUE",
            "GRUA",
            "CARRETON",
            "ORIGEN_REGISTRO"
        ]
    ].head(10)
)