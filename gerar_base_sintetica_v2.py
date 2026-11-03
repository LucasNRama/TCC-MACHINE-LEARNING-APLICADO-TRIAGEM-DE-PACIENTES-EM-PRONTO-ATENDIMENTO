from pathlib import Path

import numpy as np
import pandas as pd


# Configurações
SEED = 42
N = 5000

PASTA_SCRIPT = Path(__file__).resolve().parent
ARQUIVO_SAIDA = PASTA_SCRIPT / "base_sintetica_experimental_v2.csv"

rng = np.random.default_rng(SEED)

QUEIXAS = [
    "dor_toracica",
    "falta_de_ar",
    "febre",
    "dor_abdominal",
    "trauma",
    "mal_estar_fraqueza",
]
PESOS_QUEIXAS = [0.16, 0.16, 0.18, 0.18, 0.16, 0.16]

PRIORIDADES = ["P1", "P2", "P3", "P4", "P5"]
PESOS_PRIORIDADES = [0.20] * 5


# Sorteio da queixa e do rótulo experimental
queixa = rng.choice(QUEIXAS, size=N, p=PESOS_QUEIXAS)
prioridade = rng.choice(PRIORIDADES, size=N, p=PESOS_PRIORIDADES)

# Convenção experimental: P1 é maior prioridade sintética; P5, menor.
# Não representa classificação clínica.
nivel = np.array([int(p[1]) for p in prioridade])
gravidade_simulada = (5 - nivel) / 4.0


# Idade adulta sintética
idade = np.clip(
    np.rint(rng.normal(loc=49, scale=18, size=N)),
    18,
    95,
).astype(int)


# Geração dos sinais vitais
frequencia_cardiaca = (
    76 + 27 * gravidade_simulada + rng.normal(0, 9, N)
)
frequencia_respiratoria = (
    16 + 7 * gravidade_simulada + rng.normal(0, 3, N)
)
pressao_sistolica = (
    126 - 22 * gravidade_simulada + rng.normal(0, 13, N)
)
pressao_pulso = 46 + rng.normal(0, 9, N)
temperatura = 36.7 + rng.normal(0, 0.35, N)
saturacao_oxigenio = (
    98 - 7 * gravidade_simulada + rng.normal(0, 2, N)
)


# Efeitos artificiais associados às queixas
mascara_febre = queixa == "febre"
frequencia_cardiaca[mascara_febre] += 3
temperatura[mascara_febre] += 0.5

mascara_falta_de_ar = queixa == "falta_de_ar"
frequencia_respiratoria[mascara_falta_de_ar] += 2
saturacao_oxigenio[mascara_falta_de_ar] -= 1.5

mascara_trauma = queixa == "trauma"
frequencia_cardiaca[mascara_trauma] += 2


# Limites técnicos artificiais da simulação
frequencia_cardiaca = np.clip(frequencia_cardiaca, 40, 170)
frequencia_respiratoria = np.clip(frequencia_respiratoria, 8, 40)
pressao_sistolica = np.clip(pressao_sistolica, 75, 200)
pressao_pulso = np.clip(pressao_pulso, 25, 80)
temperatura = np.clip(temperatura, 35.0, 40.5)
saturacao_oxigenio = np.clip(saturacao_oxigenio, 75, 100)

frequencia_cardiaca = np.rint(frequencia_cardiaca).astype(int)
frequencia_respiratoria = np.rint(frequencia_respiratoria).astype(int)
pressao_sistolica = np.rint(pressao_sistolica).astype(int)
pressao_pulso = np.rint(pressao_pulso).astype(int)
temperatura = np.round(temperatura, 1)
saturacao_oxigenio = np.rint(saturacao_oxigenio).astype(int)

# Garante pressão diastólica >= 40 e sistólica > diastólica.
# Ajusta a pressão de pulso se necessário.
pressao_pulso_maxima = np.minimum(80, pressao_sistolica - 40)
pressao_pulso_maxima = np.maximum(pressao_pulso_maxima, 25)
pressao_pulso = np.minimum(pressao_pulso, pressao_pulso_maxima)

pressao_diastolica = pressao_sistolica - pressao_pulso


# Identificadores e metadado de geração
id_sintetico = [f"S{i:05d}" for i in range(1, N + 1)]

# Metadado; não usar como entrada do modelo.
cenario_geracao = np.array([f"queixa_{q}" for q in queixa])


# Construção da base
base = pd.DataFrame({
    "id_sintetico": id_sintetico,
    "idade": idade,
    "queixa_principal": queixa,
    "frequencia_cardiaca": frequencia_cardiaca,
    "frequencia_respiratoria": frequencia_respiratoria,
    "pressao_sistolica": pressao_sistolica,
    "pressao_diastolica": pressao_diastolica,
    "temperatura": temperatura,
    "saturacao_oxigenio": saturacao_oxigenio,
    "prioridade_sintetica": prioridade,
    "cenario_geracao": cenario_geracao,
})


# Auditoria automática
assert base.shape == (N, 11), (
    f"Dimensão inesperada: {base.shape}; esperado {(N, 11)}."
)
assert base["id_sintetico"].is_unique
assert not base.isna().any().any()
assert not base.duplicated().any()

assert base["idade"].between(18, 95).all()
assert base["frequencia_cardiaca"].between(40, 170).all()
assert base["frequencia_respiratoria"].between(8, 40).all()
assert base["pressao_sistolica"].between(75, 200).all()
assert base["pressao_diastolica"].between(40, 175).all()
assert base["temperatura"].between(35.0, 40.5).all()
assert base["saturacao_oxigenio"].between(75, 100).all()

assert (base["pressao_sistolica"] > base["pressao_diastolica"]).all()
assert base["prioridade_sintetica"].isin(PRIORIDADES).all()
assert base["queixa_principal"].isin(QUEIXAS).all()
assert (
    base["cenario_geracao"] == "queixa_" + base["queixa_principal"]
).all()


# Salvamento e resumo
base.to_csv(ARQUIVO_SAIDA, index=False, encoding="utf-8")

print(f"Arquivo salvo em: {ARQUIVO_SAIDA}")
print(f"Registros: {len(base)}")
print(f"Colunas: {len(base.columns)}")
print(f"Valores ausentes: {int(base.isna().sum().sum())}")
print(f"IDs únicos: {base['id_sintetico'].is_unique}")
print(f"Duplicatas completas: {int(base.duplicated().sum())}")
print(
    "Sistólica maior que diastólica em todos os registros:",
    bool((base["pressao_sistolica"] > base["pressao_diastolica"]).all()),
)

print("\nDistribuição das prioridades:")
print(
    base["prioridade_sintetica"]
    .value_counts()
    .reindex(PRIORIDADES, fill_value=0)
)

print("\nDistribuição das queixas:")
print(
    base["queixa_principal"]
    .value_counts()
    .reindex(QUEIXAS, fill_value=0)
)

colunas_numericas = [
    "idade",
    "frequencia_cardiaca",
    "frequencia_respiratoria",
    "pressao_sistolica",
    "pressao_diastolica",
    "temperatura",
    "saturacao_oxigenio",
]

print("\nResumo numérico:")
print(base[colunas_numericas].describe().round(2))