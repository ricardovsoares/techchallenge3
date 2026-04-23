# ✈️ Flight Delay ML Pipeline — Tech Challenge Fase 3

> **Machine Learning Engineering | PosTech FIAP**  
> Análise e predição de atrasos em voos nos EUA usando dados do dataset *Flight Delays and Cancellations (2015)*.

---

## 📋 Índice

1. [Visão Geral do Projeto](#visão-geral)
2. [Estrutura do Repositório](#estrutura)
3. [Instalação e Configuração](#instalação)
4. [Como Executar](#como-executar)
5. [Módulos — Lógica e Decisões](#módulos)
   - [config.py](#configpy)
   - [data_preprocessing.py](#data_preprocessingpy)
   - [eda.py](#edapy)
   - [supervised_models.py](#supervised_modelspy)
   - [unsupervised_models.py](#unsupervised_modelspy)
   - [anomaly_detection.py](#anomaly_detectionpy)
   - [evaluation.py](#evaluationpy)
   - [main.py](#mainpy)
6. [Perguntas-Guia Respondidas](#perguntas-guia)
7. [Resultados Esperados](#resultados)
8. [Limitações e Próximos Passos](#limitações)

---

## 📌 Visão Geral

O transporte aéreo nos EUA registra milhões de atrasos por ano. Este projeto constrói um **pipeline completo de Machine Learning** para:

- **Classificar** se um voo vai atrasar (≥15 minutos) ou não.
- **Prever** o tempo de atraso em minutos (regressão).
- **Agrupar** rotas com perfis similares (clusterização K-Means).
- **Detectar** voos com comportamento anômalo (Isolation Forest + LOF).
- **Explorar** a estrutura dos dados com redução de dimensionalidade (PCA).

### Dataset

| Campo | Valor |
|-------|-------|
| Fonte | [Kaggle — Flight Delays and Cancellations](https://www.kaggle.com/datasets/usdot/flight-delays) |
| Período | 2015 |
| Registros | ~5,8 milhões de voos |
| Variáveis | 31 colunas (ver `dicionario_dados_flights.pdf`) |

---

## 🗂️ Estrutura do Repositório

```
flight_delay_project/
│
├── src/                        # Código-fonte principal
│   ├── config.py               # Configurações globais (paths, hiperparâmetros)
│   ├── data_preprocessing.py   # Carregamento, limpeza e engenharia de features
│   ├── eda.py                  # Análise exploratória e visualizações
│   ├── supervised_models.py    # Classificação e regressão
│   ├── unsupervised_models.py  # Clusterização (K-Means) + PCA
│   ├── anomaly_detection.py    # Isolation Forest + LOF
│   ├── evaluation.py           # Métricas consolidadas e relatório final
│   └── main.py                 # Orquestrador do pipeline
│
├── notebooks/
│   └── flight_delay_analysis.ipynb   # Notebook interativo completo
│
├── data/
│   └── flights.csv             # ⚠️ Não incluído no repositório (ver abaixo)
│
├── outputs/                    # Gerado automaticamente pelo pipeline
│   ├── eda/                    # Figuras da EDA
│   ├── supervised/             # Figuras e CSVs de modelos supervisionados
│   ├── unsupervised/           # Figuras e CSVs de clusterização/PCA
│   ├── anomaly/                # Figuras e CSVs de anomalias
│   ├── evaluation/             # Relatório final e comparativos
│   └── pipeline.log            # Log de execução
│
├── models/                     # Modelos serializados (.pkl)
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalação e Configuração

### Pré-requisitos

- Python 3.10+
- ~4 GB de RAM (para amostra de 5% do dataset)
- ~16 GB de RAM (para dataset completo)

### Passos

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/flight-delay-ml.git
cd flight-delay-ml

# 2. Crie e ative um ambiente virtual
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Baixe o dataset
# Acesse: https://www.kaggle.com/datasets/usdot/flight-delays
# Coloque o arquivo flights.csv em: data/flights.csv
```

---

## 🚀 Como Executar

### Pipeline completo (recomendado)

```bash
cd src
python main.py
```

### Modos parciais

```bash
# Apenas EDA
python main.py --eda-only

# Apenas modelagem supervisionada
python main.py --supervised-only

# Apenas clusterização/PCA
python main.py --unsupervised-only

# Apenas detecção de anomalias
python main.py --anomaly-only

# Teste rápido com 1% dos dados (~1 minuto)
python main.py --sample 0.01
```

### Notebook interativo

```bash
cd notebooks
jupyter notebook flight_delay_analysis.ipynb
```

---

## 📦 Módulos — Lógica e Decisões

---

### `config.py`

**Responsabilidade:** Centraliza todas as configurações do projeto.

**Decisões de design:**
- Todos os *magic numbers* (thresholds, hiperparâmetros, tamanho de amostra) vivem aqui.
- A separação em arquivo único facilita reprodutibilidade — alterar `SAMPLE_FRAC` afeta todo o pipeline.
- `DELAY_THRESHOLD = 15` segue o padrão da **FAA** (Federal Aviation Administration) para definir um voo como "atrasado".

**Parâmetros principais:**

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| `DELAY_THRESHOLD` | 15 min | Padrão FAA |
| `SAMPLE_FRAC` | 0.05 | Amostra de 5% (~250k linhas) para prototipagem rápida |
| `TEST_SIZE` | 0.20 | Split 80/20 padrão |
| `RANDOM_STATE` | 42 | Reprodutibilidade |
| `OPTIMAL_K` | 4 | Definido após análise Elbow + Silhouette |

---

### `data_preprocessing.py`

**Responsabilidade:** Carrega o CSV, limpa os dados, cria features derivadas e gera a variável-alvo.

**Fluxo do pipeline:**
```
load_raw() → remove_cancelled_diverted() → impute_missing()
           → engineer_features() → create_target()
```

**Decisões de design:**

1. **Dtypes explícitos na leitura:** Reduz uso de memória em ~40% (ex.: `int8` para meses, `float32` para delays).

2. **Remoção de cancelados/desviados:** Voos com `CANCELLED=1` ou `DIVERTED=1` não possuem `ARRIVAL_DELAY` válido — incluí-los contaminaria os modelos de atraso.

3. **Imputação com mediana (numérico):** `ARRIVAL_DELAY` tem forte assimetria (cauda longa) — a mediana é mais robusta que a média.

4. **Colunas de causa de atraso → fill com 0:** `AIR_SYSTEM_DELAY`, `AIRLINE_DELAY`, etc. são nulos quando aquela causa não ocorreu — zero é semanticamente correto.

5. **Engenharia de features derivadas:**

| Feature | Lógica | Motivação |
|---------|--------|-----------|
| `PERIOD_OF_DAY` | Hora da partida → madrugada/manhã/tarde/noite | Padrões de congestionamento |
| `SEASON` | Mês → estação do ano | Sazonalidade climática |
| `ROUTE` | `ORIGIN_AIRPORT + "_" + DEST_AIRPORT` | Captura padrão de rota específica |
| `IS_PEAK_HOUR` | Flag: partida entre 7-9h ou 17-19h | Horários de pico de tráfego |
| `DEP_HOUR` | `SCHEDULED_DEPARTURE // 100` | Granularidade horária |

6. **Target binário `DELAYED`:** `ARRIVAL_DELAY >= 15` → 1, caso contrário → 0. Escolheu-se 15 min por ser o limiar operacionalmente relevante (padrão industria).

---

### `eda.py`

**Responsabilidade:** Gera visualizações para responder as perguntas-guia do projeto.

**Decisões de design:**

- Cada função produz **uma figura independente** salva em `outputs/eda/` — facilita uso em apresentação.
- Backend `matplotlib.Agg` para compatibilidade com ambientes headless (servidores, CI/CD).
- Usamos **mediana** (não média) nos boxplots de atraso por companhia — distribuição assimétrica.
- O heatmap dia × período é a visualização mais reveladora: permite identificar combinações críticas.

**Visualizações geradas:**

| Arquivo | Conteúdo |
|---------|----------|
| `01_delay_distribution.png` | Histograma + proporção atrasados/não atrasados |
| `02_delay_by_airline.png` | Boxplot de atraso por companhia aérea |
| `03_heatmap_day_period.png` | Heatmap: dia da semana × período do dia |
| `04_top_airports_delay.png` | Top 20 aeroportos com maior atraso médio |
| `05_delay_causes.png` | Contribuição média por causa de atraso |
| `06_seasonal_trends.png` | Sazonalidade mensal do atraso |
| `07_correlation_matrix.png` | Matriz de correlação das variáveis numéricas |

---

### `supervised_models.py`

**Responsabilidade:** Treina e avalia modelos de classificação e regressão.

#### Classificação (DELAYED: 0 ou 1)

**Algoritmos comparados:**
- `RandomForestClassifier` (scikit-learn)
- `LGBMClassifier` (LightGBM)

**Métricas de avaliação:**

| Métrica | Justificativa |
|---------|---------------|
| F1-Score | Dataset desbalanceado (~37% atrasados) — Accuracy seria enganosa |
| ROC-AUC | Robusta a desbalanceamento; mede capacidade discriminativa geral |
| Confusion Matrix | Visualiza trade-off entre Precision e Recall |

**Decisões de design:**

1. **`class_weight="balanced"`:** Ambos os modelos recebem pesos inversamente proporcionais à frequência de cada classe — evita o modelo aprender a prever sempre "não atrasado".

2. **`OrdinalEncoder` para categóricas:** Tree-based models não requerem one-hot encoding — OrdinalEncoder é mais eficiente e compatível com LightGBM nativo.

3. **Split estratificado:** `stratify=y` garante mesma proporção de atrasados em treino e teste.

4. **Evitamos data leakage:** Features como `ARRIVAL_TIME`, `WHEELS_ON`, `ELAPSED_TIME` foram excluídas por serem conhecidas somente após a chegada do voo.

#### Regressão (ARRIVAL_DELAY em minutos)

**Filtragem:** Usa apenas voos com `ARRIVAL_DELAY > 0` — o objetivo é prever *quanto* vai atrasar, não se vai atrasar.

**Métricas:**

| Métrica | Justificativa |
|---------|---------------|
| RMSE | Penaliza erros grandes — importante operacionalmente |
| MAE | Interpretação direta em minutos |
| R² | Proporção da variância explicada |

**Modelos:** `RandomForestRegressor` e `LGBMRegressor` com mesma arquitetura de pipeline.

---

### `unsupervised_models.py`

**Responsabilidade:** Agrupa voos em clusters e aplica PCA para visualização.

#### K-Means (Clusterização)

**Objetivo:** Agrupar voos com perfis similares de distância, duração e horário — sem usar variável de atraso como input.

**Decisões de design:**

1. **`MiniBatchKMeans`:** Versão escalável do K-Means para datasets com milhões de pontos. Treina em mini-lotes (*batches*) sem carregar tudo na RAM.

2. **Determinação do K:** Combinação de dois critérios:
   - **Elbow Method (Inertia):** Identifica o "cotovelo" da curva de inércia.
   - **Silhouette Score:** Mede coesão interna e separação entre clusters — maximizado para encontrar K ótimo.

3. **`StandardScaler` obrigatório:** K-Means é sensível à escala das features (distância euclidiana). Sem normalização, `DISTANCE` (milhas) dominaria o agrupamento.

4. **Features sem leakage:** `SCHEDULED_DEPARTURE`, `SCHEDULED_TIME`, `DISTANCE`, `MONTH`, `DAY_OF_WEEK` — informações disponíveis antes do voo.

5. **Análise de perfil:** Após clusterização, calculamos médias de cada feature por cluster + `ARRIVAL_DELAY` médio (para interpretação pós-hoc).

#### PCA (Redução de Dimensionalidade)

**Objetivo:** Visualizar a estrutura dos dados e identificar quais variáveis mais explicam a variância.

**Decisões de design:**
- Mantemos componentes que explicam ≥95% da variância total (`PCA_VARIANCE_THRESHOLD`).
- **Biplot** PC1 × PC2: setas das features originais revelam quais variáveis "puxam" cada direção — interpretabilidade essencial.

---

### `anomaly_detection.py`

**Responsabilidade:** Identifica voos com comportamento atípico (anomalias).

**Algoritmos:**

| Algoritmo | Lógica | Vantagem |
|-----------|--------|----------|
| **Isolation Forest** | Isola pontos por partições aleatórias — anomalias precisam de menos partições | Escalável, não assume distribuição |
| **LOF (Local Outlier Factor)** | Compara densidade local de cada ponto com seus vizinhos | Detecta anomalias locais/contextuais |

**Decisões de design:**

1. **`contamination=0.05`:** Assume que ~5% dos voos são anomalias. Ajustável via parâmetro.

2. **Por que Isolation Forest como primário?** Mais eficiente computacionalmente para datasets grandes (O(n log n) vs O(n²) do LOF) e não assume gaussianidade.

3. **Features multidimensionais:** Combina `DEPARTURE_DELAY`, `ARRIVAL_DELAY`, `TAXI_OUT`, `TAXI_IN`, `SCHEDULED_TIME` — captura voos anômalos em múltiplas dimensões (um voo pode ter atraso normal mas taxi_out absurdo).

4. **Visualização PCA 2D:** Reduz as features para 2D para plotar anomalias vs. normais.

---

### `evaluation.py`

**Responsabilidade:** Consolida métricas, gera comparativos visuais e produz relatório final.

**Decisões de design:**
- Gráficos comparativos lado a lado (F1 vs AUC para classificadores; RMSE/MAE/R² para regressores).
- Relatório textual (`final_report.txt`) com conclusões, limitações e próximos passos.
- Módulo autônomo: pode ser executado standalone lendo os CSVs de resultados já salvos.

---

### `main.py`

**Responsabilidade:** Orquestrador do pipeline completo com CLI.

**Decisões de design:**

1. **Argumentos de linha de comando (`argparse`):** Permite executar módulos individualmente — essencial durante desenvolvimento/debug.

2. **Logging estruturado:** Saída simultânea em console e arquivo (`outputs/pipeline.log`) — facilita auditoria e reprodução.

3. **Modularidade:** Cada etapa pode ser pulada via flags — útil quando um módulo já foi executado e não precisa re-treinar.

---

## ❓ Perguntas-Guia Respondidas

| Pergunta | Onde ver |
|----------|----------|
| Quais aeroportos são mais críticos? | `outputs/eda/04_top_airports_delay.png` |
| Que características aumentam a chance de atraso? | `outputs/supervised/fi_random_forest.png` |
| Atrasos variam por dia/horário? | `outputs/eda/03_heatmap_day_period.png` |
| É possível agrupar aeroportos com perfis similares? | `outputs/unsupervised/02_cluster_profiles_heatmap.png` |
| Até que ponto conseguimos prever atrasos? | `outputs/evaluation/final_report.txt` |

---

## 📊 Resultados Esperados

### Classificação (5% do dataset)

| Modelo | ROC-AUC | F1-Score |
|--------|---------|----------|
| LightGBM | ~0.87 | ~0.72 |
| Random Forest | ~0.85 | ~0.70 |

### Regressão (voos com atraso > 0)

| Modelo | RMSE (min) | MAE (min) | R² |
|--------|-----------|-----------|-----|
| LightGBM | ~28 | ~18 | ~0.68 |
| Random Forest | ~30 | ~20 | ~0.64 |

> ⚠️ Valores aproximados com 5% de amostra. Resultados melhoram com dataset completo.

---

## ⚠️ Limitações e Próximos Passos

### Limitações Conhecidas

1. **Dataset de 2015 apenas:** Modelos não capturam mudanças estruturais pós-2015 (pandemia, novas rotas, etc.).
2. **`DEPARTURE_DELAY` como feature:** Em produção, este dado só estaria disponível *após* a partida do voo. Para predição pré-partida, deve ser removido.
3. **Sem dados meteorológicos:** `WEATHER_DELAY` captura parcialmente o impacto do clima, mas não inclui previsão meteorológica.
4. **Aeroportos de destino não modelados geograficamente:** Um mapa geográfico de atrasos adicionaria contexto espacial.

### Próximos Passos Sugeridos

- [ ] Integrar API de clima (NOAA/OpenWeather) como feature externa
- [ ] Modelo de séries temporais (LSTM/Transformer) para efeito cascata entre voos
- [ ] Dashboard interativo com **Streamlit** ou **Dash**
- [ ] Deploy em produção com **FastAPI** + **Docker** + **MLflow** para tracking de experimentos
- [ ] Otimização de hiperparâmetros com **Optuna** (Bayesian Search)
- [ ] Detecção de data drift com **Evidently AI**

---

## 📄 Licença

Projeto educacional — Tech Challenge PosTech FIAP. Dataset público via Kaggle (U.S. DOT).
