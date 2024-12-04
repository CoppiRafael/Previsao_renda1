# Previsao_renda1
## [Link](https://previsaorenda1-ebac.streamlit.app/)

# Projeto 02 - Previsão de Renda

Este projeto tem como objetivo prever a **renda dos indivíduos** com base em diversas características demográficas e socioeconômicas. O desenvolvimento foi estruturado com base na metodologia **CRISP-DM**, abrangendo todas as etapas de análise de dados e construção de modelos preditivos.

## 🎯 Objetivo do Projeto
O objetivo principal é prever a renda dos indivíduos, fornecendo insights valiosos para aplicações em:
- **Análise de crédito**: Avaliação de risco de clientes.
- **Marketing direcionado**: Personalização de produtos e serviços.
- **Políticas sociais**: Apoio na formulação de políticas públicas.

## 🏢 Contexto do Negócio
- **Setor**: Financeiro e análise de crédito.
- **Importância**: Previsões de renda são fundamentais para entender o comportamento econômico de indivíduos.
- **Stakeholders**: Analistas de crédito, equipes de marketing, gerentes de produtos financeiros e autoridades reguladoras.

---

## 🚀 Metodologia: CRISP-DM
O projeto seguiu as etapas da metodologia **CRISP-DM**, descritas abaixo.

### 1. Business Understanding
Definição clara do problema e objetivos do projeto.

### 2. Data Understanding
Exploração e compreensão dos dados fornecidos. Os dados incluíam informações como:
| Variável              | Descrição                                          | Tipo     |
|-----------------------|--------------------------------------------------|----------|
| `data_ref`            | Data no formato Ano-Mês-Dia                      | Object   |
| `id_cliente`          | Identificação do cliente                         | Int64    |
| `sexo`                | Gênero (M = Masculino, F = Feminino)             | Object   |
| `posse_de_veiculo`    | Possui veículo? (Sim/Não)                        | Bool     |
| `posse_de_imovel`     | Possui imóvel? (Sim/Não)                         | Bool     |
| `qtd_filhos`          | Quantidade de filhos                             | Int64    |
| `tipo_renda`          | Tipo de renda (Assalariado, Autônomo, etc.)      | Object   |
| `educacao`            | Nível de educação (Secundário, Superior, etc.)   | Object   |
| `estado_civil`        | Estado civil (Casado, Solteiro, etc.)            | Object   |
| `tipo_residencia`     | Tipo de residência (Casa, Apartamento, etc.)     | Object   |
| `idade`               | Idade em anos                                    | Int64    |
| `tempo_emprego`       | Tempo de emprego em anos                         | Float64  |
| `qt_pessoas_residencia` | Número de indivíduos na residência             | Float64  |
| `mau`                 | Indicador de inadimplência (Sim/Não)             | Bool     |
| `renda`               | Renda do cliente                                 | Float64  |

### 3. Data Preparation
**Tratamento e transformação dos dados:**
- Renomeação de colunas e remoção de dados irrelevantes.
- Conversão de variáveis categóricas em variáveis dummies.
- Tratamento de dados faltantes e outliers.

**Exemplo de transformações:**
- Conversão da coluna `sexo` para valores binários (`0` para feminino, `1` para masculino).
- Criação de variáveis dummies para colunas categóricas.
- Identificação e tratamento de outliers na coluna `tempo_emprego`.

### 4. Modeling
Modelos de Machine Learning utilizados:
1. **Regressão Linear**.
2. **Árvore de Regressão**.

**Métricas de Avaliação:**
- **MSE (Erro Quadrático Médio)**.
- **R² (Coeficiente de Determinação)**.
- **RMSE (Raiz do Erro Quadrático Médio)**.
- **MAE (Erro Absoluto Médio)**.

#### Resultados
**Regressão Linear**:
- Melhor modelo: R² = 0.1917, MSE = 0.5191.

**Árvore de Regressão**:
- Melhor modelo após ajuste: R² = 0.2332, MSE = 15962324.31.

### 5. Evaluation
Os resultados mostram que tanto a Regressão Linear quanto a Árvore de Regressão possuem limitações no desempenho, com espaço para melhorias futuras na engenharia de features e uso de modelos mais complexos.

### 6. Deployment
O modelo foi implementado em um aplicativo web utilizando **Streamlit**. O app permite:
- Upload de novos dados.
- Geração de previsões de renda.
- Visualização interativa dos resultados.

---

## 📊 Visualizações e Análises
Foram realizadas diversas análises exploratórias e visualizações, como:
- **Distribuição das variáveis** (histogramas, boxplots, etc.).
- **Mapas de calor** para análise de correlação.
- **Pairplots** para visualização de relações entre variáveis.
- Análises por categoria (ex.: renda por tipo de renda, educação, etc.).

---

## 📁 Estrutura do Projeto
```plaintext
├── data/                # Dados brutos e processados
├── notebooks/           # Notebooks com análises e experimentos
├── app/                 # Código do aplicativo Streamlit
├── models/              # Modelos treinados
├── README.md            # Documentação do projeto
└── requirements.txt     # Dependências do projeto
