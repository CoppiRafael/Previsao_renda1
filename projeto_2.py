import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import time

st.set_page_config(
      page_title="Análise exploratória da previsão de renda",
      page_icon="?",
      layout="wide",
 )

@st.cache_data
def load_data():
    dataframe = pd.read_csv('./input/previsao_de_renda.csv')
    return dataframe

def describe_inicial():
      df = load_data()
      st.write(df.describe())

def distribuicao_cat():
      df = load_data()
      for row in df.columns[3:]:
            plt.figure(figsize=(10, 6))
            if row in ['tempo_emprego','renda']:
                  plt.title(f'Distribuição da coluna: {row}')
                  df[row].plot(color="#2b667c")
            else:  
                  sns.countplot(data=df, x=row, hue=row, palette="crest", legend=False)
                  plt.title(f'Distribuição da coluna: {row}')

            st.pyplot(plt)
            with st.spinner("Aguarde..."):
                time.sleep(1)
            st.markdown('---')

def pairplot_corr():
      df = load_data()    
      
      sns.heatmap(data=df.select_dtypes(['float','int']).drop(columns='Unnamed: 0').corr(),cmap='crest',vmin=-1,center=0,vmax=1,annot=True)
      st.pyplot(plt)
      with st.spinner("Aguarde..."):
            time.sleep(1)
      st.markdown('---')


def cleaning(df):
      df.rename(columns={'df_index':'index'},inplace=True)
      df.drop(columns='Unnamed: 0',inplace=True)
      df['sexo'] = df['sexo'].map({'F':0,'M':1})
      df['data_ref'] = pd.to_datetime(df['data_ref']) 
      
      def bool_to_int(df):
            columns = df.select_dtypes('bool').columns.tolist()
            for col in columns:
                  df[col] = df[col].astype(int)
            return 
      
      bool_to_int(df)
      
      def creating_dummies(df):
            columns = df.select_dtypes('object').columns.tolist()
            df = pd.get_dummies(df,columns=columns,drop_first=True)
            bool_to_int(df)
            return df  
      
      df = creating_dummies(df)
      return df

def many_charts():
      renda = load_data()
      aux = renda.drop(columns='data_ref').select_dtypes(['object']).columns.tolist()
      for row in aux:
            fig,ax = plt.subplots(ncols=4,figsize=(18, 8))
            sns.boxplot(data=renda , x=row , y='renda',ax=ax[0],color='green')  
            sns.pointplot(data=renda,x=row,y='renda',ax=ax[1],color='orange')
            sns.lineplot(data=renda,x='data_ref',y=row,color='blue',ax=ax[2])
            sns.stripplot(data=renda , x=row , y='renda',ax=ax[3],color='r')
            ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)
            ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)
            ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=45)
            ax[3].set_xticklabels(ax[3].get_xticklabels(), rotation=45)
            st.pyplot(plt)
def med_long_time():
      df = load_data()
      df['data_ref'] = pd.to_datetime(df['data_ref'])
      df['new_data'] = df['data_ref'].apply(lambda x: x.strftime('%Y-%m'))
      var_qualitative = df.drop(['mau','new_data'],axis=1).select_dtypes(['object','bool']).columns.tolist()

      for var in var_qualitative:
            plt.figure(figsize=(12,6))
            df_grouped = df[['new_data',var,'mau']].groupby(['new_data',var]).count().reset_index().rename(columns={'mau':'count'}).sort_values('new_data')
            sns.lineplot(data=df_grouped,x='new_data',y='count',hue=var,marker='o',palette='Spectral')
            plt.title(f'Média de {var} ao longo dos anos')
            plt.xticks(rotation=90)
            plt.xlabel('Ano')
            plt.ylabel('Contagem')
            st.pyplot(plt)
                

def data_preparation():
      df = cleaning(load_data())
      st.subheader("Botão da etapa Data Preparation")
      if st.button("Etapa 3 Crisp-DM: Data Preparation"):
            progress = st.progress(0)
            for i in range(100):
                  time.sleep(0.1)
                  progress.progress(i + 1)
            st.success("Carregado com sucesso")
            st.write(df)
            
      st.markdown('---')

def selecao_lateral():
      selected_option = st.sidebar.radio("Selecione uma opção", ['Distribuição Variáveis', 'Matriz de correlação', 'Médias temporáis','Multi-análise'])
      st.header("Gráficos para análises")
      st.text("Aqui você pode selecionar o tipo de análise na barra lateral na esquerda da tela")
      if selected_option:
            if selected_option == 'Distribuição Variáveis':
                  distribuicao_cat()
            elif selected_option == 'Matriz de correlação':
                  pairplot_corr()
            elif selected_option == 'Médias temporáis':
                  med_long_time()
            elif selected_option == "Multi-análise":
                  many_charts()



def layout1():
      st.sidebar.title("Projeto 2")
      st.sidebar.image("data:image/png;base64,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")
      st.sidebar.header("Análises e informações")
      st.sidebar.markdown('---')
      st.markdown("# Projeto 02 - Previsão de renda")
      st.markdown("""
## Etapa 1 Crisp-DM: Business Understanding 
#### Objetivo do Projeto:

O objetivo do projeto é prever a renda dos indivíduos com base em diversas características demográficas e socioeconômicas. Essa previsão pode ser usada para diversas finalidades, como segmentação de clientes, análise de crédito, marketing direcionado, e políticas sociais.

#### Contexto do Negócio:

- **Setor**: O projeto se insere no setor financeiro e de análise de crédito.
- **Importância**: A capacidade de prever a renda é crucial para instituições financeiras ao avaliar o risco de crédito dos clientes. Além disso, empresas podem usar essas previsões para personalizar ofertas de produtos e serviços, melhorando a experiência do cliente e aumentando a taxa de conversão.
- **Stakeholders**: Os principais interessados incluem analistas de crédito, gerentes de produtos financeiros, equipes de marketing, e autoridades reguladoras.

""")
      st.markdown("""
## Etapa 2 Crisp-DM: Data Understanding

---

### Dicionário de dados

Um breve resumo do que veremos na base de dados trabalhada nesse projeto.


| Variável                | Descrição                                           | Tipo         |
| ----------------------- |:---------------------------------------------------:| ------------:|
| data_ref                | Exibe a data no formato Ano-mês-dia                                       | object|
| id_cliente              | Número de identificação do cliente                                       | int64|
| sexo                    | M = 'Masculino'; F = 'Feminino                                       | object|
| posse_de_veiculo        | Indica se possui veículo ou não                                       | bool|
| posse_de_imovel         | Indica se possui imóvel ou não                                       | bool|
| qtd_filhos              | Quantidade de filhos                                       | int64|
| tipo_renda              | Tipo de renda (ex: assaliariado, autônomo etc)                                       | object|
| educacao                | Nível de educação (ex: secundário, superior etc)                                       | object|
| estado_civil            | Estado civíl (ex: Casado, Solteiro etc)                                        | object|
| tipo_residencia         | Tipo de residência (ex: casa/apartamento, com os pais etc)                                       | object|
| idade                   | Informa a idade                                       | int64|
| tempo_emprego           | Tempo de emprego em anos                                       | float64|
| qt_pessoas_residencia   | Número de indivíduos resident                                       | float64|
| mau                     | Indica se é inadimplente ou não                                       | bool|             
| renda                   | Renda do cliente                                       | float64|

""")
      st.markdown("""
#### Carregando os pacotes
É considerado uma boa prática carregar os pacotes que serão utilizados como a primeira coisa do programa.
""")
      st.markdown("""
```import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split

%matplotlib inline
""")
      st.text("Visualização do Data frama inicial")
      st.dataframe(load_data())
      st.markdown('Caso queira ver análises acesse a barra lateral e adicione a análise desejada.')
      st.markdown("As análises fazem parte da etapa 2 do Crisp-DM: Data Undestanding ")

def layout2():
      st.markdown("""
## Etapa 4 Crisp-DM: Modelagem
Nessa etapa que realizaremos a construção do modelo. Os passos típicos são:
- Selecionar a técnica de modelagem
- Desenho do teste
- Avaliação do modelo

""")
      st.markdown("""
### Vamos testar dois modelos de Machine Learning e ver qual tem o melhor desempenho para nossos dados. 
- Arvores de Regressão
- Regressão Linear multipla
""")

      st.header("Regressão Linear")
      st.text("Fiz 3 regressões vou exibi-las em ordem.")
      st.subheader("Regressão 1")
      st.code("""
              
      renda = renda.assign(log_renda=lambda x: np.log(x['renda']))
      
      X = renda.drop(columns=['data_ref','index','renda','log_renda'])
      y = renda[['log_renda']] 

      X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
      reg1 = sm.OLS(y_train,X_train).fit()
      print(reg1.summary())

      y_pred_linear = reg1.predict(X_test)

      mse_linear = mean_squared_error(y_test, y_pred_linear)
      r2_linear = r2_score(y_test, y_pred_linear)

      print(f'MSE: {mse_linear}')
      print(f'R²: {r2_linear}') 
      """)
      st.text("""
                                 OLS Regression Results                                
=======================================================================================
Dep. Variable:              log_renda   R-squared (uncentered):                   0.992
Model:                            OLS   Adj. R-squared (uncentered):              0.992
Method:                 Least Squares   F-statistic:                          5.748e+04
Date:                Tue, 30 Jul 2024   Prob (F-statistic):                        0.00
Time:                        12:32:23   Log-Likelihood:                         -12227.
No. Observations:               11250   AIC:                                  2.450e+04
Df Residuals:                   11225   BIC:                                  2.469e+04
Df Model:                          25                                                  
Covariance Type:            nonrobust                                                  
=================================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------------------
sexo                              0.5161      0.016     32.005      0.000       0.484       0.548
posse_de_veiculo                  0.0043      0.015      0.280      0.780      -0.026       0.034
posse_de_imovel                   0.0834      0.015      5.602      0.000       0.054       0.113
qtd_filhos                       -3.2191      0.047    -68.795      0.000      -3.311      -3.127
idade                             0.0068      0.001      7.656      0.000       0.005       0.008
tempo_emprego                     0.0481      0.001     40.676      0.000       0.046       0.050
qt_pessoas_residencia             3.2529      0.045     72.971      0.000       3.165       3.340
mau                              -0.0583      0.044     -1.326      0.185      -0.144       0.028
tipo_renda_Bolsista              -0.0147      0.415     -0.035      0.972      -0.828       0.799
tipo_renda_Empresário             0.1798      0.017     10.404      0.000       0.146       0.214
tipo_renda_Pensionista           -0.2066      0.024     -8.490      0.000      -0.254      -0.159
tipo_renda_Servidor público       0.1009      0.025      4.007      0.000       0.052       0.150
educacao_Pós graduação            1.0202      0.208      4.907      0.000       0.613       1.428
educacao_Secundário               0.3618      0.059      6.159      0.000       0.247       0.477
educacao_Superior completo        0.4753      0.059      8.005      0.000       0.359       0.592
educacao_Superior incompleto      0.3573      0.068      5.274      0.000       0.224       0.490
estado_civil_Separado             3.2146      0.054     59.144      0.000       3.108       3.321
estado_civil_Solteiro             3.2605      0.047     68.671      0.000       3.167       3.354
estado_civil_União                0.0043      0.026      0.168      0.866      -0.046       0.055
estado_civil_Viúvo                3.2507      0.057     56.873      0.000       3.139       3.363
tipo_residencia_Casa              0.2938      0.062      4.760      0.000       0.173       0.415
tipo_residencia_Com os pais       0.2155      0.069      3.107      0.002       0.080       0.351
tipo_residencia_Comunitário       0.3665      0.125      2.942      0.003       0.122       0.611
tipo_residencia_Estúdio           0.1936      0.104      1.862      0.063      -0.010       0.397
tipo_residencia_Governamental     0.1909      0.073      2.622      0.009       0.048       0.334
==============================================================================
Omnibus:                      956.341   Durbin-Watson:                   1.992
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             5725.690
Skew:                           0.148   Prob(JB):                         0.00
Kurtosis:                       6.482   Cond. No.                     2.82e+03
==============================================================================

Notes:
[1] R² is computed without centering (uncentered) since the model does not contain a constant.
[2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[3] The condition number is large, 2.82e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
MSE: 0.5191215462791032
R²: 0.19174625345228036

""")
      st.subheader("Regressão 2")
      st.code("""
X = renda.drop(columns=['data_ref','index','renda','log_renda','mau','tipo_renda_Bolsista','estado_civil_União','tipo_residencia_Estúdio'])
y = renda[['log_renda']] 

# Adicionar uma constante ao modelo
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

reg2 = sm.OLS(y_train,X_train).fit()
display(reg2.summary())

y_pred_linear = reg2.predict(X_test)

mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print(f'MSE: {mse_linear}')
print(f'R²: {r2_linear}')
              """)
      st.text("""
OLS Regression Results                                
=======================================================================================
Dep. Variable:              log_renda   R-squared (uncentered):                   0.992
Model:                            OLS   Adj. R-squared (uncentered):              0.992
Method:                 Least Squares   F-statistic:                          6.842e+04
Date:                Tue, 30 Jul 2024   Prob (F-statistic):                        0.00
Time:                        12:34:35   Log-Likelihood:                         -12230.
No. Observations:               11250   AIC:                                  2.450e+04
Df Residuals:                   11229   BIC:                                  2.466e+04
Df Model:                          21                                                  
Covariance Type:            nonrobust                                                  
=================================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------------------
sexo                              0.5164      0.016     32.066      0.000       0.485       0.548
posse_de_veiculo                  0.0050      0.015      0.329      0.742      -0.025       0.035
posse_de_imovel                   0.0831      0.015      5.589      0.000       0.054       0.112
qtd_filhos                       -3.2465      0.044    -73.850      0.000      -3.333      -3.160
idade                             0.0068      0.001      7.791      0.000       0.005       0.009
tempo_emprego                     0.0482      0.001     40.794      0.000       0.046       0.051
qt_pessoas_residencia             3.2809      0.041     79.134      0.000       3.200       3.362
tipo_renda_Empresário             0.1798      0.017     10.416      0.000       0.146       0.214
tipo_renda_Pensionista           -0.2073      0.024     -8.523      0.000      -0.255      -0.160
tipo_renda_Servidor público       0.1030      0.025      4.098      0.000       0.054       0.152
educacao_Pós graduação            1.0266      0.208      4.938      0.000       0.619       1.434
educacao_Secundário               0.3657      0.059      6.227      0.000       0.251       0.481
educacao_Superior completo        0.4799      0.059      8.089      0.000       0.364       0.596
educacao_Superior incompleto      0.3595      0.068      5.308      0.000       0.227       0.492
estado_civil_Separado             3.2418      0.052     62.559      0.000       3.140       3.343
estado_civil_Solteiro             3.2864      0.045     73.386      0.000       3.199       3.374
estado_civil_Viúvo                3.2771      0.055     59.729      0.000       3.170       3.385
tipo_residencia_Casa              0.2275      0.050      4.548      0.000       0.129       0.326
tipo_residencia_Com os pais       0.1508      0.059      2.542      0.011       0.035       0.267
tipo_residencia_Comunitário       0.3011      0.119      2.527      0.012       0.068       0.535
tipo_residencia_Governamental     0.1243      0.063      1.968      0.049       0.001       0.248
==============================================================================
Omnibus:                      969.327   Durbin-Watson:                   1.992
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             5953.052
Skew:                           0.142   Prob(JB):                         0.00
Kurtosis:                       6.552   Cond. No.                     1.44e+03
==============================================================================

Notes:
[1] R² is computed without centering (uncentered) since the model does not contain a constant.
[2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[3] The condition number is large, 1.44e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
MSE: 0.5191387016034598
R²: 0.19171954322365703
""")
      st.subheader("Regressão 3")
      st.code("""
X = renda.drop(columns=['data_ref', 'index', 'renda', 'log_renda','posse_de_veiculo','tipo_residencia_Governamental', 'mau', 'tipo_renda_Bolsista', 'estado_civil_União', 'tipo_residencia_Estúdio'])
y = renda[['log_renda']]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

reg3 = sm.OLS(y_train, X_train).fit()
display(reg3.summary())

y_pred_linear = reg3.predict(X_test)

mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print(f'MSE: {mse_linear}')
print(f'R²: {r2_linear}')
""")
      st.text("""
OLS Regression Results                                
=======================================================================================
Dep. Variable:              log_renda   R-squared (uncentered):                   0.992
Model:                            OLS   Adj. R-squared (uncentered):              0.992
Method:                 Least Squares   F-statistic:                          7.561e+04
Date:                Tue, 30 Jul 2024   Prob (F-statistic):                        0.00
Time:                        12:35:58   Log-Likelihood:                         -12232.
No. Observations:               11250   AIC:                                  2.450e+04
Df Residuals:                   11231   BIC:                                  2.464e+04
Df Model:                          19                                                  
Covariance Type:            nonrobust                                                  
================================================================================================
                                   coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------------------
sexo                             0.5161      0.015     34.087      0.000       0.486       0.546
posse_de_imovel                  0.0813      0.015      5.481      0.000       0.052       0.110
qtd_filhos                      -3.2833      0.040    -82.328      0.000      -3.361      -3.205
idade                            0.0069      0.001      7.882      0.000       0.005       0.009
tempo_emprego                    0.0482      0.001     40.813      0.000       0.046       0.051
qt_pessoas_residencia            3.3180      0.037     89.646      0.000       3.245       3.391
tipo_renda_Empresário            0.1798      0.017     10.417      0.000       0.146       0.214
tipo_renda_Pensionista          -0.2083      0.024     -8.571      0.000      -0.256      -0.161
tipo_renda_Servidor público      0.1020      0.025      4.058      0.000       0.053       0.151
educacao_Pós graduação           1.0302      0.208      4.957      0.000       0.623       1.438
educacao_Secundário              0.3675      0.059      6.260      0.000       0.252       0.483
educacao_Superior completo       0.4821      0.059      8.135      0.000       0.366       0.598
educacao_Superior incompleto     0.3599      0.068      5.315      0.000       0.227       0.493
estado_civil_Separado            3.2790      0.048     68.020      0.000       3.185       3.374
estado_civil_Solteiro            3.3232      0.041     81.702      0.000       3.243       3.403
estado_civil_Viúvo               3.3138      0.052     64.198      0.000       3.213       3.415
tipo_residencia_Casa             0.1514      0.032      4.776      0.000       0.089       0.213
tipo_residencia_Com os pais      0.0750      0.045      1.664      0.096      -0.013       0.163
tipo_residencia_Comunitário      0.2260      0.113      2.004      0.045       0.005       0.447
==============================================================================
Omnibus:                     1010.107   Durbin-Watson:                   1.991
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             6525.805
Skew:                           0.145   Prob(JB):                         0.00
Kurtosis:                       6.720   Cond. No.                     1.44e+03
==============================================================================

Notes:
[1] R² is computed without centering (uncentered) since the model does not contain a constant.
[2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[3] The condition number is large, 1.44e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
MSE: 0.5199711096924652
R²: 0.19042351349532527
""")
      
      st.header("Arvores de Regressão")
      st.subheader("Arvore 1")
      st.code("""
from sklearn.tree import DecisionTreeRegressor

X = renda.drop(columns=['data_ref', 'index', 'renda', 'log_renda'])
y = renda[['renda']]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

tree = DecisionTreeRegressor(random_state=0)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

mse_tree = mean_squared_error(y_test, y_pred_tree)
mae_tree = mean_absolute_error(y_test, y_pred_tree)
rmse_tree = np.sqrt(mse_tree)
r2_tree = r2_score(y_test, y_pred_tree)

print("Árvore de Regressão:")
print(f'MSE: {mse_tree}')
print(f'MAE: {mae_tree}')
print(f'RMSE: {rmse_tree}')
print(f'R²: {r2_tree}')
              """)
      st.text("""Árvore de Regressão:
      
MSE: 17666573.24581896
MAE: 2456.5762463613996
RMSE: 4203.16229115876
R²: 0.15138767764923622""")
      
      st.subheader("Arvore 2")
      st.code("""
from sklearn.model_selection import GridSearchCV

# Definir os parâmetros para busca
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Criar o modelo
tree = DecisionTreeRegressor(random_state=0)

# Buscar
grid_search = GridSearchCV(estimator=tree, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Melhor modelo encontrado
best_tree = grid_search.best_estimator_

# Previsões e avaliação
y_pred_tree_best = best_tree.predict(X_test)
mse_tree_best = mean_squared_error(y_test, y_pred_tree_best)
mae_tree_best = mean_absolute_error(y_test, y_pred_tree_best)
rmse_tree_best = np.sqrt(mse_tree_best)
r2_tree_best = r2_score(y_test, y_pred_tree_best)

print("Melhor Árvore de Regressão:")
print(f'MSE: {mse_tree_best}')
print(f'MAE: {mae_tree_best}')
print(f'RMSE: {rmse_tree_best}')
print(f'R²: {r2_tree_best}')

""")
      st.text("""
Melhor Árvore de Regressão:
MSE: 15962324.317389945
MAE: 2568.3142351562274
RMSE: 3995.2877640277607
R²: 0.2332511279570214
""")
      st.markdown("""
## Etapa 5 Crisp-DM: Avaliação dos resultados
Nesta etapa, avaliamos o desempenho dos modelos desenvolvidos durante o projeto. Foram utilizados dois métodos de regressão para prever a renda: Regressão Linear e Árvore de Regressão. A seguir, apresentamos uma análise detalhada dos resultados obtidos com cada um dos modelos

### Regressão Linear
Para a Regressão Linear, foram utilizados os dados de treino e teste divididos de forma aleatória. O modelo foi ajustado com todas as variáveis disponíveis, exceto data_ref, index, renda e log_renda. Abaixo estão os resultados da avaliação do modelo
                  
**Primeiro modelo:**
- MSE: 0.5191215462791032
- R²: 0.19174625345228036

**Segundo modelo:**

- MSE: 0.5191387016034598
- R²: 0.19171954322365703

**Terceiro modelo:**
                  
- MSE: 0.5199711096924652
- R²: 0.19042351349532527

### Arvores de decisão:

**Primeiro modelo**:
- MSE: 17666573.24581896
- MAE: 2456.5762463613996
- RMSE: 4203.16229115876
- R²: 0.15138767764923622

**Segundo Modelo:**
- MSE: 15962324.317389945
- MAE: 2568.3142351562274
- RMSE: 3995.2877640277607
- R²: 0.2332511279570214                  

                  """)
      st.markdown("""
### Conclusões
Os resultados mostram que a Regressão Linear e a Árvore de Regressão inicial apresentaram um desempenho similar, com valores de MSE relativamente altos e baixos valores de r2, indicando que ambos os modelos têm limitações na capacidade de explicar a variabilidade da renda.

A busca em grade para a Árvore de Regressão melhorou um pouco o desempenho do modelo, reduzindo o MSE e o RMSE e aumentando o R². No entanto, os valores ainda sugerem que há espaço para melhorias.

Para próximos passos, seria interessante explorar outros modelos de machine learning, e realizar um trabalho mais profundo na engenharia de features e pré-processamento dos dados para tentar obter melhores resultados.
""")

layout1()
data_preparation()
selecao_lateral()
layout2()