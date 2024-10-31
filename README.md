# Production Planning Optimization with Linear Programming and Machine Learning

## Objetivo do Projeto
Este projeto visa otimizar o planejamento da produção em uma empresa de equipamentos eletrônicos, utilizando uma abordagem integrada de Programação Linear e técnicas de Machine Learning. O objetivo é reduzir os custos operacionais e melhorar a assertividade na previsão de demanda, ajudando a empresa a tomar decisões mais embasadas e eficientes.

## Resumo
A interseção entre Pesquisa Operacional e Aprendizado de Máquina oferece um caminho promissor para resolver ineficiências no planejamento da produção. Neste estudo, utilizamos Programação Linear para otimização e técnicas de Machine Learning para prever a demanda futura, atingindo uma precisão de aproximadamente 9% de erro médio. O modelo proposto foi validado com stakeholders, e os resultados indicam uma significativa melhoria na eficácia do planejamento de produção.

## Etapas do Projeto
1. **Revisão de Literatura**: Pesquisa sobre Pesquisa Operacional e Machine Learning aplicados ao planejamento da produção.
2. **Coleta e Análise de Dados**: Coleta e limpeza dos dados de produção da empresa.
3. **Aplicação de Machine Learning**: Criação e comparação de modelos preditivos.
4. **Otimização com Programação Linear**: Desenvolvimento e aplicação de um modelo de Programação Linear para otimização de recursos.
5. **Análise de Resultados**: Avaliação dos resultados dos modelos e cálculo de métricas de erro.
6. **Validação**: Discussão e validação dos resultados junto aos stakeholders.

## Tecnologias e Bibliotecas
- **Linguagem**: Python
- **Bibliotecas**: Pandas, Numpy, Scikit-Learn, Matplotlib, Seaborn

## Código
O código realiza:
1. **Carregamento e preparação dos dados**: Limpeza dos dados para otimização do modelo.
2. **Construção e teste de modelos de Machine Learning**: Comparação de algoritmos de regressão, incluindo KNN e Decision Tree.
3. **Cálculo de métricas de erro**: Avaliação dos modelos com MAE, MAPE, MSE e RMSE.
4. **Otimização**: Seleção do melhor modelo e otimização de recursos com Programação Linear.

## Resultados
O modelo de Machine Learning com o menor erro médio de porcentagem absoluta (MAPE) foi selecionado e comparado ao conjunto de teste. A técnica escolhida atingiu um erro médio de previsão de cerca de 9%, mostrando-se eficaz para prever a demanda e melhorar o planejamento de produção.

## Estrutura do Projeto
- `scripts/`: Contém o script principal do projeto (`production_planning_optimization.py`).
- `data/`: Contém o dataset de produção (ou instruções para acessar os dados).
- `results/`: Resultados dos modelos, gráficos de previsão e métricas de erro.

## Artigo Publicado
- Para acesso ao artigo publicado, segue o DOI: https://doi.org/10.5935/jetia.v10i45.920
