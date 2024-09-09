# Diagnóstico de Câncer de Mama com uma Rede Neural Perceptron

Este projeto demonstra o uso de uma rede neural Perceptron para classificar diagnósticos de câncer de mama com base no Conjunto de Dados de Câncer de Mama de Wisconsin (Diagnóstico).

## Índice
- [Instalação](#instalacao)
- [Conjunto de Dados](#conjunto-de-dados)
- [Exploração e Pré-processamento](#exploracao-e-pre-processamento)
- [Detecção de Outliers](#detecao-de-outliers)
- [Rede Neural](#rede-neural)
- [Resultados](#resultados)
- [Conclusão](#conclusao)
- [Licença](#licenca)

## Instalação

Para executar este projeto, você precisará instalar as bibliotecas necessárias. Você pode fazer isso usando o pip para instalar o seguinte:

```
scikit-learn
matplotlib
numpy
pandas
```

Alternativamente, você pode usar o ambiente Anaconda.

## Conjunto de Dados

O conjunto de dados utilizado é o Conjunto de Dados de Câncer de Mama de Wisconsin. Certifique-se de ter o arquivo de dados nomeado `BreastCancerWisconsinDataSet.csv` no seu diretório de trabalho.

## Exploração e Pré-processamento

Após carregar o conjunto de dados, o projeto realiza uma exploração inicial e pré-processamento:

- **Exploração:** O cabeçalho do conjunto de dados e as informações básicas são inspecionadas para entender sua estrutura e identificar qualquer valor ausente.
- **Pré-processamento:** A primeira coluna (números de ID) e a última coluna (vazia) são removidas do conjunto de dados.

## Detecção de Outliers

Outliers (valores atípicos) são verificados usando uma função personalizada para garantir que não haja valores extremos que possam afetar o desempenho da rede neural. Após identificar colunas com muitos zeros, uma nova versão filtrada do conjunto de dados é criada.

## Rede Neural

Dois modelos Perceptron são construídos e testados:

1. **Modelo 1 (p1):** Treinado no conjunto de dados original.
2. **Modelo 2 (p2):** Treinado no conjunto de dados filtrado, excluindo linhas com valores zero nas colunas relacionadas à concavidade.

### Padronização dos Dados

As características são normalizadas usando `MinMaxScaler` para escalar os dados entre 0 e 1.

### Visualização dos Dados

Gráficos de dispersão são gerados para verificar se as classes (Maligna e Benigna) são linearmente separáveis, o que é um requisito fundamental para o uso do Perceptron.

### Treinamento e Teste

O conjunto de dados é dividido em conjuntos de treinamento (70%) e teste (30%). Os modelos são então treinados e avaliados usando acurácia, relatórios de classificação e matrizes de confusão.

## Resultados

- **Modelo 1 (p1):** Apresenta bom desempenho no conjunto de dados original.
- **Modelo 2 (p2):** Apresenta acurácia superior no conjunto de dados filtrado, com zero falsos negativos e mínimos falsos positivos.

## Conclusão

- **Modelo 1** é recomendado para pacientes sem concavidades na massa mamária (2,3% do conjunto de dados).
- **Modelo 2** é recomendado para pacientes com concavidades na massa mamária (97,7% do conjunto de dados) devido à sua maior acurácia e menor taxa de erro.

## Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---
