# Clusterização de Clientes com Apache Spark ML — Online Retail II

**Disciplina:** Processamento de Dados Massivos  
**Professor:** Alexandre Roriz — IESB  
**Aluno:** Ivan Meireles de Abreu  
**Tipo de problema:** Clusterização (aprendizado não supervisionado)  
**Algoritmo:** K-Means, biblioteca `pyspark.ml`

## Objetivo

Identificar perfis de clientes de um comércio eletrônico a partir dos indicadores **RFM** (Recência, Frequência e Valor Monetário), utilizando PySpark para processamento das transações e Spark ML para clusterização.

**Pergunta de negócio:** É possível identificar grupos de clientes com padrões distintos de recência, frequência e gasto, de modo a apoiar estratégias de relacionamento e retenção?

## Base de dados

- **Nome:** Online Retail II
- **Fonte:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- **Período:** 01/12/2009 a 09/12/2011
- **Volume original:** 1.067.371 registros e 8 colunas, em duas planilhas anuais
- **Colunas:** `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country`

A base possui mais de um milhão de linhas de itens de transações. O Spark foi empregado na leitura, exploração, limpeza e agregação dos registros. Após a agregação por cliente, o treinamento do K-Means utiliza 5.878 observações; essa etapa isoladamente não requer processamento distribuído, mas o fluxo é implementado com Spark ML e pode ser adaptado a bases maiores.

## Preparação e limpeza

Os dados Excel foram convertidos em CSV para ingestão com PySpark, usando `inferSchema=True` e verificando os tipos identificados. As exclusões foram apuradas **sequencialmente**, evitando dupla contagem de registros com mais de um problema.

| Etapa | Registros excluídos |
|---|---:|
| Duplicatas exatas | 34.335 |
| Clientes sem identificação, após retirar duplicatas | 235.151 |
| Quantidades não positivas, nas linhas restantes | 18.390 |
| Preços não positivos, nas linhas restantes | 70 |
| Notas canceladas adicionais | 0 |
| Datas nulas adicionais | 0 |
| **Total excluído** | **287.946** |
| **Registros válidos** | **779.425** |

A ausência de exclusões adicionais por cancelamento não significa que não existissem notas canceladas: suas linhas já haviam sido eliminadas por critérios anteriores. A remoção de duplicatas exatas é uma decisão metodológica que pode eliminar linhas legítimas de produtos repetidos.

## Engenharia de atributos — RFM

Para cada um dos **5.878 clientes**, foram calculados:

- **Recência:** dias desde a última compra até 10/12/2011 (um dia após a última transação da base).
- **Frequência:** quantidade de faturas distintas por cliente, usando ano e número da fatura.
- **Monetário:** soma de `Quantity × Price` por cliente.

Os três atributos receberam transformação `log1p` para reduzir assimetria, seguida de `VectorAssembler` e `StandardScaler` (`withMean=True`, `withStd=True`) em um Pipeline Spark ML.

## Seleção do número de clusters

Foram treinados modelos K-Means com `k` de 2 a 8, `seed=42` e `maxIter=50`, avaliados por **silhueta** e **inércia**.

| k | Silhueta | Inércia |
|---:|---:|---:|
| 2 | **0,6268** | 8.575,58 |
| 3 | 0,5048 | 6.335,56 |
| 4 | 0,5317 | 4.894,57 |
| 5 | 0,5093 | 4.068,48 |
| 6 | 0,4928 | 3.536,31 |
| 7 | 0,4842 | 3.185,84 |
| 8 | 0,4335 | 2.891,14 |

Embora `k=2` tenha obtido a maior silhueta, **`k=4` foi escolhido** como compromisso entre separação dos grupos, redução da inércia e interpretação de perfis comerciais mais detalhados. A inércia de `k=4` é aproximadamente 42,9% menor que a de `k=2`; essa redução, por si só, não prova superioridade do modelo.

## Perfis identificados (`k=4`)

| Cluster | Perfil interpretado | Clientes | % | Recência média (dias) | Frequência média | Gasto médio | Gasto mediano |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | Ocasionais / risco de afastamento | 1.454 | 24,74% | 231,06 | 5,03 | 1.945,21 | 1.447,74 |
| 1 | Inativos / baixo engajamento | 1.961 | 33,36% | 397,55 | 1,38 | 316,17 | 271,85 |
| 2 | Recentes / menor frequência | 1.262 | 21,47% | 29,20 | 3,04 | 854,07 | 732,07 |
| 3 | Alto valor / alta frequência | 1.201 | 20,43% | 28,44 | 19,24 | 10.698,26 | 4.967,42 |

Os nomes dos perfis são interpretações feitas após o agrupamento, não rótulos previamente conhecidos pelo algoritmo. Os valores monetários estão expressos na unidade de preço original da base.

## Persistência

O Pipeline completo (`VectorAssembler`, `StandardScaler`, `KMeans`) foi salvo com o método `save()` do Spark ML e testado por carregamento. O modelo salvo é um diretório com múltiplos arquivos, não um único arquivo `.pkl`.

## Como reproduzir

1. Abra o notebook Jupyter do projeto no Google Colab ou em um ambiente com Python e PySpark.
2. Instale as dependências utilizadas no notebook (`pyspark`, `pandas`, `openpyxl` e `matplotlib`).
3. Execute as células em sequência, começando pelo download público da base UCI e sua conversão de Excel para CSV.
4. Confira a limpeza, o cálculo RFM, os gráficos de silhueta e cotovelo, o modelo final e sua persistência.

**Importante para a entrega:** o notebook publicado deve baixar a base automaticamente da fonte original; não devem ser versionados os arquivos `.xlsx`, `.csv` nem os dados brutos. Também é necessário incluir o notebook **efetivamente executado**, com saídas e gráficos, no repositório.

## Limitações e melhorias

- A falta de identificadores de cliente levou à exclusão de parte dos registros.
- Devoluções e cancelamentos foram excluídos para modelar compras positivas; isso não representa o valor líquido de cada cliente.
- Duplicatas exatas podem representar transações legítimas.
- O conjunto é histórico e pode não refletir o comportamento atual.
- A avaliação da silhueta e dos perfis foi feita nos mesmos dados usados para ajustar o modelo; não equivale à validação de estabilidade em períodos futuros.
- O treinamento final utiliza 5.878 clientes, apesar de o pré-processamento abranger mais de um milhão de linhas.

Melhorias possíveis incluem testar a estabilidade temporal dos grupos, outros algoritmos e novos atributos de comportamento.

## Referências

- [UCI — Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- [Apache Spark — MLlib Clustering](https://spark.apache.org/docs/latest/ml-clustering.html)
- [Apache Spark — ML Pipelines](https://spark.apache.org/docs/latest/ml-pipeline.html)
