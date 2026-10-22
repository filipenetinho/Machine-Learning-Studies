# Documentação: Case G — Reinternação Hospitalar em 30 dias

Notebook: `reinternacao_hospitalar_case_G.ipynb`
Base: `Estudo_Caso_5_Base_G_Dados.txt` (10.000 internações, 21 colunas, altas de jul/2025 a jun/2026)

> Os números da parte de dados (tamanho da base, ausentes, taxas por categoria) vêm da própria base e não mudam entre execuções. Os números de modelagem dependem da execução completa no Colab (com XGBoost, LightGBM e SHAP) e estão marcados como `[preencher após execução: ...]`.

---

## Introdução

**Contexto.** Um hospital privado tem taxa alta de reinternações não programadas logo após a alta. O programa de transição de cuidado (ligações da enfermagem e consulta precoce) só comporta parte dos pacientes, então é preciso decidir, no momento da alta, quem acompanhar.

**Objetivo.** Estimar a probabilidade de cada paciente ser reinternado no hospital nos 30 dias seguintes à alta (`reinternacao_30d = 1`) e usar essa probabilidade para ordenar as altas por risco.

**Missão do enunciado.** (1) exploração inicial breve, (2) modelo de classificação binária, (3) avaliação da qualidade do modelo, com entrega em notebook compilado e apresentação baseada em explicação do código, interpretação dos resultados e conclusões de negócio.

### Principais decisões

| Tema | Decisão | Motivo |
|---|---|---|
| Ausentes em `mora_sozinho` (608) | Nova categoria `nao_informado` | Quem não informou reinterna 12,7%, contra 7,7% de quem respondeu "nao": a ausência carrega informação e remover as linhas descartaria esse sinal. |
| Ausentes em `hemoglobina_alta` (310) e `sodio_alta` (436) | Indicadora `exame_ausente` + imputação pela mediana do treino | A falta de exame se concentra em eletivas curtas (média de 1,3 dia de internação, contra 5,7 dos demais) e esse grupo reinterna só 3,2%. A mediana é calculada só no treino, dentro do `ColumnTransformer`, para não vazar informação do teste. |
| Dia da alta | `alta_fim_semana` (sim/não) no lugar de `dia_semana_alta` | Sábado e domingo concentram as maiores taxas (10,5% e 11,4%); a variável resume o efeito em uma dummy em vez de seis. |
| `id_internacao` | Excluída | É só identificador. |
| Codificação | One-hot com `drop="first"`; quantitativas sem padronização | Mesmo pré-processamento do template; modelos de árvore não dependem de escala. |
| Desbalanceamento (8,61% de eventos) | Pesos balanceados (`ModeloPonderado`) + calibração isotônica | Os pesos fazem o modelo prestar atenção à classe rara; a calibração devolve as probabilidades para a escala real. |
| Técnicas | Árvore, Random Forest, AdaBoost, Gradient Boosting, XGBoost, Hist. Gradient Boosting, LightGBM | Todas as do template, comparadas na mesma validação. |
| Métrica de seleção | AUC (média da validação cruzada), com filtro de estabilidade de 10% | O uso é ordenar pacientes por risco, e a AUC mede exatamente essa capacidade de ordenação, sem depender de ponto de corte. |
| Ponto de corte | Definido pela capacidade da equipe (20% das altas, ajustável) | Com evento raro e probabilidades calibradas, o corte de 0,5 sinaliza pouquíssimos pacientes; o que limita o programa é quantos a equipe consegue acompanhar. |

---

## 1. Bibliotecas

**O que faz.** Importa pandas, numpy, matplotlib, seaborn, scikit-learn, XGBoost, LightGBM e SHAP, fixa `OMP_NUM_THREADS=1` e a semente do numpy.

**Por que.** Mesmo conjunto do template. A única inclusão é `SimpleImputer`, usado para imputar os exames. A linha `%pip install shap xgboost lightgbm` ficou comentada como no template, porque o Colab já traz essas bibliotecas; se alguma faltar, basta descomentar.

**Resultado.** Ambiente pronto, sem saída visível.

## 2. Leitura da base de dados

**O que faz.** Procura o arquivo `Estudo_Caso_5_Base_G_Dados.txt` na pasta atual; se não encontrar (caso típico no Colab), abre a janela de upload. Depois lê com `pd.read_table` (separador tab, decimal ponto).

**Por que.** Permite rodar no Colab do início ao fim sem editar caminho.

**Resultado.** Base carregada em `dados`.

## 3. Visualização, dimensões e tipos

**O que faz.** `head(10)`, `shape` e `dtypes`.

**Por que.** Conferir se a leitura deu certo e quais colunas são texto ou número.

**Resultado.** 10.000 linhas e 21 colunas: 9 quantitativas (inteiras ou decimais), 11 de texto e a resposta `reinternacao_30d` já em 0/1.

## 4. Análise exploratória inicial

### 4.1 Duplicidades
**O que faz.** Conta linhas duplicadas e `id_internacao` repetidos.
**Por que.** A base está na visão internação; repetição indicaria erro de extração.
**Resultado.** Nenhuma duplicidade.

### 4.2 Análise de preenchimento
**O que faz.** `isna().sum()` por coluna.
**Resultado.** Ausentes só em `mora_sozinho` (608), `hemoglobina_alta` (310) e `sodio_alta` (436).

### 4.3 Taxa de reinternação com e sem preenchimento
**O que faz.** Compara a taxa do evento entre registros preenchidos e ausentes em cada uma dessas três colunas.
**Por que.** Decidir se dá para remover as linhas (como no template) ou se a ausência tem significado.
**Resultado.** `mora_sozinho` ausente: 12,7% contra 8,3% preenchido. Hemoglobina ausente: 3,5% contra 8,8%. Sódio ausente: 3,2% contra 8,9%. A ausência não é aleatória.

### 4.4 Perfil das internações sem exames
**O que faz.** Cruza a falta de exame com tipo de admissão e cirurgia, e compara o tempo de internação.
**Resultado.** 397 das 436 internações sem algum exame são eletivas (320 delas cirúrgicas). O tempo médio de internação nesse grupo é 1,3 dia (máximo 4), contra 5,7 dias no restante. São cirurgias eletivas curtas, de baixo risco.

### 4.5 Tratamento de ausentes e criação de variáveis
**O que faz.** A função `preparar_base` preenche `mora_sozinho` com `nao_informado`, cria `exame_ausente` (1 quando falta hemoglobina ou sódio) e cria `alta_fim_semana`.
**Por que.** Ver tabela de decisões. A função é reaproveitada na nova base, garantindo o mesmo tratamento na hora de usar o modelo. Essas transformações são determinísticas, linha a linha, então podem ser feitas antes da divisão sem vazamento. A imputação numérica, que usa estatística da base, fica para o pré-processamento.
**Resultado.** `mora_sozinho`: 8.402 nao, 990 sim, 608 nao_informado. `exame_ausente`: 436 casos. `alta_fim_semana`: 1.764 altas no sábado ou domingo.

### 4.6 Estatísticas descritivas
**O que faz.** `describe()` das quantitativas.
**Resultado.** Idade média de 59 anos (18 a 99). Distâncias muito assimétricas (mediana de 10,4 km, máximo de 317 km); como os modelos são de árvore, esses extremos não precisam de tratamento. Tempo de internação mediano de 4 dias (máximo 63).

### 4.7 Distribuição da variável resposta
**O que faz.** Gráfico de barras e tabela de frequência de `reinternacao_30d`.
**Resultado.** 9.139 sem reinternação (91,39%) e 861 com reinternação (8,61%). Base desbalanceada, o que justifica pesos, divisão estratificada e métricas que não dependem da acurácia simples.

### 4.8 Recodificação da resposta
**O que faz.** Confere os valores da resposta.
**Resultado.** Apenas 0 e 1; nenhuma recodificação necessária.

## 5. Definição de variáveis explicativas e resposta

**O que faz.** Separa 10 quantitativas (`idade`, `distancia_hospital_km`, `indice_charlson`, `qtd_internacoes_12m`, `qtd_atendimentos_ps_6m`, `tempo_internacao_dias`, `hemoglobina_alta`, `sodio_alta`, `qtd_medicamentos_alta`, `exame_ausente`) e 10 qualitativas (`sexo`, `categoria_plano`, `mora_sozinho`, `tipo_admissao`, `clinica_responsavel`, `passou_uti`, `realizou_cirurgia`, `alta_fim_semana`, `destino_alta`, `retorno_agendado`) e cria `X` e `y`.

**Por que.** Todas as variáveis estão disponíveis no momento da alta, que é quando o modelo será usado. Não há variável posterior à alta que cause vazamento.

## 6. Análise bivariada

**O que faz.** Boxplots das quantitativas por categoria da resposta, tabela de médias e barras com a taxa de reinternação por categoria das qualitativas.

**Por que.** Ver, antes do modelo, quais variáveis separam reinternados de não reinternados.

**Resultado.**
- Reinternados têm, em média: mais idade (66,8 contra 58,5 anos), mais comorbidades (Charlson 2,47 contra 1,10), mais internações nos 12 meses anteriores (0,90 contra 0,31), mais idas ao PS (1,36 contra 0,69), internação mais longa (7,9 contra 5,3 dias), hemoglobina mais baixa (11,0 contra 12,0) e mais medicamentos na alta (6,2 contra 4,6).
- Distância menor entre os reinternados (12,8 contra 15,1 km). Isso provavelmente reflete que quem mora longe tende a voltar a outro hospital, e essas reinternações não entram na base.
- Taxas por categoria: oncologia 19,0% (cirurgia geral 4,3%); urgência 10,2% contra eletiva 5,2%; UTI 12,3% contra 7,6%; destino instituição de longa permanência 19,7% e home care 15,5%, contra 7,6% domicílio; mora sozinho 13,4% e não informado 12,7%, contra 7,7%; sem retorno agendado 11,1% contra 7,4%; alta no fim de semana 10,9% contra 8,1%; mulheres 9,8% contra 7,4% homens.

## 7. Divisão de treino e teste externo

**O que faz.** `train_test_split` com 20% para teste externo, estratificado pela resposta, `random_state=123`.

**Por que.** O teste externo fica fora de toda a escolha de hiperparâmetros e serve como estimativa honesta do desempenho. A estratificação mantém a mesma proporção de reinternações nos dois conjuntos.

**Resultado.** 8.000 linhas no treino e 2.000 no teste; taxa de 8,61% no treino e 8,60% no teste.

## 8. Pré-processamento

**O que faz.** `ColumnTransformer` com `SimpleImputer(strategy="median")` nas quantitativas e `OneHotEncoder(drop="first", handle_unknown="ignore")` nas qualitativas. Ajustado (`fit`) só no treino e aplicado (`transform`) no teste, na base completa e depois na nova base.

**Por que.** Aplicar a mediana e as categorias aprendidas no treino evita vazamento do teste. `handle_unknown="ignore"` impede erro se aparecer categoria nova na hora de usar o modelo.

**Resultado.** 27 colunas após a codificação.

## 9. Construção dos modelos

### 9.1 Técnicas e espaço de busca
**O que faz.** Define as sete técnicas do template com as mesmas distribuições de hiperparâmetros e a classe `ModeloPonderado`, que aplica pesos balanceados dentro do `fit`.
**Por que.** Os pesos ficam dentro do ajuste para valerem também em cada fold da validação cruzada e nos reajustes da calibração. Os limites de `min_samples_leaf` (50 a 100) e `min_samples_split` (100 a 200) fazem sentido para 8.000 linhas de treino.

### 9.2 Validação cruzada e métricas
**O que faz.** `StratifiedKFold` com 5 folds, 50 iterações de busca aleatória por técnica e métricas AUC e acurácia balanceada.
**Por que.** A AUC mede a ordenação; a acurácia balanceada dá peso igual às duas classes. A acurácia simples ficaria em torno de 91% só prevendo "não reinterna" para todos, então não serve aqui.

### 9.3 Loop de construção
**O que faz.** Para cada técnica, roda o `RandomizedSearchCV` no treino, reajusta cada combinação no treino inteiro e calcula as métricas no teste externo, guardando média e desvio de treino e teste interno e as variações percentuais.
**Resultado.** 350 modelos testados (7 × 50). Tempo de execução: [preencher após execução: tempo total do loop]. Em um teste com 2 núcleos, as cinco técnicas do scikit-learn levaram cerca de 3 minutos com 10 iterações, então a execução completa deve levar algo na casa de 20 a 30 minutos no Colab.

## 10. Comparação de modelos

**O que faz.** Mantém só os modelos cuja AUC no teste interno e no teste externo não se afasta mais de 10% da AUC de treino, ordena pela AUC média do teste interno e mostra os 15 melhores em tabela e gráfico. Uma tabela extra mostra o melhor modelo de cada técnica.

**Por que.** O filtro descarta modelos sobreajustados. A tabela por técnica ajuda a explicar na apresentação se os boostings ganham das árvores simples e por quanto.

**Resultado.** Modelos que passaram no filtro: [preencher após execução: quantidade de modelos]. Melhor modelo: [preencher após execução: número e técnica], AUC média no teste interno de [preencher após execução: AUC teste interno] e no teste externo de [preencher após execução: AUC teste externo]. Melhor por técnica: [preencher após execução: tabela do melhor de cada técnica].

## 11. Análise diagnóstica

### 11.1 Escolha do modelo
**O que faz.** Pega automaticamente o primeiro do ranking filtrado (o número pode ser trocado manualmente).
**Por que.** Evita deixar um número fixo do template que não existe nesta execução.
**Resultado.** Hiperparâmetros: [preencher após execução: hiperparâmetros do modelo escolhido].

### 11.2 Medidas de qualidade de ajuste
**O que faz.** Tabela com AUC e acurácia balanceada no treino, teste interno e teste externo.
**Resultado.** AUC treino [preencher após execução: AUC treino], teste interno [preencher após execução: AUC teste interno], teste externo [preencher após execução: AUC teste externo]. Acurácia balanceada no teste externo: [preencher após execução: acurácia balanceada].

### 11.3 Reajuste no treino e gráfico da árvore
**O que faz.** Reajusta o modelo escolhido em todo o treino com os pesos; se for uma árvore de decisão, desenha a árvore.
**Resultado.** O gráfico só aparece se o vencedor for uma árvore simples.

### 11.4 Probabilidades e calibração
**O que faz.** Calcula as probabilidades no teste externo, mostra a curva de calibração e o score de Brier, aplica `CalibratedClassifierCV` (isotônica, 5 folds) e repete a análise.
**Por que.** Os pesos balanceados empurram as probabilidades para cima (o modelo "acha" que o evento é tão comum quanto o não evento). Para a equipe clínica, a probabilidade precisa ser lida como risco real.
**Resultado.** Média das probabilidades antes: [preencher após execução: média antes]; depois: [preencher após execução: média depois]; taxa real no teste: 8,6%. Brier antes [preencher após execução: Brier antes] e depois [preencher após execução: Brier depois].

### 11.5 Curva ROC e KS
**O que faz.** Calcula AUC e KS antes e depois da calibração e desenha a curva ROC e as curvas acumuladas do KS.
**Por que.** Confirmar que a calibração não piora a ordenação.
**Resultado.** AUC [preencher após execução: AUC antes/depois]; KS [preencher após execução: KS antes/depois].

### 11.6 Captura por decis de risco
**O que faz.** Ordena o teste externo da maior para a menor probabilidade calibrada, divide em 10 faixas de 200 pacientes e calcula, em cada uma, taxa observada, probabilidade média, percentual acumulado de reinternações capturadas e lift. Mostra também a curva de captura.
**Por que.** É a tradução direta do problema: se a equipe acompanha X% das altas, quantas reinternações ela alcança?
**Resultado.** Nos 10% de maior risco: [preencher após execução: captura e taxa no decil 1]. Nos 20%: [preencher após execução: captura acumulada no decil 2]. Lift do primeiro decil: [preencher após execução: lift decil 1].

### 11.7 Medidas de desempenho por ponto de corte
**O que faz.** A função `medidas_desempenho` (do template, com F1 e percentual de sinalizados adicionados) é aplicada nos cortes 0,50, 0,30, 0,20, 0,10, na taxa geral e no corte que sinaliza 20% das altas (`capacidade_acompanhamento`).
**Por que.** Mostrar o trade-off entre sensibilidade (recall) e precisão (PPV) conforme o corte e justificar o corte por capacidade.
**Resultado.** No corte de capacidade: precisão [preencher após execução: PPV], sensibilidade [preencher após execução: sensibilidade], F1 [preencher após execução: F1], especificidade [preencher após execução: especificidade]. No corte de 0,50 com probabilidades calibradas, o percentual de pacientes sinalizados é [preencher após execução: % sinalizados no corte 0,5].

### 11.8 Comparação antes e depois da calibração e matriz de confusão
**O que faz.** Compara as medidas com probabilidades ponderadas e calibradas, e mostra a matriz de confusão no corte escolhido.
**Resultado.** Matriz de confusão: [preencher após execução: VP, FP, FN, VN].

## 12. Reajuste final do modelo

**O que faz.** Reajusta o modelo com os mesmos hiperparâmetros em 100% da base (com pesos) e calibra de novo na base completa. Se for árvore, mostra o perfil de maior risco (`resumo_perfil`) e o desenho final.

**Por que.** Depois de avaliado, o modelo que vai para uso aproveita todos os dados. O pré-processamento continua o ajustado no treino, que é o mesmo aplicado na nova base.

**Resultado.** Modelo final calibrado pronto para pontuar novas altas.

## 13. Interpretabilidade

**O que faz.** Importância por redução de impureza (quando o algoritmo oferece), por permutação (queda de AUC no teste externo ao embaralhar cada variável, 30 repetições) e SHAP (barras, beeswarm e waterfall do paciente de maior risco da amostra).

**Por que.** A equipe clínica precisa entender por que um paciente foi priorizado. A permutação é medida no teste externo e na métrica de interesse; o SHAP mostra também a direção do efeito.

**Resultado.** Cinco variáveis mais importantes por permutação: [preencher após execução: top 5 permutação]. Direção dos efeitos no beeswarm: [preencher após execução: leitura do beeswarm]. Explicação do paciente de maior risco: [preencher após execução: principais contribuições no waterfall].

## 14. Uso do modelo em nova base

**O que faz.** Cria cinco altas fictícias com as colunas originais (inclusive um paciente sem exames e outro sem informação de moradia), aplica `preparar_base`, o pré-processamento e o modelo final calibrado, e indica quem seria encaminhado ao programa pelo corte escolhido.

**Por que.** Mostrar o fluxo de uso real: o dado chega no formato bruto da alta e passa pelos mesmos tratamentos.

**Resultado.** [preencher após execução: probabilidades dos cinco pacientes e encaminhamento].

## 15. Resumo dos resultados

**O que faz.** Junta em uma tabela o modelo escolhido, AUC, KS, Brier, captura nos 20% de maior risco, precisão/sensibilidade/F1 no corte e as variáveis mais importantes.

**Por que.** Serve de slide final da apresentação, direto do notebook.

---

## Respostas ao enunciado

**Exploração inicial.** Base limpa (sem duplicidades), com 8,61% de reinternações. Ausentes concentrados em três variáveis e informativos, tratados sem remover linhas. Os sinais mais claros de risco na bivariada são histórico de uso do hospital (internações e PS), comorbidades, oncologia, urgência, UTI, destino diferente de domicílio, morar sozinho, alta sem retorno agendado e alta no fim de semana.

**Modelo de classificação binária.** Sete técnicas de árvore e ensemble comparadas com busca aleatória e validação cruzada estratificada, com pesos para o desbalanceamento. Escolhido: [preencher após execução: técnica e número do modelo].

**Avaliação da qualidade.** AUC no teste externo de [preencher após execução: AUC], KS de [preencher após execução: KS], Brier calibrado de [preencher após execução: Brier], captura de [preencher após execução: %] das reinternações nos 20% de maior risco.

## Conclusão

O modelo serve como ferramenta de priorização: em vez de escolher pacientes ao acaso ou por critério único, a equipe acompanha as altas do topo do ranking e alcança uma fração bem maior das reinternações para o mesmo esforço ([preencher após execução: captura nos 20% contra 20% aleatório]). O corte deve ser fixado pela capacidade real do programa. Entre os fatores de risco, alguns são acionáveis pelo próprio hospital: sair com retorno agendado, revisar prescrições extensas e reforçar a cobertura das altas de fim de semana.

**Limitações.**
- Reinternações em outros hospitais não são captadas, então parte dos "0" pode ser reinternação real. O efeito protetor aparente da distância provavelmente vem disso.
- A base cobre um único ano, sem validação em período posterior.
- Faltam variáveis sociais e de adesão ao tratamento.
- O desempenho é moderado: muitos pacientes de alto risco não reinternam e alguns de baixo risco reinternam. O modelo deve apoiar a priorização, não substituir a avaliação clínica.

**Próximos passos.** Validar em altas posteriores a junho de 2026, monitorar a calibração mês a mês, testar o corte junto com a equipe de enfermagem e medir o efeito do programa. Quem é acompanhado tende a reinternar menos, e isso vai mudar os dados usados para treinar as próximas versões.
