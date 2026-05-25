# SemAxis の研究的背景に関する文献調査

## エグゼクティブサマリー

SemAxis は、**LLM による自然言語仮説の生成**と、**NLI による各テキスト×各仮説の entailment スコア化**を結びつけ、最終的に **人間可読な自然言語特徴行列**を通常の線形分類器へ渡す、という位置づけの手法です。研究領域としては、**LLM による interpretable feature/concept discovery**、**NLI を使った zero-shot / weakly supervised text classification**、**concept bottleneck / explanation-oriented NLP**、そして **LLM に見せる代表サンプル選択**の交差点にあります。SemAxis の構成要素自体は個別にはかなり先行研究があり、特に **FELIX**, **CHiLL**, **TBM**, **Balek et al. の LLM feature generation**, **LogiPart** は README と論文の両方で明示的に位置づけるべき近接研究です。citeturn10view0turn11search0turn14search1turn40search0turn36search0turn36search2

最も近い先行研究トップ 5 は、**FELIX**（LLM が候補特徴を生成し downstream classifier に渡す）、**TBM**（自動概念発見＋概念値計測＋線形判定）、**CHiLL**（自然言語で記述された高水準特徴を LLM で抽出し線形モデルへ渡す）、**Balek et al.**（LLM 生成の解釈可能特徴で分類・ルール発見）、**LogiPart**（自然言語による論理分割と NLI ベース割当）です。SemAxis の差分は、**仮説を宣言文として明示生成し、NLI cross-encoder で行列化し、sklearn Transformer/Pipeline と fold-local generation を前提にした汎用実装**にあります。ただし、「LLM が解釈可能な自然言語特徴を作る」「自然言語特徴を線形モデルに入れる」「自動概念発見を分類器へ接続する」という主張は既存文献でかなり満たされており、新規性主張は慎重に絞る必要があります。citeturn10view0turn11search0turn14search1turn15search8turn40search0turn36search0turn36search2

研究・実装上の重要示唆は、次の 5 点に集約されます。  
- **新規性の中心は「発想」ではなく「組み合わせと検証設計」**に置くべきです。特に FELIX, CHiLL, TBM, Balek を踏まえると、「human-readable text features の自動生成」自体は先行研究があります。citeturn10view0turn11search0turn14search1turn40search0  
- **比較実験は TBM / FELIX / zero-shot NLI / direct LLM classification を外せません。** ここを欠くと、SemAxis の位置づけが弱く見えます。citeturn10view0turn14search1turn39search0turn20search0  
- **leakage・artifact・label-name parroting の監査が必須**です。NLI データには annotation artifacts があり、概念発見型手法には redundancy / leakage の報告もあります。citeturn31search0turn15search8turn30search6turn30search1  
- **Vote-K は有力な初期ベースラインですが、過度な多様性重視で外れ値を拾う欠点が報告**されています。facility-location 系や CoverICL 系の追加が妥当です。citeturn25search11turn22search8turn25search2turn24search0turn25search0  
- **評価は accuracy だけでは足りません。** readability, faithfulness, simulatability, stability, redundancy, latency/cost をセットで測るべきです。citeturn28search0turn29search0turn28search2turn26search11turn26search14

## README 記載文献の検証

| README記載名 | 確認できた正式引用 | venue/status | 公式/一次URL | SemAxisとの関係 | 修正が必要か |
|---|---|---|---|---|---|
| Balek et al. (2025): “LLM-based feature generation for interpretable ML” | **Vojtěch Balek, Lukáš Sýkora, Vilém Sklenák, Tomáš Kliegr. 2024. _LLM-based feature generation from text for interpretable machine learning_. arXiv:2409.07132.** citeturn40search0turn40search4 | **arXiv preprint** は一次確認できる。arXiv 初出は 2024-09-11、v2 は 2025-08-27。著者告知では後続の Springer 掲載らしき記述があるが、本調査では出版社一次ページを直接確認していないため、README の参照先が arXiv である以上は **2024 arXiv** と記すのが安全。 citeturn40search2turn40search6 | arXiv:2409.07132 / DOI 10.48550/arXiv.2409.07132 citeturn40search0turn40search4 | **非常に近い。** LLM で少数の解釈可能特徴を抽出し、分類やルール発見へ使う。ただし README 記述だけでは **NLI による仮説スコア化** は入っていない。SemAxis は「仮説文の自動生成 + NLI 行列化 + sklearn transform」という設計が差分。 citeturn40search0turn40search3 | **要修正。** README の arXiv 参照なら年は **2024** に直すべき。誌面版を引用したい場合は、出版社一次ページを別途厳密確認して書誌を差し替えるべき。 |
| Yin et al. (2019): “NLI as zero-shot text classifier” | **Wenpeng Yin, Jamaal Hay, Dan Roth. 2019. _Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach_. In EMNLP-IJCNLP 2019, pp. 3914–3923.** citeturn39search0 | **査読あり会議論文**。ACL Anthology で一次確認済み。 citeturn39search0 | ACL Anthology ID D19-1404 / DOI 10.18653/v1/D19-1404 citeturn39search0 | **中核構成要素の根拠。** ラベル記述を仮説文に変換し、pretrained NLI を zero-shot classifier として使う枠組みの代表例。SemAxis はこの発想を **「生成された仮説特徴の多次元化」** に一般化している。 citeturn39search0 | **小修正推奨。** README では通称ではなく、この正式題名で記載した方がよい。 |
| LogiPart (2025): “LLM hypothesis generation + NLI propagation” | **Tiago Fernandes Tavares. 2025. _LogiPart: Local Large Language Models for Data Exploration at Scale with Logical Partitioning_. arXiv:2509.22211.** ただし arXiv の current v3 は **2026-02-17** で、コメントに「旧題からの大幅改稿」とあり、HTML/要約系では旧題・別題が混在している。 citeturn36search0turn36search2turn36search4 | **arXiv preprint**。査読採択は一次確認できず。current v3 は 2026 改訂。 citeturn36search0turn36search2 | arXiv:2509.22211 / DOI 10.48550/arXiv.2509.22211 citeturn36search0 | **近いが同一ではない。** 自然言語による論理分割を作り、**NLI-based assignment** で大規模コーパスへ割り当てる点は SemAxis と非常に近い。一方で目的は **解釈可能な thematic tree / data exploration** であり、SemAxis のようなフラットな `n_texts × n_features` 変換器ではない。 citeturn36search0turn36search2 | **要修正。** README には **正式題名・arXiv ID・preprint であること** を明示すべき。旧題や「LogiPart 2025」だけでは曖昧。 |

README の 3 件のうち、**Yin et al. は正確に実在確認でき、SemAxis の NLI 部分の基本文献**です。**Balek はリンク先 arXiv に基づくなら年は 2024** で、README の 2025 表記は修正した方が安全です。**LogiPart も実在は確認できる**ものの、**旧題・改題・2026 改訂**が混在するため、README では「arXiv preprint」であること、正式題名、arXiv ID を揃えて書くのが望ましいです。citeturn39search0turn40search0turn40search2turn36search0turn36search2

## 文献マップ

下記は、SemAxis の**設計判断**または**評価設計**に直接影響するものに絞った文献マップです。`核となる手法/知見` は原論文の主張に寄せ、`SemAxisへの具体的関連` は本調査の推論を明示的に分けています。citeturn10view0turn11search0turn14search1turn39search0

**直接近接研究**

| 優先度 | 正式引用 | 一次URL/DOI/arXiv | 核となる手法/知見 | SemAxisへの具体的関連 | 参照すべき箇所 |
|---|---|---|---|---|---|
| Must read | Balek et al., 2024, *LLM-based feature generation from text for interpretable machine learning* | arXiv:2409.07132 citeturn40search0turn40search4 | LLM で少数の解釈可能特徴を抽出し、分類と action-rule mining に利用。LLM feature only や TF-IDF/SciBERT との比較もある。 citeturn40search3 | SemAxis と最も近い「LLM-generated interpretable text features」系。NLI は使っていないので、SemAxis は **仮説生成後の score assignment を NLI に固定**する点を差分として打ち出せる。 | Sec. 3.4、Table 8、Sec. 3.5 citeturn40search3 |
| Must read | Malberg, Mosca, Groh, 2024, *FELIX: Automatic and Interpretable Feature Engineering Using LLMs* | ECML PKDD 2024, DOI 10.1007/978-3-031-70359-1_14 citeturn10view0 | LLM が labeled example pairs から candidate features を生成し、redundancy filtering と feature assignment を行い、downstream classifier に渡す。五つの分類課題で TF-IDF・embedding・zero-shot LLM・fine-tuned classifier と比較。 citeturn10view0 | SemAxis と**かなり近い主競合**。README だけでなく、可能なら再現比較を置くべき相手。 | Abstract、Sec. 1、Sec. 4 citeturn10view0 |
| Must read | McInerney et al., 2023, *CHiLL: Zero-shot Custom Interpretable Feature Extraction from Clinical Notes with Large Language Models* | Findings of EMNLP 2023 / arXiv:2302.12343 citeturn11search0turn11search1 | 自然言語で指定された高水準特徴を LLM で抽出し、得た noisy labels を線形モデルへ渡す。医療ノートで reference features と同程度の性能・高い可解釈性を報告。 citeturn11search1 | **「自然言語特徴 × 線形モデル」** の直接先行。ただし CHiLL は **専門家が特徴を手で指定**する点で SemAxis と異なる。 | Abstract、Method、weight analysis citeturn11search0turn11search1 |
| Must read | Ludan et al., 2023, *Interpretable-by-Design Text Understanding with Iteratively Generated Concept Bottleneck* | arXiv:2310.19660 citeturn14search1 | Text Bottleneck Models は、LLM で salient concepts を自動発見・計測し、線形層で最終予測する。12 データセットで black-box baselines と比較し、人手評価で概念品質も検証。 citeturn14search1 | **SemAxis と最も危険な近接研究の一つ**。差分は、SemAxis が **sklearn-compatible transformer として flat feature matrix を出す**点、NLI measurement を前面に出す点。 | Abstract、concept generation / measurement、human eval citeturn14search1turn15search8 |
| Useful | Zhou et al., 2024, *Hypothesis Generation with Large Language Models* | NLP4Science 2024, DOI 10.18653/v1/2024.nlp4science-1.10 citeturn34search0 | Labeled examplesから仮説を生成し、bandit-inspired reward で反復更新する data-driven hypothesis generation。分類性能改善と仮説の洞察性を報告。 citeturn34search0 | SemAxis の**仮説生成部**を理論的に支える。特に supervised hypothesis generation の related work に有用。 | Abstract、Method、Experiments citeturn34search0 |
| Useful | Movva et al., 2025, *Sparse Autoencoders for Hypothesis Generation* | ICML 2025, PMLR 267 citeturn8search16turn8search8 | 埋め込み上の sparse autoencoder で feature discovery を行い、予測的特徴のみ選別し、LLM で自然言語解釈を付与する。LLM-only 方式より 1–2 桁低コスト。 citeturn8search16 | SemAxis の unsupervised mode に対する**強い代替案**。NLI を使わずに概念発見する比較軸として有力。 | Abstract、Sec. 2–4、synthetic + real data experiments citeturn8search16 |
| Useful | Tavares, 2025, *LogiPart: Local Large Language Models for Data Exploration at Scale with Logical Partitioning* | arXiv:2509.22211, current v3 2026-02-17 citeturn36search0turn36search2turn36search4 | 自然言語の logical partitions を用いて interpretability を保った thematic tree を構築し、現行版では **NLI-based assignment** と local LLM による O(1) generative complexity を採用。 citeturn36search0turn36search2 | **LLM-generated question/hypothesis + NLI assignment** という構成要素の競合。SemAxis は tree ではなく flat matrix だが、README で触れないのは危険。 | Abstract、comments、architecture update note citeturn36search0turn36search2 |

**NLI による zero-shot classification と entailment scoring**

| 優先度 | 正式引用 | 一次URL/DOI/arXiv | 核となる手法/知見 | SemAxisへの具体的関連 | 参照すべき箇所 |
|---|---|---|---|---|---|
| Must read | Yin, Hay, Roth, 2019, *Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach* | EMNLP-IJCNLP 2019, DOI 10.18653/v1/D19-1404 citeturn39search0 | ラベルを hypothesis 文に変換し、NLI モデルで zero-shot classification する古典。複数データセットによる比較を提示。 citeturn39search0 | SemAxis の **NLI hypothesis scoring** の最重要出発点。 | Entire paper、especially entailment formulation citeturn39search0 |
| Must read | Ma et al., 2021, *Issues with Entailment-based Zero-shot Text Classification* | ACL 2021 Short, DOI 10.18653/v1/2021.acl-short.99 citeturn18search0 | entailment-based ZS classification の見落とされがちな問題点を整理した意見論文。仮説テンプレートやラベル記述の扱いへの警鐘。 citeturn18search0 | SemAxis でも **hypothesis wording sensitivity** と **label/hypothesis formulation bias** の確認が必要。 | Abstract、discussion sections citeturn18search0 |
| Must read | Gera et al., 2022, *Zero-Shot Text Classification with Self-Training* | EMNLP 2022 / arXiv:2210.17541 citeturn17search3turn17search6 | zero-shot NLI classifier を confident pseudo-label で self-train すると安定性・性能が改善することを示す。 citeturn17search3turn17search6 | SemAxis でも、生成済み仮説の feature matrix に対する **train-fold 内 adaptation** を検討する余地がある。 | Abstract、Method、Results citeturn17search3turn17search6 |
| Useful | Pàmies et al., 2023, *A weakly supervised textual entailment approach to zero-shot text classification* | EACL 2023 citeturn17search2 | weak supervision で textual entailment ベース分類器を構築。ゼロショット分類のための entailment formulation を補強。 citeturn17search2 | SemAxis の supervised mode で **ラベル説明 + hypothesis bank** の作り方を考える際の関連文献。 | Method、Experiments citeturn17search2 |
| Useful | Laurer et al., 2023, *Building Efficient Universal Classifiers with Natural Language Inference* | arXiv:2312.17543 citeturn17search0turn17search11 | NLI を universal classification task として整理し、33 datasets・389 classes で訓練した universal classifier を提示。効率面でも generative LLM より有利と論じる。 citeturn17search0 | SemAxis の NLI backbone の位置づけに有用。特に **direct zero-shot NLI baseline** を置く理由になる。 | Abstract、guide sections、training data summary citeturn17search0 |

**Concept bottleneck / concept-based interpretable NLP**

| 優先度 | 正式引用 | 一次URL/DOI/arXiv | 核となる手法/知見 | SemAxisへの具体的関連 | 参照すべき箇所 |
|---|---|---|---|---|---|
| Must read | Koh et al., 2020, *Concept Bottleneck Models* | ICML 2020, PMLR 119 / arXiv:2007.04612 citeturn27search0turn27search8 | 入力から高水準 concept を予測し、その concept を通して最終予測を行う CBM を提案。介入可能性と解釈性を重視。 citeturn27search0 | SemAxis は厳密な CBM ではないが、**自然言語特徴を bottleneck 的に使う**点で最上位の概念的参照先。 | Sec. 1–3、intervention results citeturn27search0 |
| Useful | Oikarinen et al., 2023, *Label-free Concept Bottleneck Models* | ICLR 2023, OpenReview citeturn27search1 | labeled concept data なしに拡張可能な CBM を構築する枠組み。自動化・スケーラビリティを重視。 citeturn27search1 | SemAxis の **concept/hypothesis を人手で作らない**立場を支える文献。 | Abstract、scalability claims citeturn27search1 |
| Useful | Yuksekgonul, Wang, Zou, 2023, *Post-hoc Concept Bottleneck Models* | ICLR 2023, OpenReview citeturn27search2turn27search6 | 既存モデルを post-hoc に concept bottleneck 化し、性能をなるべく落とさず解釈性を得る。 citeturn27search2 | SemAxis の「featureizer と classifier を分離」する発想に近い。 | Method、editing/intervention sections citeturn27search2turn27search6 |
| Useful | Tan et al., 2024, *Interpreting Pretrained Language Models via Concept Bottlenecks* | PAKDD 2024, DOI 10.1007/978-981-97-2259-4_5; arXiv:2311.05014 citeturn16search10turn16search0 | PLM 内部の semantically meaningful concepts を抽出し、診断や頑健性向上に使う C³M を提案。機械生成概念も使う。 citeturn16search0 | SemAxis の concept bottleneck 系 related work に必須。外在的 featureizer との差分を書く際にも使える。 | Abstract、real-world datasets、robustness discussion citeturn16search0turn16search10 |
| Useful | Bhan et al., 2025, *Towards Achieving Concept Completeness for Textual Concept Bottleneck Models* | Findings of EMNLP 2025 citeturn12search1turn14search17 | 完全な concept basis と reliability / leakage 問題を正面から扱う Textual CBM。human-labeled concepts や LLM annotation を不要にする方向。 citeturn12search1 | SemAxis で **concept completeness と classification leakage** をどう議論するかの直接参照先。 | Abstract、concept completeness / leakage discussion citeturn12search1 |

**Representative / diverse sampling と in-context selection**

| 優先度 | 正式引用 | 一次URL/DOI/arXiv | 核となる手法/知見 | SemAxisへの具体的関連 | 参照すべき箇所 |
|---|---|---|---|---|---|
| Must read | Liu et al., 2022, *What Makes Good In-Context Examples for GPT-3?* | DeeLIO 2022 / arXiv:2101.06804 citeturn22search6turn22search2 | semantically similar retrieval が random より一貫して良い例選択になることを示す。タスク関連 encoder の利点も報告。 citeturn22search2 | SemAxis の **embedding-based representative selection** の基本根拠。 | Retrieval strategy and main results citeturn22search2turn22search6 |
| Must read | Su et al., 2022, *Selective Annotation Makes Language Models Better Few-Shot Learners* | arXiv:2209.01975 citeturn23search3turn25search11 | annotation-efficient ICL のための unsupervised graph-based selective annotation、**vote-k** を提案。多様性と代表性の両立を狙う。 citeturn23search3 | SemAxis の Vote-K 実装の**最重要根拠文献**。 | Abstract、selection algorithm、10-dataset experiments citeturn23search3turn25search11 |
| Useful | Zhang et al., 2023, *IDEAL: Influence-Driven Selective Annotations Empower In-Context Learners in Low-Resource Data Regimes* | OpenReview / arXiv:2310.10873 citeturn22search4turn22search8 | Vote-k など既存法の限界として、代表性不足・外れ値選択・理論保証欠如を指摘し、influence-driven selection を提案。 citeturn22search8 | SemAxis に対しては **「Vote-K を default のまま固定しない」** 根拠になる。 | Motivation、comparison vs Vote-k citeturn22search4turn22search8 |
| Useful | Mavromatis et al., 2024, *CoverICL: Selective Annotation for In-Context Learning via Active Graph Coverage* | EMNLP 2024 citeturn23search15turn25search2turn25search8 | semantic similarity graph と uncertainty を組み合わせ、few-shot ICL の低予算 active selection を改善。 citeturn25search2 | SemAxis の supervised hypothesis generation で、**代表性だけでなく判別困難例も見せる**設計の候補。 | Abstract、algorithm overview citeturn25search2turn25search8 |
| Useful | Axiotis et al., 2024, *Data-Efficient Learning via Clustering-Based Sensitivity Sampling: Foundation Models and Beyond* | ICML 2024, PMLR 235 citeturn24search0 | small representative subset の理論と実証を与える data selection 研究。クラスタリング 기반の代表 subset 選択に理論的な足場を与える。 citeturn24search0 | K-Means centroid-nearest のような cheap clustering baseline を、**単なる工夫ではなく data selection 文脈**に置ける。 | Abstract、theory + data selection results citeturn24search0 |

**評価・faithfulness・stability・leakage・benchmarking**

| 優先度 | 正式引用 | 一次URL/DOI/arXiv | 核となる手法/知見 | SemAxisへの具体的関連 | 参照すべき箇所 |
|---|---|---|---|---|---|
| Must read | Li et al., 2024, *Evaluating Readability and Faithfulness of Concept-based Explanations* | EMNLP 2024 citeturn28search0turn28search3turn28search10 | concept-based explanations の readability と faithfulness を評価する体系を提案し、自動指標と human evaluation の対応も調べる。 citeturn28search0turn28search10 | SemAxis の自然言語特徴を **predictive feature** であるだけでなく **読める説明** として測る際の中核文献。 | Abstract、automatic metrics、human study sections citeturn28search0turn28search10 |
| Must read | Poché et al., 2025, *ConSim: Measuring Concept-Based Explanations’ Effectiveness with Automated Simulatability* | ACL 2025, DOI 10.18653/v1/2025.acl-long.279 citeturn29search0turn29search1 | explanation を見た利用者がモデル出力を予測できるか、という **simulatability** を LLM user-simulator で近似評価する。 citeturn29search0 | SemAxis の human-readable feature set の **human utility** を測るのに最も実装しやすい近年文献。 | Abstract、evaluation framework、ranking analysis citeturn29search0turn29search1 |
| Must read | DeYoung et al., 2020, *ERASER: A Benchmark to Evaluate Rationalized NLP Models* | ACL 2020, DOI 10.18653/v1/2020.acl-main.408 citeturn28search2turn28search5 | rationale alignment だけでなく、comprehensiveness / sufficiency を含む faithfulness-oriented benchmark と metrics を導入。 citeturn28search5 | SemAxis の feature-level explanations を **faithfulness** 観点で測る際の土台。 | Abstract、metrics sections citeturn28search2turn28search5 |
| Must read | Gururangan et al., 2018, *Annotation Artifacts in Natural Language Inference Data* | NAACL 2018 Short citeturn31search0 | SNLI/MultiNLI に hypothesis-only artifacts があることを示す代表論文。NLI モデル性能の過大評価リスクを指摘。 citeturn31search0 | SemAxis が NLI backbone を使う以上、**NLI 自体の shortcut** が feature に混入しうる。 | Abstract、artifact analysis citeturn31search0 |
| Must read | Cawley, Talbot, 2010, *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation* | JMLR 11(70):2079–2107 citeturn37search7turn37search0 | model selection を評価データに漏らすと selection bias が生じることを体系的に論じる。 citeturn37search7 | SemAxis の **fold ごとに hypothesis を train text だけから生成する設計**を正当化する最重要の一般論文。 | Entire paper、especially model selection bias discussion citeturn37search7turn37search0 |

## 競合・新規性リスク分析

SemAxis のアイデアを要素分解すると、**個別要素の多くは既出**であり、学術的ポジショニングは「first ever」ではなく、**明示的な自然言語仮説を中間表現とする sklearn-compatible, fold-safe, NLI-based featureizer** に置くのが最も安全です。特に、FELIX・CHiLL・TBM・Balek の 4 本は、README だけで軽く流すより、**設計比較実験の候補**として扱うべきです。citeturn10view0turn11search0turn14search1turn40search0

| 要素 | 既に示されていること | 組合せとして既出か | まだ検証余地があること | リスク評価 | README / 実験への示唆 |
|---|---|---|---|---|---|
| LLM が human-readable features / concepts / hypotheses をデータから生成する | FELIX, Balek, TBM, HypotheSAEs, CHiLL が既に示している。 citeturn10view0turn40search0turn14search1turn8search16turn11search0 | **はい。** | SemAxis では **宣言文仮説** に統一し、NLI で汎用スコア化する利点がどこまで出るか。 | **高** | 「自然言語特徴の自動発見」は新規性の中心にしない。 |
| テキスト×自然言語仮説を NLI で scored feature にする | Yin 2019, Laurer 2023, LogiPart 現行版で明確。 citeturn39search0turn17search0turn36search0 | **はい。** | flat feature matrix として多数仮説へ拡張したときの有効性・安定性。 | **高** | zero-shot NLI baseline と score definition ablation が必須。 |
| 自然言語特徴行列を線形モデルに渡し、係数で解釈する | CHiLL, TBM, FELIX, Aug-imodels が非常に近い。 citeturn11search0turn14search1turn10view0turn32search0 | **ほぼ既出。** | sklearn Transformer として他の分類器に差し替え可能な実用性。 | **高** | README では「package/engineering contribution」も明示する。 |
| 自動概念発見を human-labeled concept set なしで行う | Label-free CBM, TBM, CT-CBM がある。 citeturn27search1turn14search1turn12search1 | **はい。** | unsupervised / supervised で hypothesis bank の質がどう変わるか。 | **中〜高** | supervised/unsupervised 両方の比較を主実験に入れる。 |
| fold ごとに学習テキストだけで仮説生成する leakage-safe CV | 一般原理として必須だが、SemAxis のように Transformer API に埋め込んだ OSS は相対的に少ない。 citeturn37search7turn26search2 | **実装としては相対的に新しい。** | 再現性・便利さ・誤用防止の効果。 | **中** | この点は論文より OSS 価値として強い。 |
| token budget 下で代表性・多様性・判別性のあるサンプルを LLM に見せる | ICL literature で豊富。 vote-k, CoverICL, IDEAL, retrieval-based selection など。 citeturn23search3turn25search2turn22search8turn22search2 | **部品としては既出。** | hypothesis generation という用途で何が最適かは未整理。 | **中** | ここは SemAxis が比較実験で貢献しやすい。 |
| OvR/OvO multiclass の仮説生成＋NLI featureization | 零細な構成要素はあるが、SemAxis のような sklearn 互換実装はあまり見当たらない。 citeturn39search0turn14search1 | **明示的には弱い。** | OvR/OvO どちらが redundancy や interpretability に有利か。 | **中** | ここは論文の独自 ablation にしやすい。 |

直接競合としては、**FELIX と TBM は README で触れるだけでは足りず、設計比較実験が必要**です。CHiLL は医療ドメイン寄りですが、「自然言語特徴を線形モデルに入れる」設計の先行として比較枠に入れる価値があります。Balek と LogiPart は設計比較まで必須とまでは言いませんが、少なくとも README の Related Work では明確に触れるべきです。citeturn10view0turn11search0turn14search1turn40search0turn36search0

## 実験計画への落とし込み

SemAxis の実験は、**性能**, **解釈可能性**, **頑健性**, **コスト**, **leakage 安全性**を同時に比較できるように設計するのがよいです。zero-shot NLI と concept bottleneck 系の文献は、単純 accuracy だけでは手法の価値を捕まえにくいことを示しており、近年の concept-explanation 評価文献は readability / faithfulness / simulatability を外部指標として導入しています。citeturn39search0turn14search1turn28search0turn29search0turn28search2

| 項目 | 推奨内容 | 根拠文献 |
|---|---|---|
| 推奨 datasets | **AG News / DBPedia / 20 Newsgroups** で topic classification、**SST-2** で sentiment、**GoEmotions** で fine-grained emotion、**HateXplain** で explainability-aware hate speech detection を推奨。topic・sentiment・emotion・toxicity/abuse で仮説生成の性質がかなり異なるため、少なくとも 4 タスク群は欲しい。GoEmotions と HateXplain は人間可読な概念の有用性や合理性を見やすい。 citeturn39search0turn38search1turn38search0turn37search16turn38search2 | Yin 2019 は多様な ZS classification benchmark を整理。GLUE は SST-2 を含む標準基盤。GoEmotions と HateXplain は fine-grained / explainable な評価軸を提供。 citeturn39search0turn38search0turn37search16turn38search2 |
| 比較 baselines | **TF-IDF + Logistic Regression / Linear SVM**、**sentence embeddings + linear classifier**、**direct zero-shot NLI classifier**、**direct LLM classification**、可能なら **FELIX / TBM / CHiLL-like**、加えて **random hypothesis bank** を置く。 citeturn10view0turn14search1turn11search0turn39search0turn20search0 | FELIX と Balek は TF-IDF・embedding・LLM baselines と比較。TBM は few-shot GPT-4 / finetuned DeBERTa と比較。zero-shot direct LLM baseline は Gretz et al. が系統比較している。 citeturn10view0turn40search3turn14search1turn20search0 |
| 主要 ablations | **supervised vs unsupervised**, **OvR vs OvO**, **NLI backbone**（DeBERTa-v3-large vs 代替 NLI）、**LLM generator**, **feature count**, **token budget**, **subset selection**（random / centroid-nearest / Vote-K / facility-location 系 / CoverICL 風）、**score definition**（P(entail), entail-contradict margin, 3-way variants）を切る。 citeturn39search0turn18search0turn23search3turn25search2turn24search0 | NLI 文献は formulation sensitivity を示し、sample selection 文献は few-shot performance が selection に強く依存することを示している。 citeturn18search0turn22search2turn23search3turn25search2 |
| classification metrics | **accuracy, macro-F1, AUROC / AUPRC** を中心に。クラス不均衡があるタスクでは macro-F1 を主指標にし、NLI score の calibration を見るため **ECE/Brier** も補助指標に入れるのが望ましい。後者は本調査で直接の text-featureization 文献には少ないが、NLI / inference calibration 文脈では自然である。 citeturn17search3turn21search4 | Gera 2022 は安定性・性能向上、calibration 文献は inference settings の確率品質を問題化。 citeturn17search3turn21search4 |
| interpretability metrics | **readability** は Li et al. 型の自動指標 + 小規模 human rating、**faithfulness** は ERASER 的な sufficiency/comprehensiveness、**human utility** は ConSim 的な simulatability、**stability** は seed/fold 間の hypothesis overlap・係数順位相関・説明安定度を測る。 citeturn28search0turn29search0turn28search2turn26search14 | 近年の concept-based explanation 評価は readability, faithfulness, simulatability を分けて測る。 citeturn28search0turn29search0 |
| diversity / redundancy metrics | 仮説 embeddings による **pairwise cosine**, **average max similarity**, **unique coverage**, **effective feature count** を測る。TBM は redundancy / leakage 問題を報告しているため、SemAxis でも feature bank 監査が必要。 citeturn15search8turn24search0turn25search0 | TBM の人手評価と selection literature が redundancy/coverage の重要性を示す。 citeturn15search8turn25search0 |
| cost / latency metrics | **LLM prompt token 数**, **選択サンプル数**, **生成仮説数**, **NLI pair 数**, **wall-clock**, **GPU/CPU 使用量** を報告。NLI cross-encoder は `n_texts × n_features` に線形ではなく直積で効くため、feature bank size の cost 曲線は必須。 citeturn36search0turn17search0 | LogiPart は NLI-based assignment の計算設計を前面化し、Laurer は encoder-side universal classifier の効率を強調。 citeturn36search0turn17search0 |
| leakage を避けた CV protocol | **各 fold の学習テキストのみ**から sample selection・hypothesis generation・dedup・feature pruning・必要なら threshold tuning を行う。test fold は hypothesis bank 決定に一切使わない。prompt 文面も fixed し、seed と LLM version をログ化する。 citeturn37search7turn26search2 | Cawley & Talbot の selection bias 論文が最も強い一般根拠。feature engineering leakage も近年明示的に問題化されている。 citeturn37search7turn26search2 |
| leakage / artifact audit | **label 名そのもの**やデータセット shortcut をなぞった hypothesis を検出する lexical audit を追加。さらに、DISCERN や spurious correlation work を参考に、誤分類クラスタを自然言語で分析し、artifact-dependent hypotheses を除去する。 citeturn30search1turn30search6turn31search0 | generated features は dataset artifacts を拾いやすく、NLI backbone 自体も annotation artifacts をもつ可能性がある。 citeturn31search0turn30search6 |

特に重要なのは、**feature generation までを cross-validation の外で一度だけやってはいけない**という点です。SemAxis はここを設計上かなり正しく押さえていますが、論文にはその理由を一般論として書いておくべきです。加えて、TBM が報告した **redundancy / leakage**、Gururangan が示した **NLI artifacts** を踏まえると、生成特徴の監査は「付録」ではなく主実験に入れる方が説得力があります。citeturn37search7turn15search8turn31search0

## 実装に直結する提案

現在の **K-Means centroid-nearest** は、「代表性の高いサンプルを cheap に取る」実装として妥当です。ICL 研究では semantic similarity / retrieval が random を上回ることが繰り返し報告されており、data selection 理論でも clustering-based representative subset は十分に支持されています。そのため、K-Means を残す判断は文献的に十分擁護できます。citeturn22search2turn24search0

一方で **Vote-K** は、低予算下で代表性と多様性の両立を狙う強いベースラインであり、SemAxis に入っていること自体は良い判断です。ただし後続研究は、Vote-K が**多様性を重く見すぎて outlier を拾いやすい**こと、**理論保証が弱い**ことを指摘しています。したがって、Vote-K を **唯一の推奨 default** にするより、facility-location 系や uncertainty-aware graph coverage 系を並べる方が研究的にも実装的にも自然です。citeturn23search3turn25search11turn22search8turn25search2

| 追加候補 | 実装コスト | 期待効果 | 文献上の根拠 | 優先順位 |
|---|---|---|---|---|
| **facility-location / submodular selection** を embedding 上で実装 | 中 | 代表性と多様性を理論的に両立しやすい。Vote-K より安定で説明しやすい可能性。 | Submodularity / coverage の古典と、data-efficient subset selection の近年研究。 citeturn25search0turn24search0 | **最優先** |
| **Vote-K を baseline に格下げし、default は centroid + diversity hybrid** にする | 低 | 既存 API を大きく壊さず、外れ値感度を下げやすい。 | Vote-K の有効性と限界の両方が文献で出ている。 citeturn23search3turn22search8 | **高** |
| **CoverICL 風の uncertainty-aware graph coverage** | 中〜高 | supervised mode で class-discriminative な例を LLM に見せやすい。 | CoverICL は similarity graph に uncertainty を乗せて改善。 citeturn25search2turn25search8 | **高** |
| **feature pruning** として hypothesis embedding clustering + semantic dedup | 低 | redundancy を大きく減らし、NLI 計算量を直接削減。 | FELIX は redundancy filtering を組み込み、TBM は redundancy 問題を報告。 citeturn10view0turn15search8 | **高** |
| **label-name overlap / artifact overlap filter** | 低 | label leakage と dataset artifacts の露骨な混入を減らす。 | spurious correlation / DISCERN / NLI artifacts 文献。 citeturn30search1turn30search6turn31search0 | **高** |
| **train-fold 内検証による feature validation** | 中 | 「生成はされたが支持されない feature」を落とせる。 | Li 2024, ConSim, ERASER が “使える説明” と “faithful な説明” を分けて評価。 citeturn28search0turn29search0turn28search2 | **中** |
| **HypotheSAEs 風の unsupervised concept discovery を将来比較対象として実装** | 高 | NLI 不要で低コストな unsupervised alternative を持てる。 | HypotheSAEs は LLM-heavy methods より 1–2 桁低計算量を報告。 citeturn8search16 | **中** |

README の Related Work にまず追加すべき文献は、以下の 10 件です。各文献は「似ているから入れる」のではなく、**SemAxis の claim を弱めうるか、構成要素の正当化に直結するか**で選んでいます。  
- **FELIX (Malberg et al., 2024)** — LLM-generated interpretable features を downstream classifier に渡す最重要近接研究。 citeturn10view0  
- **CHiLL (McInerney et al., 2023)** — 自然言語特徴と線形モデルの実践例で、SemAxis の “feature language” 側を支える。 citeturn11search0  
- **TBM (Ludan et al., 2023)** — 自動概念発見と概念値測定による text bottleneck の近接方式。 citeturn14search1  
- **Balek et al. (2024)** — README 既掲載候補だが、書誌修正の上で継続採用すべき近接研究。 citeturn40search0  
- **Yin et al. (2019)** — NLI-based text-hypothesis scoring の原点。 citeturn39search0  
- **Ma et al. (2021)** — entailment-based classification の落とし穴を整理しており、SemAxis の注意事項に直結。 citeturn18search0  
- **Laurer et al. (2023)** — NLI を universal classifier として位置づける最近の実用的整理。 citeturn17search0  
- **LogiPart (Tavares, 2025)** — LLM-generated natural-language partition + NLI assignment の近接 preprint。 citeturn36search0turn36search2  
- **HypotheSAEs (Movva et al., 2025)** — 仮説生成を NLI 以外の概念発見ルートで行う有力代替。 citeturn8search16  
- **Li et al. (2024) / ConSim (Poché et al., 2025)** — README にも「評価軸」として入れると、accuracy 以外の価値を説明しやすい。 citeturn28search0turn29search0

## 参考文献と読む順番

以下は、本報告で主要に用いた文献の一覧です。`識別子` 欄には DOI / arXiv / Anthology ID など、検証可能な一次識別子のみを記しています。生の URL は省き、各行末の引用リンクを一次ソースへの導線として使っています。

| 参考文献 | 識別子 |
|---|---|
| Balek, Sýkora, Sklenák, Kliegr. 2024. *LLM-based feature generation from text for interpretable machine learning*. | arXiv:2409.07132 citeturn40search0turn40search4 |
| Malberg, Mosca, Groh. 2024. *FELIX: Automatic and Interpretable Feature Engineering Using LLMs*. ECML PKDD. | DOI 10.1007/978-3-031-70359-1_14 citeturn10view0 |
| McInerney, Young, van de Meent, Wallace. 2023. *CHiLL: Zero-shot Custom Interpretable Feature Extraction from Clinical Notes with Large Language Models*. Findings of EMNLP. | ACL Anthology 2023.findings-emnlp.568 / arXiv:2302.12343 citeturn11search0turn11search1 |
| Ludan, Lyu, Yang, Dugan, Yatskar, Callison-Burch. 2023. *Interpretable-by-Design Text Understanding with Iteratively Generated Concept Bottleneck*. | arXiv:2310.19660 citeturn14search1 |
| Zhou, Liu, Srivastava, Mei, Tan. 2024. *Hypothesis Generation with Large Language Models*. NLP4Science. | DOI 10.18653/v1/2024.nlp4science-1.10 citeturn34search0 |
| Movva, Peng, Garg, Kleinberg, Pierson. 2025. *Sparse Autoencoders for Hypothesis Generation*. ICML. | PMLR 267 / OpenReview 4R0pugRyN5 citeturn8search16turn8search3 |
| Tavares. 2025. *LogiPart: Local Large Language Models for Data Exploration at Scale with Logical Partitioning*. | arXiv:2509.22211 citeturn36search0turn36search2 |
| Yin, Hay, Roth. 2019. *Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach*. EMNLP-IJCNLP. | DOI 10.18653/v1/D19-1404 citeturn39search0 |
| Ma, Yao, Lin, Zhao. 2021. *Issues with Entailment-based Zero-shot Text Classification*. ACL Short. | DOI 10.18653/v1/2021.acl-short.99 citeturn18search0 |
| Gera et al. 2022. *Zero-Shot Text Classification with Self-Training*. EMNLP. | arXiv:2210.17541 / ACL Anthology 2022.emnlp-main.73 citeturn17search3turn17search6 |
| Pàmies et al. 2023. *A weakly supervised textual entailment approach to zero-shot text classification*. EACL. | ACL Anthology 2023.eacl-main.22 citeturn17search2 |
| Laurer, van Atteveldt, Casas, Welbers. 2023. *Building Efficient Universal Classifiers with Natural Language Inference*. | arXiv:2312.17543 citeturn17search0 |
| Koh et al. 2020. *Concept Bottleneck Models*. ICML. | PMLR 119 / arXiv:2007.04612 citeturn27search0turn27search8 |
| Oikarinen et al. 2023. *Label-free Concept Bottleneck Models*. ICLR. | OpenReview FlCg47MNvBA citeturn27search1 |
| Yuksekgonul, Wang, Zou. 2023. *Post-hoc Concept Bottleneck Models*. ICLR. | OpenReview nA5AZ8CEyow citeturn27search2 |
| Tan et al. 2024. *Interpreting Pretrained Language Models via Concept Bottlenecks*. PAKDD. | DOI 10.1007/978-981-97-2259-4_5 / arXiv:2311.05014 citeturn16search10turn16search0 |
| Bhan et al. 2025. *Towards Achieving Concept Completeness for Textual Concept Bottleneck Models*. Findings of EMNLP. | ACL Anthology 2025.findings-emnlp.106 citeturn12search1 |
| Liu et al. 2022. *What Makes Good In-Context Examples for GPT-3?* DeeLIO. | ACL Anthology 2022.deelio-1.10 / arXiv:2101.06804 citeturn22search6turn22search2 |
| Su et al. 2022. *Selective Annotation Makes Language Models Better Few-Shot Learners*. | arXiv:2209.01975 citeturn23search3turn25search11 |
| Zhang et al. 2023. *IDEAL: Influence-Driven Selective Annotations Empower In-Context Learners in Low-Resource Data Regimes*. | OpenReview Spp2i1hKwV / arXiv:2310.10873 citeturn22search4turn22search8 |
| Mavromatis et al. 2024. *CoverICL: Selective Annotation for In-Context Learning via Active Graph Coverage*. EMNLP. | ACL Anthology 2024.emnlp-main.1185 citeturn23search15turn25search2 |
| Axiotis et al. 2024. *Data-Efficient Learning via Clustering-Based Sensitivity Sampling: Foundation Models and Beyond*. ICML. | PMLR 235 citeturn24search0 |
| Wei, Iyer, Bilmes. 2015. *Submodularity in Data Subset Selection and Active Learning*. | PMLR v37 supplement / UW tech report citeturn25search0 |
| Li et al. 2024. *Evaluating Readability and Faithfulness of Concept-based Explanations*. EMNLP. | ACL Anthology 2024.emnlp-main.36 citeturn28search0 |
| Poché et al. 2025. *ConSim: Measuring Concept-Based Explanations’ Effectiveness with Automated Simulatability*. ACL. | DOI 10.18653/v1/2025.acl-long.279 citeturn29search0 |
| DeYoung et al. 2020. *ERASER: A Benchmark to Evaluate Rationalized NLP Models*. ACL. | DOI 10.18653/v1/2020.acl-main.408 citeturn28search2 |
| Lyu et al. 2024. *Towards Faithful Model Explanation in NLP: A Survey*. Computational Linguistics. | MIT Press / survey article citeturn26search11 |
| Gururangan et al. 2018. *Annotation Artifacts in Natural Language Inference Data*. NAACL Short. | ACL Anthology N18-2017 citeturn31search0 |
| Wang et al. 2022. *Identifying and Mitigating Spurious Correlations for Improving Robustness in NLP Models*. Findings of NAACL. | ACL Anthology 2022.findings-naacl.130 citeturn30search6 |
| Menon, Srivastava. 2024. *DISCERN: Decoding Systematic Errors in Natural Language for Text Classifiers*. EMNLP. | ACL Anthology 2024.emnlp-main.1091 citeturn30search1 |
| Cawley, Talbot. 2010. *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*. JMLR. | JMLR 11(70):2079–2107 citeturn37search7 |
| Fel et al. 2022. *How Good Is Your Explanation? Algorithmic Stability Measures To Assess Explanations*. WACV. | CVF Open Access WACV 2022 citeturn26search14 |
| Gretz et al. 2023. *Zero-shot Topical Text Classification with LLMs – an Experimental Study*. Findings of EMNLP. | ACL Anthology 2023.findings-emnlp.647 citeturn20search0 |
| Wang et al. 2018. *GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding*. | DOI 10.18653/v1/W18-5446 citeturn38search0 |
| Demszky et al. 2020. *GoEmotions: A Dataset of Fine-Grained Emotions*. ACL. | ACL Anthology 2020.acl-main.372 citeturn37search16 |
| Mathew et al. 2021. *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection*. AAAI. | DOI 10.1609/aaai.v35i17.17745 citeturn38search2turn38search8 |
| Zhang, Zhao, LeCun. 2015. *Character-level Convolutional Networks for Text Classification*. NeurIPS. | NeurIPS 2015 paper 5782 citeturn38search1 |

**読む順番**  
- **FELIX** — SemAxis の最も危ない近接競合を最初に把握するため。 citeturn10view0  
- **TBM** — text concept bottleneck と自動概念発見の最重要比較対象を押さえるため。 citeturn14search1  
- **CHiLL** — 自然言語特徴 × 線形モデルの既存成功例を見るため。 citeturn11search0  
- **Balek et al.** — LLM-generated interpretable text features の近年実装例を確認するため。 citeturn40search0  
- **Yin et al.** — NLI featureization の原型を押さえるため。 citeturn39search0  
- **Ma et al.** — entailment-based classification の落とし穴を先に知るため。 citeturn18search0  
- **Su et al.** — Vote-K の原典を読むため。 citeturn23search3  
- **CoverICL または IDEAL** — Vote-K の改善方向を知るため。 citeturn25search2turn22search8  
- **Li et al. 2024** — readability / faithfulness の評価設計を作るため。 citeturn28search0  
- **ConSim** — human utility / simulatability を実験に入れるため。 citeturn29search0

本調査の未解決点は 2 つだけです。**Balek 論文の誌面版の最終書誌**は、著者告知からは存在が示唆されるものの、本調査では出版社一次ページを直に確認していないため、本報告では **arXiv 版を確実情報**として扱いました。**LogiPart** は arXiv 改稿の過程で旧題・別題表記が混在しているため、README では **現行 PDF の題名 + arXiv ID + preprint 明記**が最も安全です。citeturn40search6turn36search0turn36search2