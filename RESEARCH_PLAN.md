# NeurIPS Workshop研究計画書
## B: CentralDogma-ΔCC / C: DeltaEvo

作成日: 2026-08-02

---

## 0. 全体方針

本計画は、Evo 2のゲノム表現とタンパク質言語モデルの表現を解釈可能な潜在変数で接続する研究Bと、疾患データによるEvo 2のfine-tuning前後を比較して、獲得された病原性表現を同定する研究Cからなる。

- **研究B:** DNAとタンパク質という異なる入力空間に対し、同一変異が引き起こすactivation差分を、共有latent・DNA固有latent・protein固有latentへ分解する。
- **研究C:** base Evo 2と疾患fine-tuned Evo 2のmatched activationをDelta-Crosscoderで比較し、fine-tuningで導入・増強されたlatentを同定し、因果介入で検証する。
- **統合研究:** Cで得られた疾患latentが、Bで得られたDNA–protein共有機序へ接続するかを検証する。

最初から全遺伝子・全変異を扱わず、BRCA1等の機能実験データが豊富な1遺伝子でend-to-end pipelineを完成させ、その後に複数遺伝子と心筋症遺伝子へ拡張する。

---

# 研究B: CentralDogma-ΔCC
## Variant-induced activation deltasを用いたGenome–Protein Crosscoder

## 1. 仮タイトル

**CentralDogma-ΔCC: Disentangling Shared and Modality-Specific Variant Mechanisms across Genome and Protein Language Models**

日本語題名:

**ゲノム・タンパク質言語モデル間の共有変異機序を抽出する差分Crosscoder**

## 2. 背景と問題設定

Evo 2はDNA配列を一塩基解像度で処理し、変異効果予測や中間層embeddingの抽出が可能である。一方、ESM-2等のタンパク質言語モデルは、アミノ酸配列から構造・機能・変異効果に関係する表現を学習する。

しかし、両モデルが同じmissense variantを評価するとき、以下は分かっていない。

1. 両モデルが共通の分子機序を表現しているか。
2. DNAモデルだけが捉える機序と、タンパク質モデルだけが捉える機序を分離できるか。
3. 共通latentがDMS等の実測機能値を説明するか。
4. 両モデルの不一致がVUSや特殊な病原性機序の検出に使えるか。

生のactivationは、hidden dimension、tokenization、入力長、学習目的が大きく異なるため、そのままCCA等で揃えると、遺伝子同一性や保存性などの大域的な情報が支配しやすい。

そこで本研究では、野生型と変異型のactivation差分を使う。

- DNA側: `Δh_DNA = h_DNA(mutant) - h_DNA(wildtype)`
- protein側: `Δh_PROT = h_PROT(mutant) - h_PROT(wildtype)`

同一変異が両モデル内部に起こした変化を比較することで、モデル固有の背景表現ではなく、変異機序に焦点を当てる。

## 3. 中心仮説

### H1: 共有latent仮説

DNAモデルとproteinモデルには、タンパク質構造破壊、保存残基破壊、binding site disruption等に対応する共有latentが存在する。

### H2: modality-private仮説

DNA固有latentはsplice、codon、RNA安定性、exon context等を、protein固有latentはfolding、active site、interface、disorder等を強く表現する。

### H3: 実測機能予測仮説

共有latentは、単独モデルのactivation、embedding concatenation、CCAよりも、gene-disjointなDMS predictionで高い説明力を持つ。

### H4: disagreement仮説

DNA–protein latentの不一致は、モデルの不確実性またはモダリティ固有の病原性機序を示し、誤分類・VUS・splice関連変異の優先順位付けに有用である。

## 4. 新規性

1. Crosscoderをbase-vs-finetuneではなく、DNA LMとprotein LMという異種生物モデル間に適用する。
2. 絶対activationではなく、同一変異によるpaired activation deltaを学習対象にする。
3. 共有成分と各モダリティ固有成分を明示的に分離する。
4. 解釈可能性だけでなく、実測DMS、cross-modal retrieval、causal ablationで評価する。
5. 中央教義の異なる階層を、操作可能な疎なlatentで接続する。

## 5. 使用モデル

### DNAモデル

第一選択:

- `evo2_7b` または `evo2_7b_base`

理由:

- 7BはBF16で実行でき、Hopper必須のFP8を避けられる。
- 公式実装が中間層embeddingの抽出を提供する。
- BRCA1 zero-shot scoring notebookがあり、最初の再現実験に向く。

### タンパク質モデル

第一選択:

- ESM-2 650M

理由:

- activation extractionが容易。
- InterPLM等のSAE先行研究との比較がしやすい。
- ESM-2 3Bより探索速度が高く、最初のlayer sweepに向く。

追加実験:

- ESM-2 3B
- ESM-Cの利用条件と実装が整う場合はrobustness check

## 6. データセット

### 6.1 MVPデータ

最初は以下の条件を満たす1遺伝子を選ぶ。

- ヒトcoding variant
- 十分なDMSまたはsaturation mutagenesisデータ
- GRCh38 transcript mappingが可能
- ClinVarラベルとの一部重複
- protein domain annotationが取得可能

第一候補:

- BRCA1

第二段階:

- ProteinGymからヒト由来assayを自動抽出し、ゲノム座標へ変換可能な遺伝子を追加
- 心筋症case studyとしてMYH7、LMNA、MYBPC3等を追加。ただしDMSや機能データの有無を事前確認する

### 6.2 ラベル

- DMS continuous fitness score
- ClinVar pathogenic / likely pathogenic / benign / likely benign
- mechanism annotation
  - protein domain
  - secondary structure
  - solvent accessibility
  - disorder
  - active/binding site
  - splice proximity
  - conservation score

### 6.3 split

結果の過大評価を避けるため、以下を明確に分ける。

- variant-random split: デバッグ用
- position-disjoint split: 同一残基の別置換を跨がせない
- gene-disjoint split: 最終主評価
- assay-disjoint split: ProteinGym拡張時の主評価

## 7. 前処理

### 7.1 transcript mapping

- MANE Select transcriptを原則使用
- HGVS protein表記をcoding DNA variantへ変換
- reference alleleをGRCh38 FASTAと照合
- strandを処理
- synonymous、nonsense、frameshiftは別カテゴリとして保持
- mapping失敗例を自動で除外せず、reason codeを保存

### 7.2 DNA入力

MVP:

- variantを中心とする8,192 bp window
- coding strandとreverse-complementの両方向を生成
- WTとmutantで長さを一致させる
- missense SNVを主対象にする

追加:

- 16,384 bp window
- exon boundaryを含むwindow
- indel用の位置alignment

### 7.3 protein入力

- WT protein sequence
- missense mutant sequence
- ESM-2の最大長を超える場合はvariant中心window
- variant residueのtoken positionを保存

## 8. activation extraction

### 8.1 保存対象

各variantについて以下を保存する。

```text
variant_id
model_name
layer_name
modality
wt_activation
mut_activation
delta_activation
pooling_method
gene
position
label
metadata
```

### 8.2 pooling

最低3種類を実装する。

1. exact-position pooling
2. local mean pooling: ±8 tokens/residues
3. attention-weighted local pooling

MVPではexact-positionとlocal meanを使う。

### 8.3 layer selection

最初から全層を保存しない。

- Evo 2: early / middle / lateから3層
- ESM-2: early / middle / lateから3層

1,000–5,000 variantsでlayer probeを行い、DMS predictionとannotation enrichmentが高い層を1–2層選ぶ。

## 9. 提案手法

## 9.1 Shared–Private Variant Delta Crosscoder

各モダリティにshared encoderとprivate encoderを持つ。

```text
Δh_DNA -> E_DNA_shared -> z_DNA_shared
        -> E_DNA_private -> z_DNA_private

Δh_PROT -> E_PROT_shared -> z_PROT_shared
         -> E_PROT_private -> z_PROT_private
```

shared latentはpaired variant間で一致するように学習する。

再構成:

```text
Δh_DNA_hat = D_DNA_shared(z_DNA_shared) + D_DNA_private(z_DNA_private)
Δh_PROT_hat = D_PROT_shared(z_PROT_shared) + D_PROT_private(z_PROT_private)
```

### loss

```text
L = L_reconstruction
  + λ_align * L_shared_alignment
  + λ_retrieval * L_contrastive
  + λ_orth * L_shared_private_orthogonality
  + λ_sparse * L_BatchTopK
  + λ_balance * L_feature_usage_balance
```

#### reconstruction loss

- normalized MSE
- modalityごとにvariance normalization

#### shared alignment loss

- cosine distanceまたはMSE
- paired DNA/proteinのみを近づける

#### contrastive loss

- batch内で同じvariantをpositive
- 他variantをnegative
- gene identityだけでretrievalしないよう、同一gene内negativeを多く含める

#### sparsity

- BatchTopKを第一選択
- latent width: input dimensionの4–16倍を探索
- target L0: 16、32、64を探索

#### private separation

- shared decoderとprivate decoderの直交正則化
- shared/private latent activationのHSICまたはcross-covariance penalty

## 9.2 Dedicated Feature branch

cross-architecture model diffingのDedicated Feature Crosscoderを参考に、以下を明示的に用意する。

- shared features
- DNA-dedicated features
- protein-dedicated features

単にdecoder normでshared/privateを事後分類するのではなく、構造として分ける。

## 10. ベースライン

必須:

1. Evo 2 delta activation単独 + linear probe
2. ESM-2 delta activation単独 + linear probe
3. activation concatenation + MLP
4. PCA + concatenation
5. linear CCA
6. PLS
7. Deep CCA
8. 独立SAE後のlatent alignment
9. shared-private VAE
10. absolute activation Crosscoder
11. delta activation Crosscoder（private branchなし）
12. 提案手法

## 11. 評価

### 11.1 representation quality

- Fraction of Variance Explained
- normalized reconstruction error
- L0 sparsity
- dead feature rate
- feature activation stability across seeds

### 11.2 cross-modal alignment

- DNA-to-protein variant retrieval Recall@1 / Recall@10
- protein-to-DNA retrieval
- 同一gene内retrieval
- mutation-type matched retrieval

### 11.3 biological prediction

- DMS Spearman correlation
- DMS MSE
- ClinVar AUROC / AUPRC
- gene-disjoint performance
- assay-disjoint performance

### 11.4 interpretability

各latentについてtop activating variantsを取得し、以下とのenrichmentを測る。

- domain
- active/binding site
- secondary structure
- disorder
- conservation bin
- splice distance
- mutation class

指標:

- purity
- normalized mutual information
- enrichment odds ratio
- annotation prediction AUPRC

### 11.5 shared/private妥当性

期待される結果:

- shared: protein structure、conserved functional site、domain disruption
- DNA-private: splice proximity、codon context、exon boundary
- protein-private: solvent exposure、binding interface、disorder

### 11.6 causal evaluation

上位latentについてdecoder directionをactivationへ介入する。

- ablation: latent contributionを除去
- injection: WT側directionをmutantへ加える
- dose response: alphaを変化

測定:

- Evo 2 variant log-likelihood deltaの変化
- ESM-2 masked-marginal variant scoreの変化
- intervention specificity
- unrelated positionsのscore変化

## 12. 成功条件

MVP成功条件:

1. gene内holdoutでcross-modal retrievalがrandomを明確に上回る。
2. shared latentがDMS predictionでCCAまたはconcat baselineを上回る。
3. 少なくとも5–10個の再現可能なbiological latentが見つかる。
4. private latentが想定したmodality-specific annotationへ有意にenrichする。
5. 上位shared latentのablationが両モデルのvariant scoreを同方向へ変化させる。

Go/No-Go:

- 共有latentが全く得られない場合、絶対activationではなくdelta、position pooling、同一gene negative samplingを再確認する。
- retrievalは成功するが解釈不能な場合、gene identity leakageを疑う。
- DMS予測が改善しない場合でも、shared/private分解とfailure analysisが明確ならWorkshop論文として成立し得る。

## 13. 論文の主要図

1. CentralDogma-ΔCC全体図
2. shared/private latentのdecoder norm・annotation map
3. cross-modal retrievalとDMS prediction benchmark
4. 代表latentのtop variantsと構造可視化
5. causal ablation / injection
6. DNA–protein disagreementによるエラー・VUS解析

## 14. 想定する主張

- 異種生物基盤モデル間でも、variant-induced deltaを用いることで疎な共有機序を抽出できる。
- 共有latentは単純なembedding fusionよりgene-disjointな変異効果を説明する。
- DNA固有・protein固有latentは異なる生物学的機序を分離する。
- モデル間不一致は、モダリティ固有機序または予測失敗を示す監査信号となる。

---

# 研究C: DeltaEvo
## 疾患fine-tuningでEvo 2が獲得する内部表現のmodel diffing

## 15. 仮タイトル

**DeltaEvo: Causal Model Diffing of Disease Knowledge Acquired by Genomic Language Models**

日本語題名:

**疾患fine-tuningでゲノム言語モデルが獲得する病原性表現の因果的差分解析**

## 16. 背景と問題設定

疾患variantでEvo 2をfine-tuningすると病原性予測性能が向上する可能性がある。しかし、その向上が以下のどれによるかは不明である。

1. 新しい分子機序latentを獲得した。
2. base modelに存在したlatentを再重み付けした。
3. 遺伝子・局所配列・データセット特有のshortcutを学習した。
4. classification headだけが変化し、backbone表現はほぼ変化していない。

本研究は、base Evo 2とdisease-adapted Evo 2のmatched activationをDelta-Crosscoderで比較し、fine-tuningで変化したlatentを抽出する。

## 17. 中心仮説

### H1: 再利用仮説

病原性fine-tuningの主要効果は、完全に新しいlatentの生成ではなく、base modelに存在するconservation、coding integrity、splice、protein-structure関連latentの増幅・再結合である。

### H2: 狭いtask固有latent仮説

narrow fine-tuningでは、変化は少数latentへ集中し、Delta-Crosscoderはstandard Crosscoderや独立SAEより高い感度で抽出できる。

### H3: 因果性仮説

fine-tune-specific latentをablateするとfine-tuned modelの性能がbase側へ戻り、同directionをbase modelへinjectするとfine-tuned behaviorの一部を再現できる。

### H4: shortcut仮説

一部の性能改善は、遺伝子・position・sequence contextに依存する非機序的latentで説明され、gene-disjoint OODで崩れる。

## 18. 新規性

1. Delta-Crosscoderをゲノム言語モデルの疾患fine-tuning解析へ適用する。
2. model diffingを因果介入、OOD評価、biological annotationと統合する。
3. fine-tuningで得た性能改善をlatent単位で分解する。
4. 「新規概念の獲得」と「既存概念の再重み付け」を区別する。
5. Bと統合し、疾患latentがprotein-level mechanismへ接続するか検証できる。

## 19. fine-tuning設計

## 19.1 モデル群

- M0: base Evo 2 7B
- M1: ClinVar pathogenicity LoRA
- M2: DMS regression LoRA
- M3: label-permutation LoRA negative control
- M4: data-size matched random-subset LoRA

Stretch:

- splice-specific LoRA
- cardiomyopathy-specific LoRA
- BRCA1-specific narrow LoRA

## 19.2 fine-tuning task

第一選択はpaired variant classification/regression。

入力:

- WT DNA window
- mutant DNA window

モデル内部のvariant-position activation差分を計算し、classification/regression headへ入力する。

```text
Δh = pool(h_mut) - pool(h_wt)
y_hat = head(Δh)
```

学習対象:

- LoRA parameters
- small prediction head

base weightsは凍結する。

loss:

```text
L = L_task
  + λ_preserve * KL(base_logits || finetuned_logits) on unlabeled genomic sequences
  + λ_norm * adapter regularization
```

preservation lossにより、疾患task以外の挙動変化を抑える。

## 19.3 LoRA挿入先

StripedHyena系でmodule名が通常のTransformerと異なる可能性があるため、Claude Codeは最初にmodel inspection scriptを作成する。

出力:

- 全linear-like module名
- parameter shape
- layer index
- forward hook可否
- LoRA target候補

MVPでは、late-middleからlate layerのprojection moduleへ限定する。

探索:

- rank: 4 / 8 / 16
- alpha: 8 / 16 / 32
- dropout: 0 / 0.05

## 20. activation dataset

同一入力をM0とM1へ与え、同一layer・positionのactivationを保存する。

```text
input_id
variant_id
label
model_id
layer_name
activation
base_activation
finetuned_activation
delta_model_activation
```

重要:

- Delta-Crosscoderのtraining dataとinterpretability evaluation dataを分離する。
- fine-tuning training variantsをCrosscoder評価へ使わない。
- gene-disjoint OODを必ず含める。

## 21. Delta-Crosscoder

### 21.1 入力

matched pair:

- `h_base(x)`
- `h_ft(x)`

同一variantのWTとmutantを別サンプルとして扱う方法と、variant deltaを入力する方法を比較する。

推奨主設定:

- `Δh_base_variant = h_base(mut) - h_base(wt)`
- `Δh_ft_variant = h_ft(mut) - h_ft(wt)`

これにより、fine-tuning差分とvariant差分の両方に焦点を当てる。

### 21.2 latent分類

- shared latent: baseとfine-tunedの両方にdecoder directionがある
- amplified latent: directionは共有だがfine-tuned activationまたはdecoder normが増加
- suppressed latent
- fine-tune-specific latent
- base-specific latent

### 21.3 objective

Delta-CrosscoderのBatchTopKとdelta-prioritized lossを実装する。

```text
L = L_reconstruct_base
  + L_reconstruct_ft
  + λ_delta * L_reconstruct_model_difference
  + λ_sparse * L_BatchTopK
  + λ_match * L_matched_pair_consistency
```

## 22. ベースライン

1. weight difference SVD
2. activation difference PCA
3. linear probe on `h_ft - h_base`
4. independent SAE + feature matching
5. standard Crosscoder
6. BatchTopK Crosscoder
7. Delta-Crosscoder
8. random adapter control
9. label-permutation adapter control

## 23. 評価

### 23.1 fine-tuning性能

- AUROC / AUPRC
- DMS Spearman
- calibration
- gene-disjoint performance
- variant-class performance

### 23.2 model diffing性能

- base/ft reconstruction FVE
- model-delta reconstruction
- fine-tune-specific latent purity
- latent stability across seeds
- latent sparsity

### 23.3 biological interpretability

上位latentを以下へ関連付ける。

- pathogenicity label
- DMS fitness
- splice distance
- coding consequence
- protein domain
- conservation
- mutation class

LLMによる自動命名は補助に限定し、定量enrichmentと専門家レビューを主とする。

### 23.4 causal test

#### ablation

fine-tuned modelから対象latent contributionを除去する。

期待:

- task performanceが低下
- base modelの予測へ近づく
- unrelated genomic LM lossは大きく悪化しない

#### injection

base modelへfine-tune-specific decoder directionを注入する。

期待:

- 一部variantでfine-tuned predictionへ近づく
- dose responseが得られる

#### latent-only classifier

上位K latentだけでfine-tuned modelの予測を再現できるか評価する。

## 24. shortcut監査

必須counterfactual:

1. gene-disjoint test
2. GC-matched control
3. synonymous control
4. variant-position distribution matched control
5. local context shuffling
6. label permutation LoRA
7. transcript strand reversal / reverse complement consistency
8. protein consequenceが同程度でDNA contextのみ異なるvariant pair

shortcut latentの定義:

- training performanceとは強く関連
- gene-disjointで崩れる
- biological annotationへ乏しい
- counterfactual context変更へ過敏

## 25. 成功条件

1. LoRAがgene-disjoint testでbaseより改善する。
2. Delta-Crosscoderがstandard Crosscoderよりmodel-delta reconstructionまたはfine-tune-specific latent recoveryで優れる。
3. 少数latentのablationで性能向上の有意な割合を除去できる。
4. 少なくとも一部latentがconservation、splice、domain等のbiological conceptへ対応する。
5. 一部のshortcut latentまたはnegative controlとの差を示す。

No-Go条件:

- prediction headだけで性能が上がり、backbone activationがほぼ変化しない。

対策:

- LoRA挿入層を増やす
- head容量を小さくする
- ranking lossをbackbone activationへ直接かける
- DMS regression等の連続taskへ変更する

## 26. 論文の主要図

1. base-to-finetune DeltaEvo pipeline
2. model performanceとOOD generalization
3. shared/amplified/specific latent taxonomy
4. biological latentとshortcut latentの比較
5. latent ablation / injectionのcausal result
6. Bのshared protein latentへの接続

## 27. 想定する主張

- narrow disease fine-tuningの変化は少数latentへ集中する。
- Delta-Crosscoderは通常のCrosscoderより局所的・非対称な変化を抽出しやすい。
- 性能改善の一部はbase modelの既存biology featureの増幅で説明される。
- 一部はshortcutであり、latent-level auditingにより発見できる。
- disease latentはDNA–protein shared latentへ接続できる場合がある。

---

# BとCの統合研究

## 28. 統合仮説

Cで得たfine-tune-specificまたはamplified latentを、BのDNA shared/private spaceへ射影する。

```text
Disease LoRAで増強されたEvo 2 latent
                ↓
       CentralDogma-ΔCC
        ↙             ↘
protein共有機序      DNA固有shortcut/機序
```

### 解釈

- protein共有spaceへ接続するlatent: 分子機序を学習した可能性
- DNA-privateでsplice annotationへ接続: 妥当なDNA固有機序
- annotationがなくgene-contextだけへ接続: shortcut候補

## 29. 統合タイトル案

**Tracing How Genomic Language Models Learn Disease across the Central Dogma**

副題:

**Delta model diffing and cross-modal sparse representations connect genomic fine-tuning to protein mechanisms**

## 30. 実施順序

### Stage 0: 1週間

- Evo 2 inferenceとembedding extractionの再現
- ESM-2 activation extraction
- BRCA1 zero-shot scoring再現
- 100 variantsのWT/mut sequence生成

### Stage 1: 2週間

- 共通data schema
- activation store
- layer probe
- CCA / concat baseline

### Stage 2: 2–3週間

- Bのdelta Crosscoder MVP
- reconstruction、retrieval、DMS prediction
- shared/private branch追加

### Stage 3: 2–3週間

- Evo 2 LoRA MVP
- gene-disjoint evaluation
- matched activation extraction

### Stage 4: 2週間

- Delta-Crosscoder
- standard Crosscoder比較
- latent taxonomy

### Stage 5: 2週間

- causal ablation / injection
- shortcut controls
- BとCの統合

### Stage 6: 1–2週間

- 複数遺伝子
- seed再現
- 図表
- Workshop原稿

---

# Claude Code向け実装仕様

## 31. リポジトリ構成

```text
central-dogma-diffing/
├── README.md
├── RESEARCH_PLAN.md
├── pyproject.toml
├── uv.lock
├── Makefile
├── configs/
│   ├── data/
│   ├── models/
│   ├── activations/
│   ├── crosscoder/
│   ├── finetune/
│   └── experiments/
├── src/
│   └── cdd/
│       ├── data/
│       │   ├── clinvar.py
│       │   ├── proteingym.py
│       │   ├── transcript_mapping.py
│       │   ├── variant_sequences.py
│       │   ├── splits.py
│       │   └── schemas.py
│       ├── models/
│       │   ├── evo2_wrapper.py
│       │   ├── esm_wrapper.py
│       │   ├── hooks.py
│       │   ├── pooling.py
│       │   └── lora.py
│       ├── activations/
│       │   ├── extract.py
│       │   ├── store.py
│       │   ├── normalize.py
│       │   └── inspect.py
│       ├── crosscoder/
│       │   ├── batch_topk.py
│       │   ├── standard.py
│       │   ├── shared_private.py
│       │   ├── dedicated_features.py
│       │   ├── delta_crosscoder.py
│       │   ├── losses.py
│       │   └── trainer.py
│       ├── finetune/
│       │   ├── dataset.py
│       │   ├── heads.py
│       │   ├── train.py
│       │   └── preservation.py
│       ├── interventions/
│       │   ├── ablate.py
│       │   ├── inject.py
│       │   └── patch.py
│       ├── eval/
│       │   ├── reconstruction.py
│       │   ├── retrieval.py
│       │   ├── variant_prediction.py
│       │   ├── interpretability.py
│       │   ├── causality.py
│       │   └── shortcuts.py
│       └── utils/
├── scripts/
│   ├── download_data.py
│   ├── build_variant_table.py
│   ├── extract_evo2_activations.py
│   ├── extract_esm_activations.py
│   ├── train_b_crosscoder.py
│   ├── train_evo2_lora.py
│   ├── train_c_delta_crosscoder.py
│   ├── evaluate_b.py
│   ├── evaluate_c.py
│   └── make_figures.py
├── tests/
│   ├── test_variant_mapping.py
│   ├── test_wt_mut_sequences.py
│   ├── test_activation_shapes.py
│   ├── test_crosscoder_forward.py
│   ├── test_batch_topk.py
│   ├── test_intervention.py
│   └── test_no_data_leakage.py
└── outputs/
```

## 32. 技術要件

- Python 3.11
- PyTorch
- `uv`によるdependency管理
- dataclassまたはPydanticによるschema validation
- activationはZarrまたはmemory-mapped arrayへ保存
- metadataはParquet
- configはYAML
- experiment trackingはWeights & Biasesをoptionalにする
- unit testはGPU不要のsmall tensor testを中心にする
- GPU integration testはmarkerで分離
- random seed、git commit、config、dataset hashを全runで保存

## 33. CLI受け入れ条件

以下のコマンドが通る状態を最初の完成条件とする。

```bash
uv sync
pytest -q

python scripts/build_variant_table.py \
  --config configs/data/brca1_mvp.yaml

python scripts/extract_evo2_activations.py \
  --config configs/experiments/brca1_evo2_small.yaml

python scripts/extract_esm_activations.py \
  --config configs/experiments/brca1_esm_small.yaml

python scripts/train_b_crosscoder.py \
  --config configs/experiments/b_mvp.yaml

python scripts/train_evo2_lora.py \
  --config configs/experiments/c_lora_mvp.yaml

python scripts/train_c_delta_crosscoder.py \
  --config configs/experiments/c_delta_mvp.yaml

python scripts/evaluate_b.py --run-dir outputs/b_mvp
python scripts/evaluate_c.py --run-dir outputs/c_delta_mvp
```

## 34. Claude Codeに最初に作らせるIssue

### Issue 1: project skeleton

- pyproject
- package layout
- config loader
- logging
- seed utility
- tests

### Issue 2: variant schema and sequence validator

- VariantRecord
- transcript mapping interface
- WT/mut generation
- reference allele validation
- reverse complement test

### Issue 3: Evo 2 wrapper

- load model
- score sequence
- extract named-layer activation
- exact/local pooling
- CPU mock model for tests

### Issue 4: ESM wrapper

- load model
- tokenize protein
- extract layer activation
- residue position mapping
- long sequence windowing

### Issue 5: activation store

- appendable Zarr
- Parquet metadata
- dataset hash
- resume support

### Issue 6: BatchTopK Crosscoder core

- encoder/decoder
- exact BatchTopK
- dead feature monitor
- FVE metric
- checkpoint

### Issue 7: B shared-private model

- modality-specific dimensions
- shared/private branches
- alignment and contrastive losses
- retrieval evaluator

### Issue 8: LoRA inspection and training

- module inventory
- configurable target modules
- paired variant head
- preservation loss

### Issue 9: Delta-Crosscoder

- base/ft matched pairs
- delta reconstruction loss
- latent classification

### Issue 10: causal intervention

- ablation hook
- injection hook
- alpha sweep
- specificity metrics

## 35. コーディング原則

1. 実データをダウンロードする処理と、研究ロジックを分離する。
2. モデルwrapperは共通protocolを実装する。
3. activation tensorのshape、dtype、position mappingを常にmetadataへ保存する。
4. 途中でOOMしてもresume可能にする。
5. 全評価をtrain dataと独立させる。
6. leakage testをunit test化する。
7. 最初は1GPU、small model、100 variantsで動くことを優先する。
8. 大規模化前に、baselineとfailure plotを作る。

## 36. Claude Codeへ貼るマスタープロンプト

```text
あなたは機械学習研究用リポジトリの主任エンジニアです。
RESEARCH_PLAN.mdを読み、CentralDogma-ΔCC（研究B）とDeltaEvo（研究C）を実装してください。

最初にコードを書き始めず、以下を実施してください。
1. 実装を10個前後のGitHub Issueへ分解する。
2. 各Issueに目的、変更ファイル、API、テスト、完了条件を書く。
3. Evo 2の実際のmodule構造とembedding APIを確認し、推測でmodule名をhardcodeしない。
4. 100 variantsとsmall tensorsで動くMVPを最優先する。
5. データリーク、WT/mut位置ずれ、strand、transcript mappingを重大リスクとして扱う。

技術要件:
- Python 3.11
- uv
- PyTorch
- type hints
- ruff
- pytest
- YAML config
- Parquet metadata
- Zarr activation store
- reproducible seeds
- resumable extraction and training

研究Bでは、DNAとproteinの絶対activationではなく、同一variantのWT-mutant activation deltaを主入力にしてください。shared、DNA-private、protein-private latentを分離し、reconstruction、cross-modal retrieval、DMS prediction、annotation enrichment、causal ablationで評価してください。

研究Cでは、base Evo 2と疾患LoRA Evo 2のmatched activation pairを保存し、Delta-Crosscoderを実装してください。standard Crosscoder、independent SAE、activation-difference PCAをbaselineにしてください。fine-tune-specific latentをablate/injectできるhook APIを作ってください。

各Issueを実装するたびに:
- unit testを追加
- READMEまたはdocsを更新
- small synthetic smoke testを実行
- 実行コマンドと結果を報告

不明なAPIは推測せず、インストール済みpackageまたは公式repositoryのコードをinspectしてからadapterを実装してください。
```

---

# 参考文献・実装基盤

1. Brixi G, et al. Genome modelling and design across all domains of life with Evo 2. Nature. 2026. DOI: 10.1038/s41586-026-10176-5.
2. Arc Institute. Evo 2 official repository. Intermediate embedding extraction、BRCA1 scoring notebook、SAE notebook、Savanna/BioNeMo fine-tuning導線を含む。
3. Lindsey J, et al. Sparse Crosscoders for Cross-Layer Features and Model Diffing. Transformer Circuits. 2024.
4. Jiralerspong T, Bricken T. Cross-Architecture Model Diffing with Crosscoders. arXiv:2602.11729. 2026.
5. Kassem A, et al. Delta-Crosscoder: Robust Crosscoder Model Diffing in Narrow Fine-Tuning Regimes. arXiv:2603.04426. 2026.
6. Minder J, et al. Robustly identifying concepts introduced during chat fine-tuning using crosscoders. arXiv:2504.02922. 2025.
7. Simon E, Zou J. InterPLM: discovering interpretable features in protein language models via sparse autoencoders. Nature Methods. 2025. DOI: 10.1038/s41592-025-02836-7.
8. Lin Z, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science. 2023. DOI: 10.1126/science.ade2574.
9. Notin P, et al. ProteinGym: Large-Scale Benchmarks for Protein Design and Fitness Prediction. 2023.
10. Hu EJ, et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR. 2022.
