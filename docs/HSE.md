# Hierarchical Sparse Experts (HSE)

## Documentation Technique v0.2 (annotée)

> Cette version **itère** sur la v0.1. Elle intègre : (1) un protocole QAP asynchrone avec budgets, (2) une clarification des Scribes (poids partagés, instances parallèles), (3) un design de caches réaliste, (4) un plan PoC, (5) une section benchmarks et métriques d’efficacité QAP, (6) une mise à jour des objectifs de performance (présentés comme **cibles**), (7) des risques & mitigations.

---

## 0) Changements clés vs v0.1

* **Scribes** : préciser que *N = ⌈contexte\_total / fenêtre⌉* correspond à **des instances parallèles partageant les mêmes poids**. Deux profils proposés :

  * **Option A (léger)** : 100–300M paramètres (bon coût/latence, routage acceptable si données bien pré-entraînées).
  * **Option B (robuste)** : 0.5–1B paramètres (meilleurs *routing\_indices* et *evidence spans*, coût plus élevé mais améliore la fiabilité du QAP).
* **QAP** : passe en **asynchrone** (requêtes parallèles, *timeouts*, *speculative prefetch*), ajout d’un **budget** par pas de décodage et de métriques dédiées.
* **Caches** : L3 conserve **tokens/indices** (pas de KV), L2 conserve **résumés/états compressés**, L1 conserve **résultats Experts récents + KV quantifié** pour la fenêtre active. Politiques d’éviction explicites.
* **Experts** : **top-2 gating** + équilibrage de charge ; possibilité d’implémenter plusieurs Experts comme **adapters** (LoRA/IA³) au-dessus d’une même base 3–7B pour simplifier le serving.
* **Orchestrateur** : fenêtre élargie (p.ex. 64–128K) via PE allongé ; QAP sert au **random access** lointain (spans ciblés) plutôt qu’à tout le contexte.
* **Benchmarks & métriques** : ajout de LongBench/PG-19/Needle-in-a-Haystack et de métriques QAP (efficacité, rappel d’évidence, latence par composant, stabilité du routage).
* **Objectifs de perf** : présentés comme **cibles** (plages), non comme *benchmarks préliminaires*.

---

## 1) Vue d’ensemble

L’architecture **HSE** vise les contextes ultra‑longs (>1M tokens) en combinant **mémoire compressive**, **sparsité d’experts**, et **accès sélectif à la demande** (QAP). Trois niveaux hiérarchiques :

1. **Scribes** — indexation sémantique parallèle + résumés + spans d’évidence.
2. **Experts** — traitement spécialisé (MoE sparse top‑k) activé par routage.
3. **Orchestrateur** — coordination globale, génération, et émission des requêtes QAP.

---

## 2) Architecture

### 2.1 Niveau 1 : Couche des Scribes

**Rôle** : traitement initial, segmentation, indexation et extraction d’évidence.

**Spécifications** :

* **Taille** : Option A 100–300M ; Option B 0.5–1B.
* **Fenêtre locale** : 2,048–4,096 tokens.
* **Instances** : $N = \lceil \text{contexte_total} / \text{fenêtre} \rceil$ **avec poids partagés**.
* **Parallélisation** : totale (data‑parallel sur chunks).
* **Index** : vecteurs *semantic\_embedding* indexés (FAISS/HNSW ou équivalent), re‑ranking léger (cross‑encodeur) facultatif.

**Outputs par chunk** :

```python
class EvidenceSpan(NamedTuple):
    start: int            # index token relatif au chunk
    end: int              # exclusif
    score: float          # confiance/spécificité

class ScribeOutput(TypedDict):
    semantic_embedding: Tensor[768]        # représentation dense
    routing_indices: Dict[str, float]      # {"code":0.8,"math":0.2,...}
    importance_score: float                # 0.0–1.0
    temporal_markers: List[int]            # positions d’événements clés
    summary_tokens: Tensor[128]            # résumé compressé
    evidence_spans: List[EvidenceSpan]     # candidats pour QAP
    index_ref: str                         # id (chunk, offsets) pour store L3
```

**Notes** :

* Les *routing\_indices* sont calibrés avec une perte d’entropie + pénalité d’over‑routage.
* Les *evidence\_spans* bornent précisément ce que QAP peut demander (évite d’extraire un chunk entier).

---

### 2.2 Niveau 2 : Couche des Experts

**Rôle** : traitement spécialisé par domaine, activé dynamiquement.

**Configuration (exemple)** :

```yaml
experts:
  - name: CodeExpert
    base: "Base-7B"            # même base possible pour plusieurs experts
    adapter: "lora:code-v1"     # option adapters pour servir léger
    parameters: 7B               # (base); <100M d'adapters
    specialization: [python, js, rust, algorithms]
    attention_span: 16K

  - name: MathExpert
    base: "Base-7B"
    adapter: "lora:math-v1"
    parameters: 7B
    specialization: [algebra, calculus, statistics, proofs]
    attention_span: 8K

  - name: DialogueExpert
    base: "Base-3B"
    adapter: "lora:dialogue-v1"
    parameters: 3B
    specialization: [conversation, qa, emotional]
    attention_span: 32K

  - name: AnalyticalExpert
    base: "Base-10B"
    adapter: "lora:analysis-v1"
    parameters: 10B
    specialization: [reasoning, logic, causality, synthesis]
    attention_span: 64K
```

**Sélection** : **top‑2 gating** (configurable), **seuil d’activation** (défaut 0.3), **équilibrage de charge** (évite le *collapse* vers un seul expert).

**QAP côté Experts** : chaque Expert peut initier des requêtes (limité par un **qap\_budget\_expert** par pas) pour récupérer des *spans* cibles auprès des Scribes.

---

### 2.3 Niveau 3 : Orchestrateur

**Rôle** : coordonner, agréger, planifier le QAP, et générer.

**Spécifications** :

* **Taille** : 10B–30B recommandé pour PoC avancé ; scalable jusqu’à 100B.
* **Fenêtre** : 64K–128K (PE étendu). QAP pour accès aléatoire lointain.
* **Entrées** : sorties Experts, graphes de dépendances, états du cache.
* **QAP** : planifie des requêtes **asynchrones** (cf. §3) avec **max\_qap\_queries** par pas et **timeouts**.

---

## 3) Protocole d’Attention à la Demande (QAP)

### 3.1 Flux asynchrone (schéma)

```mermaid
graph TD
    A[Orchestrateur] -->|QAP: VERIFICATION x2, EXPANSION x1| B[Experts]
    B -->|QAP: DETAIL(span#7), CONTEXT(span#12)| C[Scribes]
    C -->|retour SPANS ciblés + métadonnées| B
    B -->|résultats agrégés| A
    A -->|décodage stream + nouvelles QAP| A
```

### 3.2 Spécification de requête

```python
class QAPQuery(TypedDict):
    query_id: str
    source_level: Literal[3,2,1]          # 3=Orch., 2=Expert, 1=Scribe
    target_level: Literal[2,1]
    query_type: Literal["DETAIL","CONTEXT","VERIFICATION","EXPANSION"]
    resource: Literal["SPANS","TOKENS","STATE"]
    specificity: Dict[str, Any]            # {"chunk_ids":[...],"spans":[(s,e),...]}
    priority: float                        # 0.0–1.0
    deadline_ms: int                       # timeout dur
    budget_cost: float                     # coût estimé (pour throttling)
```

**Contrats** :

* **resource=SPANS** = chemin rapide (retour ≤ quelques centaines de tokens).
* **resource=TOKENS** = autoriser un *slice* continu si vraiment nécessaire.
* **resource=STATE** = résumés/états compressés (pas de KV brut depuis L3).

### 3.3 Budgets & politiques

* **Budget global par pas** : `qap_budget_step` (somme des `budget_cost`).
* **Priorités** : `VERIFICATION` > `DETAIL` > `CONTEXT` > `EXPANSION`.
* **Timeouts** : si `deadline_ms` expiré, l’Orchestrateur poursuit le décodage sans bloquer (streaming first token garanti).
* **Speculative prefetch** : l’Orchestrateur peut lancer des QAP anticipées pour les *spans* les plus probables.

### 3.4 Métriques QAP

* **qap\_efficiency** = requêtes / 1k tokens générés (plus bas = mieux, sous contrainte de qualité).
* **qap\_hit\_rate** = % de requêtes utiles (améliorent log‑prob / exactitude).
* **evidence\_recall\@k** = rappel des *spans* sources cités.
* **latency\_breakdown** = temps (Orch/Experts/QAP/IO/Decode) par token.

---

## 4) Caches hiérarchiques

| Niveau           | Contenu                                                          | Taille indicative | Politique                                     |
| ---------------- | ---------------------------------------------------------------- | ----------------- | --------------------------------------------- |
| **L1 (Orch)**    | Résultats Experts récents + **KV quantifié** (fenêtre active)    | 0.5–2 GB          | LRU + *attention‑gate* (éviction KV)          |
| **L2 (Experts)** | Résumés Scribes + **états compressés** (infini/compressive‑like) | 2–8 GB            | LRU + *age‑based decay*                       |
| **L3 (Scribes)** | **Tokens originaux + index\_ref + meta** (pas de KV)             | 8–32 GB RAM/SSD   | Importance‑score > seuil ; éviction par score |

**Notes** :

* KV long‑terme → éviter de l’entreposer ; privilégier **résumés/états**.
* KV court‑terme (Orch) → **quantif 4/8‑bit**, éviction adaptative.
* Les *evidence\_spans* servent de granule de base pour QAP/TOKENS.

---

## 5) Pipeline d’inférence

### 5.1 Phase 1 — Encodage parallèle

```python
scribe_outputs = parallel_map(
    scribe_model,           # poids partagés
    input_chunks,           # segmentation w=2–4K
    batch_size=32
)
# Indexation/stockage L3 + construction d’un graphe léger (temporal markers)
```

### 5.2 Phase 2 — Routage & Experts (top‑2)

```python
active = select_experts(
    routing=aggregate_routing(scribe_outputs),
    threshold=0.3,
    top_k=2,
    load_balance=True,
)
expert_states = run_experts(
    active,
    summaries=scribe_outputs,
    qap_budget_expert=6,
)
```

### 5.3 Phase 3 — Orchestration + QAP asynchrone + streaming

```python
with qap_controller.session(budget_step=12) as qap:
    for token in orchestrator.stream_generate(
        expert_states,
        max_qap_queries=20,
        qap=qap,
    ):
        # Le contrôleur peut lancer des QAP (prefetch, verification)
        maybe_issue_qap(token, qap, scribe_outputs)
        yield token
```

---

## 6) Entraînement

### 6.1 Stratégie multi‑phase

1. **Pré‑entraînement Scribes** : MLM + prédiction d’importance + *span extraction* supervisée.
2. **Spécialisation Experts** : fine‑tuning domaine (supervisé) ; si adapters : coût réduit, hot‑swap possible.
3. **Orchestrateur + QAP** : apprentissage de la politique de requêtes via (a) imitation (séquences d’oracle), (b) *reward modeling* sur utilité/latence, (c) RL léger (minimiser #QAP pour une qualité constante).

### 6.2 Pertes

```python
loss = (
    a * reconstruction_loss              # Scribes: MLM
  + b * routing_accuracy_loss            # qualité du routage
  + c * qap_efficiency_loss              # pénalité par requête inutile/coûteuse
  + d * generation_quality_loss          # perplexité/QA/faithfulness
  + e * load_balance_loss                # équilibre Experts (anti-collapse)
  + f * evidence_recall_loss             # encourage citation d’évidence
)
```

---

## 7) Évaluation & Benchmarks

**Datasets/Tasks (long‑contexte)** : LongBench, PG‑19 (long read), Needle‑in‑a‑Haystack, NarrativeQA‑long, arXiv summarization long.

**Métriques clés** :

* **Qualité** : exact match/F1 (QA), Rouge/L (synthèse), perplexité.
* **Mémoire** : *evidence\_recall\@k*, *source attribution precision*.
* **QAP** : *qap\_efficiency*, *qap\_hit\_rate*, *latency\_breakdown*.
* **Routage** : entropie de la passerelle, *expert hit‑rate*, *load balance*.

**Ablations** :

* QAP on/off ; SPANS vs TOKENS ; L2 compressif on/off ; top‑1 vs top‑2 Experts ; fenêtre Orch 32K vs 128K.

---

## 8) Objectifs de performance (cibles)

> Hypothèses matérielles pour PoC : 8×A100 80GB (ou équivalent) + RAM 256–512GB + SSD NVMe.

| Métrique            | Cible réaliste (PoC)                                |
| ------------------- | --------------------------------------------------- |
| Contexte adressable | ≥ 1M tokens (via Scribes + QAP SPANS)               |
| Latence 1er token   | 120–200 ms (Orch 10–15B, QAP async actif)           |
| Débit               | 60–140 tok/s (selon Experts activés)                |
| Mémoire GPU         | Orch+Expert(s) 40–120GB (avec quantif/adapters)     |
| Coût/1M tokens      | dépend fort du taux QAP ; viser < 0.01\$ en interne |

> Les chiffres v0.1 (95ms, 120 tok/s, 24GB) sont **trop optimistes** pour un système multi‑niveaux avec QAP. Garder ces valeurs comme **ambitions** à moyen terme.

---

## 9) Implémentation de référence (ébauche)

```python
class HSEModel:
    def __init__(self, cfg: HSEConfig):
        self.scribe = ScribeModel(cfg.scribe)
        self.index = EvidenceIndex(cfg.index)  # FAISS/HNSW + store L3
        self.experts = ExpertRegistry(cfg.experts)  # adapters en option
        self.orch = Orchestrator(cfg.orchestrator)
        self.qap = QAPController(cfg.qap, cache=HierarchicalCache(cfg.cache))

    def forward(self, input_ids: Tensor, attn_mask: Tensor):
        chunks = chunk(input_ids, cfg.chunk_size)  # 2–4K
        s_out = self.scribe.batch(chunks)
        self.index.upsert(s_out)

        active = route(self.experts, s_out, top_k=2, thr=0.3, balance=True)
        e_states = process_experts(active, s_out, qap_budget=cfg.qap.per_expert)

        with self.qap.session(budget_step=cfg.qap.per_step) as qap:
            return self.orch.generate(e_states, qap=qap, max_qap=cfg.qap.max_queries)
```

---

## 10) Plan PoC (3–6 semaines)

1. **Scribes v0** : modèle partagé 300M (ou 1B si budget) + *evidence\_spans* ; index FAISS ; stockage L3 (tokens + meta).
2. **Experts v0** : 2 Experts (Code, Général) comme **adapters** sur une base 3–7B ; top‑2 gating + métriques de balance.
3. **QAP v0** : requêtes **SPANS** uniquement ; budget fixe par pas ; *timeouts* ; pas de TOKENS bruts au début.
4. **Caches v0** : L1 (résultats + KV quantifié), L2 (résumés compressifs), L3 (tokens + index\_ref).
5. **Éval v0** : LongBench/Needle ; métriques QAP (efficiency/hit‑rate/latency breakdown) ; ablations (QAP on/off).

**Livrables** :

* API QAP (proto + SDK client interne)
* Dash de perf (latence par composant, qap\_efficiency, recall)
* Rapport ablations + recommandations v0.3

---

## 11) Risques & mitigations

* **Latence QAP** : asynchronisme + *prefetch* + budgets + *timeouts* ; garder l’Orch. capable d’émettre sans bloquer.
* **Collapse d’experts** : top‑2 gating + pénalité de déséquilibre + seuil dynamique.
* **QAP thrashing** : *rate limiting* + *cooldown* par *span* + *dedup* de requêtes.
* **Mémoire KV** : quantif 4/8‑bit + éviction *attention‑gate* ; éviter KV long‑terme en L3.
* **Routage bruité** : entropie/temperature reg., *teacher routing* en distillation, *calibration* périodique.

---

## 12) Roadmap mise à jour

* **v0.1 (PoC)**

  * [ ] Scribes basiques + index + *evidence\_spans*
  * [ ] QAP minimal (SPANS, async, budgets)
  * [ ] 2 Experts (Code + Général via adapters)
* **v0.2 (Alpha)**

  * [ ] Caches L1/L2/L3 complets + éviction adaptative
  * [ ] 5+ Experts spécialisés (adapters)
  * [ ] Benchmarks LongBench/PG‑19 + ablations
* **v1.0 (Prod)**

  * [ ] Optimisations CUDA/Triton
  * [ ] API OpenAI‑compatible
  * [ ] Multimodal (images côté Scribes) + sécurité/PII

---

## 13) Annexes

### 13.1 Types QAP (proto d’API interne)

```python
class QAPBudget(TypedDict):
    per_step: int
    per_expert: int
    max_queries: int

class QAPMetrics(TypedDict):
    efficiency_per_1k: float
    hit_rate: float
    evidence_recall_k: float
    latency_ms: Dict[str, float]  # {"orch":...,"expert":...,"qap":...,"io":...,"decode":...}
```

### 13.2 Contrats de cache

```yaml
cache:
  L1:
    store: [expert_results, kv_quantized]
    kv_bits: 4
    eviction: attention_gate+LRU
  L2:
    store: [scribe_summaries, compressed_states]
    eviction: LRU+age_decay
  L3:
    store: [raw_tokens, index_ref, meta]
    eviction: importance_score
```

---

**Contact** : [architecture@hse-project.ai](mailto:architecture@hse-project.ai)
**Repository** : github.com/hse-project/core
**Paper** : arxiv.org/abs/2025.xxxxx *(WIP, section QAP à détailler avec résultats PoC)*
