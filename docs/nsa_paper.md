## Native Sparse Attention: Hardware-Aligned and Natively

## Trainable Sparse Attention

```
Jingyang Yuan∗1,2, Huazuo Gao^1 , Damai Dai^1 , Junyu Luo^2 , Liang Zhao^1 , Zhengyan Zhang^1 , Zhenda Xie^1 ,
Y. X. Wei^1 , Lean Wang^1 , Zhiping Xiao^3 , Yuqing Wang^1 , Chong Ruan^1 , Ming Zhang^2 , Wenfeng Liang^1 ,
Wangding Zeng^1
```
(^1) **DeepSeek-AI**
(^2) **Key Laboratory for Multimedia Information Processing, School of Computer Science, Peking
University, PKU-Anker LLM Lab**
(^3) **University of Washington
{yuanjy, mzhang_cs}@pku.edu.cn,{zengwangding, wenfeng.liang}@deepseek.com**

## Abstract

```
Long-context modeling is crucial for next-generation language models, yet the high compu-
tational cost of standard attention mechanisms poses significant computational challenges.
Sparse attention offers a promising direction for improving efficiency while maintaining model
capabilities. We present NSA, aNatively trainableSparseAttention mechanism that integrates
algorithmic innovations with hardware-aligned optimizations to achieve efficient long-context
modeling. NSA employs a dynamic hierarchical sparse strategy, combining coarse-grained
token compression with fine-grained token selection to preserve both global context awareness
and local precision. Our approach advances sparse attention design with two key innovations:
(1) We achieve substantial speedups through arithmetic intensity-balanced algorithm design,
with implementation optimizations for modern hardware. (2) We enable end-to-end training,
reducing pretraining computation without sacrificing model performance. As shown in Figure 1,
experiments show the model pretrained with NSA maintains or exceeds Full Attention models
across general benchmarks, long-context tasks, and instruction-based reasoning. Meanwhile,
NSA achieves substantial speedups over Full Attention on 64k-length sequences across decod-
ing, forward propagation, and backward propagation, validating its efficiency throughout the
model lifecycle.
```
## 1. Introduction

```
The research community increasingly recognizes long-context modeling as a crucial capability
for next-generation large language models, driven by diverse real-world applications ranging
from in-depth reasoning (DeepSeek-AI, 2025; Zelikman et al., 2022), repository-level code gener-
ation (Zhang et al., 2023a; Zhang et al.) and multi-turn autonomous agent systems (Park et al.,
2023). Recent breakthroughs, including OpenAI’s o-series models, DeepSeek-R1 (DeepSeek-AI,
2025), and Gemini 1.5 Pro (Google et al., 2024), enabling models to process entire codebases,
lengthy documents, maintain coherent multi-turn conversations over thousands of tokens, and
perform complex reasoning across long-range dependencies. However, the high complexity (Za-
heer et al., 2020) of vanilla Attention (Vaswani et al., 2017) mechanisms emerges as a critical
```
```
*Contribution during internship at DeepSeek-AI.
```
# arXiv:2502.11089v2 [cs.CL] 27 Feb 2025


```
General LongBenchReasoning
```
```
0.
```
```
0.
```
```
0.
```
```
0.
```
```
0.
```
```
0.
```
```
Score
```
```
Performance on Benchmarks
```
```
Full Attention
NSA
```
```
Decode Forward Backward
```
```
1.
```
```
3.
```
```
5.
```
```
7.
```
```
9.
```
```
11.
```
```
13.
```
```
Speedup Ratio
```
```
11.6 ×
```
```
9.0 ×
```
```
6.0 ×
```
```
Speed on Stages
```
Figure 1|Comparison of performance and efficiency between Full Attention model and our NSA.
Left: Despite being sparse, NSA surpasses Full Attention baseline on average across general
benchmarks, long-context tasks, and reasoning evaluation. Right: For 64k-length sequence
processing, NSA achieves substantial computational speedup compared to Full Attention in all
stages: decoding, forward propagation, and backward propagation.

latency bottleneck as sequence length increases. Theoretical estimates indicate that attention
computation with softmax architectures accounts for 70–80% of total latency when decoding
64k-length contexts, underscoring the urgent need for more efficient attention mechanisms.

A natural approach to efficient long-context modeling is to take advantage of the inherent
sparsity of softmax attention (Ge et al., 2023; Jiang et al., 2023), where selectively computing
critical query-key pairs can significantly reduce computational overhead while preserving
performance. Recent advances demonstrate this potential through diverse strategies: KV-cache
eviction methods (Li et al., 2024; Zhang et al., 2023b; Zhou et al., 2024), blockwise KV-cache
selection methods (Gao et al., 2024; Tang et al., 2024; Xiao et al., 2024a), and sampling, clustering
or hashing-based selection methods (Chen et al., 2024b; Desai et al., 2024; Liu et al., 2024).
Despite these promising strategies, existing sparse attention methods often fall short in practical
deployments. Many approaches fail to achieve speedups comparable to their theoretical gains;
moreover, most methods lack effective training-time support to fully exploit the sparsity patterns
of attention.

To address these limitations, the deployment of effective sparse attention must tackle two
key challenges: (1) **Hardware-aligned inference speedup** : Converting theoretical computation
reductions into actual speed improvements requires hardware-friendly algorithm design during
both prefilling and decoding stages to mitigate memory access and hardware scheduling bottle-
necks; (2) **Training-aware algorithm design** : Enabling end-to-end computation with trainable
operators to reduce training costs while maintaining model performance. These requirements
are crucial for real-world applications to achieve fast long-context inference or training. When
considering both aspects, existing methods still exhibit a noticeable gap.

To achieve more effective and efficient sparse attention, we present NSA, a Natively trainable
Sparse Attention architecture that integrates hierarchical token modeling. As shown in Figure 2,
NSA reduces per-query computation by organizing keys and values into temporal blocks and
processing them through three attention paths: compressed coarse-grained tokens, selectively
retained fine-grained tokens, and sliding windows for local contextual information. Then


```
Concat
```
```
Compression Compress Top-n Selection Selection
```
```
Compressed Attention Sliding Attention
```
```
Sliding
```
```
Compressed Attention Mask
```
```
Sliding Attention Mask
```
```
Selected Attention
```
```
Split to Continuous Blocks
```
```
Attention Score Query Token Activated Token Evicted Token Ignored Token
```
```
Native Sparse Attention Mechanism
```
```
Selected Attention Mask
```
```
Gated Output
```
```
Output Output Output
```
```
...
```
```
Figure 2|Overview of NSA’s architecture. Left: The framework processes input sequences
through three parallel attention branches: For a given query, preceding keys and values are
processed into compressed attention for coarse-grained patterns, selected attention for important
token blocks, and sliding attention for local context. Right: Visualization of different attention
patterns produced by each branch. Green areas indicate regions where attention scores need to
be computed, while white areas represent regions that can be skipped.
```
we implement specialized kernels to maximize its practical efficiency. NSA introduces two
core innovations corresponding to the key requirements above: (1) Hardware-aligned system:
Optimize blockwise sparse attention for Tensor Core utilization and memory access, ensuring
balanced arithmetic intensity. (2) Training-aware design: Enable stable end-to-end training
through efficient algorithms and backward operators. This optimization enables NSA to support
both efficient deployment and end-to-end training.
We evaluate NSA through comprehensive experiments on real-world language corpora.
Pretraining on a 27B-parameter transformer backbone with 260B tokens, we assess NSA’s per-
formance across general language evaluations, long-context evaluations, and chain-of-thought
reasoning evaluation. We further compare the kernel speed on A100 GPUs with optimized
Triton (Tillet et al., 2019) implementations. Experimental results demonstrate that NSA achieves
comparable or superior performance to full attention baseline, while outperforming existing
sparse attention approaches. Additionally, NSA delivers substantial speedups across decoding,
forward, and backward stages compared to Full Attention, with the speedup ratio increasing for
longer sequences. These results validate that our hierarchical sparse attention design effectively
balances model capability and computational efficiency.

## 2. Rethinking Sparse Attention Methods

```
Modern sparse attention methods have made significant strides in reducing the theoretical
computational complexity of transformer models. However, most approaches predominantly
apply sparsity during inference while retaining a pretrained Full Attention backbone, poten-
tially introducing architectural bias that limits their ability to fully exploit sparse attention’s
advantages. Before introducing our native sparse architecture, we systematically analyze these
limitations through two critical lenses.
```
```
2.1. The Illusion of Efficient Inference
```
```
Despite achieving sparsity in attention computation, many methods fail to achieve correspond-
ing reductions in inference latency, primarily due to two challenges:
```

**Phase-Restricted Sparsity.** Methods such as H2O (Zhang et al., 2023b) apply sparsity
during autoregressive decoding while requiring computationally intensive pre-processing
(e.g. attention map calculation, index building) during prefilling. In contrast, approaches
like MInference (Jiang et al., 2024) focus solely on prefilling sparsity. These methods fail to
achieve acceleration across all inference stages, as at least one phase remains computational
costs comparable to Full Attention. The phase specialization reduces the speedup ability of these
methods in prefilling-dominated workloads like book summarization and code completion, or
decoding-dominated workloads like long chain-of-thought (Wei et al., 2022) reasoning.

**Incompatibility with Advanced Attention Architecture.** Some sparse attention methods
fail to adapt to modern decoding efficient architectures like Mulitiple-Query Attention (MQA)
(Shazeer, 2019) and Grouped-Query Attention (GQA) (Ainslie et al., 2023), which significantly
reduced the memory access bottleneck during decoding by sharing KV across multiple query
heads. For instance, in approaches like Quest (Tang et al., 2024), each attention head indepen-
dently selects its KV-cache subset. Although it demonstrates consistent computation sparsity
and memory access sparsity in Multi-Head Attention (MHA) models, it presents a different sce-
nario in models based on architectures like GQA, where the memory access volume of KV-cache
corresponds to the union of selections from all query heads within the same GQA group. This
architectural characteristic means that while these methods can reduce computation operations,
the required KV-cache memory access remains relatively high. This limitation forces a critical
choice: while some sparse attention methods reduce computation, their scattered memory access
pattern conflicts with efficient memory access design from advanced architectures.

These limitations arise because many existing sparse attention methods focus on KV-cache
reduction or theoretical computation reduction, but struggle to achieve significant latency
reduction in advanced frameworks or backends. This motivates us to develop algorithms that
combine both advanced architectural and hardware-efficient implementation to fully leverage
sparsity for improving model efficiency.

**2.2. The Myth of Trainable Sparsity**

Our pursuit of native trainable sparse attention is motivated by two key insights from analyzing
inference-only approaches: (1) **Performance Degradation** : Applying sparsity post-hoc forces
models to deviate from their pretrained optimization trajectory. As demonstrated by Chen et al.
(2024b), top 20% attention can only cover 70% of the total attention scores, rendering structures
like retrieval heads in pretrained models vulnerable to pruning during inference. (2) **Training
Efficiency Demands** : Efficient handling of long-sequence training is crucial for modern LLM
development. This includes both pretraining on longer documents to enhance model capacity,
and subsequent adaptation phases such as long-context fine-tuning and reinforcement learning.
However, existing sparse attention methods primarily target inference, leaving the computa-
tional challenges in training largely unaddressed. This limitation hinders the development
of more capable long-context models through efficient training. Additionally, efforts to adapt
existing sparse attention for training also expose challenges:

**Non-Trainable Components.** Discrete operations in methods like ClusterKV (Liu et al.,
2024) (includes k-means clustering) and MagicPIG (Chen et al., 2024b) (includes SimHash-based
selecting) create discontinuities in the computational graph. These non-trainable components
prevent gradient flow through the token selection process, limiting the model’s ability to learn
optimal sparse patterns.

```
Inefficient Back-propagation. Some theoretically trainable sparse attention methods suffer
```

```
from practical training inefficiencies. Token-granular selection strategy used in approaches
like HashAttention (Desai et al., 2024) leads to the need to load a large number of individual
tokens from the KV cache during attention computation. This non-contiguous memory access
prevents efficient adaptation of fast attention techniques like FlashAttention, which rely on
contiguous memory access and blockwise computation to achieve high throughput. As a result,
implementations are forced to fall back to low hardware utilization, significantly degrading
training efficiency.
```
```
2.3. Native Sparsity as an Imperative
```
These limitations in inference efficiency and training viability motivate our fundamental redesign
of sparse attention mechanisms. We propose NSA, a natively sparse attention framework that
addresses both computational efficiency and training requirements. In the following sections,
we detail the algorithmic design and operator implementation of NSA.

## 3. Methodology

```
Our technical approach spans algorithm design and kernel optimization. In the following
subsections, we first introduce the background of our methodology. Then we present the
overall framework of NSA, followed by its key algorithmic components. Finally, we detail our
hardware-optimized kernel design that maximizes practical efficiency.
```
```
3.1. Background
```
**Attention Mechanism** is widely used in language modeling where each query token **q** 𝑡computes
relevance scores against all preceding keys **k** :𝑡to generate a weighted sum of values **v** :𝑡. Formally,
for an input sequence of length𝑡, the attention operation is defined as:

```
o 𝑡=Attn
```
#### 