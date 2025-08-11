```
Shen Nie1 *† Fengqi Zhu1 *† Zebin You^1 † Xiaolu Zhang^2 ‡ Jingyang Ou^1 Jun Hu^2 ‡ Jun Zhou^2
Yankai Lin^1 ‡ Ji-Rong Wen^1 Chongxuan Li^1 ‡¶
```
## Abstract

```
Autoregressive models (ARMs) are widely re-
garded as the cornerstone of large language mod-
els (LLMs). We challenge this notion by intro-
ducingLLaDA, a diffusion model trained from
scratch under the pre-training and supervised fine-
tuning (SFT) paradigm. LLaDA models distri-
butions through a forward data masking process
and a reverse process, parameterized by a vanilla
Transformer to predict masked tokens. By op-
timizing a likelihood bound, it provides a prin-
cipled generative approach for probabilistic in-
ference. Across extensive benchmarks, LLaDA
demonstrates strongscalability, outperforming
our self-constructed ARM baselines. Remark-
ably, LLaDA 8B is competitive with strong LLMs
like LLaMA3 8B inin-context learningand, af-
ter SFT, exhibits impressiveinstruction-following
abilities in case studies such as multi-turn dia-
logue. Moreover, LLaDA addresses the rever-
sal curse, surpassing GPT-4o in a reversal poem
completion task. Our findings establish diffu-
sion models as a viable and promising alternative
to ARMs, challenging the assumption that key
LLM capabilities discussed above are inherently
tied to ARMs. Project page and codes:https:
//ml-gsai.github.io/LLaDA-demo/.
```
## 1. Introduction

```
What is now proved was once only imagined.
placeholder,placeholder,placeho—William Blake
```
```
Large language models (LLMs) (Zhao et al., 2023) fall en-
tirely within the framework ofgenerative modeling. Specifi-
cally, LLMs aim to capture the true but unknown language
*Equal contribution†Work done during an internship at Ant
Group‡Project leaders^1 Gaoling School of Artificial Intelli-
gence, Renmin University of China; Beijing Key Laboratory
of Big Data Management and Analysis Methods^2 Ant Group.
¶Correspondence to: Chongxuan Li<chongxuanli@ruc.edu.cn>.
```
```
Preprint.
```
```
46
```
```
33
```
```
50 37
```
```
19
```
```
24
```
(^3349)
50
56
39
45
60
23
29
40
59
60
65
46
52
70
27
34
47
69
70
MMLU
TruthfulQA
ARC-C
GSM8K
Math
HumanEval
MBPP
CMMLU
C-Eval
General Tasks
Mathematics
Code
Chinese
LLaDA 8B Base
LLaMA 3 8B Base
LLaMA 2 7B Base
Figure 1.Zero/Few-Shot Benchmarks.We scale LLaDA to an
unprecedented size of 8B parameters from scratch, achieving com-
petitive performance with strong LLMs (Dubey et al., 2024).
distributionpdata(·)by optimizing a model distributionpθ(·)
through maximum likelihood estimation, or equivalently KL
divergence minimization between the two distributions:
max
θ
Epdata(x)logpθ(x)⇔min
θ
KL(pdata(x)||pθ(x))
| {z }
Generative modeling principles

### . (1)

```
The predominant approach relies on theautoregressivemod-
eling (ARM)—commonly referred to as thenext-token pre-
dictionparadigm—to define the model distribution:
```
```
pθ(x) =pθ(x^1 )
```
### YL

```
i=
```
```
pθ(xi|x^1 ,...,xi−^1 )
| {z }
Autoregressive formulation
```
### , (2)

```
wherexis a sequence of lengthL, andxiis thei-th token.
This paradigm has proven remarkably effective (Radford,
2018; Radford et al., 2019; Brown, 2020; OpenAI, 2022)
and has become the foundation of current LLMs. Despite
its widespread adoption, a fundamental question remains
unanswered:Is the autoregressive paradigm the only viable
path to achieving the intelligence exhibited by LLMs?
```
# arXiv:2502.09992v2 [cs.CL] 18 Feb 2025


We argue that the answer isnota simple “yes”. The key
insight overlooked previously is: it is thegenerative mod-
eling principles(i.e., Eq. (1)),rather than the autoregres-
sive formulation(i.e., Eq. (2)) itself, that fundamentally
underpin the essential properties of LLMs, as detailed be-
low. However, certain inherent limitations of LLMs can be
directly traced to their autoregressive nature.

In particular, we argue thatscalabilityis primarily a con-
sequence of the interplay between Transformers (Vaswani,
2017), model and data size, andFisher consistency^1 (Fisher,
1922) induced by the generative principles in Eq. (1), rather
than a unique result of ARM. The success of diffusion trans-
formers (Bao et al., 2023; Peebles & Xie, 2023) on visual
data (Brooks et al., 2024) supports this claim.

Furthermore, the capabilities ofinstruction-followingand
in-context learning(Brown, 2020) appear to be intrinsic
properties of all properconditionalgenerative models on
structurally consistent linguistic tasks, rather than exclusive
advantages of ARMs. In addition, while ARMs can be
interpreted as alossless data compressor(Deletang et al.;
Huang et al., 2024b), any sufficiently expressive probabilis-
tic model can achieve similar capabilities (Shannon, 1948).

Nevertheless, the autoregressive nature of LLMs presents
notable challenges. For example, sequential token-by-token
generation incurs high computational costs, and the left-
to-right modeling limits effectiveness in reversal reasoning
tasks (Berglund et al., 2023). These inherent limitations
constrain LLMs in handling longer and more complex tasks.

Motivated by these insights, we introduceLLaDA, aLarge
LanguageDiffusion with mAsking, to investigate whether
the capabilities exhibited by LLMs can emerge from gener-
ative modeling principles beyond ARMs in Eq. (2), thereby
addressing the fundamental question posed earlier. In con-
trast to traditional ARMs, LLaDA leverages a masked diffu-
sion model (MDM) (Austin et al., 2021a; Lou et al., 2023;
Shi et al., 2024; Sahoo et al., 2024; Ou et al., 2024), which
incorporates a discrete random masking process and trains
amask predictorto approximate its reverse process. This
design enables LLaDA to construct a model distribution
with bidirectional dependencies and optimize a lower bound
of its log-likelihood, offering an unexplored and principled
alternative to existing LLMs.

We adopt the standard pipeline of data preparation, pre-
training, supervised fine-tuning (SFT), and evaluation, scal-
ing LLaDA to an unprecedented language diffusion of size
8B. In particular,LLaDA 8Bwas pre-trained from scratch
on2.3 trillion tokensusing0.13 million H800 GPU hours,
followed by SFT on4.5 million pairs. Across diverse tasks,
including language understanding, math, code, and Chinese,

(^1) It suggests the ability to recover the true data distribution with
infinite data, a sufficiently large network and optimal training.
LLaDA demonstrates the following contributions:
Scalability.LLaDA scales effectively up to a computational
budget of 1023 FLOPs, achieving comparable results to self-
constructed ARM baselines trained on the same data across
six tasks, e.g., MMLU and GSM8K.
In-Context Learning.Remarkably, LLaDA 8B surpasses
LLaMA2 7B (Touvron et al., 2023) on nearly all 15 standard
zero/few-shot learning tasks while performing on par with
LLaMA3 8B (Dubey et al., 2024).
Instruction-Following.LLaDA significantly enhances the
ability to follow instructions after SFT, as demonstrated in
case studies such as multi-turn dialogue.
Reversal Reasoning. LLaDA effectively breaks there-
versal curse(Berglund et al., 2023) with consistent per-
formance across forward and reversal tasks. Notably, it
outperforms GPT-4o in a reversal poem completion task.

## 2. Approach

```
In this section, we introduce the probabilistic formulation^2 ,
along with the pre-training, supervised fine-tuning, and in-
ference procedures for LLaDA, as illustrated in Fig. 2.
```
```
2.1. Probabilistic Formulation
```
```
Unlike ARMs in Eq. (2), LLaDA defines a model distribu-
tionpθ(x 0 )through aforward processand areverse pro-
cess(Austin et al., 2021a; Ou et al., 2024). The forward
process gradually masks tokens independently inx 0 until
the sequence is fully masked att= 1. Fort∈(0,1), the
sequencextis partially masked, with each being masked
with probabilitytor remaining unmasked with probability
1 −t. The reverse process recovers the data distribution by
iteratively predicting masked tokens astmoves from 1 to 0.
The core of LLaDA is amask predictor, a parametric model
pθ(·|xt)that takesxtas input and predicts all masked tokens
(denoted M) simultaneously. It is trained using a cross-
entropy loss computed only on the masked tokens:
```
```
L(θ)≜−Et,x 0 ,xt
```
### "

### 1

```
t
```
### XL

```
i=
```
```
1 [xit=M] logpθ(xi 0 |xt)
```
### #

### ,(3)

```
wherex 0 is sampled from the training data,tis sampled
uniformly from[0,1], andxtis sampled from the forward
process. The indicator function 1 [·]ensures that the loss is
computed only for masked tokens.
Once trained, we can simulate a reverse process (see Sec. 2.
for details) parameterized by the mask predictor and define
the model distributionpθ(x 0 )as the marginal distribution
```
(^2) Here, we focus on the approach of LLaDA. A rigorous formu-
lation of MDM is provided in Appendix A for interested readers.


```
Mask predictor
```
```
Mask all tokens independently
```
```
Mask predictor
```
```
Prompt Response
```
```
Mask predictor
```
```
Prompt Response
```
```
...
```
```
...
```
```
An intermediate step
```
```
𝑡= 0
```
```
𝑡= 1
```
```
Non-mask token Random mask
```
```
Mask token Remask
```
```
(a) (b) (c)
```
```
Remask
```
```
Mask ratio 𝑡∼𝑈( 0 , 1 )
```
Figure 2.A Conceptual Overview of LLaDA.(a) Pre-training. LLaDA is trained on text with random masks applied independently to all
tokens at the same ratiot∼U[0,1]. (b) SFT. Only response tokens are possibly masked. (c) Sampling. LLaDA simulates a diffusion
process fromt= 1(fully masked) tot= 0(unmasked), predicting all masks simultaneously at each step with flexible remask strategies.

induced att= 0. Notably, the loss in Eq. (3) has been
proven to be an upper bound on the negative log-likelihood
of the model distribution (Shi et al., 2024; Ou et al., 2024):

```
−Epdata(x 0 )[logpθ(x 0 )]≤L(θ), (4)
```
making it a principled objective for generative modeling.

Notably, LLaDA employs a masking ratio that varies ran-
domly between 0 and 1 while masked language models (De-
vlin, 2018) use a fixed ratio. The subtly differences have
significant implications, especially at scale: as shown in
Eq. (4), LLaDA is a principled generative model with the
potential to performin-context learningnaturally, akin to
LLMs. Moreover, its generative perspective ensuresFisher
consistency(Fisher, 1922) in extreme cases, suggesting
strongscalabilitywith large data and models.

2.2. Pre-training

LLaDA employs a Transformer (Vaswani, 2017) as the mask
predictor, whose architecture is similar to existing LLMs.
However, LLaDA does not use a causal mask, as its formu-
lation allows it to see the entire input for predictions.

We trained two variants of LLaDA with different sizes: 1
billion (B) and 8B. We summarize the model architecture
of LLaDA 8B and LLaMA3 8B (Dubey et al., 2024) here
and details are provided in Appendix B.2. We have ensured
consistency in most hyperparameters while making several
necessary modifications. We use vanilla multi-head atten-
tion instead of grouped query attention (Ainslie et al., 2023)
for simplicity, as LLaDA is incompatible with KV caching,
resulting in a different number of key and value heads. Con-
sequently, the attention layer has more parameters, and we
reduce the FFN dimension to maintain a comparable model
size. Additionally, the vocabulary size differs slightly due
to a tokenizer (Brown, 2020) adapted on our data.

The LLaDA model is pre-trained on a dataset comprising
2.3 trillion(T) tokens, adhering to a data protocol that aligns

```
closely with existing large language models (LLMs) (Tou-
vron et al., 2023; Chu et al., 2024), without the incorporation
of any special techniques. The data are derived from online
corpora, with low-quality content filtered through manually
designed rules and LLM-based approaches. Beyond general
text, the dataset encompasses high-quality code, math, and
multilingual data. The mixing of data sources and domains
is guided by scaled-down ARMs. The pre-training process
utilizes a fixed sequence length of 4096 tokens, incurring a
total computational cost of0.13 million H800 GPU hours,
similar to ARMs of the same scale and dataset size.
For a training sequencex 0 , we randomly samplet∈[0,1],
mask each token independently with the same probability
tto obtainxt(see Fig. 2 (a)) and estimate Eq. (3) via the
Monte Carlo method for stochastic gradient descent training.
In addition, following Nie et al. (2024), to enhance the
ability of LLaDA to handle variable-length data, we set 1%
of the pre-training data to a random length that is uniformly
sampled from the range[1,4096].
We adopted the Warmup-Stable-Decay (Hu et al., 2024)
learning rate scheduler to monitor the training progress
without interrupting continuous training. Specifically, we
linearly increased the learning rate from 0 to 4 × 10 −^4 over
the first 2000 iterations and maintained it at 4 × 10 −^4. After
processing 1.2T tokens, we decayed the learning rate to
1 × 10 −^4 and held it constant for the next 0.8T tokens to en-
sure stable training. Finally, we linearly reduced the learning
rate from 1 × 10 −^4 to 1 × 10 −^5 for the last 0.3T tokens. Fur-
thermore, we utilized the AdamW optimizer (Loshchilov,
2017) with a weight decay of 0.1, a batch size of 1280, and
a local batch size of 4 per GPU. The 8B experiment was
executed once, without any hyperparameter tuning.
```
```
2.3. Supervised Fine-Tuning
```
```
We enhance the capability of LLaDA to follow instructions
by supervised fine-tuning (SFT) with paired data(p 0 ,r 0 ),
```

wherep 0 is the prompt andr 0 denotes the response. This
is the simplest and most basic post-training method for
LLMs. Technically, this requires to model theconditional
distributionpθ(r 0 |p 0 )instead ofpθ(x 0 )in pre-training.

The implementation is similar to pre-training. As shown in
Fig. 2 (b), we leave the prompt unchanged and mask the
tokens in the response independently, as done forx 0. Then,
we feed both the prompt and the masked responsertto the
pre-trained mask predictor to compute the loss for SFT:

```
−Et,p 0 ,r 0 ,rt
```
### 

### ^1

```
t
```
### XL′

```
i=
```
```
1 [rti=M] logpθ(ri 0 |p 0 ,rt)
```
### 

### , (5)

whereL′denotes a dynamic length specified later, and all
other notations remain the same as before.

Note that this approach is fully compatible with pre-training.
Essentially, the concatenation ofp 0 andr 0 can be treated
as clean pre-training datax 0 , while the concatenation of
p 0 andrtserves as the masked versionxt. The process is
identical to pre-training, with the only difference being that
all masked tokens happen to appear in ther 0 portion.

The LLaDA 8B model undergoes SFT on a dataset com-
prising4.5 millionpairs. Consistent with the pre-training
process, both data preparation and training follow the SFT
protocols utilized in existing LLMs (Chu et al., 2024; Yang
et al., 2024), without introducing any additional techniques
to optimize LLaDA’s performance. The dataset spans mul-
tiple domains, including code, mathematics, instruction-
following, and structured data understanding. We append
|EOS|tokens to the end of short pairs in each mini-batch
to ensure equal lengths across all data. We treat|EOS|as a
normal token during training and remove it during sampling,
enabling LLaDA to control the response length automati-
cally. Please refer to Appendix B.1 for more details.

We train for 3 epochs on the SFT data using a similar sched-
ule to the pre-training phase. The learning rate is linearly
increased from 0 to 2. 5 × 10 −^5 over the first 50 iterations
and then kept constant. During the final10%of iterations,
it is linearly reduced to 2. 5 × 10 −^6. Additionally, we set
the weight decay to 0. 1 , the global batch size to 256 , and
the local batch size to 2 per GPU. The SFT experiment was
executed once, without any hyperparameter tuning.

2.4. Inference

As a generative model, LLaDA is capable of both sampling
new text and evaluating the likelihood of candidate text.

We begin with the sampling. As illustrated in Fig. 2 (c),
given a promptp 0 , we discretize the reverse process to sam-
ple from the model distributionpθ(r 0 |p 0 ), starting from a
fully masked response. The total number of sampling steps
is a hyperparameter, which naturally provides LLaDA with a

```
trade-off between efficiency and sample quality, as analyzed
in Sec. 3.3. We employ uniformly distributed timesteps by
default. In addition, the generation length is also treated as
a hyperparameter, specifying the length of the fully masked
sentence at the beginning of the sampling process. As de-
tailed in Appendix B.4, since both pre-training and SFT
are conducted using datasets with variable lengths, the final
results are insensitive to this length hyperparameter.
At an intermediate step from timet∈(0,1]tos∈[0,t), we
feed bothp 0 andrtinto the mask predictor and predict all
masked tokens simultaneously. Subsequently, weremaskst
of the predicted tokens in expectation to obtainrs, ensuring
that the transition of the reverse process aligns with the
forward process for accurate sampling (Austin et al., 2021a).
In principle, the remasking strategy should be purely ran-
dom. However, inspired by the annealing tricks of sampling
in LLMs (Holtzman et al., 2019; Brown, 2020), we explore
two deterministic yet effective remasking strategies. Specif-
ically, similarly to Chang et al. (2022), we remask thest
of predicted tokens with the lowest confidence based on
the predictions, calledlow-confidenceremasking. Addition-
ally, for LLaDA after SFT, we can divide the sequence into
several blocks and generate them from left to right, called
semi-autoregressiveremasking. Within each block, we ap-
ply the reverse process to perform sampling. We provide
more details and ablation studies in Appendix. B.3.
For conditional likelihood evaluation, we can naturally uti-
lize the upper bound in Eq. (5). However, we find that the
following equivalent form (Ou et al., 2024) exhibits lower
variance and is more stable for evaluation:
```
```
−El,r 0 ,rl
```
### "

### L

```
l
```
### XL

```
i=
```
```
1 [ril=M] logpθ(ri 0 |p 0 ,rl)
```
### #

### , (6)

```
wherelis uniformly sampled from{ 1 , 2 ,...,L}, andrlis
obtained by uniformly samplingltokens fromr 0 without
replacement for masking. In addition, we employ the unsu-
pervised classifier-free guidance (Nie et al., 2024). We refer
the readers to more details in Appendix A.2.
We present the training, sampling, and likelihood evaluation
algorithms, along with theoretical details, in Appendix A.
```
## 3. Experiments

```
We evaluate the scalability, instruction-following, and in-
context learning capabilities of LLaDA on standard bench-
marks, followed by analyses and case studies on more con-
trolled datasets to provide a comprehensive assessment.
```
```
3.1. Scalability of LLaDA on Language Tasks
```
```
We first investigate thescalabilityof LLaDA on downstream
tasks in comparison with the ARM baselines we constructed.
```

```
1020 1021 1022 1023
FLOPs
```
```
20
```
```
30
```
```
40
```
```
50
```
```
60
```
```
MMLU (5-shot)
```
```
Autoregressive Baseline
LLaDA
```
```
1020 1021 1022 1023
FLOPs
```
```
20
```
```
30
```
```
40
```
```
50
```
```
60
```
```
ARC-C (0-shot)
```
```
Autoregressive Baseline
LLaDA
```
```
1020 1021 1022 1023
FLOPs
```
```
20
```
```
30
```
```
40
```
```
50
```
```
60
```
```
CMMLU (5-shot)
```
```
Autoregressive Baseline
LLaDA
```
```
1020 1021 1022 1023
FLOPs
```
```
50
```
```
60
```
```
70
```
```
80
```
```
PIQA (0-shot)
```
```
Autoregressive Baseline
LLaDA
```
```
1020 1021 1022 1023
FLOPs
```
```
0
```
```
20
```
```
40
```
```
60
```
```
GSM8K (4-shot)
```
```
Autoregressive Baseline
LLaDA
```
```
1020 1021 1022 1023
FLOPs
```
```
0
```
```
8
```
```
16
```
```
24
```
```
HumanEval (0-shot)
```
```
Autoregressive Baseline
LLaDA
```
Figure 3.Scalability of LLaDA.We evaluate the performance of LLaDA and our ARM baselines trained on the same data across
increasing computational FLOPs. LLaDA exhibits strong scalability, matching the overall performance of ARMs on six tasks.

Specifically, at the 1B scale, we ensured that LLaDA and
ARM shared the same architecture, data, and all other config-
urations. At larger scales, we also report results for LLaDA
and ARM models of slightly different sizes trained on the
same data due to resource limit, as detailed in Appendix B.2.
We use the computational cost as a unified scaling metric.
For evaluation, we focused on six standard and diverse tasks.

As shown in Fig. 3, LLaDA demonstrates impressive scala-
bility, with its overall trend being highly competitive with
that of ARM. Notably, in tasks such as MMLU and GSM8K,
LLaDA exhibits even stronger scalability. Even on tasks like
PIQA, where performance lags, LLaDA narrows the gap
with ARMs at larger scales. To account for the significant
influence of outliers, we opted not to fit quantitative scaling
curves, avoiding potential misinterpretation. Nevertheless,
the results clearly demonstrate the scalability of LLaDA.

Nie et al. (2024) suggests that MDM requires 16 times
more computation than ARM to achieve the same likeli-
hood. However, there are key differences that make the
conclusions of this study more broadly applicable. In partic-
ular, likelihood is a relatively indirect metric for downstream
task performance, and diffusion optimizes a bound of the
likelihood, making it not directly comparable to ARM. Ad-
ditionally, we extended the scaling range from 1018 ∼ 1020
in Nie et al. (2024) to 1020 ∼ 1023 in this work.

3.2. Benchmark Results

To comprehensively evaluate thein-context learningand
instruction-followingcapabilities of LLaDA 8B, we con-
ducted detailed comparisons with existing LLMs (Touvron

```
et al., 2023; Dubey et al., 2024; Chu et al., 2024; Yang et al.,
2024; Bi et al., 2024; Jiang et al., 2023) of similar scale.
The selection of tasks and evaluation protocols was aligned
with existing studies, encompassing 15 popular benchmarks
in general tasks, mathematics, code, and Chinese. Further
details are provided in Appendix B.5. For a more direct
comparison, we re-evaluated representative LLMs (Touvron
et al., 2023; Dubey et al., 2024) in our implementation.
As shown in Tab. 1, after pretraining on 2.3T tokens,
LLaDA 8B demonstrates remarkable performance, surpass-
ing LLaMA2 7B on nearly all tasks, and is overall competi-
tive with LLaMA3 8B. LLaDA shows advantages in math
and Chinese tasks. We conjecture that the strengths stem
from the same factors as its relatively weaker performance
in some tasks—differences in data quality and distribution,
largely due to the closed-source situation of LLM datasets.
Notably, we have carefully ruled out the possibility of data
leakage by taking GSM8K as an example. First, as shown
in Fig. 3, LLaDA outperformed ARM baselines regarding
GSM8K. Moreover, the conclusion remains on a fully un-
seen GSM8K-like task (Ye et al., 2024) in Appendix B.7.
Further, Tab. 2 compares the performance of LLaDA 8B In-
struct with existing LLMs. We observed that SFT improved
LLaDA’s performance on most downstream tasks. A few
metrics, such as MMLU, showed declines, and we conjec-
ture may be due to the suboptimal quality of the SFT data.
Overall, since we did not perform alignment with reinforce-
ment learning (RL), our results are slightly behind LLaMA
8B Instruct, though the gaps in many metrics remain small.
Notably, even with only SFT, LLaDA demonstrates impres-
```

Table 1.Benchmark Results of Pre-trained LLMs.∗indicates that LLaDA 8B Base, LLaMA2 7B Base, and LLaMA3 8B Base are
evaluated under the same protocol, detailed in Appendix B.5. Results indicated by†and¶are sourced from Chu et al. (2024); Yang et al.
(2024) and Bi et al. (2024) respectively. The numbers in parentheses represent the number of shots used for evaluation. “-” indicates
unknown data.

```
LLaDA 8B∗ LLaMA3 8B∗ LLaMA2 7B∗ Qwen2 7B† Qwen2.5 7B† Mistral 7B† Deepseek 7B¶
Model Diffusion AR AR AR AR AR AR
Training tokens 2.3T 15T 2T 7T 18T - 2T
General Tasks
MMLU 65.9(5) 65.4 (5) 45.9 (5) 70.3 (5) 74.2 (5) 64.2 (5) 48.2 (5)
BBH 49.8 (3) 57.6(3) 37.3 (3) 62.3 (3) 70.4 (3) 56.1 (3) 39.5 (3)
ARC-C 47.9 (0) 53.1(0) 46.3 (0) 60.6 (25) 63.7 (25) 60.0 (25) 48.1 (0)
Hellaswag 72.5 (0) 79.1(0) 76.0 (0) 80.7 (10) 80.2 (10) 83.3 (10) 75.4 (0)
TruthfulQA 46.4(0) 44.0 (0) 39.0 (0) 54.2 (0) 56.4 (0) 42.2 (0) -
WinoGrande 74.8 (5) 77.3(5) 72.5 (5) 77.0 (5) 75.9 (5) 78.4 (5) 70.5 (0)
PIQA 74.4 (0) 80.6(0) 79.1 (0) - - - 79.2 (0)
Mathematics & Science
GSM8K 70.7(4) 53.1 (4) 14.3 (4) 80.2 (4) 85.4 (4) 36.2 (4) 17.4 (8)
Math 27.3(4) 15.1 (4) 3.2 (4) 43.5 (4) 49.8 (4) 10.2 (4) 6.0 (4)
GPQA 26.1(5) 25.9 (5) 25.7 (5) 30.8 (5) 36.4 (5) 24.7 (5) -
Code
HumanEval 33.5 (0) 34.2(0) 12.8 (0) 51.2 (0) 57.9 (0) 29.3 (0) 26.2 (0)
HumanEval-FIM 73.8(2) 73.3 (2) 26.9 (2) - - - -
MBPP 38.2 (4) 47.4(4) 18.4 (4) 64.2 (0) 74.9 (0) 51.1 (0) 39.0 (3)
Chinese
CMMLU 69.9(5) 50.7 (5) 32.5 (5) 83.9 (5) - - 47.2 (5)
C-Eval 70.5(5) 51.7 (5) 34.0 (5) 83.2 (5) - - 45.0 (5)
```
Table 2.Benchmark Results of Post-trained LLMs.LLaDA only employs an SFT procedure while other models have extra reinforcement
learning (RL) alignment.∗indicates that LLaDA 8B Instruct, LLaMA2 7B Instruct, and LLaMA3 8B Instruct are evaluated under the
same protocol, detailed in Appendix B.5. Results indicated by†and¶are sourced from Yang et al. (2024) and Bi et al. (2024) respectively.
The numbers in parentheses represent the number of shots used for in-context learning. “-” indicates unknown data.

```
LLaDA 8B∗ LLaMA3 8B∗ LLaMA2 7B∗ Qwen2 7B† Qwen2.5 7B† Gemma2 9B† Deepseek 7B¶
Model Diffusion AR AR AR AR AR AR
Training tokens 2.3T 15T 2T 7T 18T 8T 2T
Post-training SFT SFT+RL SFT+RL SFT+RL SFT+RL SFT+RL SFT+RL
Alignment pairs 4.5M - - 0.5M + - 1M + 0.15M - 1.5M + -
General Tasks
MMLU 65.5 (5) 68.4(5) 44.1 (5) - - - 49.4 (0)
MMLU-pro 37.0 (0) 41.9(0) 4.6 (0) 44.1 (5) 56.3 (5) 52.1 (5) -
Hellaswag 74.6 (0) 75.5(0) 51.5 (0) - - - 68.5 (-)
ARC-C 88.5(0) 82.4 (0) 57.3 (0) - - - 49.4 (-)
Mathematics & Science
GSM8K 78.6(4) 78.3 (4) 29.0 (4) 85.7 (0) 91.6 (0) 76.7 (0) 63.0 (0)
Math 26.6 (0) 29.6(0) 3.8 (0) 52.9 (0) 75.5 (0) 44.3 (0) 15.8 (0)
GPQA 31.8 (5) 31.9(5) 28.4 (5) 34.3 (0) 36.4 (0) 32.8 (0) -
Code
HumanEval 47.6 (0) 59.8(0) 16.5 (0) 79.9 (0) 84.8 (0) 68.9 (0) 48.2 (-)
MBPP 34.2 (4) 57.6(4) 20.6 (4) 67.2 (0) 79.2 (0) 74.9 (0) 35.2 (-)
```

```
Table 3.Comparison in the Poem Completion Task.
```
```
Forward Reversal
GPT-4o (2024-08-06) 82.7 34.
Qwen2.5 7B Instruct 75.9 38.
LLaDA 8B Instruct 48.8 42.
```
sive instruction-following abilities, as detailed in Sec. 3.4.
We leave RL-based alignment for future work.

Overall, despite the lack of data transparency, we have made
every effort to adopt standardized procedures and introduce
diverse tasks, we believe they sufficiently demonstrate the
extraordinary capabilities of LLaDA, which is the only com-
petitive non-autoregressive model to our knowledge.

3.3. Reversal Reasoning and Analyses

To quantify the reversal reasoning ability of mod-
els (Berglund et al., 2023), we follow the protocol estab-
lished in Allen-Zhu & Li (2023). Specifically, we construct
a dataset of 496 famous Chinese poem sentence pairs. Given
a sentence from a poem, models are tasked with generating
the subsequent line (forward) or the preceding line (rever-
sal) without additional fine-tuning. Examples can be found
in Appendix B.8. This setting provides a straightforward
and more realistic evaluation compared to previous stud-
ies (Nie et al., 2024; Kitouni et al., 2024).

As shown in Tab. 3, LLaDA effectively addresses therever-
sal curse(Berglund et al., 2023), demonstrating consistent
zero-shot performance across both forward and reversal
tasks. In contrast, both Qwen 2.5 and GPT-4o exhibit a
significant gap between the two. The results on forward
generation confirm that both ARMs are strong, benefiting
from significantly larger datasets and greater computational
resources than LLaDA. However, LLaDA outperforms both
by a large margin in the reversal task.

We emphasize that we did not design anything special for
reversal tasks. Intuitively, LLaDA treats tokens uniformly
without inductive bias, leading to balanced performance.
See more details in Appendix A.2.

We also analyze the effect of remasking strategies and sam-
pling steps, as detailed in Appendix B.3 and Appendix B.6.

3.4. Case Studies

We present samples generated by LLaDA 8B Instruct in
Tab. 4, showcasing its instruction-following capabilities.
First, the table illustrates LLaDA’s ability to generate co-
herent, fluent, and extended text in a non-autoregressive
manner. Second, it highlights the model’s multi-turn dia-
logue capability, effectively retaining conversation history

```
and producing contextually appropriate responses across
multiple languages. Suchchatcapabilities of LLaDA are
impressive, as it departs from conventional ARMs for the
first time, to the best of our knowledge. See more case
studies on remasking and reasoning tasks in Appendix B.9.
```
## 4. Related Work

```
Diffusion models (Sohl-Dickstein et al., 2015; Ho et al.,
2020; Song et al., 2020) have excelled in visual domains but
remain unverified for LLMs despite extensive efforts.
A simple approach is to continuousize text data and apply
diffusion models directly (Li et al., 2022; Gong et al., 2022;
Han et al., 2022; Strudel et al., 2022; Chen et al., 2022;
Dieleman et al., 2022; Richemond et al., 2022; Wu et al.,
2023; Mahabadi et al., 2024; Ye et al., 2023b). Alternatively,
some methods model continuous parameters of discrete dis-
tributions instead (Lou & Ermon, 2023; Graves et al., 2023;
Lin et al., 2023; Xue et al., 2024). However, scalability
remains a challenge, as a 1B parameter model requires 64
times the computeof an ARM to achieve comparable per-
formance (Gulrajani & Hashimoto, 2024).
Another approach replaces continuous diffusion with dis-
crete processes featuring new forward and reverse dy-
namics (Austin et al., 2021a), leading to numerous vari-
ants (Hoogeboom et al., 2021b;a; He et al., 2022; Campbell
et al., 2022; Meng et al., 2022; Reid et al., 2022; Sun et al.,
2022; Kitouni et al., 2023; Zheng et al., 2023; Chen et al.,
2023; Ye et al., 2023a; Gat et al., 2024; Zheng et al., 2024;
Sahoo et al., 2024; Shi et al., 2024). Notably, Lou et al.
(2023) showed that masked diffusion, as a special case of
discrete diffusion, achieves perplexity comparable to or sur-
passing ARMs at GPT-2 scale. Ou et al. (2024) established
fundamental theoretical results, which motivated our model
design, training, and inference (see Appendix A). Nie et al.
(2024) explored how MDM can be leveraged for language
tasks such as question answering at GPT-2 scale. Gong et al.
(2024) fine-tune ARMs in the MDM formulation. However,
improvements are confined to certain metrics, and it remains
unclear whether this approach can yield a foundation model
comparable to strong LLMs under a comprehensive evalua-
tion.
In comparison, this study scales MDM to an unprecedented
size of 8B parameters from scratch, achieving performance
comparable to leading LLMs such as LLaMA 3.
Additionally, a parallel line of work on image genera-
tion (Chang et al., 2022; 2023) aligns well with the appli-
cation of MDMs to text data. Moreover, MDMs have also
shown promise in domains such as protein generation (Wang
et al., 2024b;c), where they have achieved promising results.
Notably, Kou et al. (2024); Xu et al. (2025) demonstrate the
potential of using distillation to accelerate MDMs sampling,
```

Table 4.Visualization of the Sampling Process and a Generated Multi-round Dialogue.In the response of LLaDA, darker colors
indicate tokens predicted in the later stages of sampling, while lighter colors correspond to earlier predictions.

```
Sampling Process
User Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour.
How many kilometers can she run in 8 hours?
LLaDA
```
```
Multi-round Dialogue Case
User Could you please share with me what the initial two lines of the renowned poem ‘The
Road Not Taken’?
LLaDA Certainly! The initial two lines of the renowned poem “The Road Not Taken” by Robert
Frost are: “Two roads diverged in a yellow wood, And sorry I could not travel both.” These
lines set the stage for the poem, introducing the idea of a speaker facing a choice between
two paths.
User Please help me translate into Chinese.
LLaDA Sure! The first two lines of “The Road Not Taken” by Robert Frost can be translated into
Chinese as: “两条路分岔在黄色的树林中，遗憾我不能同时走”
User Please translate into German.
LLaDA Sure! The first two lines of “The Road Not Taken” by Robert Frost can be translated into
German as: “Zwei Wege trennten sich im gelben Wald, und leider konnte ich nicht beide
tun.”
User Please also write a poem about life choices. I require 5 sentences, and the first word of
each sentence starts with C as the first letter.
LLaDA Certainly! Here’s a poem about life choices, with each sentence starting with the letter “C”:
Choices cascade through life’s journey, Creating connections with each decision, Careful
consideration leads to better paths, Courageous choices shape our destiny, Choosing wisely
guides us through the years.
```
further enhancing their efficiency.

## 5. Conclusion and Discussion

```
In the middle of difficulty lies opportunity.
placeholder,placeholder,placeh—Albert Einstein
```
We introduceLLaDA, a principled and previously unex-
plored approach to large language modeling based on dif-
fusion models. LLaDA demonstrates strong capabilities in
scalability, in-context learning, and instruction-following,
achieving performance comparable to strong LLMs. In addi-
tion, LLaDA offers unique advantages such as bidirectional
modeling and enhanced robustness, effectively addressing
several inherent limitations of existing LLMs. Our findings
not onlyestablish diffusion models as a viable and promis-
ing alternativebut alsochallenge the prevailing assumption
that these essential capabilities are inherently tied to ARMs.

While promising, the full potential of diffusion models re-

```
mains to be fully explored. Several limitations of this work
present significant opportunities for future research.
Due to computational constraints, direct comparisons be-
tween LLaDA and ARMs—such as training on identical
datasets—were restricted to a computational budget of less
than 1023 FLOPs. To allocate resources for training the
largest possible LLaDA model and showcasing its poten-
tial, we were unable to scale the ARM baseline to the same
extent. Moreover, no specialized attention mechanisms
or position embeddings were designed for LLaDA, nor
were any system-level architectural optimizations applied.
On the inference side, our exploration of guidance mecha-
nisms (Dhariwal & Nichol, 2021; Ho & Salimans, 2022) re-
mains preliminary, and LLaDA currently exhibits sensitivity
to inference hyperparameters. Furthermore, LLaDA has yet
to undergo alignment with reinforcement learning (Ouyang
et al., 2022; Rafailov et al., 2024), which is crucial for im-
proving its performance and alignment with human intent.
```

Looking forward, the scale of LLaDA is still smaller than
leading counterparts (Achiam et al., 2023; Dubey et al.,
2024; Google, 2024; Anthropic, 2024; Yang et al., 2024;
Liu et al., 2024), highlighting the need for further scaling
to fully assess its capabilities. In addition, LLaDA’s abil-
ity to handle multi-modal data remains unexplored. The
impact of LLaDA on prompt tuning techniques (Wei et al.,
2022) and its integration into agent-based systems (Park
et al., 2023; Wang et al., 2024a) is yet to be fully under-
stood. Finally, a systematic investigation into post-training
for LLaDA could facilitate the development of O1-like sys-
tems (OpenAI, 2024; Guo et al., 2025).

Algorithm 1Pre-training of LLaDA

Require: mask predictorpθ, data distributionpdata
1:repeat
2: x 0 ∼pdata,t∼U(0,1] # with a probability of 1%, the sequence length ofx 0 follows U[1,4096]
3: xt∼qt| 0 (xt|x 0 ) #qt| 0 is defined in Eq. (7)
4: CalculateL=−t∗^1 L

### PL

```
i=1^1 [x
i
t=M] logpθ(x
i
0 |xt) #Lis the sequence length ofx^0
5: Calculate∇θLand run optimizer.
6:untilConverged
7:Returnpθ
```
Algorithm 2Supervised Fine-Tuning of LLaDA

Require: mask predictorpθ, pair data distributionpdata
1:repeat
2: p 0 ,r 0 ∼pdata,t∼U(0,1] # please refer to Appendix B.1 for details on the SFT data processing.
3: rt∼qt| 0 (rt|r 0 ) #qt| 0 is defined in Eq. (7)
4: CalculateL=−t∗^1 L′

### PL′

```
i=1^1 [r
```
```
i
t=M] logpθ(r
i
0 |p^0 ,rt) #L
′is the sequence length ofr 0
5: Calculate∇θLand run optimizer.
6:untilConverged
7:Returnpθ
```
Algorithm 3Conditional Log-likelihood Evaluation of LLaDA

Require: mask predictorpθ, promptp 0 , responser 0 , the number of Monte Carlo estimationsnmc
1:loglikelihood= 0
2:fori← 1 tonmcdo
3: l∼{ 1 , 2 ,...,L} #Lis the sequence length ofr 0
4: Obtainrlby uniformly samplingltokens fromr 0 without replacement for masking
5: loglikelihood=loglikelihood+Ll

### PL

```
i=1^1 [r
```
```
i
l=M] logpθ(r
i
0 |p^0 ,rl)
6:end for
7:loglikelihood=loglikelihood/nmc
8:Returnloglikelihood
```
## A. Formulation of Masked Diffusion Models

A.1. Training

Masked diffusion models (MDMs) (Austin et al., 2021a; Lou et al., 2023; Ou et al., 2024) define the model distribution
pθ(x 0 )in a manner distinct from autoregressive models.

These models introduce a forward process{xt}indexed by a timet∈[0,1]. This process gradually and independently
masks all tokens in the sequencex 0. At timet= 0, the data pointx 0 is fully observed with no masks, while fort∈(0,1],
xtrepresents latent variables with varying mask ratios in expectation.

Formally, the conditional distribution ofxtgivenx 0 is defined by a fully factorized form:

```
qt| 0 (xt|x 0 ) =
```
### YL

```
i=
```
```
qt| 0 (xit|xi 0 ), (7)
```
where the conditional distribution for each token is given by:

```
qt| 0 (xit|xi 0 ) =
```
### (

```
1 −t, xit=xi 0 ,
t, xit=M.
```
### (8)

Here,Mdenotes the mask token. Intuitively, each token either remains unchanged or is masked, with the probability of


Algorithm 4Reverse Process of LLaDA

Require: mask predictorpθ, promptp 0 , answer lengthL, sampling stepsN
1:Setr 1 is a fully masked sequence of lengthL.
2:fort← 1 down toN^1 stepN^1 do
3: s=t−N^1
4: r 0 = arg maxr 0 pθ(r 0 |p 0 ,rt) # we employ greedy sampling when predicting masked tokens
5: fori← 1 toLdo
6: ifrt̸=Mthen
7: ri 0 =rit
8: else
9: With probabilityst,ri 0 is set to M
10: end if
11: end for
12: rs=r 0
13:end for
14:Returnr 0

being masked increasing linearly astprogresses from 0 to 1. Att= 1, all tokens are guaranteed to be masked, meaning that
x 1 follows a Dirac distribution concentrated on a sequence of fully masked tokens. Notably, the linear masking probability
is analogous to but distinct from, the noise schedule in continuous diffusion models (Sohl-Dickstein et al., 2015; Ho et al.,
2020; Song et al., 2020). This linearity is motivated by the assumption that the information in the text is proportional to the
number of tokens on average, making it reasonable to lose information linearly during the forward process.

The forward process is not only reversible but also corresponds to a reverse process that is fully factorized across all
tokens (Austin et al., 2021a). The reverse process, from timet= 1to 0 , generates new data from sequences of fully masked
tokens. The conditional distribution for the reverse process, for 0 ≤s < t≤ 1 , is factorized as:

```
qs|t(xs|xt) =
```
### YL

```
i=
```
```
qs|t(xis|xt), (9)
```
where the conditional distribution for each token is:

```
qs|t(xis|xt) =
```
### 

### 

### 

### 

### 

```
1 , xit̸=M, xis=xit,
s
t, x
```
```
i
t=M, x
i
s=M,
t−s
t q^0 |t(x
```
```
i
s|xt), x
i
t=M, x
i
s̸=M,
0 , otherwise.
```
### (10)

Thus, the key function to estimate is the conditional distributionq 0 |t(xis|xt), which predicts the original token if it is masked
in the inputxt. This is analogous to thedata predictionform in continuous diffusion models.

As proven in (Ou et al., 2024), an equivalent yettime-freeparameterization can be derived as:

```
q 0 |t(xis|xt) =pdata(xi 0 |xUMt ), ∀isuch thatxit=M, (11)
```
wherexUMt denotes the collection of unmasked tokens inxt, which is identical to the corresponding tokens in the original
datax 0 since unmasked tokens are solely determined byx 0 and are independent of timet. Intuitively, this implies that
estimating the data prediction function is equivalent to estimating the conditional distributions on clean data, which is
time-invariant. Consequently, the timetneed not be provided as input to the parametric model.

Although the development of masked diffusion is nontrivial, the implementation is straightforward. We first introduce the
mask predictor, a parametric modelpθ(·|xt)(e.g., a Transformer without a causal mask), which takesxtfor anytas input
and predict all masked tokens simultaneously. Then, we define the model distributionpθ(x 0 )as follows: starting withx 1 as
a sequence of fully masked tokens, we simulate an approximate reverse process parameterized bypθ(·|xt)fromt= 1to 0.
The marginal distribution induced att= 0then represents the model distributionpθ(x 0 ).


Algorithm 5Low-confidence Remasking Strategy of LLaDA

Require: mask predictorpθ, promptp 0 , answer lengthL, sampling stepsN
1:Setr 1 is a fully masked sequence of lengthL.
2:fort← 1 down toN^1 stepN^1 do
3: s=t−N^1
4: fori← 1 toLdo
5: ifrti̸=Mthen
6: ri 0 =rit,ci= 1
7: else
8: ri 0 = arg maxr 0 ipθ(r 0 i|p 0 ,rt)
9: ci=pθ(ri 0 |p 0 ,rt)ri 0
10: end if
11: end for
12: nun=⌊L(1−s)⌋ # the number of unmasked tokens isnunin timesteps
13: fori← 1 toLdo
14: ifci∈Lowest−nun

### 