## BLOCK DIFFUSION: INTERPOLATING BETWEEN AU-

## TOREGRESSIVE ANDDIFFUSIONLANGUAGEMODELS

```
Marianne Arriola†∗ Aaron Kerem Gokaslan† Justin T. Chiu‡ Zhihan Yang†
```
```
Zhixuan Qi† Jiaqi Han¶ Subham Sekhar Sahoo† Volodymyr Kuleshov†
```
## ABSTRACT

```
Diffusion language models offer unique benefits over autoregressive models due
to their potential for parallelized generation and controllability, yet they lag in
likelihood modeling and are limited to fixed-length generation. In this work, we
introduce a class of block diffusion language models that interpolate between
discrete denoising diffusion and autoregressive models. Block diffusion overcomes
key limitations of both approaches by supporting flexible-length generation and
improving inference efficiency with KV caching and parallel token sampling. We
propose a recipe for building effective block diffusion models that includes an
efficient training algorithm, estimators of gradient variance, and data-driven noise
schedules to minimize the variance. Block diffusion sets a new state-of-the-art
performance among diffusion models on language modeling benchmarks and
enables generation of arbitrary-length sequences. We provide the code^1 , along
with the model weights and blog post on the project page:
https://m-arriola.com/bd3lms
```
## 1 Introduction

```
Diffusion models are widely used to generate images (Ho et al., 2020; Dhariwal & Nichol, 2021;
Sahoo et al., 2024b) and videos (Ho et al., 2022; Gupta et al., 2023), and are becoming increasingly
effective at generating discrete data such as text (Lou et al., 2024; Sahoo et al., 2024a) or biological
sequences (Avdeyev et al., 2023; Goel et al., 2024). Compared to autoregressive models, diffusion
models have the potential to accelerate generation and improve the controllability of model outputs
(Schiff et al., 2024; Nisonoff et al., 2024; Li et al., 2024; Sahoo et al., 2024c).
```
```
Discrete diffusion models currently face at least three limitations. First, in applications such as chat
systems, models must generate output sequences of arbitrary length (e.g., a response to a user’s
question). However, most recent diffusion architectures only generate fixed-length vectors (Austin
et al., 2021; Lou et al., 2024). Second, discrete diffusion uses bidirectional context during generation
and therefore cannot reuse previous computations with KV caching, which makes inference less
efficient (Israel et al., 2025). Third, the quality of discrete diffusion models, as measured by standard
metrics such as perplexity, lags behind autoregressive approaches and further limits their applicability
(Gulrajani & Hashimoto, 2024; Sahoo et al., 2024a).
```
```
This paper makes progress towards addressing these limitations by introducing Block Discrete
Denoising Diffusion Language Models (BD3-LMs), which interpolate between discrete diffusion
and autoregressive models. Specifically, block diffusion models (also known as semi-autoregressive
models) define an autoregressive probability distribution over blocks of discrete random variables
(Si et al., 2022; 2023); the conditional probability of a block given previous blocks is specified by a
discrete denoising diffusion model (Austin et al., 2021; Sahoo et al., 2024a).
```
```
Developing effective BD3-LMs involves two challenges. First, efficiently computing the training
objective for a block diffusion model is not possible using one standard forward pass of a neural
∗Correspondence to Marianne Arriola:marriola@cs.cornell.edu
†Cornell Tech, NY, USA. ¶Stanford University, CA, USA. ‡Cohere, NY, USA.
```
(^1) Code: https://github.com/kuleshov-group/bd3lms

# arXiv:2503.09573v3 [cs.LG] 17 May 2025


```
On September 17, 2016, we will be giving the release of
```
**Autoregression:**

```
There are three categories of the average
There are three categories of the average rate
There are three categories of the average rate of...
```
Generation steps

**Diffusion:**

```
the reusability will continue to the
```
```
Lower quality
```
```
Repeal the reusability cuts and the law will continue to reduce the
Repeal the reusability cuts and prove the law will continue to reduce the deficit.
```
**Block Diffusion (Ours):**

```
On September 17, we be
```
```
Parallelizable
```
```
On September 17, 2016, we will be giving the beta-release of the to our server testing ...
```
```
KV caching
```
```
KV caching
```
```
Arbitrary-length
```
```
Parallelizable
```
```
High quality Arbitrary-length
```
```
High quality
```
```
Fixed-length No KV caching
```
```
Not Parallelizable
```
```
Figure 1: Block diffusion sequentially generates blocks of tokens by performing diffusion within each
block and conditioning on previous blocks. By combining strength from autoregressive and diffusion
models, block diffusion overcomes the limitations of both approaches by supporting variable-length,
higher-quality generation and improving inference efficiency with KV caching and parallel sampling.
```
```
network and requires developing specialized algorithms. Second, training is hampered by the high
variance of the gradients of the diffusion objective, causing BD3-LMs to under-perform autoregression
even with a block size of one (when both models should be equivalent). We derive estimators of
gradient variance, and demonstrate that it is a key contributor to the gap in perplexity between
autoregression and diffusion. We then propose custom noise processes that minimize gradient
variance and make progress towards closing the perplexity gap.
```
```
We evaluate BD3-LMs on language modeling benchmarks, and demonstrate that they are able to
generate sequences of arbitrary length, including lengths that exceed their training context. In addition,
BD3-LMs achieve new state-of-the-art perplexities among discrete diffusion models. Compared to
alternative semi-autoregressive formulations that perform Gaussian diffusion over embeddings (Han
et al., 2022; 2023), our discrete approach features tractable likelihood estimates and yields samples
with improved generative perplexity using an order of magnitude fewer generation steps. In summary,
our work makes the following contributions:
```
- We introduce block discrete diffusion language models, which are autoregressive over
    blocks of tokens; conditionals over each block are based on discrete diffusion. Unlike prior
    diffusion models, block diffusion supports variable-length generation and KV caching.
- We introduce custom training algorithms for block diffusion models that enable efficiently
    leveraging the entire batch of tokens provided to the model.
- We identify gradient variance as a limiting factor of the performance of diffusion models,
    and we propose custom data-driven noise schedules that reduce gradient variance.
- Our results establish a new state-of-the-art perplexity for discrete diffusion and make
    progress toward closing the gap to autoregressive models.

## 2 Background: Language Modeling Paradigms

```
Notation We consider scalar discrete random variables withVcategories as ‘one-hot’ column
vectors in the spaceV={x∈ { 0 , 1 }V :
```
#### P

```
ixi= 1} ⊂∆
```
```
Vfor the simplex∆V. Let theV-th
category denote a special [MASK] token, wherem∈Vis its one-hot vector. We definex1:Las a
sequence ofLtokens, wherexℓ∈Vfor all tokensℓ∈{ 1 ,...,L},and useVLto denote the set of
all such sequences. Throughout the work, we simplify notation and refer to the token sequence as
xand an individual token asxℓ. Finally, letCat(·;p)be a categorical distribution with probability
p∈∆V.
```

### 2.1 Autoregressive Models

Consider a sequence ofLtokensx=

#### 

```
x^1 ,...,xL
```
#### 

drawn from the data distributionq(x). Autore-
gressive (AR) models define a factorized distribution of the form

```
logpθ(x) =
```
#### XL

```
ℓ=
```
```
logpθ(xℓ|x<ℓ), (1)
```
where eachpθ(xℓ|x<ℓ)is parameterized directly with a neural network. As a result, AR models
may be trained efficiently via next token prediction. However, AR models takeLsteps to generateL
tokens due to the sequential dependencies.

### 2.2 Discrete Denoising Diffusion Probabilistic Models

Diffusion models fit a modelpθ(x)to reverse a forward corruption processq(Sohl-Dickstein et al.,
2015; Ho et al., 2020; Sahoo et al., 2024b). This process starts with clean dataxand defines latent
variablesxt=

#### 

```
x^1 t,...,xLt
```
#### 

fort∈[0,1], which represent progressively noisier versions ofx. Given
a discretization intoTsteps, we defines(j) = (j−1)/Tandt(j) =j/T. For brevity, we dropj
fromt(j)ands(j)below; in general,sdenotes the time step precedingt.

The D3PM framework (Austin et al., 2021) definesqas a Markov forward process acting indepen-
dently on each tokenxℓ:q(xℓt|xℓs) = Cat(xℓt;Qtxℓs)whereQt∈RV×Vis the diffusion matrix.
The matrixQtcan model various transformations, including masking, random token changes, and
related word substitutions.

An ideal diffusion modelpθis the reverse of the processq. The D3PM framework definespθas

```
pθ(xs|xt) =
```
#### YL

```
ℓ=
```
```
pθ(xℓs|xt) =
```
#### X

```
x
```
#### "L

#### Y

```
ℓ=
```
```
q(xℓs|xℓt,xℓ)pθ(xℓ|xt)
```
#### #

#### , (2)

where the denoising base modelpθ(xℓ|xt)predicts clean tokenxℓgiven the noisy sequencext, and
the reverse posteriorq(xℓs|xℓt,x)is defined following Austin et al. (2021) in Suppl. B.3.

The diffusion modelpθis trained using variational inference. LetKL[·]denote the Kullback-Leibler
divergence. Then, the Negative ELBO (NELBO) is given by (Sohl-Dickstein et al., 2015):

L(x;θ) =Eq

```
"
−logpθ(x|xt(1)) +
```
```
XT
```
```
j=
```
```
DKL[q(xs(j)|xt(j),x)∥pθ(xs(j)|xt(j))] +DKL[q(xt(T)|x)∥pθ(xt(T))]
```
```
#
```
```
(3)
```
This formalism extends to continuous time via Markov chain (CTMC) theory and admits score-based
generalizations (Song & Ermon, 2019; Lou et al., 2024; Sun et al., 2022). Further simplifications
(Sahoo et al., 2024a; Shi et al., 2024; Ou et al., 2025) tighten the ELBO and enhance performance.

## 3 Block Diffusion Language Modeling

We explore a class of Block Discrete Denoising Diffusion Language Models (BD3-LMs) that
interpolate between autoregressive and diffusion models by defining an autoregressive distribution
over blocks of tokens and performing diffusion within each block. We provide a block diffusion
objective for maximum likelihood estimation and efficient training and sampling algorithms. We
show that for a block size of one, the diffusion objective suffers from high variance despite being
equivalent to the autoregressive likelihood in expectation. We identify high training variance as a
limitation of diffusion models and propose data-driven noise schedules that reduce the variance of the
gradient updates during training.

### 3.1 Block Diffusion Distributions and Model Architectures

We propose to combine the language modeling paradigms in Sec. 2 by autoregressively modeling
blocks of tokens and performing diffusion within each block. We group tokens inxintoBblocks of


lengthL′withB=L/L′(we assume thatBis an integer). We denote each blockx(b−1)L

′:bL′
from
token at positions(b−1)L′tobL′for blocksb∈ { 1 ,...,B}asxbfor simplicity. Our likelihood
factorizes over blocks as

```
logpθ(x) =
```
#### XB

```
b=
```
```
logpθ(xb|x<b), (4)
```
and eachpθ(xb|x<b)is modeled using discrete diffusion over a block ofL′tokens. Specifically, we
define a reverse diffusion process as in (2), but restricted to blockb:

```
pθ(xbs|xbt,x<b) =
```
#### X

```
xb
```
```
q(xbs|xbt,xb)pθ(xb|xbt,x<b) (5)
```
We obtain a principled learning objective by applying the NELBO in (3) to each term in (4) to obtain

```
−logpθ(x)≤LBD(x;θ) :=
```
#### XB

```
b=
```
```
L(xb,x<b;θ), (6)
```
where eachL(xb,x<b;θ)is an instance of (3) applied tologpθ(xb|x<b). Since the model is
conditioned onx<b, we make the dependence onx<b,θexplicit inL. We denote the sum of these
termsLBD(x;θ)(itself a valid NELBO).

Model Architecture Crucially, we parameterize theBbase denoiser modelspθ(xb|xbt,x<b)using
a single neural networkxθ. The neural networkxθoutputs not only the probabilitiespθ(xb|xbt,x<b),
but also computational artifacts for efficient training. This will enable us to compute the lossLBD(x;θ)
in parallel for allBblocks in a memory-efficient manner. Specifically, we parameterizexθusing a
transformer (Vaswani et al., 2017) with a block-causal attention mask. The transformerxθis applied
toLtokens, and tokens in blockbattend to tokens in blocks 1 tob. Whenxθis trained,xbθ(xbt,x<b)
yieldsL′predictions for denoised tokens in blockbbased on noisedxbtand cleanx<b.

In autoregressive generation, it is normal to cache keys and values for previously generated tokens
to avoid recomputing them at each step. Similarly, we useKb,Vbto denote the keys and values at
blockb, and we definexθto support these as input and output. The full signature ofxθis

```
xblogits,Kb,Vb←xbθ(xbt,K1:b−^1 ,V1:b−^1 ) :=xbθ(xbt,x<b), (7)
```
wherexblogitsare the predictions for the cleanxb, andKb,Vbis the key-value cache in the forward

pass ofxθ, andK1:b−^1 ,V1:b−^1 are keys and values cached on a forward pass ofxθoverx<b(hence
the inputsx<bandK1:b−^1 ,V1:b−^1 are equivalent).

### 3.2 Efficient Training and Sampling Algorithms

Ideally, we wish to compute the lossLBD(x;θ)in one forward pass ofxθ. However, observe that
denoisingxbtrequires a forward pass on this noisy input, while denoising the next blocks requires
runningxθon the clean versionxb. Thus every block has to go through the model at least twice.

Training Based on this observation, we propose a training algorithm with these minimal computa-
tional requirements (Alg. 1). Specifically, we precompute keys and valuesK1:B,V1:Bfor the full
sequencexin a first forward pass(∅,K1:B,V1:B)←xθ(x). We then compute denoised predictions
for all blocks usingxbθ(xbt,K1:b-^1 ,V1:b-^1 ). Each token passes throughxθtwice.

Vectorized Training Naively, we would compute the logits by applyingxbθ(xbt,K1:b-^1 ,V1:b-^1 )in a
loopBtimes. We propose a vectorized implementation that computesLBD(x;θ)in one forward pass
on the concatenationxnoisy⊕xof clean dataxwith noisy dataxnoisy=x^1 t 1 ⊕···⊕xBtBobtained

by applying a noise leveltbto each blockxb. We design an attention mask forxnoisy⊕xsuch that
noisy tokens attend to other noisy tokens in their block and to all clean tokens in preceding blocks
(see Suppl. B.6). Our method keeps the overhead of training BD3-LMs tractable and combines with
pretraining to further reduce costs.


Sampling We sample one block at a time, conditioned on previously sampled blocks (Alg 2).
We may use any sampling procedureSAMPLE(xbθ,K1:b-^1 ,V1:b-^1 )to sample from the conditional
distributionpθ(xbs|xbt,x<b), where the context conditioning is generated using cross-attention with
pre-computed keys and valuesK1:b−^1 ,V1:b−^1. Similar to AR models, caching the keys and values
saves computation instead of recalculating them when sampling a new block.

Notably, our block diffusion decoding algorithm enables us to sample sequences of arbitrary length,
whereas diffusion models are restricted to fixed-length generation. Further, our sampler admits parallel
generation within each block, whereas AR samplers are constrained to generate token-by-token.

Algorithm 1Block Diffusion Training

```
Input:datapointx, # of blocksB, forward
noise processqt(·|x), modelxθ, lossLBD
repeat
Samplet 1 ,...,tB∼U[0,1]
∀b∈{ 1 ,...,B}:xbtb∼qtb(·|xb)
∅,K1:B,V1:B←xθ(x) ▷KV cache
∀b:xblogit,∅,∅←xbθ(xbtb,K1:b-^1 ,V1:b-^1 )
Letxlogit←x^1 logit⊕···⊕xBlogit
Take gradient step on∇θLBD(xlogit;θ)
untilconverged
```
```
Algorithm 2Block Diffusion Sampling
Input:# blocksB, modelxθ, diffusion sam-
pling algorithm SAMPLE
x,K,V←∅ ▷output & KV cache
forb= 1toBdo
xb←SAMPLE(xbθ,K1:b-^1 ,V1:b-^1 )
∅,Kb,Vb←xbθ(xb)
x←x1:b−^1 ⊕xb
(K,V)←(K1:b−^1 ⊕Kb,V1:b−^1 ⊕Vb)
end for
returnx
```
## 4 Understanding Likelihood Gaps Between Diffusion & AR Models

### 4.1 Masked BD3-LMs

The most effective diffusion language models leverage a masking noise process (Austin et al., 2021;
Lou et al., 2024; Sahoo et al., 2024a), where tokens are gradually replaced with a special mask token.
Here, we introduce masked BD3-LMs, a special class of block diffusion models based on the masked
diffusion language modeling framework (Sahoo et al., 2024a; Shi et al., 2024; Ou et al., 2025).

More formally, we adopt a per-token noise processq(xℓt|xℓ) =Cat(xℓt;αtxℓ+ (1−αt)m)for
tokensℓ∈ { 1 ,...,L}wheremis a one-hot encoding of the mask token, andαt ∈[0,1]is a
strictly decreasing function int, withα 0 = 1andα 1 = 0. We employ the linear schedule where the
probability of masking a token at timetis 1 −αt. We adopt the simplified objective from Sahoo et al.
(2024a); Shi et al. (2024); Ou et al. (2025) (the full derivation is provided in Suppl. B.3):

```
−logpθ(x)≤LBD(x;θ) :=
```
#### XB

```
b=
```
```
Et∼[0,1]Eq
```
```
αt′
1 −αt
```
```
logpθ(xb|xbt,x<b) (8)
```
whereα′tis the instantaneous rate of change ofαtunder the continuous-time extension of (3) that
takesT→ ∞. The NELBO is tight forL′= 1but becomes a looser approximation of the true
negative log-likelihood forL′→L(see Suppl. B.5).

### 4.2 Case Study: Single Token Generation

```
Table 1: Test perplexities for single-
token generation (PPL;↓) across
16B tokens on LM1B.
```
```
PPL (↓)
AR 22.
+ random batch size 24.
BD3-LML′= 1 ≤25.
+ tuned schedule 22.
```
Our block diffusion parameterization (8) is equivalent in expec-
tation to the autoregressive NLL (1) in the limiting case where
L′= 1(see Suppl. B.4). Surprisingly, we find a two point
perplexity gap between our block diffusion model forL′= 1
and AR when training both models on the LM1B dataset.

Although the objectives are equivalent in expectation, we show
that the remaining perplexity gap is a result of high training
variance. Whereas AR is trained using the cross-entropy ofL
tokens, our block diffusion model forL′= 1only computes
the cross-entropy for masked tokensxℓt=m∀ℓ∈{ 1 ,...L}


(^3) 50k 100k 150k 200k 250k
3.
3.
3.
3.
4
Model BD3-LM (NELBO) BD3-LM (Tuned schedule) AR AR (random batch size)
Train Negative Log-Likelihood (NLL) for Single Token Generation on LM1B
Train steps
NLL
Figure 2: Train NLLs for modeling the per-token likelihood on LM1B. Models are trained on 16B
tokens. Training under the discrete diffusion NELBO, where half of the tokens in a batch are masked
on average, has similar training variance to an AR model with a random batch size.
so thatEt∼U[0,1]q(xℓt=m|xℓ) = 0. 5. Thus, training on the diffusion objective involves estimating
loss gradients with 2x fewer tokens and is responsible for higher training variance compared to AR.
To close the likelihood gap, we train a BD3-LM forL′= 1by designing the forward process to
fully mask tokens, i.e.q(xℓt=m|xℓ) = 1. Under this schedule, the diffusion objective becomes
equivalentto the AR objective (Suppl. B.4). In Table 1, we show that training under the block
diffusion objective yields the same perplexity as AR training. Empirically, we see that this reduces the
variance of the training loss in Figure 2. We verify that tuning the noise schedule reduces the variance
of the objective by measuringVarx,t[LBD(x;θ)]after training on 328M tokens: while training on the
NELBO results in a variance of 1.52, training under full masking reduces the variance to 0.11.

### 4.3 Diffusion Gap from High Variance Training

Next, we formally describe the issue of gradient variance in training diffusion models. Given our
empirical observations for single-token generation, we propose an estimator for gradient variance
that we use to minimize the variance of diffusion model training forL′≥ 1. While the NELBO is
invariant to the choice of noise schedule (Suppl. B.3), this invariance does not hold for our Monte
Carlo estimator of the loss used during training. As a result, the variance of the estimator and its
gradients are dependent on the schedule. First, we express the estimator of the NELBO with a batch

sizeK. We denote a batch of sequences asX=

#### 

```
x(1),x(2),...,x(K)
```
#### 

```
, with eachx(k)
```
iid
∼q(x). We
obtain the batch NELBO estimator below, wheret(k,b)is sampled in sequencekand blockb:

```
LBD(X;θ) :=l(X;θ) =
```
#### 1

#### K

#### XK

```
k=
```
#### XB

```
b=
```
```
α′t(k,b)
1 −αt(k,b)
```
```
logpθ
```
#### 

```
x(k),b|x(tk(k,b),b),x(k),<b
```
#### 

#### (9)

The variance of the gradient estimator overMbatches for each batchXm∀m∈{ 1 ,...,M}is:

```
VarX,t[∇θl(X;θ)]≈
```
#### 1

#### M− 1

#### XM

```
m=
```
```
∇θl(Xm;θ)−
```
#### 1

#### M

#### XM

```
m=
```
```
∇θl(Xm;θ)
```
```
2
```
```
2
```
#### (10)

## 5 Low-Variance Noise Schedules for BD3-LMs

### 5.1 Intuition: Avoid Extreme Mask Rates

We aim to identify schedules that minimize the variance of the gradient estimator and make training
most efficient. In a masked setting, we want to mask random numbers of tokens, so that the model


learns to undo varying levels of noise, which is important during sampling. However, if we mask
very few tokens, reconstructing them is easy and does not provide useful learning signal. If we mask
everything, the optimal reconstruction are the marginals of each token in the data distribution, which
is easy to learn, and again is not useful. These extreme masking rates lead to poor high-variance
gradients: we want to learn how to clip them via a simple and effective new class of schedules.

### 5.2 Clipped Schedules for Low-Variance Gradients

We propose a class of “clipped” noise schedules that sample mask rates 1 −αt∼ U[β,ω]for
0 ≤β,ω≤ 1. We argue that from the perspective of deriving Monte Carlo gradient estimates, these
schedules are equivalent to a continuous schedule where the mask probability is approximately 0
before the specified range such that 1 −α<β≈εand approximately 1 after the specified range
1 −α>ω≈ 1 −ε. Consequently,α′tis linear within the range:α′t≈ 1 /(β−ω).

### 5.3 Data-Driven Clipped Schedules Across Block Sizes

As the optimal mask rates may differ depending on the block sizeL′, we adaptively learn the schedule
during training. While Kingma et al. (2021) perform variance minimization by isolating a variance
term using their squared diffusion loss, this strategy is not directly applicable to our variance estimator
in Equation 10 since we seek to reduce variance across random batches in addition to randomtb.

Instead, we optimize parametersβ,ωto directly minimize training variance. To limit the computa-
tional burden of the optimization, we use the variance of the estimator of the diffusion ELBO as a
proxy for the gradient estimator to optimizeβ,ω:minβ,ωVarX,t[L(X;θ,β,ω)]. We perform a grid
search at regular intervals during training to find the optimalβ,ω(experimental details in Sec. 6).

In Table 2, we show that variance of the diffusion NELBO is correlated with test perplexity. Under a
range of “clipped” noise rate distributions, we find that there exists a unique distribution for each
block sizeL′∈{ 4 , 16 , 128 }that minimizes both the variance of the NELBO and the test perplexity.

Table 2: Perplexities (PPLs;↓) and variances of the NELBOVarX,t[LBD(X;θ)](Var. NELBO;↓).
Models are trained on LM1B using a linear schedule for 65B tokens, then finetuned for 10B tokens.

```
U[0,.5] U[. 3 ,.8] U[. 5 ,1] U[0,1]
L′ PPL Var. NELBO PPL Var. NELBO PPL Var. NELBO PPL Var. NELBO
128 31.72 1.03 31.78 1.35 31.92 1.83 31.78 3.
16 31.27 7.90 31.19 3.62 31.29 3.63 31.33 7.
4 29.23 32.68 29.37 10.39 29.16 8.28 29.23 23.
```
## 6 Experiments

```
Table 3: Test perplexities (PPL;↓) of mod-
els trained for 65B tokens on LM1B. Best
diffusion value is bolded.
PPL (↓)
Autoregressive
Transformer-X Base (Dai et al., 2019) 23.
Transformer (Sahoo et al., 2024a) 22.
Diffusion
D3PM (absorb) (Austin et al., 2021) ≤82.
SEDD (Lou et al., 2024) ≤32.
MDLM (Sahoo et al., 2024a) ≤31.
Block diffusion (Ours)
BD3-LMsL′= 16 ≤30.
L′= 8 ≤29.
L′= 4 ≤28.
```
We evaluate BD3-LMs across standard language
modeling benchmarks and demonstrate their ability
to generate arbitrary-length sequences uncondition-
ally. We pre-train a base BD3-LM using the maxi-
mum block sizeL′=Lfor 850K gradient steps and
fine-tune under varyingL′for 150K gradient steps
on the One Billion Words dataset (LM1B; Chelba
et al. (2014)) and OpenWebText (OWT; Gokaslan
et al. (2019)). Details on training and inference are
provided in Suppl C.

To reduce the variance of training on the diffusion
NELBO, we adaptively learn the range of masking
rates by optimizing parametersβ,ωas described in
Section 5.3. In practice, we do so using a grid search
during every validation epoch (after∼5K gradient


updates) to identifyβ,ω:minβ,ωVarX,t[L(X;θ,β,ω)]. During evaluation, we report likelihood
under uniformly sampled mask rates (8) as in Austin et al. (2021); Sahoo et al. (2024a).

### 6.1 Likelihood Evaluation

```
Table 4: Test perplexities (PPL;↓) on
OWT for models trained for 524B to-
kens. Best diffusion value is bolded.
```
```
PPL (↓)
AR (Sahoo et al., 2024a) 17.
SEDD (Lou et al., 2024) ≤24.
MDLM (Sahoo et al., 2024a) ≤22.
BD3-LMsL′= 16 ≤22.
L′= 8 ≤21.
L′= 4 ≤20.
```
On LM1B, BD3-LMs outperform all prior diffusion meth-
ods in Table 3. Compared to MDLM (Sahoo et al., 2024a),
BD3-LMs achieve up to 13% improvement in perplexity.
We observe a similar trend on OpenWebText in Table 4.

We also evaluate the ability of BD3-LMs to generalize
to unseen datasets in a zero-shot setting, following the
benchmark from Radford et al. (2019). We evaluate the
likelihood of models trained with OWT on datasets Penn
Tree Bank (PTB; (Marcus et al., 1993)), Wikitext (Merity
et al., 2016), LM1B, Lambada (Paperno et al., 2016), AG
News (Zhang et al., 2015), and Scientific Papers (Pubmed
and Arxiv subsets; (Cohan et al., 2018)). In Table 5, BD3-
LM achieves the best zero-shot perplexity on Pubmed,
surpassing AR, and the best perplexity among diffusion models on Wikitext, LM1B, and AG News.

Table 5: Zero-shot validation perplexities (↓) of models trained for 524B tokens on OWT. All
perplexities for diffusion models are upper bounds.

```
PTB Wikitext LM1B Lambada AG News Pubmed Arxiv
AR 81.07 25.32 51.14 52.13 52.11 48.59 41.
SEDD 96.33 35.98 68.14 48.93 67.82 45.39 40.
MDLM 90.96 33.22 64.94 48.29 62.78 43.13 37.
BD3-LML′= 4 96.81 31.31 60.88 50.03 61.67 42.52 39.
```
### 6.2 Sample Quality and Variable-Length Sequence Generation

```
Table 6: Generation length statistics
from sampling 500 documents from
models trained on OWT.
```
```
Median Max
# tokens # tokens
OWT train set 717 131K
AR 4008 131K
SEDD 1021 1024
BD3-LML′= 16 798 9982
```
One key drawback of many existing diffusion language mod-
els (e.g,. Austin et al. (2021); Lou et al. (2024)) is that they
cannot generate full-length sequences that are longer than
the length of the output context chosen at training time. The
OWT dataset is useful for examining this limitation, as it
contains many documents that are longer than the training
context length of 1024 tokens.

We record generation length statistics of 500 variable-length
samples in Table 6. We continue sampling tokens until an
end-of-sequence token [EOS] is generated or sample qual-
ity significantly degrades (as measured by sample entropy).
BD3-LMs generate sequences up to≈ 10 ×longer than those
of SEDD (Lou et al., 2024), which is restricted to the training context size.

We also examine the sample quality of BD3-LMs through quantitative and qualitative analyses. In
Table 7, we generate sequences of lengthsL= 1024, 2048 and measure their generative perplexity
under GPT2-Large. To sampleL= 2048tokens from MDLM, we use their block-wise decoding
technique (which does not feature block diffusion training as in BD3-LMs).

We also compare to SSD-LM (Han et al., 2022), an alternative block diffusion formulation. Unlike
our discrete diffusion framework, SSD-LM uses Gaussian diffusion and does not support likelihood
estimation. Further, BD3-LM adopts an efficient sampler from masked diffusion, where the number
of generation steps (NFEs) is upper-bounded byLsince tokens are never remasked (Sahoo et al.,
2024a; Ou et al., 2025). For SSD-LM, we compare sample quality usingT= 1K diffusion steps
per block, matching their experimental setting (yielding≥40K NFEs), andT= 25where NFEs are
comparable across methods.


```
Table 7: Generative perplexity (Gen. PPL;↓) and number of function evaluations (NFEs;↓) of 300
samples of lengthsL= 1024, 2048. All models are trained on OWT. AR, SEDD, MDLM, BD3-LMs
use 110M parameters and are trained on 524B tokens, while SSD-LM uses 400M parameters and is
pre-trained on 122B tokens. Best diffusion value is bolded. We provide further details in Suppl. C.5.
```
```
L= 1024 L= 2048
Model Gen. PPL NFEs Gen. PPL NFEs
AR 14.1 1K 13.2 2K
Diffusion
SEDD 52.0 1K – –
MDLM 46.8 1K 41.3 2K
Block Diffusion
SSD-LML′= 25 37.2 40K 35.3 80K
281.3 1K 281.9 2K
BD3-LMsL′= 16 33.4 1K 31.5 2K
L′= 8 30.4 1K 28.2 2K
L′= 4 25.7 1K 23.6 2K
```
```
BD3-LMs achieve the best generative perplexities compared to previous diffusion methods. Relative
to SSD-LM, our discrete approach yields samples with improved generative perplexity using an order
of magnitude fewer generation steps. We also qualitatively examine samples taken from BD3-LM
and baselines (AR, MDLM) trained on the OWT dataset; we report samples in Suppl. D. We observe
that BD3-LM samples have higher coherence than MDLM samples and approach the quality of AR.
```
### 6.3 Ablations

```
We assess the impact of the design choices in our proposed block diffusion recipes, namely 1)
selection of the noise schedule and 2) the efficiency improvement of the proposed training algorithm
relative to a naive implementation.
```
#### SELECTINGNOISESCHEDULES TOREDUCETRAININGVARIANCE

Compared to the linear schedule used in Lou et al. (2024); Sahoo et al. (2024a), training under
“clipped” noise schedules is the most effective for reducing the training variance which correlates with
test perplexity. In Table 8, the ideal “clipped” masking rates, which are optimized during training, are
specific to the block size and further motivate our optimization.

```
Table 8: Effect of the noise schedule on like-
lihood estimation. We finetune BD3-LMs
on 3B tokens from LM1B and evaluate on a
linear schedule. For clipped schedules, we
compare optimal clipping forL′= 4, 16.
Noise schedule PPL Var. NELBO
L’ = 4
Clipped
U[0. 45 , 0 .95] 29.21 6.
U[0. 3 , 0 .8] 29.38 10.
LinearU[0,1] 30.18 23.
Logarithmic 30.36 23.
Square root 31.41 26.
L’ = 16
Clipped
U[0. 45 , 0 .95] 31.42 3.
U[0. 3 , 0 .8] 31.12 3.
LinearU[0,1] 31.72 7.
Square 31.43 13.
Cosine 31.41 13.
```
```
Relative to other standard noise schedules (Chang et al.,
2022), “clipped” masking achieves the best perfor-
mance. As heavier masking is effective for the smaller
block sizeL′= 4, we compare with logarithmic and
square root schedules that also encourage heavy mask-
ing. As lighter masking is optimal forL′= 16, we
compare with square and cosine schedules.
```
#### EFFICIENCY OFTRAININGALGORITHM

```
In the BD3-LM training algorithm (Sec. 3.2), we com-
putexlogitusing two options. We may perform two for-
ward passes through the network (precomputing keys
and values for the full sequencex, then computing
denoised predictions), or combine these passes by con-
catenating the two inputs into the same attention kernel.
```
```
We find that a single forward pass is more efficient as
we reduce memory bandwidth bottlenecks by leverag-
ing efficient attention kernels (Dao et al., 2022; Dong
et al., 2024), see Suppl. B.7. Instead of paying the cost
```

of two passes through the network, we only pay the cost of a more expensive attention operation. Our
vectorized approach has 20-25% speed-up during training relative to performing two forward passes.

## 7 Discussion and Prior Work

Comparison to D3PM Block diffusion builds off D3PM (Austin et al., 2021) and applies it to each
autoregressive conditional. We improve over D3PM in three ways: (1) we extend D3PM beyond fixed
sequence lengths; (2) we study the perplexity gap of D3PM and AR models, identify gradient variance
as a contributor, and design variance-minimizing schedules; (3) we improve over the perplexity of
D3PM models. Our work applies to extensions of D3PM (He et al., 2022; Lou et al., 2024) including
ones in continuous time (Campbell et al., 2022; Sun et al., 2022).

Comparison to MDLM BD3-LMs further make use of the perplexity-enhancing improvements
in MDLM (Sahoo et al., 2024a; Shi et al., 2024; Ou et al., 2025). We also build upon MDLM: (1)
while Sahoo et al. (2024a) point out that their NELBO is invariant to the noise schedule, we show
that the noise schedule has a significant effect on gradient variance; (2) we push the state-of-the-art in
perplexity beyond MDLM. Note that our perplexity improvements stem not only from block diffusion,
but also from optimized schedules, and could enhance standard MDLM and D3PM models.

Comparison to Gaussian Diffusion Alternatively, one may perform diffusion over continuous
embeddings of discrete tokens (Li et al., 2022; Dieleman et al., 2022; Chen et al., 2022). This allows
using algorithms for continuous data (Song et al., 2020; Ho & Salimans, 2022), but yields worse
perplexity (Graves et al., 2023; Gulrajani & Hashimoto, 2024).

Comparison to Semi-Autoregressive Diffusion Han et al. (2022; 2023) introduced a block formu-
lation of Gaussian diffusion. BD3-LMs instead extend Austin et al. (2021), and feature: (1) tractable
likelihood estimates for principled evaluation; (2) faster generation, as our number of model calls is
bounded by the number of generated tokens, while SSD-LM performs orders of magnitude more calls;
(3) improved sample quality. AR-Diffusion (Wu et al., 2023) extends SSD-LM with a left-to-right
noise schedule; Chen et al. (2025); Ye et al. (2024) apply to decision traces and videos; Hao et al.
(2024); Kong et al. (2025) extend to latent reasoning. PARD (Zhao et al., 2024) applies discrete block
diffusion to graphs. In contrast, we (1) interpolate between AR/diffusion performance; (2) support
KV caching; (3) perform attention within noised blocks, whereas PARD injects new empty blocks.

Autoregressive diffusion models (Hoogeboom et al., 2021b;a) extend any-order AR models (AO-
ARMs; Uria et al. (2014)) to support parallel sampling. Zheng et al. (2024) prove equivalence
between MDLM and AO-ARM training. Further extensions of ARMs that compete with diffusion
include iterative editing (Gu et al., 2019), parallel and speculative decoding (Gu et al., 2017; Santilli
et al., 2023; Cai et al., 2024; Gloeckle et al., 2024), consistency training (Kou et al., 2024), guidance
(Sanchez et al., 2023), and cross-modal extensions (Liu et al., 2023; Tian et al., 2025).

Limitations Training BD3-LMs is more expensive than regular diffusion training. We propose a
vectorized algorithm that keeps training speed within <2x of diffusion training speed; in our experi-
ments, we also pre-train with a standard diffusion loss to further reduce the speed gap. Additionally,
BD3-LMs generate blocks sequentially, and hence may face the same speed and controllability con-
straints as AR especially when blocks are small. Their optimal block size is task specific (e.g., larger
for greater control). BD3-LMs are subject to inherent limitations of generative models, including
hallucinations (Achiam et al., 2023), copyright infringement (Gokaslan et al., 2024), controllability
(Schiff et al., 2024; Wang et al., 2023) and harmful outputs (Bai et al., 2022).

## 8 Conclusion

This work explores block diffusion and is motivated by two problems with existing discrete diffusion:
the need to generate arbitrary-length sequences and the perplexity gap to autoregressive models. We
introduce BD3-LMs, which represent a block-wise extension of the D3PM framework (Austin et al.,
2021), and leverage a specialized training algorithm and custom noise schedules that further improve
performance. We observe that in addition to being able to generate long-form documents, these
models also improve perplexity, setting a new state-of-the-art among discrete diffusion models.

#### 

```
x^1 ,...,xL
```
#### 

is factorized overBblocks, which we refer to asxfor simplicity,
drawn from the data distributionq(x). Specifically, we will factorize the likelihood overBblocks of
lengthL′, then perform diffusion in each block overTdiscretization steps. LetDKL[·]to denote the
Kullback-Leibler divergence,t,sbe shorthand fort(i) =i/Tands(i) = (i−1)/T∀i∈[1,T]. We
derive the NELBO as follows:

```
−logpθ(x) =−
```
#### XB

```
b=
```
```
logpθ(xb|x<b)
```
#### =−

#### XB

```
b=
```
```
logEq
```
```
pθ(xbt(1):t(T)|x<b)
q(xbt(1):t(T)|xb)
```
#### =−

#### XB

```
b=
```
```
logEq
```
```
pθ(xbt(T)|x<b)
```
#### QT

```
i=1pθ(x
```
```
b
s(i)|x
```
```
b
t(i),x
```
```
<b)
QT
i=1q(x
```
```
b
t(i)|x
```
```
b
s(i))
```
#### ≤

#### XB

```
b=
```
#### 

```
−Eqlogpθ(xb|xbt= 1
T
```
```
,x<b)
| {z }
Lrecons
+Et∈{ 2
T,...,TT−^1 ,^1 }
```
```
EqTDKL
```
#### 