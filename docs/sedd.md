```
Aaron Lou^1 Chenlin Meng1 2 Stefano Ermon^1
```
## Abstract

```
Despite their groundbreaking performance for
many generative modeling tasks, diffusion mod-
els have fallen short on discrete data domains
such as natural language. Crucially, standard dif-
fusion models rely on the well-established the-
ory of score matching, but efforts to generalize
this to discrete structures have not yielded the
same empirical gains. In this work, we bridge
this gap by proposing score entropy, a novel loss
that naturally extends score matching to discrete
spaces, integrates seamlessly to build discrete dif-
fusion models, and significantly boosts perfor-
mance. Experimentally, we test our Score Entropy
Discrete Diffusion models (SEDD) on standard
language modeling tasks. For comparable model
sizes, SEDD beats existing language diffusion
paradigms (reducing perplexity by 25 - 75 %) and
is competitive with autoregressive models, in par-
ticular outperforming GPT-2. Furthermore, com-
pared to autoregressive mdoels, SEDD generates
faithful text without requiring distribution anneal-
ing techniques like temperature scaling (around 6 -
8 ×better generative perplexity than un-annealed
GPT-2), can trade compute and quality (similar
quality with 32 ×fewer network evaluations), and
enables controllable infilling (matching nucleus
sampling quality while enabling other strategies
besides left to right prompting).
```
## 1. Introduction

```
Many recent advances in deep learning have centered around
generative modeling. Here, a model learns how to generate
novel samples from unstructured data. With the powerful
capabilities of modern neural networks, these “generative
AI” systems have developed unparalleled capabilities, such
as creating images given only text (Ramesh et al., 2022) and
answering complex questions (Brown et al., 2020).
```
(^1) Stanford University (^2) Pika Labs. Correspondence to: Aaron
Lou<aaronlou@stanford.edu>.
Proceedings of the 41 stInternational Conference on Machine
Learning, Vienna, Austria. PMLR 235, 2024. Copyright 2024 by
the author(s).
The crucial part for any deep generative model is the prob-
abilistic modeling technique. For discrete data such as
natural language, autoregressive modeling (Yule, 1971)–
arguably the simplest modeling type since it derives from
the probabilistic chain rule–has remained the only compet-
itive method for decades. Although modern autoregres-
sive transformers have produced stunning results (Vaswani
et al., 2017; Radford et al., 2019), there are limits. For ex-
ample, the sequential sampling of tokens is slow, hard to
control, and often degrades without distribution annealing
techniques like nucleus sampling (Holtzman et al., 2019).
To alleviate these issues, researchers have sought alternative
approaches to generating text data. In particular, inspired
by their success in the image domain, many works have
extended diffusion models (Sohl-Dickstein et al., 2015; Ho
et al., 2020; Song et al., 2021c) to language domains (Li
et al., 2022; Austin et al., 2021). Yet, despite considerable
effort, no such approach yet rivals autoregressive modeling,
as they are not competitive on likelihoods, are slower to sam-
ple from, and do not generate comparable samples without
resorting to heavy annealing and empirical alterations.
In our work, we challenge the longstanding dominance of
autoregressive models by introducing Score Entropy Dis-
crete Diffusion models (SEDD). SEDD parameterizes a
reverse discrete diffusion process using the ratios of the
data distribution. These are learned using score entropy, a
novel loss that is analogous to score matching for standard
diffusion models (Hyv ̈arinen, 2005; Song & Ermon, 2019)
and results in several empirical benefits:
1.On core language modeling tasks, SEDD outperforms
all existing language diffusion models (Li et al., 2022;
Austin et al., 2021; Gulrajani & Hashimoto, 2023; He
et al., 2022) by large margins and is competitive with
autoregressive models of the same size (beating GPT-
on its zero-shot perplexity tasks (Radford et al., 2019)).
2.SEDD generates high quality unconditional samples
and enables one to naturally trade off compute for qual-
ity. When measuring the generative perplexity (given
by large models) of unconditional and un-annealed
samples from similarly sized models, SEDD beats
We open source our code at github.com/louaaron/Score-
Entropy-Discrete-Diffusion

# arXiv:2310.16834v3 [stat.ML] 6 Jun 2024


```
GPT-2 by 6 - 8 ×and can match performance using 32 ×
fewer function evaluations.
```
```
3.By directly parameterizing probability ratios, SEDD
is highly controllable. In particular, one can prompt
SEDD from arbitrary positions without specialized
training. For both standard (left to right) and infill-
ing, SEDD outperforms language diffusion models
and is comparable with autoregressive models with
nucleus sampling (as measured by MAUVE score (Pil-
lutla et al., 2021)).
```
## 2. Preliminaries

2.1. Discrete Diffusion Processes

We will be modeling probability distributions over a finite
supportX={ 1 ,...,N}. As the support is discrete, note
that our probability distributions can be represented by prob-
ability mass vectorsp∈RNthat are positive and sum to
1. To define a discrete diffusion process, we evolve a fam-
ily of distributionspt∈RNaccording to the a continuous
time Markov process given by a linear ordinary differential
equation (Campbell et al., 2022; Anderson, 2012):

```
dpt
dt
```
```
=Qtpt p 0 ≈pdata (1)
```
Here,Qtare the diffusion matricesRN×Nand have non-
negative non-diagonal entries and columns which sum to
zero (so that the ratedpdttsums to 0 , meaningptdoes not
gain or lose total mass). Generally,Qtare simple (e.g.
a simple scalar factorQt=σ(t)Q) soptapproaches a
limiting distributionpbaseast→∞.

One can simulate this process by taking small∆tEuler
steps and randomly sampling the resulting transitions. In
particular, the samples are defined by transition densities
which come from the columns ofQt:

```
p(xt+∆t=y|xt=x) =δxy+Qt(y,x)∆t+O(∆t^2 )(2)
```
Finally, this process has a well known reversal (Kelly, 1980;
Sun et al., 2023) given by another diffusion matrixQt:

```
dpT−t
dt
```
```
=QT−tpT−t Qt(y,x) =
```
```
pt(y)
pt(x)
```
```
Qt(x,y)
```
```
Qt(x,x) =−
```
### X

```
y̸=x
```
```
Qt(y,x) (3)
```
This reverse process is analogous to the time reversal for typ-
ical diffusion processes onRn, with the ratiospptt((yx))(which
are collectively known as the concrete score (Meng et al.,
2022)) generalizing the typical score function∇xlogpt
(Song & Ermon, 2019)^1

(^1) The gradient operator for discrete structures is (up to some
2.2. Discrete Diffusion Models
The goal of a discrete diffusion model is to construct the
aforementioned reverse process by learning the ratiospptt((yx)).
Unlike the continuous diffusion case, which has settled
around (up to minor scaling variations) the theoretical frame-
work given by score matching (Hyvarinen, 2005), there cur- ̈
rently exist many competing methods for learning discrete
diffusion models. In particular, these tend to produce mixed
empirical results, which spurs the need for a reexamination.
Mean Prediction.Instead of directly parameterizing the
ratiospptt((yx)), Austin et al. (2021); Campbell et al. (2022)
instead follow a strategy of Ho et al. (2020) to learn the re-
verse densityp 0 |t. This actually recovers the ratiospptt((yx))in a
roundabout way (as shown in our Theorem 4.2), but comes
with several drawbacks. First, learningp 0 |tis inherently
harder since it is a density (as opposed to a general value).
Furthermore, the objective breaks down in continuous time
and must be approximated (Campbell et al., 2022). As a
result, this framework largely underperforms empirically.
Ratio Matching.Originally introduced in Hyv ̈arinen (2007)
and augmented in Sun et al. (2023), ratio matching learns
the marginal probabilities of each dimension with maximum
likelihood training. However, the resulting setup departs
from standard score matching and requires specialized and
expensive network architectures (Chen & Duvenaud, 2019).
As such, this tends to perform worse than mean prediction.
Concrete Score Matching.Meng et al. (2022) generalizes
the standard Fisher divergence in score matching, learning
sθ(x,t)≈
h
pt(y)
pt(x)
i
y̸=x
with concrete score matching:

### LCSM=

### 1

### 2

```
Ex∼pt
```
### 

### 

### X

```
y̸=x
```
### 

```
sθ(xt,t)y−
```
```
pt(y)
pt(x)
```
###  2

### 

###  (4)

```
Unfortunately, theℓ^2 loss is incompatible with the fact that
pt(y)
pt(x)must be positive. In particular, this does not suffi-
ciently penalize negative or zero values, leading to divergent
behavior. Although theoretically promising, Concrete Score
Matching struggles (as seen in Appendix D).
```
## 3. Score Entropy Discrete Diffusion Models

```
In this section, we introduce score entropy. Similar to con-
crete score matching, we learn the collected concrete score
sθ(x,t)≈
```
```
h
pt(y)
pt(x)
```
```
i
y̸=x
```
```
(sθ:X ×R→R|X|). We design
the score entropy loss to incorporate the fact that these ratios
are positive and evolve under a discrete diffusion.
scaling) defined for pairsx̸=yby∇f(xy) :=f(y)−f(x).
The score function would generalize to the normalized gradients
∇p(xy)
p(x) =
```
```
p(y)
p(x)−^1.
```

Definition 3.1. Thescore entropyLSEfor a distributionp,
weightswxy≥ 0 and a score networksθ(x)yis

Ex∼p

### 

### 

### X

```
y̸=x
```
```
wxy
```
### 

```
sθ(x)y−
```
```
p(y)
p(x)
```
```
logsθ(x)y+K
```
### 

```
p(y)
p(x)
```
### 

### 

### 

### (5)

whereK(a) =a(loga−1)is a normalizing constant func-
tion that ensures thatLSE≥ 0.

Remark. Instead of building off of Fisher divergences,
score entropy builds off of the Bregman divergence

DF

### 

```
s(x)y,pp((yx))
```
### 

```
whenF=−logis the convex function.
```
As such, score entropy is non-negative, symmetric, and con-
vex. It also generalizes standard cross entropy to general
positive values (instead of simplex-valued probabilities), in-
spiring the name. The weightswxyare used primarily when
combining score entropy with diffusion models.

While this expression is more complex than the standard
score matching variants, it satisfies several desiderata for a
discrete diffusion training objective:

3.1. Score Entropy Properties

First,score entropy is a suitable loss function that recovers
the ground truth concrete score.

Proposition 3.2(Consistency of Score Entropy).Suppose
pis fully supported andwxy> 0. As the number of samples
and model capacity approaches∞, the optimalθ∗that

minimizes Equation 5 satisfiessθ∗(x)y=pp((yx))for all pairs
x,yFurthermore,LSEwill be 0 atθ∗.

Second,score entropy directly improves upon concrete
score matching by rescaling problematic gradients. For
the weightswxy= 1,∇sθ(x)yLSE=sθ(^1 x)y∇sθ(x)yLCSM,

so the gradient signals for each pair(x,y)are scaled by a
factor ofsθ(x)yas a normalization component. As such,
this forms a natural log-barrier which keeps oursθ≥ 0.

Third,similar to concrete score matching, score entropy
can be made computationally tractable by removing the
unknownpp((xy))term. There are two alternative forms, the
first of which is analogous to the implicit score matching
loss (Hyv ̈arinen, 2005):

Proposition 3.3(Implicit Score Entropy).LSEis equal up
to a constant independent ofθto theimplicit score entropy

```
LISE=Ex∼p
```
### 

### 

### X

```
y̸=x
```
```
wxysθ(x)y−wyxlogsθ(y)x
```
### 

###  (6)

Unfortunately, a Monte Carlo estimate would require sam-
pling anxand evaluatingsθ(y)xfor all othery. For high

```
dimensions, this is intractable, which means we have to
sampleyuniformly, but this introduces additional variance
analogous to that introduced by the Hutchinson trace esti-
mator (Hutchinson, 1989) for sliced score matching (Song
et al., 2019). As a result, implicit score entropy is impracti-
cal for large-scale tasks. Instead, we work a denoising score
matching loss (Vincent, 2011) variant of score entropy:
Theorem 3.4(Denoising Score Entropy).Supposepis a
perturbation of a base densityp 0 by a transition kernel
p(·|·), iep(x) =
```
### P

```
x 0 p(x|x^0 )p^0 (x^0 ). The score entropy
LSEis equivalent (up to a constant independent ofθ) to the
denoising score entropyLDSEis
```
```
x 0 E∼p 0
x∼p(·|x 0 )
```
### 

### 

### X

```
y̸=x
```
```
wxy
```
### 

```
sθ(x)y−
```
```
p(y|x 0 )
p(x|x 0 )
logsθ(x)y
```
### 

### 

### 

### (7)

```
LDSEis scalable since Monte Carlo sampling only requires
the evaluation of onesθ(x), which gives us allsθ(x)y, and
the variance introduced byx 0 is manageable. Additionally,
it is particularly appealing for discrete diffusion since the
intermediateptare all perturbations of the base density
p 0 (resulting from Equations 1, 2), enabling us to train
withLDSEusing the diffusion transition densitiespt| 0 (·|x 0 )
(which we can make tractable).
```
```
3.2. Likelihood Bound For Score Entropy Discrete
Diffusion
```
```
Fourth,the score entropy can be used to define an ELBO
for likelihood-based training and evaluation.
Definition 3.5. For our time dependent score network
sθ(·,t), the parameterized reverse matrix isQ
```
```
θ
( t(y,x) =
sθ(x,t)yQt(x,y) x̸=y
−
```
### P

```
z̸=xQ
```
```
θ
t(z,y) x=y
```
```
found by replacing the ground
```
```
truth scores in Equation 3. Our parameterized densitiespθt
thus satisfy the following differential equation:
```
```
dpθT−t
dt
```
### =Q

```
θ
T−tp
```
```
θ
T−t p
```
```
θ
T=pbase≈pT (8)
```
```
The log likelihood of data points can be bounded using an
ELBO based off of Dynkin’s formula (Hanson, 2007), which
was derived for discrete diffusion models in Campbell et al.
(2022). Interestingly, this takes the form of our denoising
score entropy loss weighted by the forward diffusion:
Theorem 3.6(Likelihood Training and Evaluation).For
the diffusion and forward probabilities defined above,
```
```
−logpθ 0 (x 0 )≤LDWDSE(x 0 ) +DKL(pT| 0 (·|x 0 )∥pbase)
(9)
whereLDWDSE(x 0 )is thediffusion weighted denoising
score entropyfor data pointx 0
```

### ZT

```
0
```
```
Ext∼pt| 0 (·|x 0 )
```
### X

```
y̸=xt
```
```
Qt(xt,y)
sθ(xt,t)y−
```
```
pt| 0 (y|x 0 )
pt| 0 (xt|x 0 )
```
```
logsθ(xt,t)y+K
```
### 

```
pt| 0 (y|x 0 )
pt| 0 (xt|x 0 )
```
### !

```
dt (10)
```
Crucially, this result allows us to directly models based on
their likelihood values (and the related perplexity scores),
the core metric for language modeling tasks. In particular,
we can train and evaluate an upper bound.

Remark. The DWDSE (and the implicit version) can be
derived from the general framework of Benton et al. (2022)
assuming a concrete score parameterization. In particu-
lar, the implicit version coincides with the likelihood loss
introduced in Campbell et al. (2022).

3.3. Practical Implementation

Fifth,score entropy can be scaled to high dimensional tasks.

In practice, our state factorizes into sequences X =
{ 1 ,...,n}dto form sequencesx = x^1 ...xd(e.g. se-
quences of tokens or image pixel values). As a general
Qtwould be of exponential size, we instead choose a sparse
structured matrix that perturbs tokens independently with
a matrixQtokt. In particular, the nonzero entries ofQtare
given by

```
Qt(x^1 ...xi...xd,x^1 ...xbi...xd) =Qtokt (xi,bxi) (11)
```
SinceLDWDSEweights the loss byQt(x,y), this token
level transitionQtrenders most ratios irrelevant. In particu-
lar, we only need to model all ratios between sequences with
Hamming distnace 1 , so we can build our score network
sθ(·,t) :{ 1 ,...,n}d→Rd×nas a seq-to-seq map:

```
(sθ(x^1 ...xi...xd,t))i,bxi≈
pt(x^1 ...bxi...xd)
pt(x^1 ...xi...xd)
```
### (12)

To fully computeLDWDSE, we just need to calculate the
forward transitionpseqt| 0 (·|·). Luckily, this decomposes as
each token is perturbed independently:

```
pseqt| 0 (bx|x) =
```
```
Yd
```
```
i=
```
```
ptokt| 0 (bxi|xi) (13)
```
For eachptokt| 0 (·|·), we employ the previously discussed strat-

egy and setQtokt =σ(t)Qtokfor a noise levelσand a fixed
transitionQtok. This avoids numerical integration as, if we
defineσ(t)as the cumulative noise

```
Rt
0 σ(s)ds, we have:
```
```
ptokt| 0 (·|x) =x-th column ofexp
```
### 