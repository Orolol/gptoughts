arXiv:2506.21734v1 [cs.AI] 26 Jun 2025
Hierarchical Reasoning Model
Guan Wang1,†, Jin Li1, Yuhao Sun1, Xing Chen1, Changling Liu1,
Yue Wu1, Meng Lu1,†, Sen Song2,†, Yasin Abbasi Yadkori1,†
1Sapient Intelligence, Singapore
Abstract
Reasoning, the process of devising and executing complex goal-oriented action sequences,
remains a critical challenge in AI. Current large language models (LLMs) primarily employ
Chain-of-Thought (CoT) techniques, which suffer from brittle task decomposition, extensive
data requirements, and high latency. Inspired by the hierarchical and multi-timescale pro-
cessing in the human brain, we propose the Hierarchical Reasoning Model (HRM), a novel
recurrent architecture that attains significant computational depth while maintaining both train-
ing stability and efficiency. HRM executes sequential reasoning tasks in a single forward pass
without explicit supervision of the intermediate process, through two interdependent recurrent
modules: a high-level module responsible for slow, abstract planning, and a low-level mod-
ule handling rapid, detailed computations. With only 27 million parameters, HRM achieves
exceptional performance on complex reasoning tasks using only 1000 training samples. The
model operates without pre-training or CoT data, yet achieves nearly perfect performance on
challenging tasks including complex Sudoku puzzles and optimal path finding in large mazes.
Furthermore, HRM outperforms much larger models with significantly longer context windows
on the Abstraction and Reasoning Corpus (ARC), a key benchmark for measuring artificial
general intelligence capabilities. These results underscore HRM’s potential as a transformative
advancement toward universal computation and general-purpose reasoning systems.0
10
20
30
40
Accuracy %
15.8
21.0 21.2
34.5
40.3
Deepseek R1
Direct pred
Claude 3.7 8K
o3-mini-high
HRM
ARC-AGI-1
960 training examples
0
1
2
3
4
5
0.0
0.9
1.3
3.0
5.0
Direct pred
Claude 3.7 8K
Deepseek R1
o3-mini-high
HRM
ARC-AGI-2
1120 training examples
0
20
40
60
0.0 0.0 0.0 0.0
55.0
Direct pred
o3-mini-high
Claude 3.7 8K
Deepseek R1
HRM
Sudoku-Extreme (9x9)
1000 training examples
0
20
40
60
80
0.0 0.0 0.0 0.0
74.5
Direct pred
o3-mini-high
Claude 3.7 8K
Deepseek R1
HRM
Maze-Hard (30x30)
1000 training examples
Chain-of-thought, pretrained Direct prediction, small-sample learning
Figure 1: Left: HRM is inspired by hierarchical processing and temporal separation in the brain. It
has two recurrent networks operating at different timescales to collaboratively solve tasks. Right:
With only about 1000 training examples, the HRM (~27M parameters) surpasses state-of-the-art
CoT models on inductive benchmarks (ARC-AGI) and challenging symbolic tree-search puzzles
(Sudoku-Extreme, Maze-Hard) where CoT models failed completely. The HRM was randomly
initialized, and it solved the tasks directly from inputs without chain of thoughts.
2Tsinghua University † Corresponding author. Contact: research@sapient.inc
1
1 Introduction
Deep learning, as its name suggests, emerged from the idea of stacking more layers to achieve
increased representation power and improved performance1,2. However, despite the remarkable
success of large language models, their core architecture is paradoxically shallow3. This imposes
a fundamental constraint on their most sought-after capability: reasoning. The fixed depth of stan-
dard Transformers places them in computational complexity classes such as AC0 or T C0 4, prevent-
ing them from solving problems that require polynomial time5,6. LLMs are not Turing-complete
and thus they cannot, at least in a purely end-to-end manner, execute complex algorithmic rea-
soning that is necessary for deliberate planning or symbolic manipulation tasks7,8. For example,
our results on the Sudoku task show that increasing Transformer model depth can improve per-
formance,1 but performance remains far from optimal even with very deep models (see Figure 2),
which supports the conjectured limitations of the LLM scaling paradigm9.
The LLMs literature has relied largely on Chain-of-Thought (CoT) prompting for reasoning10.
CoT externalizes reasoning into token-level language by breaking down complex tasks into sim-
pler intermediate steps, sequentially generating text using a shallow model11. However, CoT for
reasoning is a crutch, not a satisfactory solution. It relies on brittle, human-defined decompositions
where a single misstep or a misorder of the steps can derail the reasoning process entirely12,13. This
dependency on explicit linguistic steps tethers reasoning to patterns at the token level. As a result,
CoT reasoning often requires significant amount of training data and generates large number of
tokens for complex reasoning tasks, resulting in slow response times. A more efficient approach is
needed to minimize these data requirements14.
Towards this goal, we explore “latent reasoning”, where the model conducts computations within
its internal hidden state space15,16. This aligns with the understanding that language is a tool for
human communication, not the substrate of thought itself17; the brain sustains lengthy, coherent
chains of reasoning with remarkable efficiency in a latent space, without constant translation back
to language. However, the power of latent reasoning is still fundamentally constrained by a model’s
effective computational depth. Naively stacking layers is notoriously difficult due to vanishing gra-
dients, which plague training stability and effectiveness1,18. Recurrent architectures, a natural al-
ternative for sequential tasks, often suffer from early convergence, rendering subsequent computa-
tional steps inert, and rely on the biologically implausible, computationally expensive and memory
intensive Backpropagation Through Time (BPTT) for training19.
The human brain provides a compelling blueprint for achieving the effective computational depth
that contemporary artificial models lack. It organizes computation hierarchically across corti-
cal regions operating at different timescales, enabling deep, multi-stage reasoning20,21,22. Recur-
rent feedback loops iteratively refine internal representations, allowing slow, higher-level areas to
guide—and fast, lower-level circuits to execute—subordinate processing while preserving global
coherence23,24,25. Notably, the brain achieves such depth without incurring the prohibitive credit-
assignment costs that typically hamper recurrent networks from backpropagation through time19,26.
Inspired by this hierarchical and multi-timescale biological architecture, we propose the Hierar-
chical Reasoning Model (HRM). HRM is designed to significantly increase the effective compu-
tational depth. It features two coupled recurrent modules: a high-level (H) module for abstract,
deliberate reasoning, and a low-level (L) module for fast, detailed computations. This structure
1Simply increasing the model width does not improve performance here.
2
27M 54M 109M 218M 436M 872M
Parameters
20
40
60
80
100
Accuracy %
Scaling Width - 8 layers fixed
Scaling Depth - 512 hidden size fixed
8 16 32 64 128 256 512
Depth / Transformer layers computed
20
40
60
80
100
Transformer
Recurrent Transformer
HRMFigure 2: The necessity of depth for complex reasoning. Left: On Sudoku-Extreme Full, which
require extensive tree-search and backtracking, increasing a Transformer’s width yields no perfor-
mance gain, while increasing depth is critical. Right: Standard architectures saturates, failing to
benefit from increased depth. HRM overcomes this fundamental limitation, effectively using its
computational depth to achieve near-perfect accuracy.
avoids the rapid convergence of standard recurrent models through a process we term “hierarchi-
cal convergence.” The slow-updating H-module advances only after the fast-updating L-module
has completed multiple computational steps and reached a local equilibrium, at which point the
L-module is reset to begin a new computational phase.
Furthermore, we proposed a one-step gradient approximation for training HRM, which offers im-
proved efficiency and eliminates the requirement for BPTT. This design maintains a constant mem-
ory footprint (O(1) compared to BPTT’s O(T ) for T timesteps) throughout the backpropagation
process, making it scalable and more biologically plausible.
Leveraging its enhanced effective depth, HRM excels at tasks that demand extensive search and
backtracking. Using only 1,000 input-output examples, without pre-training or CoT supervi-
sion, HRM learns to solve problems that are intractable for even the most advanced LLMs. For
example, it achieves near-perfect accuracy in complex Sudoku puzzles (Sudoku-Extreme Full) and
optimal pathfinding in 30x30 mazes, where state-of-the-art CoT methods completely fail (0% ac-
curacy). In the Abstraction and Reasoning Corpus (ARC) AGI Challenge27,28,29 - a benchmark
of inductive reasoning - HRM, trained from scratch with only the official dataset (~1000 exam-
ples), with only 27M parameters and a 30x30 grid context (900 tokens), achieves a performance
of 40.3%, which substantially surpasses leading CoT-based models like o3-mini-high (34.5%)
and Claude 3.7 8K context (21.2%), despite their considerably larger parameter sizes and con-
text lengths, as shown in Figure 1. This represents a promising direction toward the development
of next-generation AI reasoning systems with universal computational capabilities.
2 Hierarchical Reasoning Model
We present the HRM, inspired by three fundamental principles of neural computation observed in
the brain:
• Hierarchical processing: The brain processes information across a hierarchy of cortical ar-
eas. Higher-level areas integrate information over longer timescales and form abstract repre-
sentations, while lower-level areas handle more immediate, detailed sensory and motor process-
ing20,22,21.
3
• Temporal Separation: These hierarchical levels in the brain operate at distinct intrinsic timescales,
reflected in neural rhythms (e.g., slow theta waves, 4–8 Hz and fast gamma waves, 30–100
Hz)30,31. This separation allows for stable, high-level guidance of rapid, low-level computa-
tions32,33.
• Recurrent Connectivity: The brain features extensive recurrent connections. These feedback
loops enable iterative refinement, yielding more accurate and context-sensitive representations
at the cost of additional processing time. Additionally, the brain largely avoids the problematic
deep credit assignment problem associated with BPTT19.
The HRM model consists of four learnable components: an input network fI (·; θI ), a low-level re-
current module fL(·; θL), a high-level recurrent module fH (·; θH ), and an output network fO(·; θO).
The model’s dynamics unfold over N high-level cycles of T low-level timesteps each2. We index
the total timesteps of one forward pass by i = 1, . . . , N × T . The modules fL and fH each keep a
hidden state—zi
L for fL and zi
H for fH —which are initialized with the vectors z0
L and z0
H , respec-
tively.
The HRM maps an input vector x to an output prediction vector ˆy as follows. First, the input x is
projected into a working representation ˜x by the input network:
˜x = fI (x; θI ) .
At each timestep i, the L-module updates its state conditioned on its own previous state, the H-
module’s current state (which remains fixed throughout the cycle), and the input representation.
The H-module only updates once per cycle (i.e., every T timesteps) using the L-module’s final
state at the end of that cycle:
zi
L = fL
zi−1
L , zi−1
H , ˜x; θL
 ,
zi
H =
(
fH
zi−1
H , zi−1
L ; θH
 if i ≡ 0 (mod T ) ,
zi−1
H otherwise .
Finally, after N full cycles, a prediction ˆy is extracted from the hidden state of the H-module:
ˆy = fO(zN T
H ; θO) .
This entire N T -timestep process represents a single forward pass of the HRM. A halting mecha-
nism (detailed later in this section) determines whether the model should terminate, in which case
ˆy will be used as the final prediction, or continue with an additional forward pass.
Hierarchical convergence Although convergence is crucial for recurrent networks, standard RNNs
are fundamentally limited by their tendency to converge too early. As the hidden state settles toward
a fixed point, update magnitudes shrink, effectively stalling subsequent computation and capping
the network’s effective depth. To preserve computational power, we actually want convergence to
proceed very slowly–but engineering that gradual approach is difficult, since pushing convergence
too far edges the system toward instability.
2While inspired by temporal separation in the brain, our model’s ’high-level’ and ’low-level’ module are conceptual
abstractions and do not map directly to specific neural oscillation frequencies.
4
0 20 40 60
Step Index #
0
50
100
150
200
250
Forward residual HRM H
HRM L
0 20 40 60
Step Index #
0
50
100
150
200
250
Recurrent Neural Net
0 100 200
Layer Index #
0
50
100
150
200
250
Deep Neural Net
Principal Components Principal Components Principal Components
30
60
Step Index #
30
60
Step Index #
100
200
Layer Index #Figure 3: Comparison of forward residuals and PCA trajectories. HRM shows hierarchical conver-
gence: the H-module steadily converges, while the L-module repeatedly converges within cycles
before being reset by H, resulting in residual spikes. The recurrent neural network exhibits rapid
convergence with residuals quickly approaching zero. In contrast, the deep neural network experi-
ences vanishing gradients, with significant residuals primarily in the initial (input) and final layers.
HRM is explicitly designed to counteract this premature convergence through a process we term
hierarchical convergence. During each cycle, the L-module (an RNN) exhibits stable convergence
to a local equilibrium. This equilibrium, however, depends on the high-level state zH supplied
during that cycle. After completing the T steps, the H-module incorporates the sub-computation’s
outcome (the final state zL) and performs its own update. This zH update establishes a fresh context
for the L-module, essentially “restarting” its computational path and initiating a new convergence
phase toward a different local equilibrium.
This process allows the HRM to perform a sequence of distinct, stable, nested computations, where
the H-module directs the overall problem-solving strategy and the L-module executes the intensive
search or refinement required for each step. Although a standard RNN may approach convergence
within T iterations, the hierarchical convergence benefits from an enhanced effective depth of N T
steps. As empirically shown in Figure 3, this mechanism allows HRM both to maintain high
computational activity (forward residual) over many steps (in contrast to a standard RNN, whose
activity rapidly decays) and to enjoy stable convergence. This translates into better performance at
any computation depth, as illustrated in Figure 2.
Approximate gradient Recurrent models typically use BPTT to compute gradients. However,
BPTT requires storing the hidden states from the forward pass and then combining them with
gradients during the backward pass, which demands O(T ) memory for T timesteps. This heavy
memory burden forces smaller batch sizes and leads to poor GPU utilization, especially for large-
scale networks. Additionally, because retaining the full history trace through time is biologically
implausible, it is unlikely that the brain implements BPTT19.
Fortunately, if a recurrent neural network converges to a fixed point, we can avoid unrolling its state
sequence by applying backpropagation in a single step at that equilibrium point. Moreover, such a
mechanism could plausibly be implemented in the brain using only local learning rules34,35. Based
5
on this finding, we propose a one-step approximation of the HRM gradient–using the gradient of
the last state of each module and treating other states as constant. The gradient path is, therefore,
Output head → final state of the H-module → final state of the L-module → input embedding
The above method needs O(1) memory, does not require unrolling through time, and can be easily
implemented with an autograd framework such as PyTorch, as shown in Figure 4. Given that
each module only needs to back-propagate errors through its most recent local synaptic activity,
this approach aligns well with the perspective that cortical credit assignment relies on short-range,
temporally local mechanisms rather than on a global replay of activity patterns.
def hrm(z, x, N=2, T=2):
x = input_embedding(x)
zH, zL = z
with torch.no_grad():
for _i in range(N ∗ T − 1):
zL = L_net(zL, zH, x)
if (_i + 1) % T == 0:
zH = H_net(zH, zL)
# 1−step grad
zL = L_net(zL, zH, x)
zH = H_net(zH, zL)
return (zH, zL), output_head(zH)
# Deep Supervision
for x, y_true in train_dataloader:
z = z_init
for step in range(N_supervision):
z, y_hat = hrm(z, x)
loss = softmax_cross_entropy(y_hat, y)
z = z.detach()
loss.backward()
opt.step()
opt.zero_grad()
Figure 4: Top: Diagram of HRM with
approximate gradient. Bottom: Pseu-
docode of HRM with deep supervision
training in PyTorch.
The one-step gradient approximation is theoretically
grounded in the mathematics of Deep Equilibrium Mod-
els (DEQ)36 which employs the Implicit Function Theo-
rem (IFT) to bypass BPTT, as detailed next. Consider an
idealized HRM behavior where, during high-level cycle
k, the L-module repeatedly updates until its state zL con-
verges to a local fixed point z⋆
L. This fixed point, given
the current high-level state zk−1
H , can be expressed as
z⋆
L = fL(z⋆
L, zk−1
H , ˜x; θL) .
The H-module then performs a single update using this
converged L-state:
zk
H = fH (zk−1
H , z⋆
L; θH ) .
With a proper mapping F, the updates to the high-level
state can be written in a more compact form as zk
H =
F(zk−1
H ; ˜x, θ), where θ = (θI , θL), and the fixed-point
can be written as z⋆
H = F(z⋆
H ; ˜x, θ). Let JF = ∂F
∂zH be
the Jacobian of F, and assume that the matrix I − JF is
invertible at z⋆
H and that the mapping F is continuously
differentiable. The Implicit Function Theorem then al-
lows us to calculate the exact gradient of fixed point z⋆
H
with respect to the parameters θ without explicit back-
propagation:
∂z⋆
H
∂θ =

I − JF z⋆
H
−1 ∂F
∂θ z⋆
H
. (1)
Calculating the above gradient requires evaluating and inverting matrix (I − JF ) that can be com-
putationally expensive. Given the Neumann series expansion,
(I − JF )−1 = I + JF + J2
F + J3
F + . . . ,
the so-called 1-step gradient37 approximates the series by considering only its first term, i.e. (I −
JF )−1 ≈ I, and leads to the following approximation of Equation (1):
∂z∗
H
∂θH
≈ ∂fH
∂θH
, ∂z∗
H
∂θL
≈ ∂fH
∂z∗
L
· ∂z∗
L
∂θL
, ∂z∗
H
∂θI
≈ ∂fH
∂z∗
L
· ∂z∗
L
∂θI
. (2)
6
The gradients of the low-level fixed point, ∂z∗
L
∂θL and ∂z∗
L
∂θI , can also be approximated using another
application of the 1-step gradient:
∂z∗
L
∂θL
≈ ∂fL
∂θL
, ∂z∗
L
∂θI
≈ ∂fL
∂θI
. (3)
By substituting Equation (3) back into Equation (2), we arrive at the final simplified gradients.
Before defining our loss function, we must first introduce two key elements of our proposed
method: deep supervision and adaptive computational time.
Deep supervision Inspired by the principle that periodic neural oscillations regulate when learning
occurs in the brain38, we incorporate a deep supervision mechanism into HRM, as detailed next.
Given a data sample (x, y), we run multiple forward passes of the HRM model, each of which we
refer to as a segment. Let M denote the total number of segments executed before termination.
For each segment m ∈ {1, . . . , M }, let zm = (zmN T
H , zmN T
L ) represent the hidden state at the
conclusion of segment m, encompassing both high-level and low-level state components.
At each segment m, we apply a deep supervision step as follows:
1. Given the state zm−1 from the previous segment, compute the next state zm and its associated
output ˆym through a forward pass in the HRM model:
(zm, ˆym) ← HRM(zm−1, x; θ)
2. Compute the loss for the current segment:
Lm ← LOSS(ˆym, y)
3. Update parameters:
θ ← OPTIMIZERSTEP(θ, ∇θLm)
The crucial aspect of this procedure is that the hidden state zm is “detached” from the computa-
tion graph before being used as the input state for the next segment. Consequently, gradients from
segment m + 1 do not propagate back through segment m, effectively creating a 1-step approxi-
mation of the gradient of the recursive deep supervision process39,40. This approach provides more
frequent feedback to the H-module and serves as a regularization mechanism, demonstrating supe-
rior empirical performance and enhanced stability in deep equilibrium models when compared to
more complex, Jacobian-based regularization techniques39,41. Figure 4 shows pseudocode of deep
supervision training.
Adaptive computational time (ACT) The brain dynamically alternates between automatic think-
ing (“System 1”) and deliberate reasoning (“System 2”)42. Neuroscientific evidence shows that
these cognitive modes share overlapping neural circuits, particularly within regions such as the
prefrontal cortex and the default mode network43,44. This indicates that the brain dynamically mod-
ulates the “runtime” of these circuits according to task complexity and potential rewards45,46.
Inspired by the above mechanism, we incorporate an adaptive halting strategy into HRM that en-
ables “thinking, fast and slow”. This integration leverages deep supervision and uses the Q-learning
7
algorithm47 to adaptively determine the number of segments. A Q-head uses the final state of the
H-module to predict the Q-values ˆQm = ( ˆQm
halt, ˆQm
continue) of the “halt” and “continue” actions:
ˆQm = σ(θ⊤
QzmN T
H ) ,
where σ denotes the sigmoid function applied element-wise. The halt or continue action is chosen
using a randomized strategy as detailed next. Let Mmax denote the maximum number of segments
(a fixed hyperparameter) and Mmin denote the minimum number of segments (a random variable).
The value of Mmin is determined stochastically: with probability ε, it is sampled uniformly from the
set {2, · · · , Mmax} (to encourage longer thinking), and with probability 1 − ε, it is set to 1. The halt
action is selected under two conditions: when the segment count surpasses the maximum threshold
Mmax, or when the estimated halt value ˆQhalt exceeds the estimated continue value ˆQcontinue and the
segment count has reached at least the minimum threshold Mmin.
The Q-head is updated through a Q-learning algorithm, which is defined on the following episodic
Markov Decision Process (MDP). The state of the MDP at segment m is zm, and the action space
is {halt, continue}. Choosing the action “halt” terminates the episode and returns a binary reward
indicating prediction correctness, i.e., 1{ˆym = y}. Choosing “continue” yields a reward of 0 and
the state transitions to zm+1. Thus, the Q-learning targets for the two actions ˆGm = ( ˆGm
halt, ˆGm
continue)
are given by
ˆGm
halt = 1{ˆym = y} ,
ˆGm
continue =



ˆQm+1
halt , if m ≥ Nmax ,
max( ˆQm+1
halt , ˆQm+1
continue) , otherwise .
We can now define the loss function of our learning procedure. The overall loss for each supervision
segment combines both the Q-head loss and the sequence-to-sequence loss:
Lm
ACT = LOSS(ˆym, y) + BINARYCROSSENTROPY( ˆQm, ˆGm) .
Minimizing the above loss enables both accurate predictions and nearly optimal stopping decisions.
Selecting the “halt” action ends the supervision loop. In practice, sequences are processed in
batches, which can be easily handled by substituting any halted sample in the batch with a fresh
sample from the dataloader.
Figure 5 presents a performance comparison between two HRM variants: one incorporating ACT
and another employing a fixed computational step count equivalent to ACT’s Mmax parameter. It
shows that ACT effectively adapts its computational resources based on task complexity, achieving
significant computational savings with minimal impact on performance.
Inference-time scaling An effective neural model should exploit additional computational re-
sources during inference to enhance performance. As illustrated in Figure 5-(c), HRM seamlessly
achieves inference-time scaling by simply increasing the computational limit parameter, Mmax
without requiring further training or architectural modifications.
Additional compute is especially effective for tasks that demand deeper reasoning. On Sudoku—
a problem that often requires long-term planning—HRM exhibits strong inference-time scaling.
On the other hand, we find that extra computational resources yield minimal gains in ARC-AGI
challenge, as solutions generally require only a few transformations.
8
2 4 8
M (Fixed) or Mmax (ACT)
1
2
3
4
5
6
7
8
Mean Compute Steps
(a) ACT Compute Spent
Fixed M
ACT (Mmax limit)
2 4 8
M (Fixed) or Mmax (ACT)
82.5
85.0
87.5
90.0
92.5
95.0
97.5
100.0
Accuracy %
(b) ACT Performance
Fixed M
ACT (Mmax limit)
2 4 8 16
Inference Mmax
82.5
85.0
87.5
90.0
92.5
95.0
97.5
100.0
Accuracy %
(c) Inference-time scaling
Train Mmax = 2
Train Mmax = 4
Train Mmax = 8Figure 5: Effectiveness of Adaptive Computation Time (ACT) on the Sudoku-Extreme-Full. (a)
Mean compute steps used by models with ACT versus models with a fixed number of compute steps
(M ). ACT maintains a low and stable number of average compute steps even as the maximum limit
(Mmax) increases. (b) Accuracy comparison. The ACT model achieves performance comparable
to the fixed-compute model while utilizing substantially fewer computational steps on average. (c)
Inference-time scalability. Models trained with a specific Mmax can generalize to higher compu-
tational limits during inference, leading to improved accuracy. For example, a model trained with
Mmax = 8 continues to see accuracy gains when run with Mmax = 16 during inference.
Stability of Q-learning in ACT The deep Q-learning that underpins our ACT mechanism is
known to be prone to instability, often requiring stabilization techniques such as replay buffers
and target networks48, which are absent in our design. Our approach, however, achieves stability
through the intrinsic properties of our model and training procedure. Recent theoretical work by
Gallici et al. 49 shows that Q-learning can achieve convergence if network parameters are bounded,
weight decay is incorporated during training, and post-normalization layers are implemented. Our
model satisfies these conditions through its Post-Norm architecture that employs RMSNorm (a
layer normalization variant) and the AdamW optimizer. AdamW has been shown to solve an L∞-
constrained optimization problem, ensuring that model parameters remain bounded by 1/λ50.
Architectural details We employ a sequence-to-sequence architecture for HRM. Both input and
output are represented as token sequences: x = (x1, . . . , xl) and y = (y1, . . . , yl′ ) respectively.
The model includes an embedding layer fI that converts discrete tokens into vector representa-
tions, and an output head fO(z; θO) = softmax(θOz) that transforms hidden states into token prob-
ability distributions ˆy. For small-sample experiments, we replace softmax with stablemax51 to
improve generalization performance. The sequence-to-sequence loss is averaged over all tokens,
LOSS(ˆy, y) = 1
l′
Pl′
i=1 log p(yi), where p(yi) is the probability that distribution ˆyi assigns to token
yi. The initial hidden states z0 are initialized by sampling from a truncated normal distribution with
standard deviation of 1, truncation of 2, and kept fixed throughout training.
Both the low-level and high-level recurrent modules fL and fH are implemented using encoder-
only Transformer52 blocks with identical architectures and dimensions. These modules take mul-
tiple inputs, and we use straightforward element-wise addition to combine them, though more
sophisticated merging techniques such as gating mechanisms could potentially improve perfor-
mance and is left for future work. For all Transformer blocks in this work—including those in
the baseline models—we incorporate the enhancements found in modern LLMs (based on Llama53
architectures). These improvements include Rotary Positional Encoding54, Gated Linear Units55,
RMSNorm56, and the removal of bias terms from linear layers.
Furthermore, both HRM and recurrent Transformer models implement a Post-Norm architecture
9
(a) ARC-AGI
8 4 5 6
8 7
3 4
3 8 4 2
6 3 8
9 6
5
2 1
2 5 3 8
(b) Sudoku-Hard (c) Maze navigation (d) Sudoku-Extreme subset difficulty
7 8 4 1 2 5 9 6 3
2 6 1 3 8 9 7 4 5
3 5 9 6 4 7 8 1 2
5 3 8 4 9 6 1 2 7
4 1 6 2 7 3 5 9 8
9 7 2 8 5 1 4 3 6
6 9 3 5 1 8 2 7 4
8 4 7 9 6 2 3 5 1
1 2 5 7 3 4 6 8 9Figure 6: Left: Visualization of benchmark tasks. Right: Difficulty of Sudoku-Extreme examples.
with weights initialized via truncated LeCun Normal initialization57,58,59, while the scale and bias
parameters are excluded from RMSNorm. All parameters are optimized using the Adam-atan2 op-
timizer60, a scale-invariant variant of Adam61, combined with a constant learning rate that includes
linear warm-up.
3 Results
This section begins by describing the ARC-AGI, Sudoku, and Maze benchmarks, followed by an
overview of the baseline models and their results. Figure 6-(a,b,c) presents a visual representa-
tion of the three benchmark tasks, which are selected to evaluate various reasoning abilities in AI
models.
3.1 Benchmarks
ARC-AGI Challenge The ARC-AGI benchmark evaluates general fluid intelligence through IQ-
test-like puzzles that require inductive reasoning27. The initial version, ARC-AGI-1, presents chal-
lenges as input-output grid pairs that force AI systems to extract and generalize abstract rules from
just a few examples. Each task provides a few input–output example pairs (usually 2–3) and a test
input. An AI model has two attempts to produce the correct output grid. Although some believe
that mastering ARC-AGI would signal true artificial general intelligence, its primary purpose is
to expose the current roadblocks in AGI progress. In fact, both conventional deep learning meth-
ods and chain-of-thought techniques have faced significant challenges with ARC-AGI-1, primarily
because it requires the ability to generalize to entirely new tasks28.
Addressing the limitations identified in ARC-AGI-1, ARC-AGI-2 significantly expands the bench-
mark by providing a more comprehensive and carefully refined collection of tasks. These new
tasks emphasize deeper compositional reasoning, multi-step logic, contextual rule application, and
symbolic abstraction. Human calibration studies show these tasks are challenging but doable for
people, while being much harder for current AI systems, offering a clearer measure of general
reasoning abilities29.
10
Sudoku-Extreme Sudoku is a 9×9 logic puzzle, requiring each row, column, and 3×3 block to
contain the digits 1–9 exactly once. A prediction is determined correct if it exactly matches the
puzzle’s unique solution. Sudoku’s complex logical structure makes it a popular benchmark for
evaluating logical reasoning in machine learning62,63,64.
The most frequently used Sudoku datasets in research, namely the Kaggle dataset65, can be fully
solved using elementary single-digit techniques66. The minimal 17-clue puzzles62, another widely-
used collection, might seem more challenging due to its small number of clues. However, this
perception is misleading—since 17 represents the minimum number of clues required to guarantee
a unique Sudoku solution, these hints need to be highly orthogonal of each other. This orthogonal
arrangement leads to many direct, easily-resolved solution paths67.
We introduce Sudoku-Extreme, a more challenging dataset that is compiled from the aforemen-
tioned easy datasets as well as puzzles recognized by the Sudoku community as exceptionally
difficult for human players:
• Easy puzzles compiled from Kaggle, 17-clue, plus unbiased samples from the Sudoku puzzle
distribution67: totaling 1 149 158 puzzles.
• Challenging puzzles compiled from Magictour 1465, Forum-Hard and Forum-Extreme subsets:
totaling 3 104 157 puzzles.
The compiled data then undergo a strict 90/10 train-test split, ensuring that the test set puzzles
cannot be derived through equivalent transformations of any training samples. Sudoku-Extreme is
a down-sampled subset of this data containing 1000 training examples. We use Sudoku-Extreme in
our main experiments (Figure 1), which focuses on small-sample learning scenarios. To guarantee
convergence and control overfitting effects in our analysis experiments (Figures 2, 3 and 5), we use
the complete training data, Sudoku-Extreme-Full, containing 3 831 994 examples.
We measure puzzle difficulty by counting the number of search backtracks ("guesses") required
by a smart Sudoku solver program tdoku, which uses propositional logic to reduce the number of
guesses67. Our Sudoku-Extreme dataset exhibits a mean difficulty of 22 backtracks per puzzle, sig-
nificantly higher than existing datasets, including recent handmade puzzles Sudoku-Bench68 which
average just 0.45 backtracks per puzzle. These subset complexity levels are shown in Figure 6-(d).
Maze-Hard This task involves finding the optimal path in a 30×30 maze, making it interpretable
and frequently used for training LLMs in search tasks69,70,71. We adopt the instance generation
procedure of Lehnert et al. 71 , but introduce an additional filter to retain only those instances whose
difficulty exceeds 110. Here, “difficulty” is defined as the length of the shortest path, which aligns
with the linear time complexity of the wavefront breadth-first search algorithm on GPUs72. A path
is considered correct if it is valid and optimal—that is, the shortest route from the start to the goal.
The training and test set both include 1000 examples.
3.2 Evaluation Details
For all benchmarks, HRM models were initialized with random weights and trained in the sequence-
to-sequence setup using the input–output pairs. The two-dimensional input and output grids were
flattened and then padded to the maximum sequence length. The resulting performance is shown in
Figure 1. Remarkably, HRM attains these results with just ~1000 training examples per task—and
without pretraining or CoT labels.
11
For ARC-AGI challenge, we start with all input-output example pairs in the training and the evalua-
tion sets. The dataset is augmented by applying translations, rotations, flips, and color permutations
to the puzzles. Each task examples is prepended with a learnable special token that represents the
puzzle it belongs to. At test time, we proceed as follows for each test input in the evaluation set: (1)
Generate and solve 1000 augmented variants and, for each, apply the inverse-augmentation trans-
form to obtain a prediction. (2) Choose the two most popular predictions as the final outputs.3 All
results are reported on the evaluation set.
We augment Sudoku puzzles by applying band and digit permutations, while data augmentation is
disabled for Maze tasks. Both tasks undergo only a single inference pass.
For ARC-AGI, the scores of the CoT models are taken from the official leaderboard29, while for
Sudoku and Maze, the scores are obtained by evaluating through the corresponding API.
In Figure 1, the baselines are grouped based on whether they are pre-trained and use CoT, or neither.
The "Direct pred" baseline means using "direct prediction without CoT and pre-training", which
retains the exact training setup of HRM but swaps in a Transformer architecture. Interestingly, on
ARC-AGI-1, "Direct pred" matches the performance of Liao and Gu 73 , who built a carefully de-
signed, domain-specific equivariant network for learning the ARC-AGI task from scratch, without
pre-training. By substituting the Transformer architecture with HRM’s hierarchical framework and
implementing ACT, we achieve more than a twofold performance improvement.
On the Sudoku-Extreme and Maze-Hard benchmarks, the performance gap between HRM and the
baseline methods is significant, as the baselines almost never manage to solve the tasks. These
benchmarks that demand lengthy reasoning traces are particularly difficult for CoT-based methods.
With only 1000 training examples, the “Direct pred” baseline—which employs an 8-layer Trans-
former identical in size to HRM—fails entirely on these challenging reasoning problems. When
trained on the larger Sudoku-Extreme-Full dataset, however, “Direct pred” can solve some easy
Sudoku puzzles and reaches 16.9% accuracy (see Figure 2). Lehnert et al. 71 showed that a large
vanilla Transformer model with 175M parameters, trained on 1 million examples across multiple
trials, achieved only marginal success on 30x30 Maze tasks, with accuracy below 20% using the
pass@64 evaluation metric.
3.3 Visualization of intermediate timesteps
Although HRM demonstrates strong performance on complex reasoning tasks, it raises an intrigu-
ing question: what underlying reasoning algorithms does the HRM neural network actually imple-
ment? Addressing this question is important for enhancing model interpretability and developing a
deeper understanding of the HRM solution space.
While a definitive answer lies beyond our current scope, we begin our investigation by analyzing
state trajectories and their corresponding solution evolution. More specifically, at each timestep
i and given the low-level and high-level state pair (zi
L and zi
H ) we perform a preliminary forward
pass through the H-module to obtain ¯zi = fH (zi
H , zi
L; θH ) and its corresponding decoded prediction
¯yi = fO(¯zi; θO). The prediction ¯yi is then visualized in Figure 7.
In the Maze task, HRM appears to initially explore several potential paths simultaneously, subse-
quently eliminating blocked or inefficient routes, then constructing a preliminary solution outline
followed by multiple refinement iterations. In Sudoku, the strategy resembles a depth-first search
3The ARC-AGI allows two attempts for each test input.
12
Timestep i = 0 Timestep i = 1 Timestep i = 2 Timestep i = 3 Timestep i = 4 Timestep i = 5 Timestep i = 64 8 9
7 3 1
2
6 7
3 4
1 6 4 2 3
2 7 3
4 6 1 2
3 7 6 1
Initial
2 4 3 5 7 1 8 9 6
6 7 8 6 3 4 1 5 4
6 5 1 2 7 9 7 3 4
8 3 4 8 6 7 2 1 2
7 2 8 3 1 8 6 4 8
1 5 6 4 9 2 3 7 9
8 1 2 7 9 3 4 6 6
4 6 8 1 2 8 7 3 7
3 9 7 9 9 6 4 2 1
Timestep i = 0
2 4 3 5 7 1 8 9 6
6 7 8 6 3 4 1 5 4
6 5 1 2 8 9 7 3 4
5 3 4 8 6 7 2 1 9
7 2 5 3 1 5 6 4 8
1 9 6 4 5 2 3 7 7
9 1 2 7 4 3 9 8 6
4 6 5 1 2 8 9 7 3
3 8 7 5 5 6 4 2 1
Timestep i = 1
2 4 3 5 7 1 8 9 3
8 7 9 6 3 4 1 5 2
6 5 1 2 8 8 7 3 4
5 3 4 8 6 7 2 1 5
7 2 5 3 1 5 6 4 9
1 9 6 4 5 2 3 7 7
9 1 2 7 4 3 5 6 8
4 6 5 1 2 8 9 7 3
3 8 7 9 5 6 4 2 1
Timestep i = 2
2 4 3 5 7 1 8 9 3
8 7 9 6 3 4 1 5 2
6 5 1 2 8 9 7 3 4
5 3 4 8 6 7 2 1 5
7 2 5 3 1 1 6 4 6
1 9 6 4 5 2 3 7 7
9 1 2 7 4 3 5 6 8
4 6 8 1 2 8 9 7 3
3 8 7 9 5 6 4 2 1
Timestep i = 3
2 4 1 6 7 5 8 9 3
6 7 9 6 3 4 1 5 2
6 5 3 2 1 8 7 6 4
5 3 4 9 6 7 2 1 5
7 2 5 3 8 1 6 4 6
1 9 6 4 5 2 3 8 7
9 1 2 7 4 3 5 6 8
4 6 8 1 2 8 9 7 3
3 8 7 5 8 6 4 2 1
Timestep i = 4
2 4 1 6 7 5 8 9 3
6 7 9 9 3 4 1 5 2
9 5 1 2 8 8 7 3 4
5 3 4 8 6 7 2 1 8
7 2 5 3 1 1 6 4 6
1 9 6 4 5 2 3 7 7
9 1 2 7 4 3 5 8 6
4 6 8 1 2 8 9 3 7
3 8 7 5 8 6 4 2 1
Timestep i = 5
2 4 1 6 7 5 8 9 3
6 7 9 8 3 4 1 5 2
8 5 3 2 1 9 7 6 4
5 3 4 9 6 7 2 1 8
7 2 8 3 8 1 6 4 6
1 9 6 4 8 2 3 7 5
9 1 2 7 4 3 5 8 6
4 6 5 1 2 8 9 3 7
3 8 7 5 9 6 4 2 1
Timestep i = 6
2 4 1 6 7 5 8 9 3
6 7 9 8 3 4 1 5 2
8 5 3 2 1 9 7 6 4
5 3 4 9 6 7 2 1 8
7 2 8 3 5 1 6 4 9
1 9 6 4 8 2 3 7 5
9 1 2 7 4 3 5 8 6
4 6 5 1 2 8 9 3 7
3 8 7 5 9 6 4 2 1
Timestep i = 7[7666fa5d] Example Input [7666fa5d] Example Output [7666fa5d] Test Input Timestep i = 0 Timestep i = 1 Timestep i = 2 Timestep i = 3 Timestep i = 4[7b80bb43] Example Input [7b80bb43] Example Output
[7b80bb43] Test Input Timestep i = 0 Timestep i = 1 Timestep i = 2 Timestep i = 3 Timestep i = 4 Timestep i = 5 Timestep i = 6Figure 7: Visualization of intermediate predictions by HRM on benchmark tasks. Top: Maze-
Hard—blue cells indicate the predicted path. Middle: Sudoku-Extreme—bold cells represent ini-
tial givens; red highlights cells violating Sudoku constraints; grey shading indicates changes from
the previous timestep. Bottom: ARC-AGI-2 Task—left: provided example input-output pair; right:
intermediate steps solving the test input.
approach, where the model appears to explore potential solutions and backtracks when it hits dead
ends. HRM uses a different approach for ARC tasks, making incremental adjustments to the board
and iteratively improving it until reaching a solution. Unlike Sudoku, which involves frequent
backtracking, the ARC solution path follows a more consistent progression similar to hill-climbing
optimization.
Importantly, the model shows it can adapt different reasoning approaches, likely choosing an ef-
fective strategy for each particular task. Further research is needed to gain more comprehensive
insights into these solution strategies.
4 Brain Correspondence
A key principle from systems neuroscience is that a brain region’s functional repertoire—its ability
to handle diverse and complex tasks—is closely linked to the dimensionality of its neural represen-
tations75,76. Higher-order cortical areas, responsible for complex reasoning and decision-making,
must handle a wide variety of tasks, demanding more flexible and context-dependent processing77.
In dynamical systems, this flexibility is often realized through higher-dimensional state-space tra-
jectories, which allow for a richer repertoire of potential computations78. This principle gives
rise to an observable dimensionality hierarchy, where a region’s position in the processing hierar-
chy correlates with its effective dimensionality. To quantify this phenomena, we can examine the
13
(c)
(d)
(a)
(b)
Position in the hierarchy
Participation Ratio (PR)
20 400
2.0
2.5
3.0
3.5
4.0
4.5
5.0
(e)
(f)Figure 8: Hierarchical Dimensionality Organization in the HRM and Mouse Cortex. (a,b) are
adapted from Posani et al. 74 . (a) Anatomical illustration of mouse cortical areas, color-coded by
functional modules. (b) Correlation between Participation Ratio (PR), a measure of effective neural
dimensionality, and hierarchical position across different mouse cortical areas. Higher positions in
the hierarchy (e.g., MOs, ACAd) exhibit significantly higher PR values compared to lower sensory
areas (e.g., SSp-n), with a Spearman correlation coefficient of ρ=0.79 (P=0.0003). (c,d) Trained
HRM. (c) PR scaling of the trained HRM with task diversity. The dimensionality of the high-
level module (zH ) scales with the number of unique tasks (trajectories) included in the analysis,
indicating an adaptive expansion of its representational capacity. In contrast, the low-level module’s
(zL) dimensionality remains stable. (d) PR values for the low-level (zL, PR = 30.22) and high-
level (zH , PR = 89.95) modules of the trained HRM, computed from neural activity during 100
unique Sudoku-solving trajectories. A clear dimensionality hierarchy is observed, with the high-
level module operating in a substantially higher-dimensional space. (e,f) Analysis of Untrained
Network. To verify that the dimensionality hierarchy is an emergent property of training, the same
analyses were performed on an untrained HRM with random weights. (e) In contrast to the trained
model’s scaling in (c), the dimensionality of both modules in the untrained model remains low and
stable, failing to scale with the number of tasks. (f) Similarly, contrasting with the clear separation
in (d), the PR values for the untrained model’s modules (zL, PR = 42.09; zH , PR = 40.75) are
low and nearly identical, showing no evidence of hierarchical separation. This confirms that the
observed hierarchical organization of dimensionality is a learned property that emerges through
training, not an artifact of the model’s architecture.
14
Participation Ratio (PR) which serves as a standard measure of the effective dimensionality of a
high-dimensional representation79. The PR is calculated using the formula
PR = (P
i λi)2
P
i λ2
i
,
where {λi} are the eigenvalues of the covariance matrix of neural trajectories. Intuitively, a higher
PR value signifies that variance is distributed more evenly across many dimensions, corresponding
to a higher-dimensional representation. Conversely, a lower PR value indicates that variance is
concentrated in only a few principal components, reflecting a more compact, lower-dimensional
structure.
The dimensionality hierarchy can be observed, for example, in the mouse cortex, where the PR of
population activity increases monotonically from low-level sensory areas to high-level associative
areas, supporting this link between dimensionality and functional complexity74 (Figure 8 a,b).
We evaluated whether HRM reproduces this neuroscientific principle by calculating the PR for
both recurrent modules after training on the Sudoku-Extreme Full dataset. The PR computation
used the covariance matrix derived from neural states gathered across multiple Sudoku-solving
trajectories. The results show a striking parallel to the biological findings. The low-level module’s
state (zL) occupies a relatively small subspace with a participation ratio of 30.22, whereas the high-
level module’s state (zH ) operates in a substantially larger subspace with a participation ratio of
89.95, as shown in Figure 8-(c). Furthermore, Figure 8-(d) shows that increasing the number of
unique tasks (trajectories) from 10 to 100 causes zH dimensionality to scale up accordingly, while
zL dimensionality remains stable. These results suggest an emergent separation of representational
capacity between the modules that parallels their functional roles.
To confirm that this hierarchical organization is an emergent property of training, and not an artifact
of the network’s architecture, we performed a control analysis using an identical but untrained
network with random weights.
We initialized an identical HRM architecture with random weights and, without any training, mea-
sured the PR of its modules as the network processed the same task-specific inputs given to the
trained model.
The results, shown in Figure 8-(e,f), reveal a stark contrast: the high-level and low-level modules of
the untrained network exhibit no hierarchical separation, with their PR values remaining low and
nearly indistinguishable from each other. This control analysis validates that the dimensionality
hierarchy is an emergent property that arises as the model learns to perform complex reasoning.
The high-to-low PR ratio in HRM (zH /zL ≈ 2.98) closely matches that measured in the mouse
cortex (≈ 2.25). In contrast, conventional deep networks often exhibit neural collapse, where
last-layer features converge to a low-dimensional subspace80,81,82. HRM therefore departs from the
collapse pattern and instead fosters a high-dimensional representation in its higher module. This
is significant because such representations are considered crucial for cognitive flexibility and are a
hallmark of higher-order brain regions like the prefrontal cortex (PFC), which is central to complex
reasoning.
This structural parallel suggests the model has discovered a fundamental organizational principle.
By learning to partition its representations into a high-capacity, high-dimensional subspace (zH )
and a more specialized, low-dimensional one (zL), HRM autonomously discovers an organizational
15
principle that is thought to be fundamental for achieving robust and flexible reasoning in biological
systems. This provides a potential mechanistic explanation for the model’s success on complex,
long-horizon tasks that are intractable for models lacking such a differentiated internal structure.
We emphasize, however, that this evidence is correlational. While a causal link could be tested
via intervention (e.g., by constraining the H-module’s dimensionality), such methods are difficult
to interpret in deep learning due to potential confounding effects on the training process itself.
Thus, the causal necessity of this emergent hierarchy remains an important question for future
investigation.
5 Related Work
Reasoning and algorithm learning Given the central role of reasoning problems and their close
relation to algorithms, researchers have long explored neural architectures that enable algorithm
learning from training instances. This line of work includes Neural Turing Machines (NTM)83,
the Differentiable Neural Computer (DNC)84, and Neural GPUs85–all of which construct iterative
neural architectures that mimic computational hardware for algorithm execution, and are trained to
learn algorithms from data. Another notable work in this area is Recurrent Relational Networks
(RRN)62, which executes algorithms on graph representations through graph neural networks.
Recent studies have integrated algorithm learning approaches with Transformer-based architec-
tures. Universal Transformers extend the standard Transformer model by introducing a recurrent
loop over the layers and implementing an adaptive halting mechanism. Geiping et al. 86 demonstrate
that looped Transformers can generalize to a larger number of recurrent steps during inference than
what they were trained on. Shen et al. 16 propose adding continuous recurrent reasoning tokens
to the Transformer. Finally, TransNAR8 combine recurrent graph neural networks with language
models.
Building on the success of CoT-based reasoning, a line of work have introduced fine-tuning meth-
ods that use reasoning paths from search algorithms (like A*) as SFT targets87,71,70.
We also mention adaptive halting mechanisms designed to allocate additional computational re-
sources to more challenging problems. This includes the Adaptive Computation Time (ACT) for
RNNs88 and follow-up research like PonderNet89, which aims to improve the stability of this allo-
cation process.
HRM further pushes the boundary of algorithm learning through a brain-inspired computational
architecture that achieves exceptional data efficiency and model expressiveness, successfully dis-
covering complex and diverse algorithms from just 1000 training examples.
Brain-inspired reasoning architectures Developing a model with the reasoning power of the
brain has long been a goal in brain-inspired computing. Spaun90 is one notable example, which uses
spiking neural networks to create distinct modules corresponding to brain regions like the visual
cortex and prefrontal cortex. This design enables an architecture to perform a range of cognitive
tasks, from memory recall to simple reasoning puzzles. However, its reasoning relies on hand-
designed algorithms, which may limit its ability to learn new tasks. Another significant model is the
Tolman-Eichenbaum Machine (TEM)91, which is inspired by the hippocampal-entorhinal system’s
role in spatial and relational memory tasks. TEM proposes that medial entorhinal cells create a
basis for structural knowledge, while hippocampal cells link this basis to sensory information. This
allows TEM to generalize and explains the emergence of various cell types like grid, border, and
16
place cells. Another approach involves neural sampling models92, which view the neural signaling
process as inference over a distribution, functioning similarly to a Boltzmann machine. These
models often require hand-made rules to be set up for solving a specific reasoning task. In essence,
while prior models are restricted to simple reasoning problems, HRM is designed to solve complex
tasks that are hard for even advanced LLMs, without pre-training or task-specific manual design.
Hierarchical memory The hierarchical multi-timescale structure plays also an important role in
how the brain processes memory. Models such as Hierarchical Sequential Models93 and Clockwork
RNN94 use multiple recurrent modules that operate at varying time scales to more effectively cap-
ture long-range dependencies within sequences, thereby mitigating the forgetting issue in RNNs.
Similar mechanisms have also been adopted in linear attention methods for memorizing long con-
texts (see the Discussions section). Since HRM focuses on reasoning, full attention is applied for
simplicity. Incorporating hierarchical memory into HRM could be a promising future direction.
6 Discussions
Turing-completeness of HRM Like earlier neural reasoning algorithms including the Universal
Transformer95, HRM is computationally universal when given sufficient memory and time con-
straints. In other words, it falls into the category of models that can simulate any Turing machine,
overcoming the computational limitations of standard Transformers discussed previously in the in-
troduction. Given that earlier neural algorithm reasoners were trained as recurrent neural networks,
they suffer from premature convergence and memory intensive BPTT. Therefore, in practice, their
effective computational depth remains limited, though still deeper than that of a standard Trans-
former. By resolving these two challenges and being equipped with adaptive computation, HRM
could be trained on long reasoning processes, solve complex puzzles requiring intensive depth-first
search and backtracking, and move closer to practical Turing-completeness.
Reinforcement learning with chain-of-thought Beyond fine-tuning using human-annotated CoT,
reinforcement learning (RL) represents another widely adopted training methodology. However,
recent evidence suggests that RL primarily unlock existing CoT-like capabilities rather than dis-
covering fundamentally new reasoning mechanisms96,97,98,99. Additionally, CoT-training with RL
is known for its instability and data inefficiency, often requiring extensive exploration and careful
reward design. In contrast, HRM takes feedback from dense gradient-based supervision rather than
relying on a sparse reward signal. Moreover, HRM operates naturally in a continuous space, which
is biologically plausible and avoids allocating same computational resources to each token, even
though tokens vary in their reasoning and planning complexity16.
Linear attention Recurrence has been explored not only for its capability in universal computa-
tion, but also as a means to replace the attention mechanism in Transformers, which suffers from
quadratic time and memory complexity100. Recurrent alternatives offer a more efficient design by
processing input tokens sequentially and predicting the next token at each time step, similar to early
RNN-based language models.
Some linear-attention variants, such as Log-linear Attention101, share an RNN-like state-update that
can be interpreted as propagating multi-timescale summary statistics, thereby retaining long-range
context without the quadratic memory growth of standard self-attention. However, substituting the
attention mechanism alone does not change the fact that Transformers are still fixed-depth, and
require CoT as a compensatory mechanism. Notably, linear attention can operate with a reduced
17
key-value cache over extended contexts, making them more suitable for deployment on resource-
constrained edge devices.
7 Conclusion
This work introduces the Hierarchical Reasoning Model, a brain-inspired architecture that lever-
ages hierarchical structure and multi-timescale processing to achieve substantial computational
depth without sacrificing training stability or efficiency. With only 27M parameters and train-
ing on just 1000 examples, HRM effectively solves challenging reasoning problems such as ARC,
Sudoku, and complex maze navigation–tasks that typically pose significant difficulties for contem-
porary LLM and chain-of-thought models.
Although the brain relies heavily on hierarchical structures to enable most cognitive processes,
these concepts have largely remained confined to academic literature rather than being translated
into practical applications. The prevailing AI approach continues to favor non-hierarchical models.
Our results challenge this established paradigm and suggest that the Hierarchical Reasoning Model
represents a viable alternative to the currently dominant chain-of-thought reasoning methods, ad-
vancing toward a foundational framework capable of Turing-complete universal computation.
Acknowledgements We thank Mingli Yuan, Ahmed Murtadha Hasan Mahyoub and Hengshuai
Yao for their insightful discussions and valuable feedback throughout the course of this work.
18
