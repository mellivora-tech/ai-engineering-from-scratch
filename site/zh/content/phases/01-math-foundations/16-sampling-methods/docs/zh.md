# 采样方法

> 采样是 AI 探索可能性空间的方式。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 1，第 06-07 课（概率、贝叶斯定理）
**时间：** ~120 分钟

## 学习目标

- 只使用 uniform random numbers，从零实现 inverse CDF、rejection 和 importance sampling
- 为语言模型 token generation 构建 temperature、top-k 和 top-p（nucleus）sampling
- 解释 reparameterization trick，以及它为什么能让 VAE 中的 sampling 支持 backpropagation
- 运行 Metropolis-Hastings MCMC，从 unnormalized target distribution 中采样

## 问题

一个语言模型处理完你的 prompt，输出一个包含 50,000 个 logits 的向量。词表中每个 token 一个。现在它必须选一个。怎么选？

如果它总是选最高概率 token，每次回复都会一样。确定。无聊。如果它均匀随机选择，输出就是乱码。答案在这两个极端之间，而那个位置由 sampling 控制。

Sampling 不限于文本生成。Reinforcement learning 通过采样 trajectories 估计 policy gradients。VAEs 通过从学到的分布采样并让 randomness 支持 backpropagation 来学习 latent representations。Diffusion models 通过采样噪声并迭代去噪来生成图像。Monte Carlo methods 估计没有闭式解的积分。MCMC algorithms 探索无法枚举的高维 posterior distributions。

每个 generative AI system 都是 sampling system。Sampling strategy 决定输出的质量、多样性和可控性。本课从 uniform random numbers 开始，从零构建每种主要 sampling method，一直到驱动现代 LLMs 和 generative models 的技术。

## 概念

### 为什么 Sampling 重要

Sampling 在 AI 和机器学习中有四个基础角色：

**生成。** Language models、diffusion models 和 GANs 都通过 sampling 产生输出。Sampling algorithm 直接控制创造性、连贯性和多样性。Temperature、top-k 和 nucleus sampling 是工程师每天会调整的旋钮。

**训练。** Stochastic gradient descent 采样 mini-batches。Dropout 采样要停用的 neurons。Data augmentation 采样随机变换。Importance sampling 重新加权 samples，以降低 reinforcement learning（PPO、TRPO）中的 gradient variance。

**估计。** ML 中许多量没有 closed-form solution。数据分布上的 expected loss、energy-based model 的 partition function、Bayesian inference 中的 evidence。Monte Carlo estimation 通过对 samples 求平均来近似所有这些。

**探索。** MCMC algorithms 探索 Bayesian inference 中的 posterior distributions。Evolutionary strategies 采样 parameter perturbations。Thompson sampling 在 bandits 中平衡 exploration 和 exploitation。

核心挑战：你只能直接从简单分布（uniform、normal）中采样。其他所有分布，都需要一种方法把简单 samples 转换成目标分布的 samples。

### Uniform Random Sampling

每个 sampling method 都从这里开始。Uniform random number generator 在 [0, 1) 中产生值，长度相同的每个子区间有相同概率。

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

Properties:
  E[U] = 0.5
  Var(U) = 1/12
```

要从 n 个 items 的离散集合均匀采样，生成 U 并返回 floor(n * U)。要从连续区间 [a, b] 采样，计算 a + (b - a) * U。

关键洞见：一个 uniform random number 正好包含生成任意分布一个 sample 所需的随机性。技巧是找到正确变换。

### Inverse CDF Method（Inverse Transform Sampling）

Cumulative distribution function（CDF）把值映射到概率：

```
F(x) = P(X <= x)

Properties:
  F is non-decreasing
  F(-inf) = 0
  F(+inf) = 1
  F maps the real line to [0, 1]
```

Inverse CDF 把概率映射回值。如果 U ~ Uniform(0, 1)，那么 X = F_inverse(U) 服从目标分布。

```
Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**Exponential distribution 例子：**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Solve F(x) = u for x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Since (1 - U) and U have the same distribution:
  x = -ln(u) / lambda
```

当你能写出闭式 F_inverse 时，这个方法完美可用。Normal distribution 没有 closed-form inverse CDF，所以我们用其他方法（Box-Muller 或数值近似）。

**离散版本：** 对离散分布，构建 cumulative sum 作为 CDF，生成 U，并找到 cumulative sum 第一个超过 U 的 index。这就是第 06 课 `sample_categorical` 的工作方式。

### Rejection Sampling

当你无法反转 CDF，但可以在差一个常数的情况下评估 target PDF，rejection sampling 就能工作。

```
Target distribution: p(x)  (can evaluate, possibly unnormalized)
Proposal distribution: q(x)  (can sample from)
Bound: M such that p(x) <= M * q(x) for all x

Algorithm:
  1. Sample x ~ q(x)
  2. Sample u ~ Uniform(0, 1)
  3. If u < p(x) / (M * q(x)), accept x
  4. Otherwise, reject and go to step 1

Acceptance rate = 1/M
```

Bound M 越紧，acceptance rate 越高。在低维（1-3）中，rejection sampling 很好用。在高维中，acceptance rate 会指数下降，因为 proposal volume 的大部分都会被拒绝。这是 rejection sampling 的维度灾难。

**例子：从 truncated normal 采样。** 使用截断范围上的 uniform proposal。Envelope M 是该范围内 normal PDF 的最大值。

**例子：从半圆采样。** 在 bounding rectangle 中均匀 proposal。如果点落在半圆内就 accept。这就是 Monte Carlo 计算 pi 的方式：acceptance rate 等于面积比 pi/4。

### Importance Sampling

有时你不需要来自 target distribution p(x) 的 samples。你需要估计 p(x) 下的 expectation，而你有来自另一个分布 q(x) 的 samples。

```
Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx

Rewrite:
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

where w(x) = p(x) / q(x)  are the importance weights.

Estimator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    where x_i ~ q(x)
```

这在 reinforcement learning 中很关键。在 PPO（Proximal Policy Optimization）中，你用 old policy pi_old 收集 trajectories，但想优化 new policy pi_new。Importance weight 是 pi_new(a|s) / pi_old(a|s)。PPO 会 clip 这些 weights，防止新 policy 偏离旧 policy 太远。

Importance sampling estimator 的方差取决于 q 与 p 有多相似。如果 q 与 p 很不同，少数 samples 会得到巨大 weights 并主导估计。Self-normalized importance sampling 会除以 weights 总和来缓解这个问题：

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### Monte Carlo Estimation

Monte Carlo estimation 通过对 random samples 求平均来近似积分。大数定律保证收敛。

```
Goal: estimate I = integral of g(x) dx over domain D

Method:
  1. Sample x_1, ..., x_N uniformly from D
  2. I ~ (Volume of D / N) * sum(g(x_i))

Error: O(1 / sqrt(N))   regardless of dimension
```

误差率与维度无关。这就是为什么 Monte Carlo methods 在高维中占主导，而 grid-based integration 不可能。

**估计 pi：**

```
Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)
```

**估计 expectations：**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    where x_i ~ p(x)

The sample mean converges to the true expectation.
Variance of the estimator = Var(f(X)) / N
```

### Markov Chain Monte Carlo（MCMC）：Metropolis-Hastings

MCMC 构造一个 Markov chain，使其 stationary distribution 是目标分布 p(x)。经过足够多步后，链中的 samples（近似）来自 p(x)。

```
Target: p(x)  (known up to a normalizing constant)
Proposal: q(x'|x)  (how to propose the next state given the current state)

Metropolis-Hastings algorithm:
  1. Start at some x_0
  2. For t = 1, 2, ..., T:
     a. Propose x' ~ q(x'|x_t)
     b. Compute acceptance ratio:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Accept with probability min(1, alpha):
        - If u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Otherwise: x_{t+1} = x_t
  3. Discard first B samples (burn-in)
  4. Return remaining samples
```

对 symmetric proposals（q(x'|x) = q(x|x')），ratio 简化为 p(x')/p(x)。这就是原始 Metropolis algorithm。

**为什么有效。** Acceptance rule 确保 detailed balance：位于 x 并移动到 x' 的概率，等于位于 x' 并移动到 x 的概率。Detailed balance 蕴含 p(x) 是链的 stationary distribution。

**实践注意事项：**
- Burn-in：丢弃链到达 equilibrium 之前的早期 samples
- Thinning：每隔 k 个 sample 保留一个，以减少 autocorrelation
- Proposal scale：太小，链移动慢（高 acceptance，慢探索）；太大，大多数 proposals 被拒绝（低 acceptance，卡住）
- 高维中 Gaussian proposal 的最优 acceptance rate 约为 0.234

### Gibbs Sampling

Gibbs sampling 是 multivariate distributions 的 MCMC 特例。它不是一次性在所有维度 proposal，而是每次从 conditional distribution 更新一个变量。

```
Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

Gibbs sampling 要求你能从每个 conditional distribution p(x_i | x_{-i}) 中采样。对许多模型来说这很直接：
- Bayesian networks：conditionals 来自 graph structure
- Gaussian mixtures：conditionals 是 Gaussian
- Ising models：每个 spin 的 conditional 只依赖 neighbors

Acceptance rate 总是 1（每个 proposal 都被 accept），因为从精确 conditional 采样会自动满足 detailed balance。

**局限。** 当变量高度相关时，Gibbs sampling mixes slowly，因为一次只更新一个变量，无法沿分布做大的对角移动。

### Temperature Sampling（用于 LLMs）

语言模型为词表中每个 token 输出 logits z_1, ..., z_V。Softmax 把这些转成概率。Temperature 会在 softmax 前重新缩放 logits：

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standard softmax (original distribution)
T -> 0:  argmax (deterministic, always picks highest logit)
T -> inf: uniform (all tokens equally likely)
T < 1.0: sharpens the distribution (more confident, less diverse)
T > 1.0: flattens the distribution (less confident, more diverse)
```

**为什么有效。** 用 T < 1 除以 logits 会放大 logits 之间的差异。如果 z_1 = 2 且 z_2 = 1，用 T = 0.5 除后得到 z_1/T = 4、z_2/T = 2，差距变大。经过 softmax，最高 logit token 会获得更大概率质量。

**实践中：**
- T = 0.0：greedy decoding，适合 factual Q&A
- T = 0.3-0.7：略有创造性，适合 code generation
- T = 0.7-1.0：平衡，适合 general conversation
- T = 1.0-1.5：creative writing、brainstorming
- T > 1.5：越来越随机，通常没用

Temperature 不会改变哪些 tokens 可能出现。它改变的是分配给每个 token 的概率质量。

### Top-k Sampling

Top-k sampling 把候选集合限制为概率最高的 k 个 tokens，然后重新归一化并从这个限制集合中采样。

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Keep only the top k tokens
  4. Renormalize: p_i' = p_i / sum(p_j for j in top-k)
  5. Sample from the renormalized distribution

k = 1:  greedy decoding
k = V:  no filtering (standard sampling)
k = 40: typical setting, removes long tail of unlikely tokens
```

Top-k 防止模型选择 vocabulary distribution 长尾中极不可能的 tokens（拼写错误、乱码）。问题是：k 与上下文无关，固定不变。当模型很自信（一个 token 有 95% 概率）时，k = 40 仍允许 39 个替代项。当模型不确定（概率分散到 1000 个 tokens）时，k = 40 会切掉合理选项。

### Top-p（Nucleus）Sampling

Top-p sampling 动态调整候选集合大小。它不是保留固定数量 tokens，而是保留累计概率超过 p 的最小 token 集合。

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Find smallest k such that sum of top-k probabilities >= p
  4. Keep only those k tokens
  5. Renormalize and sample

p = 0.9:  keeps tokens covering 90% of probability mass
p = 1.0:  no filtering
p = 0.1:  very restrictive, nearly greedy
```

当模型很自信时，nucleus sampling 保留很少 tokens（可能 2-3 个）。当模型不确定时，它保留很多（可能 200 个）。这种自适应行为就是 nucleus sampling 通常比 top-k 生成更好文本的原因。

**常见组合：**
- Temperature 0.7 + top-p 0.9：通用设置
- Temperature 0.0（greedy）：最适合 deterministic tasks
- Temperature 1.0 + top-k 50：Fan et al. (2018) 原始论文设置

Top-k 和 top-p 可以组合。先应用 top-k，再在剩余集合上应用 top-p。

### Reparameterization Trick（用于 VAEs）

Variational autoencoders（VAEs）通过把输入编码成 latent space 中的一个分布、从该分布采样、再解码 sample 来学习。问题是：你不能通过 sampling operation 做 backpropagation。

```
Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???
```

Reparameterization trick 把 randomness 与 parameters 分开：

```
Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.
```

这可行，因为 N(mu, sigma^2) 与 mu + sigma * N(0, 1) 分布相同。关键洞见：把 randomness 移到无参数来源（epsilon），再把 sample 表示成 parameters 的可微变换。

**VAE training loop 中：**
1. Encoder 为每个输入输出 mu 和 log(sigma^2)
2. Sample epsilon ~ N(0, 1)
3. 计算 z = mu + sigma * epsilon
4. Decode z 以重构输入
5. 通过步骤 4、3、2、1 做 backpropagation（可行，因为第 3 步可微）

没有 reparameterization trick，VAEs 无法用标准 backpropagation 训练。这个单一洞见让 VAEs 变得实用。

### Gumbel-Softmax（可微 Categorical Sampling）

Reparameterization trick 适用于连续分布（Gaussian）。对离散 categorical distributions，我们需要不同方法。Gumbel-Softmax 提供 categorical sampling 的可微近似。

**Gumbel-Max trick（不可微）：**

```
To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.
```

**Gumbel-Softmax（可微近似）：**

```
Replace the hard argmax with a soft softmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperature) controls the approximation:
  tau -> 0:  approaches a one-hot vector (hard categorical)
  tau -> inf: approaches uniform (1/k, 1/k, ..., 1/k)
  tau = 1.0: soft approximation
```

Gumbel-Softmax 产生离散 sample 的连续松弛。输出是 probability vector（soft one-hot），而不是 hard one-hot。Gradients 会通过 softmax 流动。训练的 forward pass 中，你可以使用 “straight-through” estimator：forward pass 用 hard argmax，但 backward pass 用 soft Gumbel-Softmax gradients。

**应用：**
- VAEs 中的离散 latent variables
- Neural architecture search（选择离散 operations）
- Hard attention mechanisms
- 带离散 actions 的 reinforcement learning

### Stratified Sampling

标准 Monte Carlo sampling 可能因随机性在 sample space 中留下空洞。Stratified sampling 通过把空间分成 strata 并从每个 stratum 采样，强制均匀覆盖。

```
Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

Stratified sampling 的方差总是低于或等于标准 Monte Carlo：

```
Var(stratified) <= Var(standard Monte Carlo)

The improvement is largest when f(x) varies smoothly.
For piecewise-constant functions, stratified sampling is exact.
```

**应用：**
- Numerical integration（quasi-Monte Carlo）
- 训练数据 splits（确保每个 fold class balance）
- 带 stratification 的 importance sampling（结合两种技术）
- NeRF（Neural Radiance Fields）沿 camera rays 使用 stratified sampling

### 与 Diffusion Models 的连接

Diffusion models 通过 sampling process 生成图像。Forward process 在 T 步中向图像加入 Gaussian noise，直到变成纯噪声。Reverse process 学习去噪，逐步恢复原图。

```
Forward process (known):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  After T steps: x_T ~ N(0, I)  (pure noise)

Reverse process (learned):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  Each denoising step is a sampling step.
```

与本课方法的连接：
- 每个 denoising step 使用 reparameterization trick（采样噪声，应用 deterministic transform）
- Noise schedule {alpha_t} 控制一种 temperature annealing
- 训练使用 Monte Carlo estimation 近似 ELBO（evidence lower bound）
- Diffusion models 中的 ancestral sampling 是 Markov chain（每一步只依赖当前状态）

整个图像生成过程都是迭代采样：从噪声开始，每一步基于学到的 denoising model，采样一个噪声略少的版本。

## 构建它

### 第 1 步：Uniform 和 inverse CDF sampling

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

生成 10,000 个 exponential samples，并验证均值为 1/lambda。

### 第 2 步：Rejection sampling

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

使用 rejection sampling 从 truncated normal distribution 中采样。通过 histogram samples 验证形状。

### 第 3 步：Importance sampling

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

使用 uniform proposal 估计 normal distribution 下的 E[X^2]。与已知答案（mu^2 + sigma^2）比较。

### 第 4 步：Monte Carlo 估计 pi

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### 第 5 步：Metropolis-Hastings MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

从 bimodal distribution（两个 Gaussians 的 mixture）中采样。可视化 chain trajectory。

### 第 6 步：Gibbs sampling

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### 第 7 步：Temperature sampling

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

展示 temperature 如何改变一组 token logits 的输出分布。

### 第 8 步：Top-k 和 top-p sampling

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### 第 9 步：Reparameterization trick

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

演示 gradients 可以流过 reparameterized sample，但不能流过 direct sampling。

### 第 10 步：Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

展示降低 temperature 如何让输出接近 one-hot vector。

包含所有可视化的完整实现位于 `code/sampling.py`。

## 使用它

使用 NumPy 和 SciPy 的生产版本：

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

大规模 MCMC 使用专门库：
- PyMC：使用 NUTS（adaptive HMC）的完整 Bayesian modeling
- emcee：ensemble MCMC sampler
- NumPyro/JAX：GPU-accelerated MCMC

你已经从零构建了这些方法。现在你知道库调用内部在做什么。

## 练习

1. 为 Cauchy distribution 实现 inverse CDF sampling。CDF 是 F(x) = 0.5 + arctan(x)/pi。生成 10,000 个 samples，并将 histogram 与真实 PDF 比较。注意 heavy tails（远离中心的极端值）。

2. 使用 Uniform(0, 1) proposal，通过 rejection sampling 生成 Beta(2, 5) distribution 的 samples。把 accepted samples 与真实 Beta PDF 作图。理论 acceptance rate 是多少？

3. 用 Monte Carlo 估计 sin(x) 从 0 到 pi 的积分，samples 数分别为 1,000、10,000 和 100,000。比较每个级别的误差。验证误差按 O(1/sqrt(N)) 缩放。

4. 实现 Metropolis-Hastings，从 2D distribution p(x, y) proportional to exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2) 中采样。绘制 samples 和 chain trajectory。尝试不同 proposal standard deviations。

5. 构建完整文本生成 demo：给定 10 个词的 vocabulary 和 logits，使用 (a) greedy、(b) temperature=0.7、(c) top-k=3、(d) top-p=0.9 生成 20 tokens 序列。比较 5 次运行的输出多样性。

## 关键术语

| 术语 | 人们常说 | 它实际意味着什么 |
|------|----------------|----------------------|
| Sampling | “抽随机值” | 按概率分布生成值。所有 generative AI 背后的机制 |
| Uniform distribution | “都一样可能” | [a, b] 中每个值有相等 probability density 1/(b-a)。所有 sampling methods 的起点 |
| Inverse CDF | “概率变换” | F_inverse(U) 把 uniform sample 转换为任意已知 CDF 分布的 sample。精确且高效 |
| Rejection sampling | “提出并 accept/reject” | 从简单 proposal 生成，以 target/proposal ratio 成比例的概率 accept。精确但浪费 samples |
| Importance sampling | “重加权 samples” | 用 q(x) 的 samples 估计 p(x) 下的 expectations，每个 sample 权重为 p(x)/q(x)。RL 中 PPO 的核心 |
| Monte Carlo | “平均 random samples” | 用 sample averages 近似积分。误差 O(1/sqrt(N))，与维度无关 |
| MCMC | “会收敛的随机游走” | 构造 stationary distribution 为目标分布的 Markov chain。Metropolis-Hastings 是基础算法 |
| Metropolis-Hastings | “接受上坡，有时接受下坡” | 提出 moves，按 density ratio accept。Detailed balance 确保收敛到 target distribution |
| Gibbs sampling | “一次一个变量” | 固定其他变量，从每个变量的 conditional distribution 更新。100% acceptance rate |
| Temperature | “confidence knob” | Softmax 前用 T 除 logits。T<1 sharpen（更自信），T>1 flatten（更多样） |
| Top-k sampling | “保留 k 个最好” | 把除 k 个最高概率 tokens 以外的概率归零，重新归一化并采样。固定候选集大小 |
| Nucleus sampling（top-p） | “保留有概率质量的那些” | 保留 cumulative probability 超过 p 的最小 token 集合。候选集大小自适应 |
| Reparameterization trick | “把 randomness 移出去” | 写成 z = mu + sigma * epsilon，其中 epsilon ~ N(0,1)。让 sampling 可微。VAE training 必需 |
| Gumbel-Softmax | “Soft categorical sampling” | 使用 Gumbel noise + 带 temperature 的 softmax，对 categorical sampling 做可微近似 |
| Stratified sampling | “强制覆盖” | 把 sample space 分成 strata，并从每个 stratum 采样。方差总是低于 naive Monte Carlo |
| Burn-in | “预热期” | MCMC 链到达 stationary distribution 前丢弃的初始 samples |
| Detailed balance | “可逆性条件” | p(x) * T(x->y) = p(y) * T(y->x)。p 成为 Markov chain stationary distribution 的充分条件 |
| Diffusion sampling | “迭代去噪” | 从噪声开始，通过 learned denoising steps 生成数据。每一步都是 conditional sampling operation |

## 延伸阅读

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - MCMC 基础的详细教程
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - 原始 Gumbel-Softmax 论文
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - nucleus (top-p) sampling 论文
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - 引入 reparameterization trick 的 VAE 论文
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - DDPM 把 sampling 连接到图像生成
