\section*{Generative Modeling via Drifting}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/14b7cf7f-b68f-4015-b13d-5b756b999ce4-01.jpg?height=472&width=1505&top_left_y=476&top_left_x=279}
\captionsetup{labelformat=empty}
\caption{Figure 1. Drifting Model. A network $f$ performs a pushforward operation: $q=f_{\#} p_{\text {prior }}$, mapping a prior distribution $p_{\text {prior }}$ (e.g., Gaussian, not shown here) to a pushforward distribution $q$ (orange). The goal of training is to approximate the data distribution $p_{\text {data }}$ (blue). As training iterates, we obtain a sequence of models $\left\{f_{i}\right\}$, which corresponds to a sequence of pushforward distributions $\left\{q_{i}\right\}$. Our Drifting Model focuses on the evolution of this pushforward distribution at training-time. We introduce a drifting field (detailed in main text) that approaches zero when $q$ matches $p_{\text {data }}$. This drifting field provides a loss function ( y -axis, in log-scale) for training.}
\end{figure}

\begin{abstract}
Generative modeling can be formulated as learning a mapping $f$ such that its pushforward distribution matches the data distribution. The pushforward behavior can be carried out iteratively at inference time, e.g., in diffusion/flow-based models. In this paper, we propose a new paradigm called Drifting Models, which evolve the pushforward distribution during training and naturally admit one-step inference. We introduce a drifting field that governs the sample movement and achieves equilibrium when the distributions match. This leads to a training objective that allows the neural network optimizer to evolve the distribution. In experiments, our one-step generator achieves state-of-the-art results on ImageNet $256 \times 256$, with FID 1.54 in latent space and 1.61 in pixel space. We hope that our work opens up new opportunities for high-quality one-step generation.
\end{abstract}

\section*{1. Introduction}

Generative models are commonly regarded as more challenging than discriminative models. While discriminative modeling typically focuses on mapping individual samples to their corresponding labels, generative modeling concerns mapping from one distribution to another. This can be expressed as learning a mapping $f$ such that the pushforward

\footnotetext{
${ }^{1}$ MIT ${ }^{2}$ Harvard University.
Project page:
lambertae.github.io/projects/drifting
}
of a prior distribution $p_{\text {prior }}$ matches the data distribution, namely, $f_{\#} p_{\text {prior }} \approx p_{\text {data }}$. Conceptually, generative modeling learns a functional (here, $f_{\#}$ ) that maps from one function (here, a distribution) to another.

The "pushforward" behavior can be realized iteratively at inference time, e.g., in prevailing paradigms such as Diffusion (Sohl-Dickstein et al., 2015) and Flow Matching (Lipman et al., 2022). When generating, these models map noisier samples to slightly cleaner ones, progressively evolving the sample distribution toward the data distribution. This modeling philosophy can be viewed as decomposing a complex pushforward map (i.e., $f_{\#}$ ) into a chain of more feasible transformations, applied at inference time.

In this paper, we propose Drifting Models, a new paradigm for generative modeling. Drifting Models are characterized by learning a pushforward map that evolves during training time, thereby removing the need for an iterative inference procedure. The mapping $f$ is represented by a single-pass, non-iterative network. As the training process is inherently iterative in deep learning optimization, it can be naturally viewed as evolving the pushforward distribution, $f_{\#} p_{\text {prior }}$, through the update of $f$. See Fig. 1.

To drive the evolution of the training-time pushforward, we introduce a drifting field that governs the sample movement. This field depends on the generated distribution and the data distribution. By definition, this field becomes zero when the two distributions match, thereby reaching an equilibrium in which the samples no longer drift.

Building on this formulation, we propose a simple training objective that minimizes the drift of the generated sam-
ples. This objective induces sample movements and thereby evolves the underlying pushforward distribution through iterative optimization (e.g., SGD). We further introduce the designs of the drifting field, the neural network model, and the training algorithm.

Drifting Models naturally perform single-step ("1-NFE") generation and achieve strong empirical performance. On ImageNet $256 \times 256$, we obtain a 1-NFE FID of $\mathbf{1 . 5 4}$ under the standard latent-space generation protocol, achieving a new state-of-the-art among single-step methods. This result remains competitive even when compared with multistep diffusion-/flow-based models. Further, under the more challenging pixel-space generation protocol (i.e., without latents), we reach a 1-NFE FID of $\mathbf{1 . 6 1}$, substantially outperforming previous pixel-space methods. These results suggest that Drifting Models offer a promising new paradigm for high-quality, efficient generative modeling.

\section*{2. Related Work}

Diffusion-/Flow-based Models. Diffusion models (e.g., Sohl-Dickstein et al. 2015; Ho et al. 2020; Song et al. 2020) and their flow-based counterparts (e.g., Lipman et al. 2022; Liu et al. 2022; Albergo et al. 2023) formulate noise-to-data mappings through differential equations (SDEs or ODEs). At the core of their inference-time computation is an iterative update, e.g., of the form $\mathbf{x}_{i+1}=\mathbf{x}_{i}+\Delta \mathbf{x}_{i}$, such as with an Euler solver. The update $\Delta \mathrm{x}_{i}$ depends on the neural network $f$, and as a result, generation involves multiple steps of network evaluations.

A growing body of work has focused on reducing the steps of diffusion-/flow-based models. Distillation-based methods (e.g., Salimans \& Ho 2022; Luo et al. 2023; Yin et al. 2024; Zhou et al. 2024) distill a pretrained multi-step model into a single-step one. Another line of research aims to train one-step diffusion/flow models from scratch (e.g., Song et al. 2023; Frans et al. 2024; Boffi et al. 2025; Geng et al. 2025a). To achieve this goal, these methods incorporate the SDE/ODE dynamics into training by approximating the induced trajectories. In contrast, our work presents a conceptually different paradigm and does not rely on SDE/ODE formulations as in diffusion/flow models.

Generative Adversarial Networks (GANs). GANs (Goodfellow et al., 2014) are a classical family of models that train a generator by discriminating generated samples from real data. Like GANs, our method involves a single-pass network $f$ that maps noise to data, whose "goodness" is evaluated by a loss function; however, unlike GANs, our method does not rely on adversarial optimization.

Variational Autoencoders (VAEs). VAEs (Kingma \& Welling, 2013) optimize the evidence lower bound (ELBO), which consists of a reconstruction loss and a KL divergence
term. Classical VAEs are one-step generators when using a Gaussian prior. Today's prevailing VAE applications often resort to priors learned from other methods, e.g., diffusion (Rombach et al., 2022) or autoregressive models (Esser et al., 2021), where VAEs effectively act as tokenizers.

Normalizing Flows (NFs). NFs (Rezende \& Mohamed, 2015; Dinh et al., 2016; Zhai et al., 2024) learn mappings from data to noise and optimize the log-likelihood of samples. These methods require invertible architectures and computable Jacobians. Conceptually, NFs operate as onestep generators at inference, with computation performed by the inverse of the network.

Moment Matching. Moment-matching methods (Dziugaite et al., 2015; Li et al., 2015) seek to minimize the Maximum Mean Discrepancy (MMD) between the generated and data distributions. Moment Matching has recently been extended to one-/few-step diffusion (Zhou et al., 2025). Related to MMD, our approach also leverages the concepts of kernel functions and positive/negative samples. However, our approach focuses on a drifting field that explicitly governs the sample drifts at training time. Further discussion is in C.2.

Contrastive Learning. Our drifting field is driven by positive samples from the data distribution and negative samples from the generated distribution. This is conceptually related to the positive and negative samples in contrastive representation learning (Hadsell et al., 2006; Oord et al., 2018). The idea of contrastive learning has also been extended to generative models, e.g., to GANs (Unterthiner et al., 2017; Kang \& Park, 2020) or Flow Matching (Stoica et al., 2025).

\section*{3. Drifting Models for Generation}

We propose Drifting Models, which formulate generative modeling as a training-time evolution of the pushforward distribution via a drifting field. Our model naturally performs one-step generation at inference time.

\subsection*{3.1. Pushforward at Training Time}

Consider a neural network $f: \mathbb{R}^{C} \mapsto \mathbb{R}^{D}$. The input of $f$ is $\boldsymbol{\epsilon} \sim p_{\boldsymbol{\epsilon}}($ e.g., any noise of dimension $C)$, and the output is denoted by $\mathbf{x}=f(\boldsymbol{\epsilon}) \in \mathbb{R}^{D}$. In general, the input and output dimensions need not be equal.

We denote the distribution of the network output by $q$, i.e., $\mathbf{x}=f(\boldsymbol{\epsilon}) \sim q$. In probability theory, $q$ is referred to as the pushforward distribution of $p_{\boldsymbol{\epsilon}}$ under $f$, denoted by:
$$
\begin{equation*}
q=f_{\#} p_{\boldsymbol{\epsilon}} . \tag{1}
\end{equation*}
$$

Here, " $f_{\#}$ " denotes the pushforward induced by $f$. Intuitively, this notation means that $f$ transforms a distribution $p_{\epsilon}$ into another distribution $q$. The goal of generative modeling is to find $f$ such that $f_{\#} p_{\boldsymbol{\epsilon}} \approx p_{\text {data }}$.

Since neural network training is inherently iterative (e.g., SGD), the training process produces a sequence of models $\left\{f_{i}\right\}$, where $i$ denotes the training iteration. This corresponds to a sequence of pushforward distributions $\left\{q_{i}\right\}$ during training, where $q_{i}=\left[f_{i}\right]_{\#} p_{\epsilon}$ for each $i$. The training process progressively evolves $q_{i}$ to match $p_{\text {data }}$.
When the network $f$ is updated, a sample at training iteration $i$ is implicitly "drifted" as: $\mathrm{x}_{i+1}=\mathrm{x}_{i}+\Delta \mathrm{x}_{i}$, where $\Delta \mathrm{x}_{i}:= f_{i+1}(\boldsymbol{\epsilon})-f_{i}(\boldsymbol{\epsilon})$ arises from parameter updates to $f$. This implies that the update of $f$ determines the "residual" of $\mathbf{x}$, which we refer to as the "drift".

\subsection*{3.2. Drifting Field for Training}

Next, we define a drifting field to govern the training-time evolution of the samples $\mathbf{x}$ and, consequently, the pushforward distribution $q$. A drifting field is a function that computes $\Delta \mathrm{x}$ given x . Formally, denoting this field by $\mathbf{V}_{p, q}(\cdot): \mathbb{R}^{d} \rightarrow \mathbb{R}^{d}$, we have:
$$
\begin{equation*}
\mathbf{x}_{i+1}=\mathbf{x}_{i}+\mathbf{V}_{p, q_{i}}\left(\mathbf{x}_{i}\right) \tag{2}
\end{equation*}
$$

Here, $\mathbf{x}_{i}=f_{i}(\boldsymbol{\epsilon}) \sim q_{i}$ and after drifting we denote $\mathbf{x}_{i+1} \sim q_{i+1}$. The subscripts $p, q$ denote that this field depends on $p$ (e.g., $p=p_{\text {data }}$ ) and the current distribution $q$.

Ideally, when $p=q$, we want all $\mathbf{x}$ to stop drifting i.e., $\mathbf{V}=\mathbf{0}$. In this paper, we consider the following proposition:
Proposition 3.1. Consider an anti-symmetric drifting field:
$$
\begin{equation*}
\mathbf{V}_{p, q}(\mathbf{x})=-\mathbf{V}_{q, p}(\mathbf{x}), \quad \forall \mathbf{x} \tag{3}
\end{equation*}
$$

Then we have: $\quad q=p \quad \Rightarrow \quad \mathbf{V}_{p, q}(\mathbf{x})=\mathbf{0}, \forall \mathbf{x}$.
The proof is straightforward ${ }^{1}$. Intuitively, anti-symmetry means that swapping $p$ and $q$ simply flips the sign of the drift. This proposition implies that if the pushforward distribution $q$ matches the data distribution $p$, the drift is zero for any sample and the model achieves an equilibrium.

We note that the converse implication, i.e., $\mathbf{V}_{p, q}=\mathbf{0} \Rightarrow q=p$, is false in general for arbitrary choices of $\mathbf{V}$. For our kernelized formulation (Sec. 3.3), we give sufficient conditions under which $\mathbf{V}_{p, q} \approx \mathbf{0}$ implies $q \approx p$ (Appendix C.1).

Training Objective. The property of equilibrium motivates a definition of a training objective. Let $f_{\theta}$ be a network parameterized by $\theta$, and $\mathbf{x}=f_{\theta}(\boldsymbol{\epsilon})$ for $\boldsymbol{\epsilon} \sim p_{\boldsymbol{\epsilon}}$. At the equilibrium where $\mathbf{V}=\mathbf{0}$, we set up the following fixedpoint relation:
$$
\begin{equation*}
f_{\hat{\theta}}(\boldsymbol{\epsilon})=f_{\hat{\theta}}(\boldsymbol{\epsilon})+\mathbf{V}_{p, q_{\hat{\theta}}}\left(f_{\hat{\theta}}(\boldsymbol{\epsilon})\right) \tag{4}
\end{equation*}
$$

Here, $\hat{\theta}$ denotes the optimal parameters that can achieve the equilibrium, and $q_{\hat{\theta}}$ denotes the pushforward of $f_{\hat{\theta}}$.
$$
{ }^{1} q=p \Rightarrow \mathbf{V}_{p, q}=\mathbf{V}_{q, p}=-\mathbf{V}_{p, q} \Rightarrow \mathbf{V}_{p, q}=\mathbf{0}
$$

This equation motivates a fixed-point iteration during training. At iteration $i$, we seek to satisfy:
$$
\begin{equation*}
f_{\theta_{i+1}}(\boldsymbol{\epsilon}) \leftarrow f_{\theta_{i}}(\boldsymbol{\epsilon})+\mathbf{V}_{p, q_{\theta_{i}}}\left(f_{\theta_{i}}(\boldsymbol{\epsilon})\right) \tag{5}
\end{equation*}
$$

We convert this update rule into a loss function:
$\mathcal{L}=\mathbb{E}_{\boldsymbol{\epsilon}}[\|\underbrace{f_{\theta}(\boldsymbol{\epsilon})}_{\text {prediction }}-\underbrace{\operatorname{stopgrad}\left(f_{\theta}(\boldsymbol{\epsilon})+\mathbf{V}_{p, q_{\theta}}\left(f_{\theta}(\boldsymbol{\epsilon})\right)\right)}_{\text {frozen target }}\|^{2}]$.

Here, the stop-gradient operation provides a frozen state from the last iteration, following (Chen \& He, 2021; Song \& Dhariwal, 2023). Intuitively, we compute a frozen target and move the network prediction toward it.

We note that the value of our loss function $\mathcal{L}$ is equal to $\mathbb{E}_{\boldsymbol{\epsilon}}\left[\|\mathbf{V}(f(\boldsymbol{\epsilon}))\|^{2}\right]$, that is, the squared norm of the drifting field $\mathbf{V}$. With the stop-gradient formulation, our solver does not directly back-propagate through $\mathbf{V}$, because $\mathbf{V}$ depends on $q_{\theta}$ and back-propagating through a distribution is nontrivial. Instead, our formulation minimizes this objective indirectly: it moves $\mathbf{x}=f_{\theta}(\boldsymbol{\epsilon})$ towards its drifted version, i.e., towards $\mathrm{x}+\Delta \mathrm{x}$ that is frozen at this iteration.

\subsection*{3.3. Designing the Drifting Field}

The field $\mathbf{V}_{p, q}$ depends on two distributions $p$ and $q$. To obtain a computable formulation, we consider the form:
$$
\begin{equation*}
\mathbf{V}_{p, q}(\mathbf{x})=\mathbb{E}_{\mathbf{y}^{+} \sim p} \mathbb{E}_{\mathrm{y}^{-} \sim q}\left[\mathcal{K}\left(x, \mathbf{y}^{+}, \mathrm{y}^{-}\right)\right] \tag{7}
\end{equation*}
$$
where $\mathcal{K}(\cdot, \cdot, \cdot)$ is a kernel-like function describing interactions among three sample points. $\mathcal{K}$ can optionally depend on $p$ and $q$. Our framework supports a broad class of functions $\mathcal{K}$, as long as $\mathbf{V}=0$ when $p=q$.

For the instantiation in this work, we introduce a form of $\mathbf{V}$ driven by attraction and repulsion. We define the following fields inspired by the mean-shift method (Cheng, 1995):
$$
\begin{align*}
& \mathbf{V}_{p}^{+}(\mathbf{x}):=\frac{1}{Z_{p}} \mathbb{E}_{p}\left[k\left(\mathbf{x}, \mathbf{y}^{+}\right)\left(\mathbf{y}^{+}-\mathbf{x}\right)\right], \\
& \mathbf{V}_{q}^{-}(\mathbf{x}):=\frac{1}{Z_{q}} \mathbb{E}_{q}\left[k\left(\mathbf{x}, \mathbf{y}^{-}\right)\left(\mathbf{y}^{-}-\mathbf{x}\right)\right] . \tag{8}
\end{align*}
$$

Here, $Z_{p}$ and $Z_{q}$ are normalization factors:
$$
\begin{align*}
Z_{p}(\mathbf{x}) & :=\mathbb{E}_{p}\left[k\left(\mathbf{x}, \mathbf{y}^{+}\right)\right]  \tag{9}\\
Z_{q}(\mathbf{x}) & :=\mathbb{E}_{q}\left[k\left(\mathbf{x}, \mathbf{y}^{-}\right)\right] .
\end{align*}
$$

Intuitively, Eq. (8) computes the weighted mean of the vector difference $\mathbf{y}-\mathbf{x}$. The weights are given by a kernel $k(\cdot, \cdot)$ normalized by (9). We then define $\mathbf{V}$ as:
$$
\begin{equation*}
\mathbf{V}_{p, q}(\mathbf{x}):=\mathbf{V}_{p}^{+}(\mathbf{x})-\mathbf{V}_{q}^{-}(\mathbf{x}) \tag{10}
\end{equation*}
$$

Intuitively, this field can be viewed as attracting by the data distribution $p$ and repulsing by the sample distribution $q$. This is illustrated in Fig. 2.
- Positive samples $\mathbf{y}^{+} \sim p$
- Negative samples $\mathbf{y}^{-} \sim q$

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/14b7cf7f-b68f-4015-b13d-5b756b999ce4-04.jpg?height=263&width=295&top_left_y=449&top_left_x=464}
\captionsetup{labelformat=empty}
\caption{Figure 2. Illustration of drifting a sample. A generated sample $\mathbf{x}$ (black) drifts according to a vector: $\mathbf{V}=\mathbf{V}_{p}^{+}-\mathbf{V}_{q}^{-}$. Here, $\mathbf{V}_{p}^{+}$is the mean-shift vector of the positive samples (blue) and $\mathbf{V}_{q}^{-}$is the mean-shift vector of the negative samples (orange): see Eq. (8). $\mathbf{x}$ is attracted by $\mathbf{V}_{p}^{+}$and repulsed by $\mathbf{V}_{q}^{-}$.}
\end{figure}

Substituting Eq. (8) into Eq. (10), we obtain:
$$
\begin{equation*}
\mathbf{V}_{p, q}(\mathbf{x})=\frac{1}{Z_{p} Z_{q}} \mathbb{E}_{p, q}\left[k\left(\mathbf{x}, \mathbf{y}^{+}\right) k\left(\mathbf{x}, \mathbf{y}^{-}\right)\left(\mathbf{y}^{+}-\mathbf{y}^{-}\right)\right] . \tag{11}
\end{equation*}
$$

Here, the vector difference reduces to $\mathbf{y}^{+}-\mathbf{y}^{-}$; the weight is computed from two kernels and normalized jointly. This form is an instantiation of Eq. (7). It is easy to see that $\mathbf{V}$ is anti-symmetric: $\mathbf{V}_{p, q}=-\mathbf{V}_{q, p}$. In general, our method does not require $\mathbf{V}$ to be decomposed into attraction and repulsion; it only requires $\mathbf{V}=0$ when $p=q$.

Kernel. The kernel $k(\cdot, \cdot)$ can be a function that measures the similarity. In this paper, we adopt:
$$
\begin{equation*}
k(\mathbf{x}, \mathbf{y})=\exp \left(-\frac{1}{\tau}\|\mathbf{x}-\mathbf{y}\|\right), \tag{12}
\end{equation*}
$$
$\underset{\sim}{\text { where }} \tau$ is a temperature and $\|\cdot\|$ is $\ell_{2}$-distance. We view $\tilde{k}(\mathbf{x}, \mathbf{y}) \triangleq \frac{1}{Z} k(\mathbf{x}, \mathbf{y})$ as a normalized kernel, which absorbs the normalization in Eq. (11).
In practice, we implement $\tilde{k}$ using a softmax operation, with logits given by $-\frac{1}{\tau}\|\mathbf{x}-\mathbf{y}\|$, where the softmax is taken over $\mathbf{y}$. This softmax operation is similar to that of InfoNCE (Oord et al., 2018) in contrastive learning. In our implementation, we further apply an extra softmax normalization over the set of $\{\mathbf{x}\}$ within a batch, which slightly improves performance in practice. This additional normalization does not alter the antisymmetric property of the resulting $\mathbf{V}$.

Equilibrium and Matched Distributions. Since our training loss in Eq. (6) encourages minimizing $\|\mathbf{V}\|^{2}$, we hope

Algorithm 1 Training Loss. Note: for brevity, here the negative samples y_neg are from the same batch of generated data, though they can include other source of negatives.
```
# f: generator
# y_pos: [N_pos, D], data samples
e = randn([N, C]) # noise
x = f(e) # [N, D], generated samples
y_neg = x # reuse x as negatives
V = compute_V(x, y_pos, y_neg)
x_drifted = stopgrad(x + V)
loss = mse_loss(x - x_drifted)
```

that $\mathbf{V} \approx \mathbf{0}$ leads to $q \approx p$. While this implication does not hold for arbitrary choices of $\mathbf{V}$, we empirically observe that decreasing the value of $\|\mathbf{V}\|^{2}$ correlates with improved generation quality. In Appendix C.1, we provide an identifiability heuristic: for our kernelized construction, the zerodrift condition imposes a large set of bilinear constraints on $(p, q)$, and under mild non-degeneracy assumptions this forces $p$ and $q$ to match (approximately).

Stochastic Training. In stochastic training (e.g., mini-batch optimization), we estimate $\mathbf{V}$ by approximating the expectations in Eq. (11) with empirical means. For each training step, we draw $N$ samples of noise $\boldsymbol{\epsilon} \sim p_{\boldsymbol{\epsilon}}$ and compute a batch of $\mathbf{x}=f_{\theta}(\boldsymbol{\epsilon}) \sim q$. The generated samples also serve as the negative samples in the same batch, i.e., $\mathbf{y}^{-} \sim q$. On the other hand, we sample $N_{\text {pos }}$ data points $\mathbf{y}^{+} \sim p_{\text {data }}$. The drifting field $\mathbf{V}$ is computed in this batch of positive and negative samples. Alg. 1 provide the pseudocode for such a training step, where compute_V is given in Section A.1.

\subsection*{3.4. Drifting in Feature Space}

Thus far, we have defined the objective (6) directly in the raw data space. Our formulation can be extended to any feature space. Let $\phi$ denote a feature extractor (e.g., an image encoder) operating on real or generated samples. We rewrite the loss (6) in the feature space as:
$$
\begin{equation*}
\mathbb{E}\left[\| \phi(\mathbf{x})-\text { stopgrad }(\phi(\mathbf{x})+\mathbf{V}(\phi(\mathbf{x}))) \|^{2}\right] . \tag{13}
\end{equation*}
$$

Here, $\mathbf{x}=f_{\theta}(\boldsymbol{\epsilon})$ is the output (e.g., images) of the generator. $\mathbf{V}$ is defined in the feature space: in practice, this means that $\phi\left(\mathbf{y}^{+}\right)$and $\phi\left(\mathbf{y}^{-}\right)$serve as the positive/negative samples. It is worth noting that feature encoding is a training-time operation and is not used at inference time.

This can be further extended to multiple features, e.g., at
multiple scales and locations:
$$
\begin{equation*}
\sum_{j} \mathbb{E}\left[\left\|\phi_{j}(\mathbf{x})-\operatorname{stopgrad}\left(\phi_{j}(\mathbf{x})+\mathbf{V}\left(\phi_{j}(\mathbf{x})\right)\right)\right\|^{2}\right] . \tag{14}
\end{equation*}
$$

Here, $\phi_{j}$ represents the feature vectors at the $j$-th scale and/or location from an encoder $\phi$. With a ResNet-style image encoder (He et al., 2016), we compute drifting losses across multiple scales and locations, which provides richer gradient information for training.

The feature extractor plays an important role in the generation of high-dimensional data. As our method is based on the kernel $k(\cdot, \cdot)$ for characterizing sample similarities, it is desired for semantically similar samples to stay close in the feature space. This goal is aligned with self-supervised learning (e.g., He et al. 2020; Chen et al. 2020a). We use pre-trained self-supervised models as the feature extractor.

Relation to Perceptual Loss. Our feature-space loss is related to perceptual loss (Zhang et al., 2018) but is conceptually different. The perceptual loss minimizes: $\left\|\phi(\mathbf{x})-\phi\left(\mathbf{x}_{\text {target }}\right)\right\|_{2}^{2}$, that is, the regression target is $\phi\left(\mathbf{x}_{\text {target }}\right)$ and requires pairing $\mathbf{x}$ with its target. In contrast, our regression target in (13) is $\phi(\mathbf{x})+\mathbf{V}(\phi(\mathbf{x}))$, where the drifting is in the feature space and requires no pairing. In principle, our feature-space loss aims to match the pushforward distributions $\phi_{\#} q$ and $\phi_{\#} p$.

Relation to Latent Generation. Our feature-space loss is orthogonal to the concept of generators in the latent space (e.g., Latent Diffusion (Rombach et al., 2022)). In our case, when using $\phi$, the generator $f$ can still produce outputs in the pixel space or the latent space of a tokenizer. If the generator $f$ is in the latent space and the feature extractor $\phi$ is in the pixel space, the tokenizer decoder is applied before extracting features from $\phi$.

\subsection*{3.5. Classifier-Free Guidance}

Classifier-free guidance (CFG) (Ho \& Salimans, 2022) improves generation quality by extrapolating between classconditional and unconditional distributions. Our method naturally supports a related form of guidance.

In our model, given a class label $c$ as the condition, the underlying target distribution $p$ now becomes $p_{\text {data }}(\cdot \mid c)$, from which we can draw positive samples: $\mathbf{y}^{+} \sim p_{\text {data }}(\cdot \mid c)$. To achieve guidance, we draw negative samples either from generated samples or real samples from different classes. Formally, the negative sample distribution is now:
$$
\begin{equation*}
\tilde{q}(\cdot \mid c) \triangleq(1-\gamma) q_{\theta}(\cdot \mid c)+\gamma p_{\text {data }}(\cdot \mid \varnothing) \tag{15}
\end{equation*}
$$

Here, $\gamma \in[0,1)$ is a mixing rate, and $p_{\text {data }}(\cdot \mid \varnothing)$ denotes the unconditional data distribution ${ }^{2}$.

The goal of learning is to find $\tilde{q}(\cdot \mid c)=p_{\text {data }}(\cdot \mid c)$. Substitut-
ing it into (15), we obtain:
$$
\begin{equation*}
q_{\theta}(\cdot \mid c)=\alpha p_{\text {data }}(\cdot \mid c)-(\alpha-1) p_{\text {data }}(\cdot \mid \varnothing) \tag{16}
\end{equation*}
$$
where $\alpha=\frac{1}{1-\gamma} \geq 1$. This implies that $q_{\theta}(\cdot \mid c)$ is to approximate a linear combination of conditional and unconditional data distributions. This follows the spirit of original CFG.

In practice, Eq. (15) means that we sample extra negative examples from the data in $p_{\text {data }}(\cdot \mid \varnothing)$, in addition to the generated data. The distribution $q_{\theta}(\cdot \mid c)$ corresponds to a classconditional network $f_{\theta}(\cdot \mid c)$, similar to common practice (Ho \& Salimans, 2022). We note that, in our method, CFG is a training-time behavior by design: the one-step (1-NFE) property is preserved at inference time.

\section*{4. Implementation for Image Generation}

We describe our implementation for image generation on ImageNet (Deng et al., 2009) at resolution $256 \times 256$. Full implementation details are provided in Appendix A.

Tokenizer. By default, we perform generation in latent space (Rombach et al., 2022). We adopt the standard SDVAE tokenizer, which produces a $32 \times 32 \times 4$ latent space in which generation is performed.

Architecture. Our generator ( $f_{\theta}$ ) has a DiT-like (Peebles \& Xie, 2023) architecture. Its input is $32 \times 32 \times 4$-dim Gaussian noise $\boldsymbol{\epsilon}$, and its output is the generated latent $\mathbf{x}$ of the same dimension. We use a patch size of 2, i.e., like DiT/2. Our model uses adaLN-zero (Peebles \& Xie, 2023) for processing class-conditioning or other extra conditioning.

CFG conditioning. We follow (Geng et al., 2025b) and adopt CFG-conditioning. At training time, a CFG scale $\alpha$ (Eq. (16)) is randomly sampled. Negative samples are prepared based on $\alpha$ (Eq. (15)), and the network is conditioned on this value. At inference time, $\alpha$ can be freely specified and varied without retraining. Details are in A.7.

Batching. The pseudo-code in Alg. 1 describes a batch of $N=N_{\text {neg }}$ generated samples. In practice, when class labels are involved, we sample a batch of $N_{\mathrm{c}}$ class labels. For each label, we perform Alg. 1 independently. Accordingly, the effective batch size is $B=N_{\mathrm{c}} \times N$, which consists of $N_{\mathrm{c}} \times N$ negatives and $N_{\mathrm{c}} \times N_{\text {pos }}$ positives.

We define a "training epoch" based on the number of generated samples $\mathbf{x}$. In particular, each iteration generates $B$ samples, and one epoch corresponds to $N_{\text {data }} / B$ iterations for a dataset of size $N_{\text {data }}$.

Feature Extractor. Our model is trained with drifting loss in a feature space (Sec. 3.4). The feature extractor $\phi$ is an image encoder. We mainly consider a ResNet-style (He

\footnotetext{
${ }^{2}$ This should be the data distribution excluding the class $c$. For simplicity, we use the unconditional data distribution.
}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/14b7cf7f-b68f-4015-b13d-5b756b999ce4-06.jpg?height=644&width=821&top_left_y=229&top_left_x=189}
\captionsetup{labelformat=empty}
\caption{Figure 3. Evolution of the generated distribution. The distribution $q$ (orange) evolves toward a bimodal target $p$ (blue) during training. We show three initializations of $q$ : (top): initialized between the two modes; (middle): initialized far from both modes; (bottom): initialized collapsed onto one mode. Across all initializations, our method approximates the target distribution without mode collapse.}
\end{figure}
et al., 2016) encoder, pre-trained by self-supervised learning, e.g., MoCo (He et al., 2020) and SimCLR (Chen et al., 2020a). When these pre-trained models operate in pixel space, we apply the VAE decoder to map our generator's latent-space output back to pixel space for feature extraction. Gradients are backpropagated through the feature encoder and VAE decoder. We also study an MAE (He et al., 2022) pre-trained in latent space (detailed in A.3).

For all ResNet-style models, features are extracted from multiple stages (i.e., multi-scale feature maps). The drifting loss in (13) is computed at each scale and then combined. We elaborate on the details in A.6.

Pixel-space Generation. While our experiments primarily focus on latent-space generation, our models support pixelspace generation. In this case, $\boldsymbol{\epsilon}$ and $\mathbf{x}$ are both $256 \times 256 \times 3$. We use a patch size of 16 (i.e., DiT/16). The feature extractor $\phi$ is directly on the pixel space.

\section*{5. Experiments}

\subsection*{5.1. Toy Experiments}

Evolution of the generated distribution. Figure 3 visualizes a 2D toy case, where $q$ evolves toward a bimodal distribution $p$ at training time, under three initializations.

In this toy example, our method approximates the target distribution without exhibiting mode collapse. This holds even when $q$ is initialized in a collapsed single-mode state (bottom). This provides intuition into why our method is robust to mode collapse: if $q$ collapses onto one mode,

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/14b7cf7f-b68f-4015-b13d-5b756b999ce4-06.jpg?height=429&width=822&top_left_y=227&top_left_x=1065}
\captionsetup{labelformat=empty}
\caption{Figure 4. Evolution of samples. We show generated points sampled at different training iterations, along with their loss values. The loss (whose value equals $\|V\|^{2}$ ) decreases as the distribution converges to the target. (y-axis is log-scale.)}
\end{figure}

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 1. Importance of anti-symmetry: breaking the antisymmetry leads to failure. Here, the anti-symmetric case is defined in Eq. (10) and Eq. (11); other destructive cases are defined in similar ways. (Setting: B/2 model, 100 epochs)}
\begin{tabular}{|l|l|l|}
\hline case & drifting field V & FID \\
\hline anti-symmetry (default) & $\mathbf{V}^{+}-\mathbf{V}^{-}$ & 8.46 \\
\hline $1.5 \times$ attraction & $1.5 \mathrm{~V}^{+}-\mathrm{V}^{-}$ & 41.05 \\
\hline $1.5 \times$ repulsion & $\mathbf{V}^{+}-1.5 \mathbf{V}^{-}$ & 46.28 \\
\hline $2.0 \times$ attraction & $2 \mathbf{V}^{+}-\mathbf{V}^{-}$ & 86.16 \\
\hline $2.0 \times$ repulsion & $\mathbf{V}^{+}-2 \mathbf{V}^{-}$ & 112.84 \\
\hline attraction-only & $\mathbf{V}^{+}$ & 177.14 \\
\hline
\end{tabular}
\end{table}
other modes of $p$ will attract the samples, allowing them to continue moving and pushing $q$ to continue evolving.

Evolution of the samples. Figure 4 shows the training process on two 2D cases. A small MLP generator is trained. The loss (whose value equals $\|\mathbf{V}\|^{2}$ ) decreases as the generated distribution converges to the target. This is in line with our motivation that reducing the drift and pushing towards the equilibrium will approximately yield $p=q$.

\subsection*{5.2. ImageNet Experiments}

We evaluate our models on ImageNet $256 \times 256$. Ablation studies use a B/2 model on the SD-VAE latent space, trained for 100 epochs. The drifting loss is in a feature space computed by a latent-MAE encoder. We report FID (Heusel et al., 2017) on 50 K generated images. We analyze the results as follows.

Anti-symmetry. Our derivation of equilibrium requires the drifting field to be anti-symmetric; see Eq. (3). In Table 1, we conduct a destructive study that intentionally breaks this anti-symmetry. The anti-symmetric case (our ablation default) works well, while other cases fail catastrophically.

Intuitively, for a sample $\mathbf{x}$, we want attraction from $p$ to be canceled by repulsion from $q$ when $p$ and $q$ match. This equilibrium is not achieved in the destructive cases.

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 2. Allocation of positive and negative samples. In both subtables, we control the total compute by fixing the epochs (100) and the batch size $B=N_{\mathrm{c}} \times N_{\text {pos }}$ (4096). Here, $N_{\mathrm{c}}$ is for class labels. Under the same budget, increasing positive samples (left) and negative samples (right) improves generation quality. (Setting: B/2 model, 100 epochs)}
\begin{tabular}{ccc|c|r}
\hline$N_{\mathrm{c}}$ & $N_{\text {pos }}$ & $N_{\text {neg }}$ & $B$ & FID \\
\hline 64 & 1 & 64 & 4096 & 20.43 \\
64 & 16 & 64 & 4096 & 10.39 \\
64 & 32 & 64 & 4096 & 8.97 \\
64 & $\mathbf{6 4}$ & 64 & 4096 & $\mathbf{8 . 4 6}$ \\
\hline
\end{tabular}
\end{table}

\begin{tabular}{ccc|c|r}
\hline$N_{\mathrm{c}}$ & $N_{\text {pos }}$ & $N_{\text {neg }}$ & $B$ & FID \\
\hline 512 & 8 & 8 & 4096 & 11.82 \\
256 & 16 & 16 & 4096 & 10.16 \\
128 & 32 & 32 & 4096 & 9.32 \\
64 & 64 & $\mathbf{6 4}$ & 4096 & $\mathbf{8 . 4 6}$ \\
\hline
\end{tabular}

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 3. Feature space for drifting. We compare self-supervised learning (SSL) encoders. Standard SimCLR and MoCo encoders achieve competitive results, whereas our customized latent-MAE performs best and benefits from increased width and longer training. (Generator setting: B/2 model, 100 epochs)}
\begin{tabular}{|l|l|l|l|l|l|}
\hline \multirow[b]{2}{*}{SSL method} & \multicolumn{4}{|c|}{feature encoder ( $\phi$ )} & \multirow[b]{2}{*}{FID} \\
\hline & arch & block & width & SSL ep. & \\
\hline SimCLR & ResNet & bottleneck & 256 & 800 & 11.05 \\
\hline MoCo-v2 & ResNet & bottleneck & 256 & 800 & 8.41 \\
\hline latent-MAE (default) & ResNet & basic & 256 & 192 & 8.46 \\
\hline latent-MAE & ResNet & basic & 384 & 192 & 7.26 \\
\hline latent-MAE & ResNet & basic & 512 & 192 & 6.49 \\
\hline latent-MAE & ResNet & basic & 640 & 192 & 6.30 \\
\hline latent-MAE & ResNet & basic & 640 & 1280 & 4.28 \\
\hline latent-MAE + cls ft & ResNet & basic & 640 & 1280 & 3.36 \\
\hline
\end{tabular}
\end{table}

Allocation of Positive and Negative Samples. Our method samples positive and negative examples to estimate $\mathbf{V}$ (see Alg. 1). In Table 2, we study the effect of $N_{\text {pos }}$ and $N_{\text {neg }}$, under fixed epochs and fixed batch size $B$.

Table 2 shows that using larger $N_{\text {pos }}$ and $N_{\text {neg }}$ is beneficial. Larger sample sizes are expected to improve the accuracy of the estimated $\mathbf{V}$ and hence the generation quality. This observation aligns with results in contrastive learning (Oord et al., 2018; He et al., 2020; Chen et al., 2020a), in which larger sample sets improve representation learning.

Feature Space for Drifting. Our model computes the drifting loss in a feature space (Sec. 3.4). Table 3 compares the feature encoders. Using the public pre-trained encoders from SimCLR (Chen et al., 2020a) and MoCo v2 (Chen et al., 2020b), our method obtains decent results.

These standard encoders operate in the pixel domain, which requires running the VAE decoder at training. To circumvent this, we pre-train a ResNet-style model with the MAE objective (He et al., 2022), directly on the latent space. The feature space produced by this "latent-MAE" performs strongly (Table 3). Increasing the MAE encoder width and the number of pre-training epochs both improve generation quality; fine-tuning it with a classifier ('cls ft') boosts the results further to 3.36 FID.

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 4. From ablation to final setting. We train our model for more epochs, adjust hyper-parameters for this regime, and use a larger model size.}
\begin{tabular}{l|lr|l}
\hline case & arch & ep & FID \\
\hline (a) baseline (from Table 3) & $\mathrm{B} / 2$ & 100 & 3.36 \\
\hline (b) longer & $\mathrm{B} / 2$ & 320 & 2.51 \\
(c) longer + hyper-param. & $\mathrm{B} / 2$ & 1280 & 1.75 \\
(d) larger model & $\mathrm{L} / 2$ & 1280 & $\mathbf{1 . 5 4}$ \\
\hline
\end{tabular}
\end{table}

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 5. System-level comparison: ImageNet $\mathbf{2 5 6} \boldsymbol{\times} \mathbf{2 5 6}$ generation in latent space. FID is on 50K images, all reported with CFG if applicable. The parameter numbers are "generator + decoder". All generators are trained from scratch (i.e., not distilled).}
\begin{tabular}{|l|l|l|l|l|l|}
\hline method & space & params & NFE & FID↓ & IS ↑ \\
\hline \multicolumn{6}{|l|}{Multi-step Diffusion/Flows} \\
\hline DiT-XL/2 (Peebles \& Xie, 2023) & SD-VAE & $675 \mathrm{M}+49 \mathrm{M}$ & $250 \times 2$ & 2.27 & 278.2 \\
\hline SiT-XL/2 (Ma et al., 2024) & SD-VAE & $675 \mathrm{M}+49 \mathrm{M}$ & $250 \times 2$ & 2.06 & 270.3 \\
\hline SiT-XL/2+REPA (Yu et al., 2024) & SD-VAE & $675 \mathrm{M}+49 \mathrm{M}$ & $250 \times 2$ & 1.42 & 305.7 \\
\hline LightningDiT-XL/2 (Yao et al., 2025) & VA-VAE & $675 \mathrm{M}+70 \mathrm{M}$ & $250 \times 2$ & 1.35 & 295.3 \\
\hline $\mathrm{RAE}+\mathrm{DiT}^{\mathrm{DH}}-\mathrm{XL} / 2$ (Zheng et al., 2025) & RAE & $839 \mathrm{M}+415 \mathrm{M}$ & $50 \times 2$ & 1.13 & 262.6 \\
\hline \multicolumn{6}{|l|}{Single-step Diffusion/Flows} \\
\hline iCT-XL/2 (Song \& Dhariwal, 2023) & SD-VAE & $675 \mathrm{M}+49 \mathrm{M}$ & 1 & 34.24 & - \\
\hline Shortcut-XL/2 (Frans et al., 2024) & SD-VAE & $675 \mathrm{M}+49 \mathrm{M}$ & 1 & 10.60 & - \\
\hline MeanFlow-XL/2 (Geng et al., 2025a) & SD-VAE & $676 \mathrm{M}+49 \mathrm{M}$ & 1 & 3.43 & - \\
\hline AdvFlow-XL/2 (Lin et al., 2025) & SD-VAE & $673 \mathrm{M}+49 \mathrm{M}$ & 1 & 2.38 & 284.2 \\
\hline iMeanFlow-XL/2 (Geng et al., 2025b) & SD-VAE & $610 \mathrm{M}+49 \mathrm{M}$ & 1 & 1.72 & 282.0 \\
\hline \multicolumn{6}{|l|}{Drifting Models} \\
\hline Drifting Model, B/2 & SD-VAE & $133 \mathrm{M}+49 \mathrm{M}$ & 1 & 1.75 & 263.2 \\
\hline Drifting Model, L/2 & SD-VAE & $463 \mathrm{M}+49 \mathrm{M}$ & 1 & 1.54 & 258.9 \\
\hline
\end{tabular}
\end{table}

The comparison in Table 3 shows that the quality of the feature encoder plays an important role. We hypothesize that this is because our method depends on a kernel $k(\cdot, \cdot)$ (see Eq. (12)) to measure sample similarity. Samples that are closer in feature space generally yield stronger drift, providing richer training signals. This goal is aligned with the motivation of self-supervised learning. A strong feature encoder reduces the occurrence of a nearly "flat" kernel (i.e., $k(\cdot, \cdot)$ vanishes because all samples are far away).

On the other hand, we report that we were unable to make our method work on ImageNet without a feature encoder. In this case, the kernel may fail to effectively describe similarity, even in the presence of a latent VAE. We leave further study of this limitation for future work.

System-level Comparisons. In addition to the ablation setting, we train stronger variants and summarize them in Table 4. We compare with previous methods in Table 5.

Our method achieves 1.54 FID with native 1-NFE generation. It outperforms all previous 1-NFE methods, which are based on approximating diffusion-/flow-based trajectories. Notably, our Base-size model competes with previous XLsize models. Our best model (FID 1.54) uses a CFG scale of 1.0, which corresponds to "no CFG" in diffusion-based methods. Our CFG formulation exhibits a tradeoff between

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 6. System-level comparison: ImageNet $\mathbf{2 5 6} \boldsymbol{\times} \mathbf{2 5 6}$ generation in pixel space. FID is on 50 K images, all reported with CFG if applicable. The parameter numbers are of the generator. All generators are trained from scratch (i.e., not distilled).}
\begin{tabular}{|l|l|l|l|l|l|}
\hline method & space & params & NFE & FID↓ & IS ↑ \\
\hline \multicolumn{6}{|l|}{Multi-step Diffusion/Flows} \\
\hline ADM-G (Dhariwal \& Nichol, 2021) & pix & 554 M & $250 \times 2$ & 4.59 & 186.7 \\
\hline SiD, UViT/2 (Hoogeboom et al., 2023) & pix & 2.5 B & $1000 \times 2$ & 2.44 & 256.3 \\
\hline VDM++, UViT/2 (Kingma \& Gao, 2023) & pix & 2.5 B & $256 \times 2$ & 2.12 & 267.7 \\
\hline SiD2, UViT/2 (Hoogeboom et al., 2024) & pix & - & $512 \times 2$ & 1.73 & - \\
\hline SiD2, UViT/1 (Hoogeboom et al., 2024) & pix & - & $512 \times 2$ & 1.38 & - \\
\hline JiT-G/16 (Li \& He, 2025) & pix & 2B & $100 \times 2$ & 1.82 & 292.6 \\
\hline PixelDiT/16 (Yu et al., 2025) & pix & 797 M & $200 \times 2$ & 1.61 & 292.7 \\
\hline \multicolumn{6}{|l|}{Single-step Diffusion/Flows} \\
\hline EPG-L/16 (Lei et al., 2025) & pix & 540M & 1 & 8.82 & - \\
\hline \multicolumn{6}{|l|}{GANs} \\
\hline BigGAN (Brock et al., 2018) & pix & 112 M & 1 & 6.95 & 152.8 \\
\hline GigaGAN (Kang et al., 2023) & pix & 569 M & 1 & 3.45 & 225.5 \\
\hline StyleGAN-XL (Sauer et al., 2022) & pix & 166 M & 1 & 2.30 & 265.1 \\
\hline \multicolumn{6}{|l|}{Drifting Models} \\
\hline Drifting Model, B/16 & pix & 134 M & 1 & 1.76 & 299.7 \\
\hline Drifting Model, L/16 & pix & 464 M & 1 & 1.61 & 307.5 \\
\hline
\end{tabular}
\end{table}

FID and IS (see B.3), similar to standard CFG.
We provide uncurated qualitative results in Appendix B.5, Fig. 7-10, with CFG 1.0. Moreover, Fig. 11-15 show a side-by-side comparison with improved MeanFlow (iMF) (Geng et al., 2025b), a recent state-of-the-art one-step method.

Pixel-space Generation. Our method can naturally work without the latent VAE, i.e., the generator $f$ directly produces $256 \times 256 \times 3$ images. The feature encoder is applied on the generated images for computing drifting loss. We adopt a configuration similar to that of the latent variant; implementation details are in Appendix A.

Table 6 compares different pixel-space generators. Our one-step, pixel-space method achieves $\mathbf{1 . 6 1}$ FID, which outperforms or competes with previous multi-step methods. Comparing with other one-step, pixel-space methods (GANs), our method achieves 1.61 FID using only 87 G FLOPs; by comparison, StyleGAN-XL produces 2.30 FID using 1574G FLOPs. More ablations are in B.1.

\subsection*{5.3. Experiments on Robotic Control}

Beyond image generation, we further evaluate our method on robotics control. Our experiment designs and protocols follow Diffusion Policy (Chi et al., 2023). At the core of Diffusion Policy is a multi-step, diffusion-based generator; we replace it with our one-step Drifting Model. We directly compute drifting loss on the raw representations for control, using no feature space. Results are in Table 7. Our 1-NFE model matches or exceeds the state-of-the-art Diffusion Policy that uses 100 NFE. This comparison suggests that Drifting Models can serve as a promising generative model

Table 7. Robotics Control: Comparison with Diffusion Policy. The evaluation protocol follows Diffusion Policy (Chi et al., 2023). This table involves four single-stage tasks and two multi-stage tasks. "Drifting Policy" (ours) replaces the multi-step Diffusion Policy generator with our one-step generator. Success rates are reported as the average over the last 10 checkpoints.

\begin{tabular}{|l|l|l|l|}
\hline \multirow[b]{2}{*}{Task} & \multirow[b]{2}{*}{Setting} & Diffusion Policy & Drifting Policy \\
\hline & & NFE: 100 & NFE: 1 \\
\hline \multicolumn{4}{|l|}{Single-Stage Tasks (State \& Visual Observation)} \\
\hline \multirow[b]{2}{*}{Lift} & State & 0.98 & 1.00 \\
\hline & Visual & 1.00 & 1.00 \\
\hline \multirow{2}{*}{Can} & State & 0.96 & 0.98 \\
\hline & Visual & 0.97 & 0.99 \\
\hline \multirow[b]{2}{*}{ToolHang} & State & 0.30 & 0.38 \\
\hline & Visual & 0.73 & 0.67 \\
\hline \multirow{2}{*}{PushT} & State & 0.91 & 0.86 \\
\hline & Visual & 0.84 & 0.86 \\
\hline \multicolumn{4}{|l|}{Multi-Stage Tasks (State Observation)} \\
\hline \multirow[t]{3}{*}{BlockPush} & Phase 1 & 0.36 & 0.56 \\
\hline & Phase 2 & 0.11 & 0.16 \\
\hline & Phase 1 & 1.00 & 1.00 \\
\hline \multirow[t]{3}{*}{Kitchen} & Phase 2 & 1.00 & 1.00 \\
\hline & Phase 3 & 1.00 & 0.99 \\
\hline & Phase 4 & 0.99 & 0.96 \\
\hline
\end{tabular}
across different domains.

\section*{6. Discussion and Conclusion}

We present Drifting Models, a new paradigm for generative modeling. At the core of our model is the idea of modeling the evolution of pushforward distributions during training. This allows us to focus on the update rule, i.e., $\mathbf{x}_{i+1}= \mathbf{x}_{i}+\Delta \mathbf{x}_{i}$, during the iterative training process. This is in contrast with diffusion-/flow-based models, which perform the iterative update at inference time. Our method naturally performs one-step inference.

Given that our methodology is substantially different, many open questions remain. For example, although we show that $q=p \Rightarrow \mathbf{V}=\mathbf{0}$, the converse implication does not generally hold in theory. While our designed $\mathbf{V}$ performs well empirically, it remains unclear under what conditions $\mathbf{V} \rightarrow \mathbf{0}$ leads to $q \rightarrow p$.

From a practical standpoint, although our paper presents an effective instantiation of drifting modeling, many of our design decisions may remain sub-optimal. For example, the design of the drifting field and its kernels, the feature encoder, and the generator architecture remain open for future exploration.

From a broader perspective, our work reframes iterative neural network training as a mechanism for distribution evolution, in contrast to the differential equations underlying diffusion-/flow-based models. We hope that this perspective will inspire the exploration of other realizations of this mechanism in future work.

\section*{Acknowledgements}

We greatly thank Google TPU Research Cloud (TRC) for granting us access to TPUs. We thank Michael Albergo, Ziqian Zhong, Yilun Xu, Zhengyang Geng, Hanhong Zhao, Jiangqi Dai, Alex Fan, and Shaurya Agrawal for helpful discussions. Mingyang Deng is partially supported by funding from MIT-IBM Watson AI Lab.

\section*{References}

Albergo, M. S., Boffi, N. M., and Vanden-Eijnden, E. Stochastic interpolants: A unifying framework for flows and diffusions. arXiv preprint arXiv:2303.08797, 2023.

Boffi, N. M., Albergo, M. S., and Vanden-Eijnden, E. Flow map matching with stochastic interpolants: A mathematical framework for consistency models. TMLR, 2025.

Brock, A., Donahue, J., and Simonyan, K. Large scale GAN training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096, 2018.

Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. A simple framework for contrastive learning of visual representations. In ICML, 2020a.

Chen, X. and He, K. Exploring simple siamese representation learning. In CVPR, pp. 15750-15758, 2021.

Chen, X., Fan, H., Girshick, R., and He, K. Improved baselines with momentum contrastive learning. arXiv preprint arXiv:2003.04297, 2020b.

Cheng, Y. Mean shift, mode seeking, and clustering. TPAMI, 1995.

Chi, C., Feng, S., Du, Y., Xu, Z., Cousineau, E., Burchfiel, B., and Song, S. Diffusion policy: Visuomotor policy learning via action diffusion. In RSS, 2023.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. ImageNet: A large-scale hierarchical image database. In CVPR, pp. 248-255. Ieee, 2009.

Dhariwal, P. and Nichol, A. Diffusion models beat GANs on image synthesis. NeurIPS, 34:8780-8794, 2021.

Dinh, L., Sohl-Dickstein, J., and Bengio, S. Density estimation using real NVP. arXiv preprint arXiv:1605.08803, 2016.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.

Dziugaite, G. K., Roy, D. M., and Ghahramani, Z. Training generative neural networks via maximum mean discrepancy optimization. arXiv preprint arXiv:1505.03906, 2015.

Esser, P., Rombach, R., and Ommer, B. Taming transformers for high-resolution image synthesis. In CVPR, pp. 1287312883, 2021.

Frans, K., Hafner, D., Levine, S., and Abbeel, P. One step diffusion via shortcut models. arXiv preprint arXiv:2410.12557, 2024.

Geng, Z., Deng, M., Bai, X., Kolter, J. Z., and He, K. Mean flows for one-step generative modeling. arXiv preprint arXiv:2505.13447, 2025a.

Geng, Z., Lu, Y., Wu, Z., Shechtman, E., Kolter, J. Z., and He , K. Improved mean flows: On the challenges of fastforward generative models. arXiv preprint arXiv:2512.02012, 2025b.

Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y. Generative adversarial nets. NeurIPS, 2014.

Hadsell, R., Chopra, S., and LeCun, Y. Dimensionality reduction by learning an invariant mapping. In CVPR, pp. 1735-1742, 2006.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In CVPR, pp. 770-778, 2016.

He, K., Fan, H., Wu, Y., Xie, S., and Girshick, R. Momentum contrast for unsupervised visual representation learning. In CVPR, pp. 9729-9738, 2020.

He, K., Chen, X., Xie, S., Li, Y., Dollár, P., and Girshick, R. Masked autoencoders are scalable vision learners. In CVPR, 2022.

Henry, A., Dachapally, P. R., Pawar, S. S., and Chen, Y. Query-key normalization for transformers. In EMNLP, pp. 4246-4253, 2020.

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., and Hochreiter, S. GANs trained by a two time-scale update rule converge to a local nash equilibrium. NeurIPS, 2017.

Ho, J. and Salimans, T. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. NeurIPS, 33:6840-6851, 2020.

Hoogeboom, E., Heek, J., and Salimans, T. Simple diffusion: End-to-end diffusion for high resolution images. In ICML, pp. 13213-13232. PMLR, 2023.

Hoogeboom, E., Mensink, T., Heek, J., Lamerigts, K., Gao, R., and Salimans, T. Simpler diffusion (SiD2): 1.5 fid on ImageNet512 with pixel-space diffusion. arXiv preprint arXiv:2410.19324, 2024.

Ioffe, S. and Szegedy, C. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In ICML, pp. 448-456. pmlr, 2015.

Kang, M. and Park, J. ContraGAN: Contrastive learning for conditional image generation. NeurIPS, 33:21357-21369, 2020.

Kang, M., Zhu, J.-Y., Zhang, R., Park, J., Shechtman, E., Paris, S., and Park, T. Scaling up GANs for text-to-image synthesis. In CVPR, pp. 10124-10134, 2023.

Kingma, D. and Gao, R. Understanding diffusion objectives as the ELBO with simple data augmentation. NeurIPS, 36:65484-65516, 2023.

Kingma, D. P. and Welling, M. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

Lei, J., Liu, K., Berner, J., Yu, H., Zheng, H., Wu, J., and Chu, X. There is no VAE: End-to-end pixel-space generative modeling via self-supervised pre-training. arXiv preprint arXiv:2510.12586, 2025.
$\mathrm{Li}, \mathrm{T}$. and He, K. Back to basics: Let denoising generative models denoise. arXiv preprint arXiv:2511.13720, 2025.

Li, Y., Swersky, K., and Zemel, R. Generative moment matching networks. In ICML, pp. 1718-1727. PMLR, 2015.

Lin, S., Yang, C., Lin, Z., Chen, H., and Fan, H. Adversarial flow models. arXiv preprint arXiv:2511.22475, 2025.

Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., and Le, M. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Liu, X., Gong, C., and Liu, Q. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. In ICLR, 2019.

Luo, W., Hu, T., Zhang, S., Sun, J., Li, Z., and Zhang, Z. Diff-Instruct: A universal approach for transferring knowledge from pre-trained diffusion models. NeurIPS, 36:76525-76546, 2023.

Ma, N., Goldstein, M., Albergo, M. S., Boffi, N. M., VandenEijnden, E., and Xie, S. SiT: Exploring flow and diffusionbased generative models with scalable interpolant transformers. In ECCV, pp. 23-40. Springer, 2024.

Oord, A. v. d., Li, Y., and Vinyals, O. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.

Peebles, W. and Xie, S. Scalable diffusion models with transformers. In CVPR, pp. 4195-4205, 2023.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning transferable visual models from natural language supervision. In ICML, pp. 8748-8763. PmLR, 2021.

Rezende, D. and Mohamed, S. Variational inference with normalizing flows. In ICML, pp. 1530-1538. PMLR, 2015.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In CVPR, pp. 10684-10695, 2022.

Ronneberger, O., Fischer, P., and Brox, T. U-Net: Convolutional networks for biomedical image segmentation. In MICCAI, 2015.

Salimans, T. and Ho, J. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.

Sauer, A., Schwarz, K., and Geiger, A. StyleGAN-XL: Scaling StyleGAN to large diverse datasets. In SIGGRAPH, pp. 1-10, 2022.

Shazeer, N. GLU variants improve transformer. arXiv preprint arXiv:2002.05202, 2020.

Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., and Ganguli, S. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, pp. 2256-2265. pmlr, 2015.

Song, Y. and Dhariwal, P. Improved techniques for training consistency models. arXiv preprint arXiv:2310.14189, 2023.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

Song, Y., Dhariwal, P., Chen, M., and Sutskever, I. Consistency models. 2023.

Stoica, G., Ramanujan, V., Fan, X., Farhadi, A., Krishna, R., and Hoffman, J. Contrastive flow matching. arXiv preprint arXiv:2506.05350, 2025.

Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and Liu, Y. Roformer: Enhanced transformer with totary position embedding. IJON, 568:127063, 2024.

Unterthiner, T., Nessler, B., Seward, C., Klambauer, G., Heusel, M., Ramsauer, H., and Hochreiter, S. Coulomb GANs: Provably optimal nash qquilibria via potential fields. arXiv preprint arXiv:1708.08819, 2017.

Woo, S., Debnath, S., Hu, R., Chen, X., Liu, Z., Kweon, I. S., and Xie, S. ConvNeXt V2: Co-designing and scaling ConvNets with masked autoencoders. In CVPR, pp. 16133-16142, 2023.

Wu, Y. and He, K. Group normalization. In ECCV, pp. 3-19, 2018.

Yao, J., Yang, B., and Wang, X. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In CVPR, pp. 15703-15712, 2025.

Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W. T., and Park, T. One-step diffusion with distribution matching distillation. In CVPR, pp. 66136623, 2024.

Yu, S., Kwak, S., Jang, H., Jeong, J., Huang, J., Shin, J., and Xie, S. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940, 2024.

Yu, Y., Xiong, W., Nie, W., Sheng, Y., Liu, S., and Luo, J. PixelDiT: Pixel diffusion transformers for image generation. arXiv preprint arXiv:2511.20645, 2025.

Zhai, S., Zhang, R., Nakkiran, P., Berthelot, D., Gu, J., Zheng, H., Chen, T., Bautista, M. A., Jaitly, N., and Susskind, J. Normalizing flows are capable generative models. arXiv preprint arXiv:2412.06329, 2024.

Zhang, B. and Sennrich, R. Root mean square layer normalization. NeurIPS, 32, 2019.

Zhang, R., Isola, P., Efros, A. A., Shechtman, E., and Wang, O. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018.

Zheng, B., Ma, N., Tong, S., and Xie, S. Diffusion transformers with representation autoencoders. arXiv preprint arXiv:2510.11690, 2025.

Zhou, L., Ermon, S., and Song, J. Inductive moment matching. arXiv preprint arXiv:2503.07565, 2025.

Zhou, M., Zheng, H., Wang, Z., Yin, M., and Huang, H. Score identity distillation: Exponentially fast distillation of pretrained diffusion models for one-step generation. In ICML, 2024.

\section*{A. Additional Implementation Details}

Table 8 summarizes the configurations and hyper-parameters for ablation studies and system-level comparisons. We provide detailed experimental configurations for reproducibility. All ablation studies share a common default setup, while system-level comparisons use scaled-up configurations. More implementation details are described as follows.

\section*{A.1. Pseudo-code for Computing Drifting Field V}

Alg. 2 provides the pseudo-code for computing V. The computation is based on taking empirical means in Eq. (11) and (12), which are implemented as softmax over $\boldsymbol{y}$-sample axis. In practice, we further normalize over the $\mathbf{x}$-sample axis, also implemented as softmax on the same logit matrix. We ablate its influence in B.2.

It is worth noting that this implementation preserves the desired property of $\mathbf{V}$. In principle, this implementation can be viewed as a Monte Carlo estimation of a drifting field:
$$
\begin{equation*}
\mathbf{V}_{p, q}(\mathbf{x})=\mathbb{E}_{\mathcal{B}, p, q}\left[\tilde{K}_{\mathcal{B}}\left(\mathbf{x}, \mathbf{y}^{+}\right) \tilde{K}_{\mathcal{B}}\left(\mathbf{x}, \mathbf{y}^{-}\right)\left(\mathbf{y}^{+}-\mathbf{y}^{-}\right)\right], \tag{17}
\end{equation*}
$$
where $\mathcal{B}$ consists of other samples in the batch and $\tilde{K}_{\mathcal{B}}$ denote normalizing the distance based on statistics within $\mathcal{B}$. This $\mathbf{V}$ also satisfies $\mathbf{V}_{p, p}(\mathbf{x})=\mathbf{0}$, since when $p=q$, the term $\tilde{K}_{\mathcal{B}}\left(\mathbf{y}^{+}, x\right) \tilde{K}_{\mathcal{B}}\left(\mathbf{y}^{-}, x\right)\left(\mathbf{y}^{+}-\mathbf{y}^{-}\right)$cancels out with the term $\tilde{K}_{\mathcal{B}}\left(\mathbf{y}^{-}, x\right) \tilde{K}_{\mathcal{B}}\left(\mathbf{y}^{+}, x\right)\left(\mathbf{y}^{-}-\mathbf{y}^{+}\right)$.

\section*{A.2. Generator Architecture}

Input and output. The input to the generator consists of random noise along with conditioning:
$$
f_{\theta}:(\boldsymbol{\epsilon}, c, \alpha) \mapsto \mathbf{x}
$$
where $\boldsymbol{\epsilon}$ denotes random variables, $c$ is a class label, and $\alpha$ is the CFG strength. $\boldsymbol{\epsilon}$ may consist of both continuous random variables (e.g., Gaussian noise) and discrete ones (e.g., uniformly distributed integers; see random style embeddings). For latent-space models, the output $\mathbf{x} \in \mathbb{R}^{32 \times 32 \times 4}$ is in the SD-VAE latent space. For pixel-space models, the output $\mathbf{x} \in \mathbb{R}^{256 \times 256 \times 3}$ is directly an image.

Transformer. We adopt a DiT-style Transformer (Peebles \& Xie, 2023). Following (Yao et al., 2025), we use SwiGLU (Shazeer, 2020), RoPE (Su et al., 2024), RMSNorm (Zhang \& Sennrich, 2019), and QK-Norm (Henry et al., 2020). The input Gaussian noise is patchified into $256=16 \times 16$ tokens (patch size $2 \times 2$ for latent, $16 \times 16$ for pixel). Conditioning ( $c, \alpha$ ) is processed by adaLN, as well as by in-context conditioning tokens. The output tokens are unpatchified back to the target shape.

In-context tokens. Following (Li \& He, 2025), we prepend 16 learnable tokens to the sequence for in-context conditioning (Peebles \& Xie, 2023). These tokens are formed by
```
Algorithm 2 Computing the drifting field V.
def compute_V(x, y_pos, y_neg, T):
    # x: [N, D]
    # y_pos: [N_pos, D]
    # y_neg: [N_neg, D]
    # T: temperature
    # compute pairwise distance
    dist_pos = cdist(x, y_pos) # [N, N_pos]
    dist_neg = cdist(x, y_neg) # [N, N_neg]
    # ignore self (if y_neg is x)
    dist_neg += eye(N) * 1e6
    # compute logits
    logit_pos = -dist_pos / T
    logit_neg = -dist_neg / T
    # concat for normalization
    logit = cat([logit_pos, logit_neg], dim=1)
    # normalize along both dimensions
    A_row = logit.softmax(dim=-1)
    A_col = logit.softmax(dim=-2)
    A = sqrt(A_row * A_col)
    # back to [N, N_pos] and [N, N_neg]
    A_pos, A_neg = split(A, [N_pos,], dim=1)
    # compute the weights
    W_pos = A_pos # [N, N_pos]
    W_neg = A_neg # [N, N_neg]
    W_pos *= A_neg.sum(dim=1,keepdim=True)
    W_neg *= A_pos.sum(dim=1,keepdim=True)
    drift_pos = W_pos @ y_pos # [N_x, D]
    drift_neg = W_neg @ y_neg # [N_x, D]
    V = drift_pos - drift_neg
    return V
```

summing the projected conditioning vector with positional embeddings. Random style embeddings. Our framework allows arbitrary noise distributions beyond Gaussians. Inspired by StyleGAN (Sauer et al., 2022), we introduce an additional 32 "style tokens": each of which is a random index into a codebook of 64 learnable embeddings. These are summed and added to the conditioning vector. This does not change the sequence length and introduces negligible overhead in terms of parameters and FLOPs. This table reports the effect of style embeddings on our ablation default:

\begin{tabular}{c|cc}
\hline & w/o style & w/ style \\
\hline FID & 8.86 & $\mathbf{8 . 4 6}$ \\
\hline
\end{tabular}

In contrast to diffusion-/flow-based methods, our method can naturally handle different types of noise or random variables. With random style embeddings, the input random variables consist of two parts: (1) Gaussian noise, and

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 8. Configurations for ImageNet $\mathbf{2 5 6} \boldsymbol{\times} \mathbf{2 5 6}$.}
\begin{tabular}{|l|l|l|l|l|l|}
\hline & ablation default & B/2, latent (Table 5) & L/2, latent (Table 5) & B/16, pixel (Table 6) & L/16, pixel (Table 6) \\
\hline \multicolumn{6}{|l|}{Generator Architecture} \\
\hline arch & DiT-B/2 & DiT-B/2 & DiT-L/2 & DiT-B/16 & DiT-L/16 \\
\hline input size & $32 \times 32 \times 4$ & $32 \times 32 \times 4$ & $32 \times 32 \times 4$ & $32 \times 32 \times 4$ & $32 \times 32 \times 4$ \\
\hline patch size & $2 \times 2$ & $2 \times 2$ & $2 \times 2$ & $16 \times 16$ & $16 \times 16$ \\
\hline hidden dim & 768 & 768 & 1024 & 768 & 1024 \\
\hline depth & 12 & 12 & 24 & 12 & 24 \\
\hline register tokens & 16 & 16 & 16 & 16 & 16 \\
\hline style embedding tokens & 32 & 32 & 32 & 32 & 32 \\
\hline \multicolumn{6}{|l|}{Feature Encoder for Drifting Loss} \\
\hline arch & ResNet & ResNet & ResNet & ResNet + ConvNeXt-V2 & ResNet + ConvNeXt-V2 \\
\hline SSL pre-train method & latent-MAE & latent-MAE & latent-MAE & pixel-MAE & pixel-MAE \\
\hline ResNet: input size & $32 \times 32 \times 4$ & $32 \times 32 \times 4$ & $32 \times 32 \times 4$ & $256 \times 256 \times 3$ & $256 \times 256 \times 3$ \\
\hline ResNet: conv ${ }_{1}$ stride & 1 & 1 & 1 & 8 & 8 \\
\hline ResNet: base width & 256 & 640 & 640 & 640 & 640 \\
\hline ResNet: block type & \multicolumn{5}{|c|}{bottleneck} \\
\hline ResNet: blocks / stage & \multicolumn{5}{|c|}{[3, 4, 6, 3]} \\
\hline ResNet: size / stage & \multicolumn{5}{|c|}{$\left[32^{2}, 16^{2}, 8^{2}, 4^{2}\right]$} \\
\hline MAE: masking ratio & \multicolumn{5}{|c|}{50\%} \\
\hline MAE: pre-train epochs & 192 & 1280 & 1280 & 1280 & 1280 \\
\hline classification finetune & No & 3k steps & 3k steps & 3k steps & 3k steps \\
\hline \multicolumn{6}{|l|}{Generator Optimizer} \\
\hline optimizer & \multicolumn{5}{|c|}{AdamW ( $\beta_{1}=0.9, \beta_{2}=0.95$ )} \\
\hline learning rate & 2e-4 & 4e-4 & 4e-4 & 2e-4 & 4e-4 \\
\hline weight decay & 0.01 & 0.0 & 0.01 & 0.01 & 0.01 \\
\hline warmup steps & 5k & 10k & 10k & 10k & 10 k \\
\hline gradient clip & 2.0 & 2.0 & 2.0 & 2.0 & 2.0 \\
\hline training steps & 30k & 200k & 200k & 100k & 100k \\
\hline training epochs & 100 & 1280 & 1280 & 640 & 640 \\
\hline EMA decay & 0.999 & \multicolumn{4}{|c|}{\{0.999, 0.9995, 0.9998, 0.9999\}} \\
\hline \multicolumn{6}{|l|}{Drifting Loss Computation} \\
\hline class labels $N_{\mathrm{c}}$ & 64 & 128 & 128 & 128 & 128 \\
\hline positive samples $N_{\text {pos }}$ & 64 & 128 & 64 & 128 & 128 \\
\hline generated samples $N_{\text {neg }}$ & 64 & 64 & 64 & 64 & 64 \\
\hline effective batch $B\left(N_{\mathrm{c}} \times N_{\text {neg }}\right)$ & 4096 & 8192 & 8192 & 8192 & 8192 \\
\hline temperatures $\tau$ & \multicolumn{5}{|c|}{$\{0.02,0.05,0.2\}$ : one loss per $\tau$, sum all loss terms} \\
\hline \multicolumn{6}{|l|}{CFG Configuration} \\
\hline train: CFG $\alpha$ range & [1,4] & $[1,4]$
$p(\alpha) \propto \alpha^{-5}$ & [1, 4] & $[1,4]$
$p(\alpha) \propto \alpha^{-5}$ & [1, 4] \\
\hline train: $\mathrm{CFG} \alpha$ sampling & $[1,4]$
$p(\alpha) \propto \alpha^{-3}$ & & $[1,4]$
$50 \%: \alpha=1,50 \%: p(\alpha) \propto \alpha^{-3}$ & & $p(\alpha) \propto \alpha^{-5}$ \\
\hline train: uncond samples $N_{\text {uncond }}$ & 16 & 32 & 32 & 32 & \\
\hline inference: CFG $\alpha$ search & & & [1.0, 3.5] & & \\
\hline
\end{tabular}
\end{table}
(2) discrete indices for style embeddings. Our model $f$ produces the pushforward distribution of their joint distribution.

\section*{A.3. Implementation of ResNet-style MAE}

In addition to standard self-supervised learning models (MoCo (He et al., 2020), SimCLR(Chen et al., 2020a)), we develop a customized ResNet-style MAE model as the feature encoder for drifting loss.

Overview. Unlike standard MAE (He et al., 2022), which is based on ViT (Dosovitskiy et al., 2021), our MAE trains a convolutional ResNet that provides multi-scale features. For latent-space models, the input and output have dimension $32 \times 32 \times 4$; for pixel-space models, the input and output have dimension $256 \times 256 \times 3$.

Our MAE consists of a ResNet-style encoder paired with a deconvolutional decoder in a U-Net-style (Ronneberger et al., 2015) encoder-decoder architecture. We only use the ResNet-style encoder for feature extraction when computing the drifting loss.

MAE Encoder. The encoder follows a classical ResNet (He et al., 2016) design. It maps an input to multi-scale feature maps (4 scales in ResNet):
$$
\text { Encoder : } \mathbf{x} \mapsto\left\{\mathbf{f}_{1}, \mathbf{f}_{2}, \mathbf{f}_{3}, \mathbf{f}_{4}\right\}
$$

Here, a feature map $\mathbf{f}_{i}$ has dimension $H_{i} \times W_{i} \times C_{i}$, with $H_{i} \times W_{i} \in\left\{32^{2}, 16^{2}, 8^{2}, 4^{2}\right\}$ and $C_{i} \in\{C, 2 C, 4 C, 8 C\}$ for a base width $C$.

The architecture follows standard ResNet (He et al., 2016) design, with GroupNorm (GN) (Wu \& He, 2018) used in place of BatchNorm (BN) (Ioffe \& Szegedy, 2015). All residual blocks are "basic" blocks (i.e., each consisting of two $3 \times 3$ convolutions). Following the standard ResNet34 (He et al., 2016): the encoder has a $3 \times 3$ convolution (without downsampling) and 4 stages with $[3,4,6,3]$ blocks; downsampling (stride 2 ) happens at the first block of stages 2 to 4 .

For latent-space (i.e., latent-MAE), the input of this ResNet is $32 \times 32 \times 4$; for pixel-space, the $256 \times 256 \times 3$ input is first
patchified (by a $8 \times 8$ patch) into $32 \times 32 \times 192$. The ResNet operates on the input with $H \times W=32 \times 32$.

MAE Decoder. The decoder returns to the input shape via deconvolutions and skip connections:
$$
\text { Decoder : }\left\{\mathbf{f}_{4}, \mathbf{f}_{3}, \mathbf{f}_{2}, \mathbf{f}_{1}\right\} \mapsto \hat{\mathbf{x}}
$$

It starts with a $3 \times 3$ convolutional block on $\mathbf{f}_{4}$, followed by 4 upsampling blocks. Each upsampling block performs: bilinear $2 \times 2$ upsampling → concatenating with encoder's skip connection $\rightarrow \mathrm{GN} \rightarrow$ two $3 \times 3$ convolutions with GN and ReLU. A final $1 \times 1$ convolution produces the output channels. For the pixel-space, the decoder unpatchifies back to the original resolution after the last layer.

Masking. The MAE is trained to reconstruct randomly masked inputs. Unlike the ViT-based MAE (He et al., 2022), which removes the masked tokens from the sequence, we simply zero out masked patches. For the input of a shape $H \times W=32 \times 32$ (in either the latent- or pixel-based case), we mask $2 \times 2$ patches by zeroing. Each patch is independently masked with $50 \%$ probability.

MAE training. We minimize the $\ell_{2}$ reconstruction loss on the masked regions. We use AdamW (Loshchilov \& Hutter, 2019) with learning rate $4 \times 10^{-3}$ and a batch size of 8192. EMA with decay 0.9995 is used. Following (He et al., 2022), we apply random resized crop augmentation to the input (for the latent setting, images are augmented before being passed through the VAE encoder).

Classification fine-tuning. For our best feature encoder (last row of Table 3), we fine-tune the MAE model with a linear classifier head. The loss is $\lambda \mathcal{L}_{\text {cls }}+(1-\lambda) \mathcal{L}_{\text {recon }}$. We fine-tune all parameters in this MAE for 3k iterations, where $\lambda$ follows a linear warmup schedule, increasing from 0 to 0.1 over the first 1 k iterations and remaining constant at 0.1 for the rest of the training.

\section*{A.4. Other Pretrained Feature Encoders}

In addition to our customized MAE, we also evaluate other feature encoders for computing the drifting loss.

MoCo and SimCLR. We evaluate publicly available selfsupervised encoders trained on ImageNet in pixel space: MoCo (He et al., 2020; Chen et al., 2020b) SimCLR (Chen et al., 2020a). We use the ResNet-50 variant. For latentspace generation, we apply the VAE decoder to map generator outputs from latent space ( $32 \times 32 \times 4$ ) to pixel space $(256 \times 256 \times 3)$ before feature extraction. Gradients are backpropagated through both the feature extractor and the VAE decoder.

MAE with ConvNeXt-V2. In our pixel-space generator, we also investigate ConvNeXt-V2 (Woo et al., 2023) as the feature encoder. We note that ConvNeXt-V2 is a
self-supervised pre-trained model using the MAE objective, followed by classification fine-tuning. Like ResNet, ConvNeXt-V2 is a multi-stage architecture.

\section*{A.5. Multi-scale Features for Drifting Loss}

Given an image, the feature encoder produces feature maps at multiple scales, with multiple spatial locations per scale. We compute one drifting loss per feature (e.g., per scale and/or per location). Specifically, we compute the kernel, the drift, and the resulting loss independently for each feature. The resulting losses are summed.

For each stage in a ResNet, we extract features from the output of every 2 residual blocks, together with the final output. This yields a set of feature maps, each of shape $H_{i} \times W_{i} \times C_{i}$. For each feature map, we produce:
(a) $H_{i} \times W_{i}$ vectors, one per location (each $C_{i}$-dim);
(b) 1 global mean and 1 global std (each $C_{i}$-dim);
(c) $\frac{H_{i}}{2} \times \frac{W_{i}}{2}$ vectors of means and $\frac{H_{i}}{2} \times \frac{W_{i}}{2}$ vectors of stds (each $C_{i}$-dim), computed over $2 \times 2$ patches;
(d) $\frac{H_{i}}{4} \times \frac{W_{i}}{4}$ vectors of means and $\frac{H_{i}}{4} \times \frac{W_{i}}{4}$ vectors of stds (each $C_{i}$-dim), computed over $4 \times 4$ patches.

In addition, for the encoder's input ( $H_{0} \times W_{0} \times C_{0}$ ), we compute the mean of squared values ( $x^{2}$ ) per channel and obtain a $C_{0}$-dim vector.

All resulting vectors here are $C_{i}$-dim. We compute one drifting loss for each of these $C_{i}$-dim vectors. All these losses, in addition to the vanilla drifting loss without $\phi$, are summed. This table compares the effect of these designs on our ablation default:

\begin{tabular}{c|ccc}
\hline & $(\mathrm{a}, \mathrm{b})$ & $(\mathrm{a}-\mathrm{c})$ & $(\mathrm{a}-\mathrm{d})$ \\
\hline FID & 9.58 & 9.10 & $\mathbf{8 . 4 6}$ \\
\hline
\end{tabular}

This shows that our method benefits from richer feature sets. We note that once the feature encoder is run, the computational cost of our drifting loss is negligible: computing multi-scale, multi-location losses incurs little overhead compared to computing a single loss.

\section*{A.6. Feature and Drift Normalization}

To balance the multiple loss terms from multiple features, we perform normalization for each feature $\phi_{j}$, where, $\phi_{j}$ denotes a feature at a specific spatial location within a given scale (see A.5). Intuitively, we want to perform normalization such that the kernel $k(\cdot, \cdot)$ and the drift $\mathbf{V}$ are insensitive to the absolute magnitude of features. This allows our model to robustly support different feature encoders (see Table 3) as well as a rich set of features from one encoder.

Feature Normalization. Consider a feature $\phi_{j} \in \mathbb{R}^{C_{j}}$. We
define a normalization scale $S_{j} \in \mathbb{R}^{1}$ and the normalized feature is denoted by:
$$
\begin{equation*}
\tilde{\phi}_{j}:=\phi_{j} / S_{j} . \tag{18}
\end{equation*}
$$

When using $\tilde{\phi}_{j}$, the $\ell_{2}$ distance computed in Eq. (12) is:
$$
\begin{equation*}
\operatorname{dist}_{j}(\mathbf{x}, \mathbf{y})=\left\|\tilde{\phi}_{j}(\mathbf{x})-\tilde{\phi}_{j}(\mathbf{y})\right\|, \tag{19}
\end{equation*}
$$
where $\mathbf{x}$ denotes a generated sample and $\mathbf{y}$ denotes a positive/negative sample, and $\tilde{\phi}_{j}(\cdot)$ means extracting their feature at $j$. We want the average distance to be $\sqrt{C_{j}}$ :
$$
\begin{equation*}
\mathrm{E}_{\mathbf{x}} \mathrm{E}_{\mathbf{y}}\left[\operatorname{dist}_{j}(\mathbf{x}, \mathbf{y})\right] \approx \sqrt{C_{j}} \tag{20}
\end{equation*}
$$

To achieve this, we set the normalization scale $S_{j}$ as:
$$
\begin{equation*}
S_{j}=\frac{1}{\sqrt{C_{j}}} \mathrm{E}_{\mathbf{x}} \mathrm{E}_{\mathbf{y}}\left[\left\|\phi_{j}(\mathbf{x})-\phi_{j}(\mathbf{y})\right\|\right] \tag{21}
\end{equation*}
$$

In practice, we use all $\mathbf{x}$ and $\mathbf{y}$ samples in a batch to compute the empirical mean in place of the expectation. We reuse the cdist computation in Alg. 2 for computing the pairwise distances. We apply stop-gradient to $S_{j}$, because this scalar is conceptually computed from samples from the previous batch.

With the normalized feature, the kernel in Eq. (12) is set as:
$$
\begin{equation*}
k(\mathbf{x}, \mathbf{y})=\exp \left(-\frac{1}{\tilde{\tau}_{j}}\left\|\tilde{\phi}_{j}(\mathbf{x})-\tilde{\phi}_{j}(\mathbf{y})\right\|\right) \tag{22}
\end{equation*}
$$
where $\tilde{\tau}_{j}:=\tau \cdot \sqrt{C_{j}}$. By doing so, the value of temperature $\tau$ does not depend on the feature magnitude or feature dimensionality. We set $\tau \in\{0.02,0.05,0.2\}$ (discussed next).

Drift Normalization. When using the feature $\phi_{j}$, the resulting drift is in the same feature space as $\phi_{j}$, denoted as $\mathbf{V}_{j}$. We perform a drift normalization on $\mathbf{V}_{j}$, for each feature $\phi_{j}$. Formally, we define a normalization scale $\lambda_{j} \in \mathbb{R}^{1}$ and denote:
$$
\begin{equation*}
\tilde{\mathbf{V}}_{j}:=\mathbf{V}_{j} / \lambda_{j} \tag{23}
\end{equation*}
$$

Again, we want the normalized drift to be insensitive to the feature magnitude:
$$
\begin{equation*}
\mathbb{E}\left[\frac{1}{C_{j}}\left\|\tilde{\mathbf{V}}_{j}\right\|^{2}\right] \approx 1 \tag{24}
\end{equation*}
$$

To achieve this, we set $\lambda_{j}$ as:
$$
\begin{equation*}
\lambda_{j}=\sqrt{\mathbb{E}\left[\frac{1}{C_{j}}\left\|\mathbf{V}_{j}\right\|^{2}\right]} . \tag{25}
\end{equation*}
$$

In practice, the expectation is replaced with the empirical mean computed over the entire batch.

With the normalized feature and normalized drift, the drifting loss of the feature $\phi_{j}$ is:
$$
\begin{equation*}
\mathcal{L}_{j}=\operatorname{MSE}\left(\tilde{\phi}_{j}(\mathbf{x})-\operatorname{sg}\left(\tilde{\phi}_{j}(\mathbf{x})+\tilde{\mathbf{V}}_{j}\right)\right), \tag{26}
\end{equation*}
$$
where MSE denotes mean squared error. The overall loss is the sum across all features: $\mathcal{L}=\sum_{j} \mathcal{L}_{j}$.
Multiple temperatures. Using normalized feature distances, the value of temperature $\tau$ determines what is considered "nearby". To improve robustness across different features and across different pretrained models we study, we adopt multiple temperatures.

Formally, for each $\tau$ value, we compute the normalized drift as described above, denoted by $\tilde{\mathbf{V}}_{j, \tau}$. Then we compute an aggregated field: $\tilde{\mathbf{V}}_{j} \leftarrow \sum_{\tau} \tilde{\mathbf{V}}_{j, \tau}$, and use it for the loss in Equation (26).

This table shows the effect of multiple temperatures on our ablation default:

\begin{tabular}{l|ccc|c}
\hline$\tau$ & 0.02 & 0.05 & 0.2 & $\{0.02,0.05,0.2\}$ \\
\hline FID & 10.62 & $\mathbf{8 . 6 7}$ & 8.96 & $\mathbf{8 . 4 6}$ \\
\hline
\end{tabular}

Using multiple temperatures can achieve slightly better results than using a single optimal temperature. We fix $\tau \in \{0.02,0.05,0.2\}$ and do not require tuning this hyperparameter across different configurations.

Normalization across spatial locations. For a feature map of resolution $H_{i} \times W_{i}$, there are $H_{i} \times W_{i}$ per-location features. Separately computing the normalization for each location would be slow and unnecessary. We assume that features at different locations within the same feature map share the same normalization scale. Accordingly, we concatenate all $H_{i} \times W_{i}$ locations and compute the normalization scale over all of them. The feature normalization and drift normalization are both performed in this way.

\section*{A.7. Classifier-Free Guidance (CFG)}

To support CFG, at training time, we include $N_{\text {unc }}$ additional unconditional samples (real images from random classes) as extra negatives. These samples are weighted by a factor $w$ when computing the kernel. For a generated sample $\mathbf{x}$, the effective negative distribution it compares with is:
$$
\tilde{q}(\cdot \mid c) \triangleq \frac{\left(N_{\mathrm{neg}}-1\right) \cdot q_{\theta}(\cdot \mid c)+N_{\mathrm{unc}} w \cdot p_{\mathrm{data}}(\cdot \mid \varnothing)}{\left(N_{\mathrm{neg}}-1\right)+N_{\mathrm{unc}} w} .
$$

Comparing this equation with Eq. (15)(16), we have:
$$
\gamma=\frac{N_{\mathrm{unc}} w}{\left(N_{\mathrm{neg}}-1\right)+N_{\mathrm{unc}} w}
$$
and
$$
\alpha=\frac{1}{1-\gamma}=\frac{\left(N_{\mathrm{neg}}-1\right)+N_{\mathrm{unc}} w}{N_{\mathrm{neg}}-1} .
$$

Given a CFG strength $\alpha$, we compute $w$ accordingly, which is used to weight the kernel. The same weighting $w$ is also applied when computing the global distance normalization.

We train our model with CFG-conditioning (Geng et al., 2025b). At each iteration, we randomly sample $\alpha$ following a pre-defined distribution (see Table 8) and compute the resulting $w$ for weighting the unconditional samples. The value of $\alpha$ is a condition input to the network $f_{\theta}(\boldsymbol{\epsilon}, c, \alpha)$, alongside the class label $c$.

At inference time, we specify a value of $\alpha$. The inferencetime computation remains to be one-step (1-NFE).

\section*{A.8. Sample Queue}

Our method requires access to randomly sampled real (positive/unconditional) data. This can be implemented using a specialized data loader. Instead, we adopt a sample queue of cached data, similar to the queue used in MoCo (He et al., 2020). This implementation samples data in a statistically similar way to a specialized data loader. For completeness, we describe our implementation as follows, while noting that a data loader would be a more principled solution.

For each class label, we keep a queue of size 128; for unconditional samples (used in CFG), we maintain a separate global queue of size 1000. At each training step, we push the latest 64 new real (positive/unconditional) samples, alongside their labels, into the corresponding queues; the earliest ones are dequeued. When sampling, positive samples are drawn from the queue of the corresponding class, and unconditional samples are drawn from the global queue. We sample without replacement.

\section*{A.9. Training Loop}

In summary, in the training loop, each step proceeds as:
1. Sample a batch ( $N_{c}$ ) of class labels.
2. For each label $c$, sample a CFG scale $\alpha$.
3. Sample a batch ( $N_{\text {neg }}$ ) of noise $\boldsymbol{\epsilon}$. Feed $(\boldsymbol{\epsilon}, c, \alpha)$ to the generator $f$ to produce generated samples;
4. Sample positive samples (same class, $N_{\text {pos }}$ ) and unconditional samples (for CFG, $N_{\text {unc }}$ );
5. Extract features on all generated, positive, and unconditional samples
6. Compute the drifting loss using the features.
7. Run backpropagation and parameter update.

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 9. Ablations on pixel-space generation. We study generation directly in pixel space (without VAE). Applying the same MAE recipe as in latent space yields higher FID, indicating that pixel-space generation is more challenging. Combining MAE with ConvNeXt-V2 helps close this gap. Latent-space results shown for reference. The results below follow the ablation setting (B/16 model for pixel-space, 100 epochs).}
\begin{tabular}{lrr}
\hline & \multicolumn{2}{c}{ FID (100-epoch) } \\
\cline { 2 - 3 } feature encoder $\phi$ & latent (B/2) & pixel (B/16) \\
\hline MAE (width 256, epoch 192) & 8.46 & 32.11 \\
MAE (width 640, epoch 1280) + cls ft. & 3.36 & 9.35 \\
+ MAE w/ ConvNeXt-V2 & - & $\mathbf{3 . 7 0}$ \\
\hline
\end{tabular}
\end{table}

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 10. Pixel-space generation: from ablation to final setting. Beyond the ablation setting, we compare the settings that lead to the results in Table 6.}
\begin{tabular}{l|cc|c}
\hline case & arch & ep & FID \\
\hline (a) baseline (from Table 9) & B/16 & 100 & 3.70 \\
\hline (b) longer + hyper-param. & B/16 & 320 & 2.19 \\
(c) longer & B/16 & 640 & 1.76 \\
(d) larger model & L/16 & 640 & $\mathbf{1 . 6 1}$ \\
\hline
\end{tabular}
\end{table}

\section*{B. Additional Experimental Results}

\section*{B.1. Ablations on Pixel-Space Generation}

We provide more ablations on pixel-space generation in Table 9 and 10. Table 9 compares the effect of the feature encoder on the pixel-space generator. It shows that the choice of feature encoder plays a more significant role in pixel-space generation quality. A weaker MAE encoder yields an FID of 32.11, whereas a stronger MAE encoder improves performance to an FID of 9.35. We further add another feature encoder, ConvNeXt-V2 (Woo et al., 2023), which is also pre-trained with the MAE objective. This further improves the result to an FID of 3.70.

Table 10 reports the results of training longer and using a larger model. Due to limited time, we train pixel-space models for 640 epochs (vs. the latent counterpart's 1280); we expect that longer training would yield further improvements. We achieve an FID of 1.61 for pixel-space generation. This is our result in the main paper (Table 6).

\section*{B.2. Ablation on Kernel Normalization}

In Eq. (11), our drifting field is weighted by normalized kernels, which can be written as:
$$
\begin{equation*}
\mathbf{V}(\mathbf{x})=\mathbb{E}_{p, q}\left[\tilde{k}\left(\mathbf{x}, \mathbf{y}^{+}\right) \tilde{k}\left(\mathbf{x}, \mathbf{y}^{-}\right)\left(\mathbf{y}^{+}-\mathbf{y}^{-}\right)\right], \tag{27}
\end{equation*}
$$
where $\tilde{k}(\cdot, \cdot)=\frac{1}{Z} k(\cdot, \cdot)$ denotes the normalized kernel. In principle, this normalization is approximated by a softmax operation over the axis of $\mathbf{y}$ samples. Our implementation (Alg. 2) further applies softmax over the axis of $\mathbf{x}$ samples. We compare these designs, along with another variant

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 11. Ablation on kernel normalization. Softmax normalization over both the $\mathbf{x}$ and $\mathbf{y}$ axes performs better. On the other hand, even using no normalization performs decently, showing the robustness of our method. (Setting: B/2 model, 100 epochs)}
\begin{tabular}{lr}
\hline kernel normalization & FID \\
\hline softmax over $\mathbf{x}$ and $\mathbf{y}$ (default) & $\mathbf{8 . 4 6}$ \\
softmax over $\mathbf{y}$ & 8.92 \\
no normalization & 10.54 \\
\hline
\end{tabular}
\end{table}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/14b7cf7f-b68f-4015-b13d-5b756b999ce4-17.jpg?height=1069&width=528&top_left_y=627&top_left_x=330}
\captionsetup{labelformat=empty}
\caption{Figure 5. Effect of CFG scale $\alpha$. (a): FID vs. $\alpha$. (b): IS vs. $\alpha$. (c): IS vs. FID. We show the L/2 (solid) and B/2 (dashed) models. Consistent with common observations in diffusion-/flow-based models, the CFG scale effectively trades off distributional coverage (as reflected by FID) against per-image quality (measured by IS). Notably, with the L/2 model, the optimal FID is achieved at $\alpha=1.0$, which is often regarded as "w/o CFG" in diffusion-/flow-based models. For $\mathrm{B} / 2$, the optimal FID is achieved at $\alpha=1.1$.}
\end{figure}
without normalization ( $Z=1$ ).
Table 11 compares the three designs. Using the $\mathbf{y}$-only softmax performs well (8.92 FID), whereas using both $\mathbf{x}$ and $\mathbf{y}$ softmax improves the result ( 8.46 FID). On the other hand, even without normalization, performance remains decent, demonstrating the robustness of our method.

We note that all three variants satisfy the equilibrium condition $\mathbf{V}_{p, q}(\mathbf{x})=\mathbf{0}$ when $p=q$. This explains why all variants perform reasonably well and why even the destructive setting (no normalization) avoids catastrophic failure.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/14b7cf7f-b68f-4015-b13d-5b756b999ce4-17.jpg?height=771&width=835&top_left_y=216&top_left_x=1054}
\captionsetup{labelformat=empty}
\caption{Figure 6. Nearest neighbor analysis. Each panel shows a generated sample together with its top-10 nearest real images. The nearest neighbors are retrieved from the ImageNet training set based on the cosine similarity using a CLIP encoder (Radford et al., 2021). Our method generates novel images that are visually distinct from their nearest neighbors.}
\end{figure}

\section*{B.3. Ablation on CFG}

In Figure 5, we investigate the CFG scale $\alpha$ used at inference time. It shows that the CFG formulation developed for our models exhibits behavior similar to that observed in diffusion-/flow-based models. Increasing the CFG scale leads to higher IS values, whereas beyond the FID sweet spot, further increases in IS come at the cost of worse FID.

Notably, with our best model (L/2), the optimal FID is achieved at $\alpha=1.0$, which is often regarded as "w/o CFG" in diffusion-/flow-based models (even though their "w/o CFG" setting can reduce NFE by half). While our method need not run an unconditional model at inference time (in contrast to standard CFG), training is influenced by the use of unconditional real samples as negatives.

\section*{B.4. Nearest Neighbor Analysis}

In Figure 6, we show generated images together with their nearest real images. The nearest neighbors are retrieved from the ImageNet training set using CLIP features. These visualizations suggest that our method generates novel images that are visually distinct from their nearest neighbors, rather than merely memorizing training samples.

\section*{B.5. Qualitative Results}

Fig. 7-10 show uncurated samples from our model. Fig. 1115 provide side-by-side comparison with improved MeanFlow (iMF) (Geng et al., 2025b), the current state-of-the-art one-step method.

\section*{C. Additional Derivations}

\section*{C.1. On Identifiability of the Zero-Drift Equilibrium}

In Sec. 3, we showed that anti-symmetry implies $p=q \Rightarrow \mathbf{V}(\mathbf{x}) \equiv \mathbf{0}$. Here we investigate the converse: under what conditions does $\mathbf{V}(\mathbf{x}) \approx \mathbf{0}$ imply $p \approx q$ ? Generally, this is not guaranteed for arbitrary vector fields. However, we argue that for our specific construction, the zero-drift condition imposes strong constraints on the distributions.

To avoid boundary issues, we assume that $p$ and $q$ have full support on $\mathbb{R}^{d}$ (e.g., via infinitesimal Gaussian smoothing). Consequently, ensuring the equilibrium condition $\mathbf{V}(\mathbf{x}) \approx \mathbf{0}$ for generated samples $\mathbf{x} \sim q$ effectively enforces $\mathbf{V}(\mathbf{x}) \approx \mathbf{0}$ for all $\mathbf{x} \in \mathbb{R}^{d}$.

Setup. Consider a general interaction kernel $K\left(\mathbf{x}, \mathbf{y}^{+}, \mathbf{y}^{-}\right) \in \mathbb{R}^{d}$ and the drifting field
$$
\begin{equation*}
\mathbf{V}_{p, q}(\mathbf{x}):=\mathbb{E}_{\mathbf{y}^{+} \sim p, \mathbf{y}^{-} \sim q}\left[K\left(\mathbf{x}, \mathbf{y}^{+}, \mathbf{y}^{-}\right)\right] . \tag{28}
\end{equation*}
$$

We assume that $p$ and $q$ belong to a finite-dimensional model class spanned by a linearly independent basis $\left\{\varphi_{i}\right\}_{i=1}^{m}$ :
$$
\begin{equation*}
p(\mathbf{y})=\sum_{i=1}^{m} a_{i} \varphi_{i}(\mathbf{y}), \quad q(\mathbf{y})=\sum_{i=1}^{m} b_{i} \varphi_{i}(\mathbf{y}), \tag{29}
\end{equation*}
$$
where $\mathbf{a}, \mathbf{b} \in \mathbb{R}^{m}$ are coefficient vectors.
Bilinear expansion over test locations. Consider a set of test locations (probes) $\mathcal{X}=\left\{\mathbf{x}_{k}\right\}_{k=1}^{N}$ with sufficiently large $N$ (e.g., $N \gg m^{2}$ ). For each pair of basis indices ( $i, j$ ), we define the induced interaction vector $\mathbf{U}_{i j} \in \mathbb{R}^{d \times N}$ by computing its column:
$$
\begin{equation*}
\mathbf{U}_{i j}[:, \mathbf{x}] \triangleq \iint K\left(\mathbf{x}, \mathbf{y}^{+}, \mathbf{y}^{-}\right) \varphi_{i}\left(\mathbf{y}^{+}\right) \varphi_{j}\left(\mathbf{y}^{-}\right) d \mathbf{y}^{+} d \mathbf{y}^{-} \tag{30}
\end{equation*}
$$
evaluated at all $\mathbf{x} \in \mathcal{X}$. Substituting the basis expansion into Eq. (28), the drifting field evaluated on $\mathcal{X}$ (stored as a matrix $\mathbf{V}_{\mathcal{X}}$ ) is a bilinear combination:
$$
\begin{equation*}
\mathbf{V}_{\mathcal{X}} \triangleq \sum_{i=1}^{m} \sum_{j=1}^{m} a_{i} b_{j} \mathbf{U}_{i j} \tag{31}
\end{equation*}
$$

Here, $\mathbf{V}_{\mathcal{X}} \in \mathbb{R}^{d \times N}$. At the equilibrium, we have $\mathbf{V}_{\mathcal{X}}=\mathbf{0}$, which yields $d N$ linear equations.

Linear independence assumption. Our anti-symmetry condition implies that switching $p$ and $q$ negates the field. In terms of basis interactions, this means $\mathbf{U}_{i j}=-\mathbf{U}_{j i}$ (and consequently $\mathbf{U}_{i i}=\mathbf{0}$ ). We make the generic nondegeneracy assumption: The set of vectors $\left\{\mathbf{U}_{i j}\right\}_{1 \leq i<j \leq m}$ is linearly independent in $\mathbb{R}^{d N}$. This assumption requires the probes $\mathcal{X}$ and kernel $K$ to be non-degenerate; if all $\mathbf{x}$ yield identical constraints, independence would fail. For generic choices of $K$ and sufficiently diverse probes $\mathcal{X}$
with $d N \gg m^{2}$, such linear independence is a natural nondegeneracy condition.

Uniqueness of the equilibrium. The zero-drift condition $\mathbf{V}(\mathbf{x}) \equiv \mathbf{0}$ implies $\mathbf{V}_{\mathcal{X}}=\mathbf{0}$. Grouping terms by the independent basis vectors $\left\{\mathbf{U}_{i j}\right\}_{i<j}$, we have:
$$
\begin{equation*}
\sum_{1 \leq i<j \leq m}\left(a_{i} b_{j}-a_{j} b_{i}\right) \mathbf{U}_{i j}=\mathbf{0} . \tag{32}
\end{equation*}
$$

By the linear independence assumption, the coefficients must vanish: $a_{i} b_{j}-a_{j} b_{i}=0$ for all $i, j$. This implies that the vector $\mathbf{a}$ is parallel to $\mathbf{b}$ (i.e, $\mathbf{a} \propto \mathbf{b}$ ). Since $p$ and $q$ are probability densities (implying $\int p=\int q=1$ ), we must have $\mathbf{a}=\mathbf{b}$, and thus $p=q$.

Connection to the mean shift field. The mean-shift field fits this framework. The update vector (before normalization) is $\mathbb{E}_{p, q}\left[k\left(\mathbf{x}, \mathbf{y}^{+}\right) k\left(\mathbf{x}, \mathbf{y}^{-}\right)\left(\mathbf{y}^{+}-\mathbf{y}^{-}\right)\right]$. Assuming the normalization factors $Z_{p}$ and $Z_{q}$ are finite, the condition $\mathbf{V}(\mathbf{x})=\mathbf{0}$ implies the numerator integral vanishes, which corresponds to an interaction kernel of the form:
$$
\begin{equation*}
K\left(\mathbf{x}, \mathbf{y}^{+}, \mathbf{y}^{-}\right)=k\left(\mathbf{x}, \mathbf{y}^{+}\right) k\left(\mathbf{x}, \mathbf{y}^{-}\right)\left(\mathbf{y}^{+}-\mathbf{y}^{-}\right) . \tag{33}
\end{equation*}
$$

This kernel generates the bilinear structure analyzed above. Since we can choose $N$ such that $d N \gg m^{2}$, the dimension of the test space is much larger than the number of basis pairs. Thus, the linear independence of $\left\{\mathbf{U}_{i j}\right\}$ is expected to hold for generic configurations. Finally, for general distributions $p$ and $q$, we can approximate them using a sufficiently large basis expansion, turning into $\tilde{p}$ and $\tilde{q}$. When the basis approximation is sufficiently accurate, $\tilde{p} \approx p$ and $\tilde{q} \approx q$, and the drift field $\mathbf{V}_{\tilde{p}, \tilde{q}} \approx \mathbf{V}_{p, q} \approx 0$. By the argument above, $\tilde{p} \approx \tilde{q}$, and thus $p \approx q$.

The argument above works for general form of drifting field, under mild anti-degeneracy assumptions.

\section*{C.2. The Drifting Field of MMD}

In principle, if a method minimizes a discrepancy between two distributions $p$ and $q$ and reaches minimum at $p=q$, then from the perspective of our framework, a drifting field $\mathbf{V}$ exists that governs sample movement: we can let $\mathbf{V} \propto-\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$, which is zero when $p=q$. We discuss the formulation of this $\mathbf{V}$ for a loss based on Maximum Mean Discrepancy (MMD) (Li et al., 2015; Dziugaite et al., 2015).

Gradients of Drifting Loss. With $\mathbf{x}=f_{\theta}(\boldsymbol{\epsilon})$, our drifting loss in Eq. (6) can be written as:
$$
\begin{equation*}
\mathcal{L}=\mathbb{E}_{\mathbf{x} \sim q}[\mathcal{L}(\mathbf{x})]=\mathbb{E}_{\mathbf{x} \sim q}\left[\|\mathbf{x}-\operatorname{sg}(\mathbf{x}+\mathbf{V}(\mathbf{x}))\|^{2}\right] \tag{34}
\end{equation*}
$$
where "sg" is short for stop-gradient. The gradient w.r.t. the parameters $\theta$ is computed by:
$$
\begin{equation*}
\frac{\partial \mathcal{L}}{\partial \theta}=\mathbb{E}_{\mathbf{x} \sim q}\left[\frac{\partial \mathcal{L}(\mathbf{x})}{\partial \mathbf{x}} \frac{\partial \mathbf{x}}{\partial \theta}\right] \tag{35}
\end{equation*}
$$
where $\frac{\partial \mathcal{L}(\mathbf{x})}{\partial \mathbf{x}}=2(\mathbf{x}-\mathrm{sg}(\mathbf{x}+\mathbf{V}(\mathbf{x})))=-2 \mathbf{V}(\mathbf{x})$. This gives:
$$
\begin{equation*}
\mathbf{V}(\mathbf{x})=-\frac{1}{2} \frac{\partial \mathcal{L}(\mathbf{x})}{\partial \mathbf{x}} \tag{36}
\end{equation*}
$$

We note that this formulation is general and imposes no constraints on $\mathbf{V}$, except that $\mathbf{V}=\mathbf{0}$ when $p=q$.

Our method does not require $\mathcal{L}$ to define a discrepancy between $p$ and $q$. However, for other methods that depend on minimizing a discrepancy $\mathcal{L}$, we can induce a drifting field via (36). This is valid if $\mathcal{L}$ is minimized when $p=q$.

Gradients of MMD Loss. In MMD-based methods (e.g., Li et al. 2015), the difference between two distributions $p$ and $q$ is measured by squared MMD:
$$
\begin{align*}
\mathcal{L}_{\mathrm{MMD}^{2}}(p, q) & =\mathbb{E}_{\mathbf{x}, \mathbf{x}^{\prime} \sim q}\left[\xi\left(\mathbf{x}, \mathbf{x}^{\prime}\right)\right]-2 \mathbb{E}_{\mathbf{y} \sim p, \mathbf{x} \sim q}[\xi(\mathbf{y}, \mathbf{x})] \\
& + \text { const. } \tag{37}
\end{align*}
$$

Here, the constant term is $\mathbb{E}_{\mathbf{y}, \mathbf{y}^{\prime} \sim p}\left[\xi\left(\mathbf{y}, \mathbf{y}^{\prime}\right)\right]$, which depends only on the target distribution $p$ and remains unchanged. $\xi$ is a kernel function.

Consider $\mathbf{x}=f_{\theta}(\boldsymbol{\epsilon})$ with $\boldsymbol{\epsilon} \sim p_{\boldsymbol{\epsilon}}$. The gradient estimation performed in (Li et al., 2015) corresponds to:
$$
\begin{equation*}
\frac{\partial \mathcal{L}_{\mathrm{MMD}^{2}}}{\partial \theta}=\mathbb{E}_{\mathbf{x} \sim q}\left[\frac{\partial \mathcal{L}_{\mathrm{MMD}^{2}}(\mathbf{x})}{\partial \mathbf{x}} \frac{\partial \mathbf{x}}{\partial \theta}\right] \tag{38}
\end{equation*}
$$
where the gradient w.r.t $\mathbf{x}$ is computed by:
$$
\begin{equation*}
\frac{\partial \mathcal{L}_{\mathrm{MMD}^{2}}(\mathbf{x})}{\partial \mathbf{x}}=2 \mathbb{E}_{\mathbf{x}^{\prime} \sim q}\left[\frac{\partial \xi\left(\mathbf{x}, \mathbf{x}^{\prime}\right)}{\partial \mathbf{x}}\right]-2 \mathbb{E}_{\mathbf{y} \sim p}\left[\frac{\partial \xi(\mathbf{x}, \mathbf{y})}{\partial \mathbf{x}}\right] . \tag{39}
\end{equation*}
$$

Using our notation of positives and negatives, we rename the variables and rewrite as:
$\frac{\partial \mathcal{L}_{\mathrm{MMD}^{2}}(\mathbf{x})}{\partial \mathbf{x}}=2 \mathbb{E}_{\mathbf{y}^{-} \sim q}\left[\frac{\partial \xi\left(\mathbf{x}, \mathbf{y}^{-}\right)}{\partial \mathbf{x}}\right]-2 \mathbb{E}_{\mathbf{y}^{+} \sim p}\left[\frac{\partial \xi\left(\mathbf{x}, \mathbf{y}^{+}\right)}{\partial \mathbf{x}}\right]$.

Comparing with Eq. (36), we obtain:
$$
\begin{equation*}
\mathbf{V}_{\mathrm{MMD}}(\mathbf{x}) \triangleq \mathbb{E}_{\mathbf{y}^{+} \sim p}\left[\frac{\partial \xi\left(\mathbf{x}, \mathbf{y}^{+}\right)}{\partial \mathbf{x}}\right]-\mathbb{E}_{\mathbf{y}^{-} \sim q}\left[\frac{\partial \xi\left(\mathbf{x}, \mathbf{y}^{-}\right)}{\partial \mathbf{x}}\right] \tag{41}
\end{equation*}
$$

This is the underlying drifting field that corresponds to the MMD loss $\mathcal{L}_{\mathrm{MMD}^{2}}$.

For a radial kernel $\xi(\mathbf{x}, \mathbf{y})=\xi(R)$ where $R=\|\mathbf{x}-\mathbf{y}\|^{2}$, the gradient of kernel is:
$$
\begin{equation*}
\frac{\partial \xi(\mathbf{x}, \mathbf{y})}{\partial \mathbf{x}}=2 \xi^{\prime}\left(\|\mathbf{x}-\mathbf{y}\|^{2}\right)(\mathbf{x}-\mathbf{y}) \tag{42}
\end{equation*}
$$
where $\xi^{\prime}$ is the derivative of the function $\xi(R)$. Accordingly, Eq. (41) becomes:
$$
\begin{align*}
\mathbf{V}_{\mathrm{MMD}}(\mathbf{x}) & =\mathbb{E}_{\mathbf{y}^{+} \sim p}\left[2 \xi^{\prime}\left(\left\|\mathbf{x}-\mathbf{y}^{+}\right\|^{2}\right)\left(\mathbf{x}-\mathbf{y}^{+}\right)\right]  \tag{43}\\
& -\mathbb{E}_{\mathbf{y}^{-} \sim q}\left[2 \xi^{\prime}\left(\left\|\mathbf{x}-\mathbf{y}^{-}\right\|^{2}\right)\left(\mathbf{x}-\mathbf{y}^{-}\right)\right]
\end{align*}
$$

In (Li et al., 2015), the Gaussian kernel is used: $\xi(\mathbf{x}, \mathbf{y})=\exp \left(-\frac{1}{2 \sigma^{2}}\|\mathbf{x}-\mathbf{y}\|^{2}\right)$, leading to $\xi^{\prime}\left(\|\mathbf{x}-\mathbf{y}\|^{2}\right)= -\frac{1}{2 \sigma^{2}} \exp \left(-\frac{1}{2 \sigma^{2}}\|\mathbf{x}-\mathbf{y}\|^{2}\right)$.

Relations and Differences. When using our definition of $\mathbf{V}=\mathbf{V}^{+}-\mathbf{V}^{-}$(i.e., Eq. (10)), we have:
$$
\begin{align*}
\mathbf{V}(\mathbf{x}) & =\mathbb{E}_{\mathbf{y}^{+} \sim p}\left[\tilde{k}\left(\mathbf{x}, \mathbf{y}^{+}\right)\left(\mathbf{y}^{+}-\mathbf{x}\right)\right] \\
& -\mathbb{E}_{\mathbf{y}^{-} \sim q}\left[\tilde{k}\left(\mathbf{x}, \mathbf{y}^{-}\right)\left(\mathbf{y}^{-}-\mathbf{x}\right)\right] \tag{44}
\end{align*}
$$

Comparing (43) with (44), we show that the underlying kernel used to build the drifting field of MMD is:
$$
\begin{equation*}
\tilde{k}_{\mathrm{MMD}}(\mathbf{x}, \mathbf{y})=-2 \xi^{\prime}\left(\|\mathbf{x}-\mathbf{y}\|^{2}\right) . \tag{45}
\end{equation*}
$$

When $\xi$ is a Gaussian function, we have: $\tilde{k}(\mathbf{x}, \mathbf{y})= \frac{1}{\sigma^{2}} \exp \left(-\frac{1}{2 \sigma^{2}}\|\mathbf{x}-\mathbf{y}\|^{2}\right)$. Without normalization, the resulting drift no longer satisfies the assumptions underlying Alg. 2, and the mean-shift interpretation breaks down.

As a comparison, our general formulation enables to use normalized kernels:
$$
\begin{equation*}
\tilde{k}(\mathbf{x}, \mathbf{y})=\frac{1}{Z(\mathbf{x})} k(\mathbf{x}, \mathbf{y})=\frac{1}{\mathbb{E}_{\mathbf{y}}[k(\mathbf{x}, \mathbf{y})]} k(\mathbf{x}, \mathbf{y}), \tag{46}
\end{equation*}
$$
where the expectation is over $p$ or $q$. Only when we use normalized kernels, we have (see Eq. (11)):
$$
\begin{equation*}
\mathbf{V}(\mathbf{x})=\mathbb{E}_{p, q}\left[\tilde{k}\left(\mathbf{x}, \mathbf{y}^{+}\right) \tilde{k}\left(\mathbf{x}, \mathbf{y}^{-}\right)\left(\mathbf{y}^{+}-\mathbf{y}^{-}\right)\right] \tag{47}
\end{equation*}
$$
on which our Alg. 2 is based.
Given this relation, we summarize the key differences between our model and the MMD-based methods as follows:
(i) Our method is formulated around the drifting field $\mathbf{V}$, which is more flexible and general.
(ii) Our method supports and leverages normalized kernels $\frac{1}{Z} k(\mathbf{x}, \mathbf{y})$ that cannot be naturally derived from the MMD perspective.
(iii) Our $\mathbf{V}$-centric formulation enables a flexible step size for drifting (i.e., $\mathbf{x} \leftarrow \mathbf{x}+\eta \mathbf{V}$ ) and therefore naturally supports $\mathbf{V}$-normalization (see A.6).
(iv) Our $\mathbf{V}$-centric formulation allows the equilibrium concept to be naturally extended to support CFG, whereas a CFG variant for MMD remains unexplored.

In summary, although a special case of our method reduces to MMD, our V-centric framework is more general and enables unique possibilities that are important in practice. In our experiments, we were not able to obtain reasonable results using the MMD framework.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/14b7cf7f-b68f-4015-b13d-5b756b999ce4-20.jpg?height=2027&width=1713&top_left_y=313&top_left_x=180}
\captionsetup{labelformat=empty}
\caption{Figure 7. Uncurated samples from our latent- $\mathbf{L} / \mathbf{2}$ model with $\mathbf{C F G}=\mathbf{1 . 0}$ (page 1/4). FID $=1.54$, IS $=258.9$.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/14b7cf7f-b68f-4015-b13d-5b756b999ce4-21.jpg?height=2036&width=1709&top_left_y=311&top_left_x=180}
\captionsetup{labelformat=empty}
\caption{Figure 8. Uncurated samples from our latent-L/2 model with $\mathbf{C F G} \boldsymbol{=} \mathbf{1 . 0}$ (page 2/4). FID $=1.54$, IS $=258.9$.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/14b7cf7f-b68f-4015-b13d-5b756b999ce4-22.jpg?height=2029&width=1713&top_left_y=311&top_left_x=180}
\captionsetup{labelformat=empty}
\caption{Figure 9. Uncurated samples from our latent-L/2 model with $\mathbf{C F G} \boldsymbol{=} \mathbf{1 . 0}$ (page 3/4). FID $=1.54$, IS $=258.9$.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/14b7cf7f-b68f-4015-b13d-5b756b999ce4-23.jpg?height=2031&width=1713&top_left_y=311&top_left_x=180}
\captionsetup{labelformat=empty}
\caption{Figure 10. Uncurated samples from our latent-L/2 model with $\mathbf{C F G} \boldsymbol{=} \mathbf{1 . 0}$ (page 4/4). $\mathrm{FID}=1.54$, $\mathrm{IS}=258.9$.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/14b7cf7f-b68f-4015-b13d-5b756b999ce4-24.jpg?height=1790&width=1705&top_left_y=363&top_left_x=182}
\captionsetup{labelformat=empty}
\caption{Figure 11. Side-by-side comparison with improved MeanFlow (iMF) (Geng et al., 2025b) (page 1/5). Uncurated samples from our method (left) and iMF (right) on all ImageNet classes visualized in the iMF paper. Both methods generate images with a single neural function evaluation (1-NFE). The iMF visualizations use CFG $\omega=6.0$ and interval $\left[t_{\text {min }}, t_{\text {max }}\right]=[0.2,0.8]$, achieving FID 3.92 and IS 348.2 (DiT-XL/2). For fair comparison, we set the CFG scale to match the IS of iMF visualizations, which leads to FID 3.01 and IS 354.4 (at $\mathrm{CFG}=1.5$ ) for our method (DiT-L/2).}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/14b7cf7f-b68f-4015-b13d-5b756b999ce4-25.jpg?height=1784&width=1713&top_left_y=363&top_left_x=180}
\captionsetup{labelformat=empty}
\caption{Figure 12. Side-by-side comparison with improved MeanFlow (iMF) (Geng et al., 2025b) (page 2/5). Uncurated samples from our method (left) and iMF (right) on all ImageNet classes visualized in the iMF paper. Both methods generate images with a single neural function evaluation (1-NFE). The iMF visualizations use CFG $\omega=6.0$ and interval $\left[t_{\text {min }}, t_{\text {max }}\right]=[0.2,0.8]$, achieving FID 3.92 and IS 348.2 (DiT-XL/2). For fair comparison, we set the CFG scale to match the IS of iMF visualizations, which leads to FID 3.01 and IS 354.4 (at $\mathrm{CFG}=1.5$ ) for our method (DiT-L/2).}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/14b7cf7f-b68f-4015-b13d-5b756b999ce4-26.jpg?height=1782&width=1713&top_left_y=365&top_left_x=180}
\captionsetup{labelformat=empty}
\caption{Figure 13. Side-by-side comparison with improved MeanFlow (iMF) (Geng et al., 2025b) (page 3/5). Uncurated samples from our method (left) and iMF (right) on all ImageNet classes visualized in the iMF paper. Both methods generate images with a single neural function evaluation (1-NFE). The iMF visualizations use CFG $\omega=6.0$ and interval $\left[t_{\text {min }}, t_{\text {max }}\right]=[0.2,0.8]$, achieving FID 3.92 and IS 348.2 (DiT-XL/2). For fair comparison, we set the CFG scale to match the IS of iMF visualizations, which leads to FID 3.01 and IS 354.4 (at $\mathrm{CFG}=1.5$ ) for our method (DiT-L/2).}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/14b7cf7f-b68f-4015-b13d-5b756b999ce4-27.jpg?height=1778&width=1715&top_left_y=367&top_left_x=178}
\captionsetup{labelformat=empty}
\caption{Figure 14. Side-by-side comparison with improved MeanFlow (iMF) (Geng et al., 2025b) (page 4/5). Uncurated samples from our method (left) and iMF (right) on all ImageNet classes visualized in the iMF paper. Both methods generate images with a single neural function evaluation (1-NFE). The iMF visualizations use CFG $\omega=6.0$ and interval $\left[t_{\text {min }}, t_{\text {max }}\right]=[0.2,0.8]$, achieving FID 3.92 and IS 348.2 (DiT-XL/2). For fair comparison, we set the CFG scale to match the IS of iMF visualizations, which leads to FID 3.01 and IS 354.4 (at $\mathrm{CFG}=1.5$ ) for our method (DiT-L/2).}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/14b7cf7f-b68f-4015-b13d-5b756b999ce4-28.jpg?height=1097&width=1715&top_left_y=706&top_left_x=178}
\captionsetup{labelformat=empty}
\caption{Figure 15. Side-by-side comparison with improved MeanFlow (iMF) (Geng et al., 2025b) (page 5/5). Uncurated samples from our method (left) and iMF (right) on all ImageNet classes visualized in the iMF paper. Both methods generate images with a single neural function evaluation (1-NFE). The iMF visualizations use CFG $\omega=6.0$ and interval $\left[t_{\text {min }}, t_{\text {max }}\right]=[0.2,0.8]$, achieving FID 3.92 and IS 348.2 (DiT-XL/2). For fair comparison, we set the CFG scale to match the IS of iMF visualizations, which leads to FID 3.01 and IS 354.4 (at $\mathrm{CFG}=1.5$ ) for our method (DiT-L/2).}
\end{figure}