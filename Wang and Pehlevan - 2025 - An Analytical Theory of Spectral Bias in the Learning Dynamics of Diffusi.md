\title{
An Analytical Theory of Spectral Bias in the Learning Dynamics of Diffusion Models
}

\author{
Binxu Wang \\ Kempner Institute, Harvard University \\ Boston, MA, USA \\ binxu_wang@hms.harvard.edu
}

\author{
Cengiz Pehlevan \\ SEAS, Harvard University \\ Cambridge, MA, USA \\ cpehlevan@seas.harvard.edu
}

\begin{abstract}
We develop an analytical framework for understanding how the generated distribution evolves during diffusion model training. Leveraging a Gaussian-equivalence principle, we solve the full-batch gradient-flow dynamics of linear and convolutional denoisers and integrate the resulting probability-flow ODE, yielding analytic expressions for the generated distribution. The theory exposes a universal inverse-variance spectral law: the time for an eigen- or Fourier mode to match its target variance scales as $\tau \propto \lambda^{-1}$, so high-variance (coarse) structure is mastered orders of magnitude sooner than low-variance (fine) detail. Extending the analysis to deep linear networks and circulant full-width convolutions shows that weight sharing merely multiplies learning rates-accelerating but not eliminating the bias-whereas local convolution introduces a qualitatively different bias. Experiments on Gaussian and natural-image datasets confirm the spectral law persists in deep MLP-based UNet. Convolutional U-Nets, however, display rapid near-simultaneous emergence of many modes, implicating local convolution in reshaping learning dynamics. These results underscore how data covariance governs the order and speed with which diffusion models learn, and they call for deeper investigation of the unique inductive biases introduced by local convolution.
\end{abstract}

\section*{1 Introduction}

Diffusion models create rich data by gradually transforming Gaussian noise into signal, a paradigm that now drives state-of-the-art generation in vision, audio, and molecular design [1, 2, 3]. Yet two basic questions remain open. (i) Which parts of the data distribution do these models learn first, and which linger un-learned-risking artefacts under early stopping? (ii) How does architectural inductive bias shape this learning trajectory? Addressing both questions demands that we track the evolution of the full generated distribution during training and relate it to the network's parameterization.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/11e476a8-86be-4d33-b2f0-7738bec1c9ed-01.jpg?height=437&width=755&top_left_y=1787&top_left_x=992}
\captionsetup{labelformat=empty}
\caption{Figure 1: Spectral-bias schematic. Learning and sampling together impose a variance-ordered bias along covariance eigenmodes.}
\end{figure}

We tackle the learning puzzle through the simplest tractable setting-linear denoisers-where datasets become equivalent to a Gaussian with matched mean and covariance. In this regime we solve, in closed form, the nested dynamics of
gradient-flow of the weights and the probability-flow ODE that carries noise into data, leading to an analytical characterization of the evolution of the generated distribution. The analysis exposes an inverse-variance spectral law: the time required for an eigen-mode to match target variance scales like $\tau_{k} \propto \lambda_{k}^{-\alpha}$, so high-variance directions corresponding to global structure are mastered orders of magnitude sooner than low-variance, fine-detail directions. Extending the analysis to deep linear and linear convolutional nets, we show how convolutional architecture redirect this bias to Fourier or patch space, and accelerate convergence via weight sharing.

Main contributions 1. Closed-form distribution dynamics. We derive exact weight and distributional trajectories for one-layer, two-layer linear, and convolutional denoisers under full-batch DSM training. 2. Inverse-variance spectral bias. The theory reveals and quantifies a spectral-law ordering of mode convergence, offering one mechanistic explanation for early-stop errors. 3. Empirical validation in nonlinear neural nets. Experiments on Gaussian and natural-image datasets confirm the spectral-law in deep MLP-based diffusion. 4. Convolutional architectural shape learning dynamics. Experiments on convolutional UNet, showing rapid patch-first learning dynamics different from fully-connected architectures.

\section*{2 Related Work and Motivation: Spectral Bias in Distribution Learning}

Spectral structure of natural data Many natural signals have interesting spectral structures (e.g. image [4], sound [5], video [6]). For natural images, their covariance eigenvalues decay as a power law, and the corresponding eigenvectors can align with semantically meaningful patterns [4]. For faces, for instance, leading eigenmodes capture coarse, low-frequency shape variations, whereas tail modes encode fine-grained textures [7, 8]. Analyzing spectral effect on diffusion learning can therefore show which type of features the model acquires first and which remain slow to learn.

Hidden Gaussian Structure in Diffusion Model Recent work has shown, for most diffusion times, the learned neural score is closely approximated by the linear score of a Gaussian fit to the data, which is usually the best linear approximation [9, 10]. Crucially, this Gaussian linear score admits a closed-form solution to the probability-flow ODE, which can be exploited to accelerate sampling and improve its quality [11]. Moreover, this same linear structure has been linked to the generalization-memorization transition in diffusion models [10]. In sum, across many noise levels, the Gaussian linear approximation is a predominant structure in the learned score. Thus, we hypothesize it will have a significant effect on the learning dynamics of score approximator. From this perspective, our contribution is to elucidate the learning process of this linear structure.

Learning theory for regression and deep linear networks Gradient dynamics in regression are well-studied, with spectral bias and implicit regularisation emerging as central themes [12, 13, 14]. In Sec. 4.1, we show that the loss of a linear diffusion model reduces to ridge regression, letting us import those results directly. Our analysis also builds on learning theory of deep linear networks (including linear-convolutional and denoising autoencoders) [15, 16, 17, 18]. We extend these insights to modern diffusion-based generative models, offering closed-form description of how the generated distribution itself evolves during training.

Diffusion learning theory Several recent theory studies address diffusion models from a spectral perspective but tackle different questions. [19, 20, 21, 22] document spectral bias in the sampling process after training; our focus is on how that bias arises during training. [23] study stochastic sampling assuming an optimal score, orthogonal to our analysis of training dynamics. Sharing our interest in training, [24] analyze learning of mixtures of spherical Gaussians to recover component means, whereas we tackle anisotropic covariances and track reconstruction of the full covariance. [25] characterises optimal score and distribution under constraints; results from our convolutional setup can be viewed through that lens.

\section*{3 Background}

\subsection*{3.1 Score-based Diffusion Models}

Let $p_{0}(\mathbf{x})$ be the data distribution of interest, and for each noise level $\sigma>0$ define $p(\mathbf{x} ; \sigma)= \left(p_{0} * \mathcal{N}\left(0, \sigma^{2} \mathbf{I}\right)\right)(\mathbf{x})=\int p_{0}(\mathbf{y}) \mathcal{N}\left(\mathbf{x} \mid \mathbf{y}, \sigma^{2} \mathbf{I}\right) d \mathbf{y}$. The associated score function is $\nabla_{\mathbf{x}} \log p(\mathbf{x} ; \sigma)$,
i.e the gradient of the log-density at noise scale $\sigma$. In the EDM framework [26], one shows that the "probability flow" ODE
$$
\begin{equation*}
\frac{d \mathbf{x}}{d \sigma}=-\sigma \nabla_{\mathbf{x}} \log p(\mathbf{x} ; \sigma) \tag{1}
\end{equation*}
$$
exactly transports samples from $p\left(\cdot ; \sigma_{T}\right)$ to $p(\cdot ; \sigma)$ as $\sigma$ decreases. In particular, integrating from $\sigma_{T}$ down to $\sigma=0$ recovers clean data samples from $p_{0}$. We adopt the EDM parametrization for its notational simplicity; other common diffusion formalisms are equivalent up to simple rescalings of space and time [26]. To learn the score of a data distribution $p_{0}(\mathbf{x})$, we minimize the denoising score matching (DSM) objective [27] with a function approximator. We reparametrize the score function with the 'denoiser' $\mathbf{s}_{\theta}(\mathbf{x}, \sigma)=\left(\mathbf{D}_{\theta}(\mathbf{x}, \sigma)-\mathbf{x}\right) / \sigma^{2}$, then at noise level $\sigma$ the DSM objective reads
$$
\begin{equation*}
\mathcal{L}_{\sigma}=\mathbb{E}_{\mathbf{x}_{0} \sim p_{0}, \mathbf{z} \sim \mathcal{N}(0, \mathbf{I})}\left\|\mathbf{D}_{\theta}\left(\mathbf{x}_{0}+\sigma \mathbf{z} ; \sigma\right)-\mathbf{x}_{0}\right\|_{2}^{2} . \tag{2}
\end{equation*}
$$

To balance the loss and importance of different noise scales, practical diffusion models all adopt certain weighting functions in their overall loss $\mathcal{L}=\int_{\sigma} d \sigma w(\sigma) \mathcal{L}_{\sigma}$.

\subsection*{3.2 Gaussian Data and Optimal Denoiser}

To motivate our linear score approximator set up, it is useful to consider the optimal score and the denoiser of a Gaussian distribution. For Gaussian data $\mathbf{x}_{0} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}), \mathbf{x}_{0} \in \mathbb{R}^{d}$ and $\boldsymbol{\Sigma}$ is a positive semi-definite matrix. When noising $\mathbf{x}_{0}$ by Gaussian noise at scale $\sigma$, the corrupted $\mathbf{x}$ satisfies $\mathbf{x} \sim \mathcal{N}\left(\boldsymbol{\mu}, \boldsymbol{\Sigma}+\sigma^{2} \mathbf{I}\right)$, for which the Bayes-optimal denoiser is an affine function of $\mathbf{x}$.
$$
\begin{equation*}
\mathbf{D}^{*}(\mathbf{x} ; \sigma)=\boldsymbol{\mu}+\left(\boldsymbol{\Sigma}+\sigma^{2} \mathbf{I}\right)^{-1} \boldsymbol{\Sigma}(\mathbf{x}-\boldsymbol{\mu}) \tag{3}
\end{equation*}
$$

For Gaussian data, minimizing (2) yields $\mathbf{D}^{*}$. This solution has an intuitive interpretation, i.e. the difference of the state $\mathbf{x}$ and distribution mean was projected onto the eigenbasis and shrinked mode-by-mode by $\lambda_{k} /\left(\lambda_{k}+\sigma^{2}\right)$. Thus, according to the variance $\lambda_{k}$ along target axis, modes with variance significantly higher than noise $\lambda_{k} \gg \sigma^{2}$ will be retained; modes with variance much smaller than noise will be "shrinked" out. Effectively $\sigma^{2}$ defines a threshold of signal and noise, and modes below which will be removed. This intuition similar to Ridge regression is made exact in Sec. 4.1.

\section*{4 Learning in Diffusion Models with a Linear Denoiser}

Problem set-up. Throughout the paper, we assume the denoiser at each noise scale is linear (affine) and independent across scales:
$$
\begin{equation*}
\mathbf{D}(\mathbf{x} ; \sigma)=\mathbf{W}_{\sigma} \mathbf{x}+\mathbf{b}_{\sigma} . \tag{4}
\end{equation*}
$$

Since the parameters $\left\{\mathbf{W}_{\sigma}, \mathbf{b}_{\sigma}\right\}$ are decoupled across noise scales, each $\sigma$ can be analysed independently. Through further parametrization, this umbrella form captures linear residual nets, deep linear nets, and linear convolutional nets (see Sec. 5).

We train on an arbitrary distribution $p_{0}$ with mean $\boldsymbol{\mu}$ and covariance $\Sigma$ by gradient flow on the full-batch DSM loss, i.e. the exact expectation over data and noise (2). (In practice, one cannot sample all $\mathbf{z}$ values, but the full-batch limit yields clean closed-form dynamics.)

This setting lets us dissect analytically the role of data spectrum, model architecture ( $\mathbf{W}_{\sigma}$ parametrisation), and loss variant in shaping diffusion learning.

\subsection*{4.1 Diffusion learning as ridge regression}

Gaussian equivalence. For any joint distribution $p(X, Y)$ the quadratic loss
$$
\mathcal{L}(\mathbf{W}, \mathbf{b})=\mathbb{E}_{p(X, Y)}\|\mathbf{W} X+\mathbf{b}-Y\|^{2}
$$
depends on $p$ only through the first two moments of $(X, Y)$; see App. C.1.1 for proof. Hence a linear denoiser trained on arbitrary $p_{0}$ interacts with the data solely via its mean $\boldsymbol{\mu}$ and covariance $\Sigma$.

Instance for diffusion. Under EDM loss (2), the noisy input-target pair is $X=\mathbf{x}_{0}+\sigma \mathbf{z}, Y=\mathbf{x}_{0}$, giving $\Sigma_{X X}=\Sigma+\sigma^{2} I, \Sigma_{Y X}=\Sigma$.

Gradient and optimum. Differentiating and setting gradients to zero yields
$$
\begin{align*}
\nabla_{\mathbf{W}_{\sigma}} \mathcal{L}_{\sigma} & =-2 \Sigma+2 \mathbf{W}_{\sigma}\left(\Sigma+\sigma^{2} \mathbf{I}\right)+\nabla_{\mathbf{b}_{\sigma}} \mathcal{L}_{\sigma} \boldsymbol{\mu}^{\top}, & \nabla_{\mathbf{b}_{\sigma}} \mathcal{L}_{\sigma} & =2\left(\mathbf{b}_{\sigma}-\left(\mathbf{I}-\mathbf{W}_{\sigma}\right) \boldsymbol{\mu}\right)  \tag{5}\\
\mathbf{W}_{\sigma}^{*} & =\Sigma\left(\Sigma+\sigma^{2} \mathbf{I}\right)^{-1}, & \mathbf{b}_{\sigma}^{*} & =\left(\mathbf{I}-\mathbf{W}_{\sigma}^{*}\right) \boldsymbol{\mu}
\end{align*}
$$
$$
\min \mathcal{L}_{\sigma}=\sigma^{2} \operatorname{Tr}\left[\Sigma\left(\Sigma+\sigma^{2} \mathbf{I}\right)^{-1}\right] .
$$

Thus the optimal linear denoiser reproduces the denoiser for the Gaussian approximation of data (3), and its best achievable loss is set purely by the data spectrum.
Other objectives. While the main text focuses on the EDM loss (2), we have worked out the gradients, optima, and learning dynamics for several popular variants used in diffusion and flow-matching [28] literature; these results are summarised in Tab. C.4 (derivations in App. C.4).
Ridge viewpoint. Because
$$
\mathcal{L}_{\sigma}=\mathbb{E}_{\mathbf{x} \sim p_{0}}\|\mathbf{W} \mathbf{x}+\mathbf{b}-\mathbf{x}\|^{2}+\sigma^{2}\|\mathbf{W}\|_{F}^{2},
$$
full-batch diffusion at noise scale $\sigma$ is simply auto-encoding with ridge regularisation of strength $\sigma^{2}$ (App. C.2.1, cf. [29]). We will exploit classic ridge-regression results when analyzing learning dynamics in the following sections.

\subsection*{4.2 Weight Learning Dynamics of a Linear Denoiser}

With the gradient structure in hand, we solve the full-batch gradient-flow ODE,
$$
\begin{equation*}
\frac{d \mathbf{W}_{\sigma}}{d \tau}=-\eta \nabla_{\mathbf{W}_{\sigma}} \mathcal{L}_{\sigma}, \quad \frac{d \mathbf{b}_{\sigma}}{d \tau}=-\eta \nabla_{\mathbf{b}_{\sigma}} \mathcal{L}_{\sigma}, \tag{GF}
\end{equation*}
$$
where $\tau$ is training time and $\eta$ the learning-rate.
Zero-mean data $(\boldsymbol{\mu}=0)$ : Exponential convergence mode-by-mode Because the gradients to W, b decouple (5), the dynamics is simplified on the eigenbasis of the covariance. We diagonalize the covariance, $\Sigma=\sum_{k=1}^{d} \lambda_{k} \mathbf{u}_{k} \mathbf{u}_{k}^{\top}$, with orthonormal principal components (PC) $\mathbf{u}_{k}$ and eigenvalues $\lambda_{k} \geq 0$ (the mode variances). Projecting (GF) onto this basis yields the closed-form solution (derivation in App. D.1):
$$
\begin{equation*}
\mathbf{b}_{\sigma}(\tau)=\mathbf{b}_{\sigma}(0) e^{-2 \eta \tau}, \quad \mathbf{W}_{\sigma}(\tau)=\mathbf{W}_{\sigma}^{*}+\sum_{k=1}^{d}\left[\mathbf{W}_{\sigma}(0)-\mathbf{W}_{\sigma}^{*}\right] \mathbf{u}_{k} \mathbf{u}_{k}^{\top} e^{-2 \eta \tau\left(\sigma^{2}+\lambda_{k}\right)} \tag{7}
\end{equation*}
$$

Interpretation. Each eigenmode projection of the weight $\mathbf{W}_{\sigma} \mathbf{u}_{k}$ converges to the optimal value $\mathbf{W}_{\sigma}^{*} \mathbf{u}_{k}$ exponentially with rate ( $\sigma^{2}+\lambda_{k}$ ); hence (i) the weights at larger noise $\sigma$ generally converge faster; (ii) at a fixed $\sigma$, high-variance $\lambda_{k}$ modes converge first, while modes buried beneath the noise floor ( $\lambda_{k} \ll \sigma^{2}$ ) share the same slower timescale. Fig. 2 A illustrates this spectrum-ordered convergence, with high-variance modes reaching their optima before the low-variance ones (see also 5 A ).

Non-centred data $(\mu \neq 0)$ : Interaction of mean and covariance learning. A non-zero mean introduces a rank-one coupling between $\mathbf{W}$ and $\mathbf{b}$ (matrix $M$ in Prop D.2). Eigenmodes of weights overlapping with the mean ( $\mathbf{u}_{k}^{\top} \boldsymbol{\mu} \neq 0$ ) now interact with $\mathbf{b}$, producing transient overshoots and other non-monotonic effects; orthogonal modes retain the exponential

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/11e476a8-86be-4d33-b2f0-7738bec1c9ed-04.jpg?height=566&width=780&top_left_y=1604&top_left_x=982}
\captionsetup{labelformat=empty}
\caption{Figure 2: Learning dynamics per eigenmode. Top: one-layer linear denoiser. Bottom: two-layer symmetric denoiser. (A,D) Weight trajectories $\mathbf{u}_{k}^{\top} \mathbf{W}_{\sigma}(\tau) \mathbf{u}_{k}(\sigma=$ 1). (B,E) Generated-variance $\tilde{\lambda}_{k}$ versus target variance $\lambda_{k}$. (C,F) Power-law relation between emergence time $\tau_{k}^{*}$ and $\lambda_{k}$.}
\end{figure}
convergnece above. App. D. 2 gives the full linear-system analysis and two-dimensional visualisations (Fig. 21).

\subsection*{4.3 Sampling Dynamics during Training}

For diffusion models, our goal is the generated distribution, obtained by integrating the probabil-ity-flow ODE (PF-ODE) backwards from a large $\sigma_{T}$ to a $\sigma_{\text {min }} \approx 0$,
$$
\begin{equation*}
\frac{d \mathbf{x}}{d \sigma}=-\sigma^{-1}\left[\left(\mathbf{W}_{\sigma}-I\right) \mathbf{x}+\mathbf{b}_{\sigma}\right] \tag{PF}
\end{equation*}
$$
initialized with Gaussian noise $\mathbf{x}_{T} \sim \mathcal{N}\left(0, \sigma_{T}^{2} \mathbf{I}\right)$. For linear denoiser, the PF-ODE is an inhomogeneous affine system, so its solution $\mathbf{x}_{\sigma}$ is necessarily an affine function of the initial state $\mathbf{x}_{T}$ [30], $\mathbf{x}\left(\sigma_{0}\right)=A\left(\sigma_{0} ; \sigma_{T}\right) \mathbf{x}\left(\sigma_{T}\right)+c\left(\sigma_{0} ; \sigma_{T}\right)$. Since the map is affine, the distribution of $\mathbf{x}\left(\sigma_{0}\right)$ remains Gaussian, with covariance $\sigma_{T}^{2} A\left(\sigma_{0} ; \sigma_{T}\right) A^{\top}\left(\sigma_{0} ; \sigma_{T}\right)$.
However, in general, the state-transition matrix $A\left(\sigma_{0} ; \sigma_{T}\right)$ is hard to evaluate, as it involves time-ordered matrix exponential, and the weight matrices at different noise scales $\mathbf{W}_{\sigma}$ may not commute. The analysis below-and our closed-form results-hinges on situations where commutativity is maintained by gradient flow or architectural bias, thus removing the time-ordering operator.
Lemma 4.1 (PF-ODE solution for commuting weights). If the linear denoiser $\mathbf{D}(\mathbf{x} ; \sigma)=\mathbf{W}_{\sigma} \mathbf{x}+\mathbf{b}_{\sigma}$ satisfies $\left[\mathbf{W}_{\sigma}, \mathbf{W}_{\sigma^{\prime}}\right]=0$ for all $\sigma, \sigma^{\prime}$, then for any $0<\sigma_{0}<\sigma_{T}$,
$$
\mathbf{x}\left(\sigma_{0}\right)=A\left(\sigma_{0}, \sigma_{T}\right) \mathbf{x}\left(\sigma_{T}\right)+c\left(\sigma_{0}, \sigma_{T}\right), \quad A\left(\sigma_{0}, \sigma_{T}\right)=\exp \left[-\int_{\sigma_{0}}^{\sigma_{T}} \frac{\mathbf{W}_{s}-\mathbf{I}}{s} d s\right]
$$

Interpretation. For each common eigenvector $\mathbf{u}_{k}$, the term $\left(\mathbf{u}_{k}^{\top} \mathbf{W}_{\sigma} \mathbf{u}_{k}-1\right) / \sigma$ is the instantaneous expansion (or contraction) rate of the sample variance along $\mathbf{u}_{k}$; the final variance is obtained by integrating this rate over noise scales $\sigma$ (see App. C.5).

When does commutativity hold? This arises in three common settings. (i) At convergence, this is satisfied by the optimal weights $\mathbf{W}_{\sigma}^{*}$ (6), which jointly diagonalize on eigenbasis of $\Sigma$. In such case, we recover the the closed-form solution to PF-ODE for Gaussian data, as found by [9, 31]. (ii) During training of linear denoisers, if weights are initialized to be aligned with eigenbasis of $\Sigma$, then gradient flow keeps them aligned, preserving commutativity (iii) For linear convolutional denoisers, circulant weights share the Fourier basis and commute by construction (see Sec.5.2). In these cases, the sampling process can be understood mode-by-mode. Here we show the explicit solution for one layer linear denoiser.
Proposition 4.2 (Dynamics of generated distribution in one layer case). Assume (i) zero-mean data, (ii) aligned initialization $\mathbf{W}_{\sigma}(0)=\sum_{k} Q_{k} \mathbf{u}_{k} \mathbf{u}_{k}^{\top}$, and (iii) gradient flow, full-batch training with learning rate $\eta$. Then, while training the one-layer linear denoiser, the generated distribution at time $\tau$ is $\mathcal{N}(\tilde{\mu}, \tilde{\Sigma})$ with $\tilde{\Sigma}=\sum_{k} \tilde{\lambda}_{k}(\tau) \mathbf{u}_{k} \mathbf{u}_{k}^{\top}$ and
$\tilde{\lambda}_{k}(\tau)=\sigma_{T}^{2} \frac{\Phi_{k}^{2}\left(\sigma_{0}, \tau\right)}{\Phi_{k}^{2}\left(\sigma_{T}, \tau\right)}, \quad \Phi_{k}(\sigma, \tau)=\sqrt{\lambda_{k}+\sigma^{2}} \exp \left[\frac{1-Q_{k}}{2} \operatorname{Ei}\left(-2 \eta \tau \sigma^{2}\right) e^{-2 \eta \tau \lambda_{k}}-\frac{1}{2} \operatorname{Ei}\left(-2 \eta \tau\left(\sigma^{2}+\lambda_{k}\right)\right)\right]$ where Ei is the exponential-integral function. (derivation in App.D.3)

Spectral bias. Figure 2 B traces the variance trajectory $\tilde{\lambda}_{k}(\tau)$ for each eigen-mode. All modes begin with the same initialization-induced level, then follow sigmoidal curves to their targets, but in descending order of $\lambda_{k}$ We define the first-passage time $\tau_{k}^{*}$ as the training time at which $\tilde{\lambda}_{k}(\tau)$ reaches the geometric (or harmonic) mean of its initial and target values. We find the first-passage time obeys an inverse law $\tau_{k}^{*} \propto \lambda_{k}^{-\alpha}, \alpha \approx 1$, (Fig. 2C), which implies that learning a mode with variance $1 / 10$ smaller takes roughly 10 times longer to converge. With larger weight initialization (larger $Q_{k}$ ), the initial variance is closer to the target variance of some modes, then the inverse law splits into separate branches for modes with rising vs. decaying variance (Fig. 5B, Fig. 6).
Practical implication. This suggests when training stops earlier, the distribution in higher variance PC spaces have already converged, while low-variance ones-often the perceptual finer points such as letter strokes or finger joints-are under-trained. This could be an explanation for the familiar "wrong detail" artefacts in diffusion samples.

\section*{5 Deep and Convolutional Extensions}

After analyzing the simplest linear denoiser, we set out to examine the effect of architectures via different parametrizations of the weights, specifically deeper linear models and linear convolutional networks. In the following, we will assume $\mu=0$ and focus on learning of covariance.

\subsection*{5.1 Deeper linear network}

Consider a depth- $L$ linear denoiser $\mathbf{D}(\mathbf{x}, \sigma)=\mathbf{W}_{L} \cdots \mathbf{W}_{1} \mathbf{x}$, where-for notational clarity-we suppress the explicit $\sigma$-dependence of weights. We assume aligned initialization, where for singular decomposition of each matrix, $\mathbf{W}_{\ell}(0)=U_{\ell} \Lambda_{\ell} V_{\ell}^{\top}$, the right basis of each layer matching the left basis of the next, $V_{\ell+1}=U_{\ell}, \forall \ell=1, \ldots, L-1$, and with $U_{L}=V_{1}=U$ where $U$ diagonalizes data covariance $\Sigma$. Then the total weight at initialization is $\mathbf{W}_{\text {tot }}(0)=\prod_{\ell=1}^{L} \mathbf{W}_{\ell}(0)=U\left(\prod_{\ell=1}^{L} \Lambda_{\ell}\right) U^{\top}$, With aligned initialization, every eigenmode learns independently-mirroring classical results [15], 32]. In our case, this also implies that the total weight $\prod_{l} \mathbf{W}_{l}$ shares the eigenbasis $U$ across training and noise scales, thus commute, making sampling tractable.
One especially illuminating case is the two-layer symmetric network, where $\mathbf{D}(\mathbf{x}, \sigma)=P_{\sigma} P_{\sigma}^{\top} \mathbf{x}$.
Proposition 5.1 (Dynamics of weight and distribution in two layer linear model). Assume (i) centered data $\mu=0$; (ii) the weight matrix is initialized aligned, i.e. $P_{\sigma}(0) P_{\sigma}(0)^{\top}=\sum_{k} Q_{k} \mathbf{u}_{k} \mathbf{u}_{k}^{\top}$, then the gradient flow ODE admits a closed-form solution (derivation in App. E.1)
$$
\begin{equation*}
\mathbf{W}_{\sigma}(\tau)=P_{\sigma}(\tau) P_{\sigma}(\tau)^{\top}=\sum_{k} \frac{\lambda_{k}}{\sigma^{2}+\lambda_{k}} \mathbf{u}_{k} \mathbf{u}_{k}^{\top}\left(\frac{Q_{k}}{\left(\frac{\lambda_{k}}{\sigma^{2}+\lambda_{k}}-Q_{k}\right) e^{-8 \eta \lambda_{k} \tau}+Q_{k}}\right) \tag{8}
\end{equation*}
$$

The generated distribution at time $\tau$ is $\mathcal{N}(\tilde{\mu}, \tilde{\Sigma})$ with $\tilde{\Sigma}=\sum_{k} \tilde{\lambda}_{k}(\tau) \mathbf{u}_{k} \mathbf{u}_{k}^{\top}$ and $\tilde{\lambda}_{k}(\tau)=\sigma_{T}^{2} \frac{\Phi_{k}^{2}\left(\sigma_{0}\right)}{\Phi_{k}^{2}\left(\sigma_{T}\right)}$
$$
\Phi_{k}(\sigma)=(\sigma)^{\frac{\left(1-Q_{k}\right) e^{-8 \eta \tau \lambda_{k}}}{Q_{k}+\left(1-Q_{k}\right) e^{-8 \eta \tau \lambda_{k}}}}\left[\lambda_{k} e^{-8 \eta \tau \lambda_{k}}+Q_{k}\left(1-e^{-8 \eta \tau \lambda_{k}}\right)\left(\lambda_{k}+\sigma^{2}\right)\right]^{\frac{Q_{k}}{2 Q_{k}+2\left(1-Q_{k}\right) e^{-8 \eta \tau \lambda_{k}}}}
$$

Interpretation. The learning dynamics of weights and variance along different principal components are visualized in Fig 2 D-F. Compared to one-layer case, here, the weight converges along the PCs via sigmoidal dynamics, with the emergence time (reaching harmonic mean of initial and final value) $\tau_{k}^{*}=\ln 2 /\left(8 \eta \lambda_{k}\right)$. As for generated distribution, we find similar relationship between the target variance and emergence time $\tau_{k}^{*} \propto \lambda_{k}^{-\alpha}, \alpha \approx 1$. For the more general non-aligned initialization, we show the non-aligned parts of weight will follow non-monotonic rise-and-fall dynamics (App. E.1.2). Extensions to non-symmetric two layer model and deeper model were studied in App. F. which have similar bias but lack clean expressions.

\subsection*{5.2 Linear convolutional network}

We consider a linear denoiser with convolutional architecture, $\mathbf{D}(\mathbf{x}, \sigma)=\mathbf{w}_{\sigma} * \mathbf{x}$ where samples $\mathbf{x} \in \mathbb{R}^{N}$ have 1d spatial structure, and a width $K$ convolution filter $\mathbf{w}_{\sigma}$ operates on it. The analysis could be easily generalized to 2 d convolution. With circular boundary condition, $\mathbf{w}_{\sigma}$ defines a circulant weight matrix $\mathbf{W}_{\sigma} \in \mathbb{R}^{N \times N}$, where $\mathbf{w}_{\sigma} * \mathbf{x}=\mathbf{W}_{\sigma} \mathbf{x}$. One favorable property of circulant matrices is that they are diagonalized by discrete Fourier transform $F$ [33].
$$
\begin{equation*}
\mathbf{W}_{\sigma}=F \Gamma_{\sigma} F^{*} \quad F_{m k}:=\frac{1}{\sqrt{N}} \exp \left(-2 \pi i \frac{m k}{N}\right) \tag{9}
\end{equation*}
$$

Thus all weights $\mathbf{W}_{\sigma}$ commutes, which allows us to leverage Lemma 4.1, and solve the sampling dynamics mode-by-mode on the Fourier basis, leading to following result.
Proposition 5.2. Linear convolutional denoisers with circular boundary can only model stationary Gaussian processes (GP), with independent Fourier modes, proof in App.G.2.

Learning dynamics of full-width filter $K=N$ When convolution filter $\mathbf{w}_{\sigma}$ is as large as the signal, the gradient flow is diagonal and unconstrained in the Fourier domain. Thus, the analyses in Sec. 4 re-emerge with variance of Fourier mode $\tilde{\Sigma}_{k k}$ taking the place of $\lambda_{k}$.
Proposition 5.3 (Full-width circular convolution learning dynamics). Let $\mathbf{D}(\mathbf{x}, \sigma)=\mathbf{w}_{\sigma} * \mathbf{x}$, with full-width filter $K=N$, and train $\mathbf{w}_{\sigma}$ by full-batch gradient flow at rate $\eta$. Then the weights at noise $\sigma$ and its spectral representation $\gamma$ evolves as
$$
\begin{equation*}
\mathbf{w}_{\sigma}(\tau)=\frac{1}{\sqrt{N}} F^{*} \gamma(\tau, \sigma) \quad ; \quad \gamma_{k}(\tau, \sigma)=\gamma_{k}^{*}(\sigma)+\left(\gamma_{k}(\tau, \sigma)-\gamma_{k}^{*}(\sigma)\right) e^{-2 N \eta\left(\sigma^{2}+\tilde{\Sigma}_{k k}\right) \tau} \tag{10}
\end{equation*}
$$
where $\gamma_{k}^{*}(\sigma)=\tilde{\Sigma}_{k k} /\left(\sigma^{2}+\tilde{\Sigma}_{k k}\right)$ and $\tilde{\Sigma}_{k k}=\left[F^{*} \Sigma F\right]_{k k}$ is the variance of Fourier mode.
The generated distribution has diagonal covariance in the Fourier basis and follows exactly Prop. 4.2 after the replacement $\lambda_{k} \rightarrow \tilde{\Sigma}_{k k}, \eta \rightarrow N \eta, U \rightarrow F$. (derivation in App. G.3)

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 1: Summary of theory. exp. and sigm. denotes exponential and sigmoidal convergence. xN denotes the $N$ time speed up due to weight sharing.}
\begin{tabular}{|l|l|l|l|l|}
\hline & One layer & Sym. two layer & Full-width linear conv & Patch linear conv \\
\hline Param. & W & $P P^{T}$ & $\mathbf{W}=\operatorname{Circ}(\mathbf{w})$ & $\mathbf{W}=\operatorname{Circ}(\mathbf{w}), K<N$ \\
\hline Weight dyn. & PC, exp. [7] & PC, sigm. 5.1 & Fourier mode, exp. xN 5.3 & PC of patches, exp. xN 5.4 \\
\hline Learned distr. & Gaussian & Gaussian & Stationary GP 5.2 & Stationary GP \\
\hline Distr. dyn. & PC, sigm., power law 4.2 & PC, sigm., power law 5.1 & Fourier, sigm., xN, power law 5.3 & Fourier, N.S. \\
\hline
\end{tabular}
\end{table}

Interpretation. The weight and distribution dynamics mirror the fully-connected case, with spectral bias towards higher variance Fourier modes; convolutional weight sharing simply multiplies every rate by $N$, accelerating convergence without altering the inverse-variance law.

Notably, the learned distribution is asymptotically equivalent to the Gaussian approximation to the original training data with all possible spatial shifts as augmentations (proof in App. G.3.2). This is one case where equivariant architectural constraints facilitates creativity as discussed in [25]. Similarly, two-layer linear conv net with full-width filter can be treated as in Sec 5.1.

Learning dynamics of local filter $K<N$ When the convolution filter has a limited bandwidth $K \neq N$, the Fourier domain dynamics get constrained, so it is easier to work with the filter weights. Let $r$ be the half-width of the kernel ( $K=2 r+1$ ). Define the circular patch extractor $\mathcal{P}_{r}(\mathbf{x})= \left[\mathbf{x}_{i-r: i+r}\right]_{i=1}^{N} \in \mathbb{R}^{K \times N}$, and the patch covariance $\Sigma_{\text {patch }}=\frac{1}{N} \mathbb{E}_{\mathbf{x}}\left[\mathcal{P}_{r}(\mathbf{x}) \mathcal{P}_{r}(\mathbf{x})^{\top}\right] \in \mathbb{R}^{K \times K}$.
Proposition 5.4 (Patch-convolution learning dynamics). For the circular convolutional denoiser, $\mathbf{D}(\mathbf{x}, \sigma)=\mathbf{w}_{\sigma} * \mathbf{x}$ trained by full-batch gradient flow with step size $\eta$. Let $\mathbf{e}_{0} \in \mathbb{R}^{K}$ be the one-hot vector with a single 1 at the center position $r+1$ (1-indexed). (derivation in App. G.4)
$$
\mathbf{w}_{\sigma}(\tau)=\mathbf{w}_{\sigma}^{*}+\exp \left[-2 N \eta \tau\left(\sigma^{2} I+\Sigma_{\text {patch }}\right)\right]\left(\mathbf{w}_{\sigma}(0)-\mathbf{w}_{\sigma}^{*}\right), \quad \mathbf{w}_{\sigma}^{*}=\left(\sigma^{2} I+\Sigma_{\text {patch }}\right)^{-1} \Sigma_{\text {patch }} \mathbf{e}_{0} .
$$

Interpretation. Training with a narrow convolutional filter reduces to ridge regression in patch space. Under gradient flow, filter converges along eigenmodes of patch $\Sigma_{\text {patch }}$ : modes with larger variance converges sooner, those with smaller variance later, preserving the inverse-variance law. It also enjoys the $N$ times speed up given by weight sharing, accelerating progress without altering the ordering. The sampling ODE remains diagonal in Fourier space, so the generated distribution will be a stationary Gaussian process with local covariance structure shaped by the learned patch denoiser, though its exact form needs numerical integration to spell out. This setting is similar to the equivariant and local score machine described in [25].

\section*{6 Empirical Validation of the Theory in Practical Diffusion Model Training}

General Approach To test our theoretical predictions about the evolution of generated distribution (esp. covariance), we resort to the following method: 1) we fix a training dataset $\left\{\mathbf{x}_{i}\right\}$ and compute its empirical mean $\boldsymbol{\mu}$ and covariance $\boldsymbol{\Sigma}$. We then perform an eigen-decomposition of $\boldsymbol{\Sigma}$, obtaining eigenvalues $\lambda_{k}$ and eigenvectors $\mathbf{u}_{k}$. 2) Next, we train a diffusion model on this dataset by optimizing the DSM objective with a neural network denoiser $\mathbf{D}_{\theta}(\mathbf{x}, \sigma)$. 3) During training, at certain steps $\tau$, we generate samples $\left\{\mathbf{x}_{i}^{\tau}\right\}$ from the diffusion model by integrating the PF-ODE (1). We then estimate the sample mean $\tilde{\boldsymbol{\mu}}^{\tau}$ and sample covariance $\tilde{\boldsymbol{\Sigma}}^{\tau}$. Finally, we compute the variance of the generated samples along the eigenbasis of training data, $\tilde{\lambda}_{k}^{\tau}=\mathbf{u}_{k}^{\top} \tilde{\boldsymbol{\Sigma}}^{\tau} \mathbf{u}_{k}$. To stress test our theory and maximize its relevance, we'd keep most of the training hyperparameters as practical ones.

\subsection*{6.1 Multi-Layer Perceptron (MLP)}

To test our theory about linear and deep linear network (Prop 4.2|5.1), we used a Multi-Layer Perceptron (MLP) inspired by the SongUnet in EDM [26, 34] (details in App. I.2). We found this architecture effective in learning distribution like point cloud data (Fig. 23). We kept the preconditioning, loss weighting and initialization the same as in [26].

Experiment 1: Zero-mean Gaussian Data x MLP We first consider a zero mean Gaussian $\mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$ in $d$ dimension as training distribution, with covariance defined as a randomly rotated diagonal matrix with log normal spectrum (details in App. I.4). During training, the generated variance of each eigenmode follows a sigmoidal trajectory toward its target value $\lambda_{k}$; modes with larger $\lambda_{k}$ cross the plateau sooner (Fig.9/4). We mark the emergence time $\tau^{*}$ as the step at which the variance reaches the geometric mean of its initial and asymptotic values (Fig. 9B). Across both

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/11e476a8-86be-4d33-b2f0-7738bec1c9ed-08.jpg?height=401&width=1342&top_left_y=214&top_left_x=384}
\captionsetup{labelformat=empty}
\caption{Figure 3: Spectral Learning Dynamics of MLP-UNet (FFHQ32). A. Generated samples during training. B. Evolution of sample variance $\tilde{\lambda}_{k}(\tau)$ across eigenmodes during training. C. Heatmap of variance trajectories along all eigenmodes, with dots marking mode emergence times $\tau^{*}$ (first-passage time at the geometric mean of initial and final variances). The gray zone ( $0.5-2 \times$ target variance) indicates modes starting too close to their target, causing unreliable $\tau^{*}$ estimates. D. Power-law scaling of $\tau^{*}$ versus target variance $\lambda_{k}$. A separate law was fit for modes with increasing and decreasing variance, excluding the middle gray-zone eigenmodes for stability.}
\end{figure}
high- and low-variance modes, $\tau^{*}$ obeys an inverse power-law, $\tau^{*} \propto \lambda_{k}^{-\alpha}$. With higher-dimensional Gaussians the exponent is estimated more precisely and remains close to 1: for $\mathrm{d}=256, \alpha=1.08$; for $\mathrm{d}=512, \alpha_{\text {incr }}=1.05$ and $\alpha_{\text {decr }}=1.13$ (Fig. 9 C ). The scaling breaks down only for modes whose initial variance is already near $\lambda_{k}$; in that regime the trajectory is less sigmoidal and $\tau^{*}$ becomes ill-defined. This result shows that despite many non-idealistic conditions e.g. deeper network, nonlinear activation function, residual connections, normal weights initialization, shared parametrization of denoisers at different noise level, the prediction from the linear network theory is still quantitatively correct.

Experiment 2: Natural Image Datasets x MLP Next, we validated our theory on natural image datasets. We flattened the images as a vectors, and trained a deeper and wider MLPUNet to learn the distribution. Using FFHQ as our running example, monitoring the generated samples throughout training (Fig. 3A), despite heavy noise early on, the coarse facial con-tours-corresponding to the mean and top principal components of human face distribution [7]-emerge quickly, whereas high-frequency details (lower PCs) only appear later. We note that

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/11e476a8-86be-4d33-b2f0-7738bec1c9ed-08.jpg?height=433&width=847&top_left_y=1370&top_left_x=913}
\captionsetup{labelformat=empty}
\caption{Figure 4: Learning dynamics of UNet differs I FFHQ32. A. Sample trajectory from CNN-UNet. B. Variance evolution along covariance eigenmodes. (c.f. Fig. 3A.C.)}
\end{figure}
this spectral ordering effect of training dynamics is reminiscent and similar to that in the sampling dynamics after training [19, 22].
Quantitatively, the sample covariance $\tilde{\Sigma}^{\tau}$ rapidly aligns with and becomes close to diagonal in the data eigenbasis $U$ (Fig. 10). The top eigenmodes' variances, $\tilde{\lambda}_{k}(\tau)$, follow sigmoidal trajectories converging to their targets, and their "emergence times" $\tau_{k}^{*}$ increase down the spectrum (Fig. 3B,C). We exclude a central band of modes whose initial variances lie within $0.5-2 \times$ the target, since their undulating learning dynamics make first-passage time estimates unreliable. After this exclusion, modes with increasing and decreasing variance each exhibit a clear power-law scaling between emergence step $\tau^{*}$ and target variance $\lambda_{k}$, with exponents $-0.48\left(R^{2}=0.97, N=57\right)$ and -0.35 ( $R^{2}=0.92, N=2,914$ ), respectively (Fig. 3D). Although the observed spectral bias is slightly attenuated relative to the Gaussian case and linear theory prediction, it remains robust and consistent across datasets (MNIST, CIFAR-10, FFHQ32 and AFHQ32) (App. B.2.2). This shows that even with natural image data, the distributional learning dynamics of MLP-based diffusion still suffers from slower convergence speed for lower eigenmodes.

\subsection*{6.2 Convolutional Neural Networks (CNNs)}

Next we turn to the convolutional U-Net-the work-horse of image-diffusion models [34, 35]. For a full-width linear convolutional network our analysis predicts an inverse-variance law in Fourier space (Prop. 5.3). The patch-convolution variant lacks a clear forecast on distribution, so the following experiments probe empirically whether-and how-its learning dynamics is affected by spectral bias.

Experiment 3: Natural Image Datasets x CNN UNet Training on the same FFHQ dataset, the distributional learning trajectory of CNN-UNet is markedly different from the MLPs: early in training, we do not see contour of face, but locally coherent patches, reminiscent of Ising models (Fig. 4 A .). Visually and variance-wise, the CNN-based UNet converge much faster and better than the MLP-based UNet, matching the N-fold speed-up from weight sharing (Prop. 5.4 , Fig. $4 \mathbf{B}$ ). When projecting onto the data eigenbasis, all eigenmodes with increasing variance rise simultaneously, while eigenmodes with decreasing variance co-decay at a later time, giving an effective power-law exponent $\alpha \approx 0$; Thus, spectral bias is essentially absent (Fig. 12C.D.).
The likely cause is locality: local convolutional filters couple neighbouring pixels, binding many Fourier modes into one learning unit. Sampling remains diagonal in Fourier space, so a broad band of modes is amplified simultaneously. The early CNN denoiser is indeed well-approximated by a local linear filter (Fig. 14). A full analytic treatment of convolutional U-Net training dynamics is deferred to future work.

\section*{7 Discussion}

In summary, we presented closed-form solutions for training denoisers with linear, deep linear or linear convolutional architecture, under the DSM objective on arbitrary data. This setup allows for a precise mode-wise understanding of the gradient flow dynamics of the denoiser and the evolution of the learned distribution: covariance eigenmode for deep linear network and Fourier mode for convolutional networks. For both the weights and the distribution, we showed analytical evidence of spectral bias, i.e. weights converge faster along the eigenmodes or Fourier modes with high variance, and the learned distribution recovers the true variance first along the top eigenmodes. These theoretical results are summarized in Tab. 1 .

We hope these results can serve as a solvable model for spectral bias in the diffusion models through the nested training and sampling dynamics. Furthermore, our analysis is not limited to the diffusion and the DSM loss, in App. H. we showed a similar derivation for the spectral bias in flow matching models [28, 36].

Relevance of our theoretical assumptions We found, for the purpose of analytical tractability, we made many idealistic assumptions about neural network training, 1) linear neural network, 2) small or orthogonal weight initialization, 3) "full-batch" gradient flow, 4) independent evolution of weights at each noise scale. In our MLP experiments, we found even when all of these assumptions were somewhat violated, the general theoretical prediction is correct, with modified power coefficients. This shows most of these assumptions could be relaxed in real life, and the spectrum of data indeed have a large effect on the learning dynamics, esp. for fully connected networks.

Inductive bias of the local convolution In our CNN experiments, however, the theoretical predictions from linear models deviate: the spectral bias in learning speed does not directly apply to the distribution of full images. Although our theory predicts that filter-weight learning dynamics are governed by the patch covariance, the ultimate image distribution is shaped by the convolution of those filters. To date, many learning-theory analyses for diffusion models assume MLP-like architectures [24]. For future theoretical work on the learning dynamics of practical diffusion models, a rigorous treatment of the local convolutional structure-and its frequency-coupling effects-will likely be essential, rather than relying on full-width convolution analyses [37].

Broader Impact Although our work is primarily theoretical, the inverse scaling law could offer valuable insights into how to improve the training of large-scale diffusion or flow generative models.

\section*{References}
[1] Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In Proceedings of the 32nd International Conference on Machine Learning (ICML), 2015.
[2] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems (NeurIPS), 2020.
[3] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In Advances in Neural Information Processing Systems (NeurIPS), 2021.
[4] Antonio Torralba and Aude Oliva. Statistics of natural image categories. Network: computation in neural systems, 14(3):391, 2003.
[5] Hagai Attias and Christoph Schreiner. Temporal low-order statistics of natural sounds. Advances in neural information processing systems, 9, 1996.
[6] Dawei W Dong and Joseph J Atick. Statistics of natural time-varying images. Network: computation in neural systems, 6(3):345, 1995.
[7] Matthew Turk and Alex Pentland. Eigenfaces for recognition. Journal of cognitive neuroscience, 3(1):71-86, 1991.
[8] Binxu Wang and Carlos R Ponce. A geometric analysis of deep generative image models and its applications. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=GH7QRzUDdXG.
[9] Binxu Wang and John J. Vastola. The Hidden Linear Structure in Score-Based Models and its Application. arXiv e-prints, art. arXiv:2311.10892, November 2023. doi: 10.48550/arXiv.2311. 10892.
[10] Xiang Li, Yixiang Dai, and Qing Qu. Understanding generalizability of diffusion models requires rethinking the hidden gaussian structure. arXiv preprint arXiv:2410.24060, 2024.
[11] Binxu Wang and John Vastola. The unreasonable effectiveness of gaussian score approximation for diffusion models and its applications. Transactions on Machine Learning Research, December 2024. arXiv preprint arXiv:2412.09726.
[12] Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred Hamprecht, Yoshua Bengio, and Aaron Courville. On the spectral bias of neural networks. In International Conference on Machine Learning, pages 5301-5310. PMLR, 2019.
[13] Blake Bordelon, Abdulkadir Canatar, and Cengiz Pehlevan. Spectrum dependent learning curves in kernel regression and wide neural networks. In Hal Daumé III and Aarti Singh, editors, Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 1024-1034. PMLR, 13-18 Jul 2020. URL https://proceedings.mlr.press/v119/bordelon20a.html.
[14] Abdulkadir Canatar, Blake Bordelon, and Cengiz Pehlevan. Spectral bias and task-model alignment explain generalization in kernel regression and infinitely wide neural networks. Nature Communications, 12(1):2914, May 2021. ISSN 2041-1723. doi: 10.1038/s41467-021-23103-1. URL https://doi.org/10.1038/s41467-021-23103-1
[15] Andrew M Saxe, James L McClelland, and Surya Ganguli. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. arXiv preprint arXiv:1312.6120, 2013.
[16] Suriya Gunasekar, Jason D Lee, Daniel Soudry, and Nati Srebro. Implicit bias of gradient descent on linear convolutional networks. Advances in neural information processing systems, 31, 2018.
[17] Arnu Pretorius, Steve Kroon, and Herman Kamper. Learning dynamics of linear denoising autoencoders. In International Conference on Machine Learning, pages 4141-4150. PMLR, 2018.
[18] Chulhee Yun, Shankar Krishnan, and Hossein Mobahi. A unifying view on implicit bias in training linear neural networks. arXiv preprint arXiv:2010.02501, 2020.
[19] Sander Dieleman. Diffusion is spectral autoregression, 2024. URLhttps://sander.ai/2024/ 09/02/spectral-autoregression.html.
[20] Severi Rissanen, Markus Heinonen, and Arno Solin. Generative modelling with inverse heat dissipation. arXiv preprint arXiv:2206.13397, 2022.
[21] Florentin Guth, Simon Coste, Valentin De Bortoli, and Stephane Mallat. Wavelet score-based generative modeling. Advances in neural information processing systems, 35:478-491, 2022.
[22] Binxu Wang and John J Vastola. Diffusion models generate images like painters: an analytical theory of outline first, details later. arXiv preprint arXiv:2303.02490, 2023.
[23] Giulio Biroli, Tony Bonnaire, Valentin De Bortoli, and Marc Mézard. Dynamical regimes of diffusion models. Nature Communications, 15(1):9957, 2024.
[24] Kulin Shah, Sitan Chen, and Adam Klivans. Learning mixtures of gaussians using the ddpm objective. Advances in Neural Information Processing Systems, 36:19636-19649, 2023.
[25] Mason Kamb and Surya Ganguli. An analytic theory of creativity in convolutional diffusion models. arXiv preprint arXiv:2412.20292, 2024.
[26] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. arXiv preprint arXiv:2206.00364, 2022.
[27] Pascal Vincent. A connection between score matching and denoising autoencoders. Neural Computation, 23(7):1661-1674, 2011.
[28] Yaron Lipman, Marton Havasi, Peter Holderrieth, Neta Shaul, Matt Le, Brian Karrer, Ricky TQ Chen, David Lopez-Paz, Heli Ben-Hamu, and Itai Gat. Flow matching guide and code. arXiv preprint arXiv:2412.06264, 2024.
[29] Chris M Bishop. Training with noise is equivalent to tikhonov regularization. Neural computation, 7(1):108-116, 1995.
[30] Philip Hartman. Ordinary differential equations. SIAM, 2002.
[31] Emile Pierret and Bruno Galerne. Diffusion models for gaussian distributions: Exact solutions and wasserstein errors. arXiv preprint arXiv:2405.14250, 2024.
[32] Kenji Fukumizu. Effect of batch learning in multilayer neural networks. Gen, 1(04):1E-03, 1998.
[33] Robert M Gray et al. Toeplitz and circulant matrices: A review. Foundations and Trends® in Communications and Information Theory, 2(3):155-239, 2006.
[34] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021. URL https://openreview. net/forum?id=PxTIG12RRHS.
[35] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention-MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pages 234-241. Springer, 2015.
[36] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.
[37] Suriya Gunasekar, Jason D Lee, Daniel Soudry, and Nati Srebro. Implicit bias of gradient descent on linear convolutional networks. Advances in neural information processing systems, 31, 2018.
[38] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840-6851, 2020.
[39] Marvin Li, Aayush Karan, and Sitan Chen. Blink of an eye: a simple theory for feature localization in generative models. arXiv preprint arXiv:2502.00921, 2025.
[40] Minyoung Huh, Hossein Mobahi, Richard Zhang, Brian Cheung, Pulkit Agrawal, and Phillip Isola. The low-rank simplicity bias in deep networks. arXiv preprint arXiv:2103.10427, 2021.
[41] Li Jing, Jure Zbontar, et al. Implicit rank-minimizing autoencoder. Advances in Neural Information Processing Systems, 33:14736-14746, 2020.
[42] Shuxuan Guo, Jose M Alvarez, and Mathieu Salzmann. Expandnets: Linear overparameterization to train compact convolutional networks. Advances in Neural Information Processing Systems, 33:1298-1310, 2020.
[43] Meena Jagadeesan, Ilya Razenshteyn, and Suriya Gunasekar. Inductive bias of multi-channel linear convolutional networks with bounded weight norm. In Conference on Learning Theory, pages 2276-2325. PMLR, 2022.
[44] Kathlén Kohn, Thomas Merkh, Guido Montúfar, and Matthew Trager. Geometry of linear convolutional networks. SIAM Journal on Applied Algebra and Geometry, 6(3):368-406, 2022.
[45] Kathlén Kohn, Guido Montúfar, Vahid Shahverdi, and Matthew Trager. Function space and critical points of linear convolutional networks. SIAM Journal on Applied Algebra and Geometry, 8(2):333-362, 2024.
[46] Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky. Deep image prior. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 9446-9454, 2018.
[47] Prithvijit Chakrabarty and Subhransu Maji. The spectral bias of the deep image prior. arXiv preprint arXiv:1912.08905, 2019.
[48] Zezhou Cheng, Matheus Gadelha, Subhransu Maji, and Daniel Sheldon. A bayesian perspective on the deep image prior. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5443-5451, 2019.
[49] Basri Ronen, David Jacobs, Yoni Kasten, and Shira Kritchman. The convergence rate of neural networks for learned functions of different frequencies. Advances in Neural Information Processing Systems, 32, 2019.
[50] Ronen Basri, Meirav Galun, Amnon Geifman, David Jacobs, Yoni Kasten, and Shira Kritchman. Frequency bias in neural networks for input of non-uniform density. In International conference on machine learning, pages 685-694. PMLR, 2020.
[51] Mingchen Li, Mahdi Soltanolkotabi, and Samet Oymak. Gradient descent with early stopping is provably robust to label noise for overparameterized neural networks. In International conference on artificial intelligence and statistics, pages 4313-4324. PMLR, 2020.
[52] Reinhard Heckel and Mahdi Soltanolkotabi. Denoising and regularization via exploiting the structural bias of convolutional generators. arXiv preprint arXiv:1910.14634, 2019.
[53] James R Bunch, Christopher P Nielsen, and Danny C Sorensen. Rank-one modification of the symmetric eigenproblem. Numerische Mathematik, 31(1):31-48, 1978.
[54] Ming Gu and Stanley C Eisenstat. A stable and efficient algorithm for the rank-one modification of the symmetric eigenproblem. SIAM journal on Matrix Analysis and Applications, 15(4): 1266-1276, 1994.
[55] Gene H Golub. Some modified matrix eigenvalue problems. SIAM review, 15(2):318-334, 1973.
[56] Gilbert Strang. A proposal for Toeplitz matrix calculations. Studies in Applied Mathematics, 74 (2):171-176, 1986.
[57] Mingkui Chen. On the solution of circulant linear system. Technical Report TR-401, Yale University, Department of Computer Science, 1985. URL https://cpsc.yale.edu/sites/ default/files/files/tr401.pdf.
[58] Michela Mastronardi. Presentation slides (mastronardi.pdf). https://www.math.unipd.it/ ~michela/2gg07/TALKS/Mastronardi.pdf, 2007. Accessed: 2025-05-23.
[59] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper/2019/file/ 3001ef257407d5a371a96dcd947c7d93-Paper.pdf.
Appendix
Contents
1 Introduction ..... 1
2 Related Work and Motivation: Spectral Bias in Distribution Learning ..... 2
3 Background ..... 2
3.1 Score-based Diffusion Models ..... 2
3.2 Gaussian Data and Optimal Denoiser ..... 3
4 Learning in Diffusion Models with a Linear Denoiser ..... 3
4.1 Diffusion learning as ridge regression ..... 3
4.2 Weight Learning Dynamics of a Linear Denoiser ..... 4
4.3 Sampling Dynamics during Training ..... 5
5 Deep and Convolutional Extensions ..... 5
5.1 Deeper linear network ..... 6
5.2 Linear convolutional network ..... 6
6 Empirical Validation of the Theory in Practical Diffusion Model Training ..... 7
6.1 Multi-Layer Perceptron (MLP) ..... 7
6.2 Convolutional Neural Networks (CNNs) ..... 9
7 Discussion ..... 9
A Extended Related works ..... 17
B Extended Results ..... 19
B. 1 Extended Visualization of Theoretical Results ..... 19
B.1.1 Scaling curves of diffusion training (EDM) ..... 19
B.1.2 Scaling curves of flow matching ..... 21
B. 2 Extended Empirical Results ..... 22
B.2.1 MLP-UNet Gaussian training experiments ..... 22
B.2.2 MLP-UNet natural image training experiments ..... 22
B.2.3 CNN-UNet training experiment ..... 25
C Detailed Derivation: General analysis ..... 32
C. 1 General Property of Linear Regression ..... 32
C.1.1 Gaussian equivalence ..... 32
C. 2 General analysis of the gradient learning of linear predictor ..... 33
C.2.1 Denoising as Ridge Regression ..... 33
C. 3 General Analysis of the Denoising Score Matching Objective ..... 34
C. 4 Gradient Structure of Other Variants of Loss ..... 36
C. 5 General Analysis of the Sampling ODE ..... 39
C. 6 KL Divergence Computation ..... 40
D Detailed Derivations for the One-Layer Linear Model ..... 42
D. 1 Zero-mean data: Exponential Converging Training Dynamics ..... 42
D.1.1 Discrete time Gradient descent dynamics ..... 43
D.1.2 Special parametrization: residual connection ..... 44
D. 2 General non-centered distribution: Interaction of mean and covariance learning ..... 45
D.2.1 Special case: low dimensional interaction of mean and variance learning ..... 48
D. 3 Sampling ODE and Generated Distribution ..... 49
E Detailed Derivations for Two-Layer Symmetric Parameterization ..... 51
E. 1 Symmetric parametrization zero mean gradient dynamics ..... 51
E.1.1 Simplifying assumption: orthogonal initialization $q_{k}^{T} q_{m}=0$ ..... 52
E.1.2 Beyond Aligned Initialization : qualitative analysis of off diagonal dynamics ..... 55
E. 2 Sampling ODE and Generated Distribution ..... 56
F Detailed Derivations for Deep Linear network ..... 59
F. 1 Aligned assumption ..... 59
F. 2 Two layer linear network ..... 60
F.2.1 Special case: homogeneous initialization $\Lambda_{1}=\Lambda_{2}$ (symmetric two layer case) ..... 61
F.2.2 General Case: general initialization (general two layer $\mathbf{W}=P Q$ ) ..... 61
F. 3 General deep linear network ..... 62
F.3.1 Deep Residual network ..... 63
G Detailed Derivations for Linear Convolutional Network ..... 65
G. 1 General set up ..... 65
G. 2 General analysis of sampling dynamics ..... 65
G. 3 Full width linear convolutional network ..... 66
G.3.1 Training dynamics of full width linear convolutional network ..... 66
G.3.2 Sampling dynamics of full width linear convolutional network ..... 69
G. 4 Local patch linear convolutional network ..... 72
G.4.1 Training dynamics of patch linear convolutional net ..... 72
G.4.2 Sampling dynamics of patch linear convolutional net ..... 76
G. 5 Appendix: Useful math ..... 76
H Detailed derivation of Flow Matching model ..... 79
H. 1 Solution to the flow matching sampling ODE with optimal solution ..... 80
H. 2 Learning dynamics of flow matching objective (single layer) ..... 81
H.2.1 Interaction of weight learning and flow sampling ..... 82
H. 3 Learning dynamics of flow matching objective (two-layers) ..... 83
I Detailed Experimental Procedure ..... 86
I. 1 Computational Resources ..... 86
I. 2 MLP architecture inspired by UNet ..... 86
I. 3 EDM Loss Function ..... 87
I. 4 Experiment 1: Diffusion Learning of High-dimensional Gaussian Data ..... 89
I.4.1 Data Generation and Covariance Specification ..... 89
I.4.2 Network Architecture and Training Setup ..... 89
I.4.3 Sampling and Trajectory Visualization ..... 89
I.4.4 Covariance Evaluation in the True Eigenbasis ..... 89
I. 5 Experiment 2: Diffusion Learning of MNIST I MLP ..... 90
I.5.1 Data Preprocessing ..... 90
I.5.2 Network Architecture and Training Setup ..... 90
I.5.3 Sampling and Analysis ..... 90
I. 6 Experiment 3: Diffusion learning of Image Datasets with EDM-style CNN UNet ..... 90

\section*{A Extended Related works}

Beyond the closely related works reviewed in the main text, here we are some spiritually related lines of works that inspired ours.

Spectral effect in the sampling process of diffusion models Many works have observed that during the sampling process of diffusion models [22, 38]: low spatial frequency aspects of the sample (e.g. layout) were specified first in the denoiser, before the higher frequency ones (e.g. textures). This phenomenon has been understood through the natural statistics of images (e.g. power-law spectrum) [19] and theory of diffusion [11], and recently through the lens of stochastic localization [39]. Basically, low frequency aspects usually have higher variance, thus were later to be corrupted by noise, so earlier to be generated during sampling process.

In our current work, we extend this line of thought to consider the spectral effects on the training dynamics of diffusion models.

Inductive bias of deep networks There as been a rich history of studying the inductive bias or implicit regularization effect of deep neural network and gradient descent. Deep neural networks have been reported to tend to find low-rank solutions of the task [40], and deeper networks could find it difficult to learn higher-rank target functions. This finding has also been leveraged to facilitate low-rank solutions by over-parameterizing linear operations in a deep networks (e.g. linear [41] or convolution [42] layers).

Implicit bias of convolutional neural networks When the neural network has convolutional structures in it, what kind of inductive bias or regularization effect does it bring to the function approximator?

People have attacked this by analyzing (deep) linear networks. [37] analyzed the inductive bias of gradient learning of the linear convolution network. In their case, the kernel is as wide as the signal and with circular boundary condition, thus convolution is equivalent to pointwise multiplication in Fourier space, which simplified the problem a lot. Then they can derive the learning dynamics of each Fourier mode. This result can be unified with other linear network approaches [18].
[43] further analyzed the inductive bias of the linear convolutional network with non-trivial local kernel size (neither pointwise nor full image) and multiple channels, and provided analytical statements about the inductive bias. However, they also found less success for closed form solutions for even two-layer convolutional networks with finite kernel width.

From an algebraic and geometric perspective, [44, 45] have analyzed the geometry of the function space of the deep linear convolutional network, which is equivalent to the space of polynomials that can be factorized into shorter polynomials.

Deep image prior and spectral bias in CNN On the empirical side, one intriguing result comes from the famous Deep Image Prior (DIP) experiment of Ulyanov et al. [46]. They showed that if a deep convolutional network (e.g. UNet) is used to regress a noisy image as target with pure noise as input, then when we employ early stopping in the optimization, the neural network will produce a cleaner image, thus performing denoising. To understand this method, [47] showed empirical evidence that deep convolutional networks tend to fit the low spatial frequency aspect of the data first. Thus, given the different spectral signature of natural images and noise, networks will fit the natural image before the noise. As a corollary, they showed that if the noise has more low frequency components, then neural network will fail to denoise those low frequency corruptions from image.

People have also looked at the inductive bias of untrained convolutional neural networks. Theoretically, [48] showed that infinite-width convolutional network at initialization is equivalent to spatial Gaussian process (random field), and the authors used this Bayesian perspective to understand the Deep image prior.

We noticed that this line of works in deep image prior has intriguing conceptual connection to our current work, i.e. the spectral bias of learning a function with convolutional architecture tend to learn lower frequency aspect first. Comparing diffusion models to DIP, diffusion models regress clean images from many randomly sampled noisy images; on the contrary DIP regress the clean images on a single noise pattern.

Neural Tangent Kernel A widely recognized technique for analyzing the learning dynamics of deep neural network is the neural tangent kernel. For example, an infinitely wide network would be similar to a kernel machine, where the learning dynamics will be linearized and reduce to exponential convergence along different eigenmode of the tangent kernel.

Using neural tangent kernel (NTK) techniques, by inspecting the eigenvalues of the NTK associated with functions of different frequency, [49] has been able to show that given uniform data on sphere assumption, and simple neural network architectures (two-layer fully connected network with ReLU nonlinearity), neural networks learn lower-frequency functions faster, with learning speed quadratically related to the frequency. Later they lifted the spatial uniformity assumption [50], and derived how convergence speed and eigenvalues depend on the local data density.
These insights have been leveraged in classification problems to show that early stopping can lead neural networks to learn smoother functions, thus being robust to labeling noise [51].
What about convolutional architecture? With some similar NTK techniques, using a simplified architecture, [52] proved that the learning dynamics of the convolutional network will preferably learn the lower spatial frequency aspect of target image first. Their proof technique is also based on the relationship between over-parametrized neural network and the tangent kernel. The proof is based on a simpler generator architecture: one convolutional layer with ReLU nonlinearity and fixed readout vector. They numerically showed the same effect for deeper architectures. This result provided further theoretical foundation for the Deep Image Prior.

\section*{B Extended Results}

\section*{B. 1 Extended Visualization of Theoretical Results}

\section*{B.1.1 Scaling curves of diffusion training (EDM)}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/11e476a8-86be-4d33-b2f0-7738bec1c9ed-19.jpg?height=1034&width=1111&top_left_y=468&top_left_x=507}
\captionsetup{labelformat=empty}
\caption{Figure 5: Learning dynamics of the weight and variance of the generated distribution per eigenmode (continued) Top Single layer linear denoiser. Bottom Symmetric two-layer denoiser. A.C. Learning dynamics of $\mathbf{u}_{k}^{\top} \mathbf{W}(\tau) \mathbf{u}_{k}$. B.D. Learning dynamics of the variance of the generated distribution $\tilde{\lambda}_{k}$, as a function of the variance of the target eigenmode $\lambda_{k}$. This case with larger amplitude weight initialization $Q_{k}=0.5$.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/11e476a8-86be-4d33-b2f0-7738bec1c9ed-20.jpg?height=1093&width=1393&top_left_y=603&top_left_x=367}
\captionsetup{labelformat=empty}
\caption{Figure 6: Power law relationship between mode emergence time and target mode variance for one-layer and two-layer linear denoisers. Panels (A) and (B) respectively plot the Mode variance against the Emergence Step for different values of weight initialization $Q_{k} \in \{0.0,0.1,0.5,0.6,1.0\}$ (columns), for one layer and two layer linear denoser (rows). We used $\sigma_{0}=0.002$ and $\sigma_{T}=80$. The emergence steps were quantified via different criterions, via harmonic mean in $\mathbf{A}$, and geometric mean in $\mathbf{B}$. Within each panel, red markers and lines denote the modes where their variance increases; blue markers and lines denote modes that "decrease" their variance. The solid lines show least-squares fits on log-log scale, giving rise to the $y=a x^{b}$ type relation. Comparisons reveal a systematic power-law decay of variance with respect to the Emergence Step under both the harmonic-mean and geometric-mean definitions. Note, the $Q_{k}=0$ and two layer case was empty since zero initialization is an (unstable) fixed point, thus it will not converge.}
\end{figure}

\section*{B.1.2 Scaling curves of flow matching}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/11e476a8-86be-4d33-b2f0-7738bec1c9ed-21.jpg?height=312&width=1393&top_left_y=322&top_left_x=365}
\captionsetup{labelformat=empty}
\caption{Figure 7: Learning dynamics of the weight and variance of the generated distribution per eigenmode, for one layer linear flow matching model Similar plotting format as Fig. 2. A. Learning dynamics of weights $\mathbf{u}_{k}^{\top} \mathbf{W}(\tau ; t) \mathbf{u}_{k}$ for various time point $t \in\{0.1,0.5,0.9,0.99\}$. B. Learning dynamics of the variance of the generated distribution $\tilde{\lambda}_{k}$, as a function of the variance of the target eigenmode $\lambda_{k} \in\{10,3,1,0.3,0.1,0.03,0.01,0.001\}$. Weight initialization is set at $Q_{k}=0.1$ for every mode.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/11e476a8-86be-4d33-b2f0-7738bec1c9ed-21.jpg?height=596&width=1401&top_left_y=971&top_left_x=363}
\captionsetup{labelformat=empty}
\caption{Figure 8: Power law relationship between mode emergence time and target mode variance for one-layer linear flow matching. Panels (A) and (B) respectively plot the Mode variance against the Emergence Step for different values of weight initialization $Q_{k} \in\{0.0,0.1,0.5,0.6,1.0\}$ (columns), for one layer linear flow model. The emergence steps were quantified via different criterions, via harmonic mean in $\mathbf{A}$, and geometric mean in $\mathbf{B}$. We used the same plotting format as in Fig. 6. Comparisons reveal a systematic power-law decay of variance with respect to the Emergence Step under both the harmonic-mean and geometric-mean definitions.}
\end{figure}

\section*{B. 2 Extended Empirical Results}

\section*{B.2.1 MLP-UNet Gaussian training experiments}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/11e476a8-86be-4d33-b2f0-7738bec1c9ed-22.jpg?height=1234&width=1359&top_left_y=378&top_left_x=382}
\captionsetup{labelformat=empty}
\caption{Figure 9: Spectral Learning Dynamics of MLP-UNet (Gaussian-rotated). (same layout and analysis procedure as main Fig. 3) Top, middle, bottom show cases for 128d, 256d and 512d Gaussian. A. Evolution of sample variance $\tilde{\lambda}_{k}(\tau)$ across eigenmodes during training. B. Heatmap of variance trajectories along all eigenmodes, with dots marking mode emergence times $\tau^{*}$ (first-passage time at the geometric mean of initial and final variances). The gray zone ( $0.5-2 \times$ target variance) indicates modes starting too close to their target, causing unreliable $\tau^{*}$ estimates. C. Power-law scaling of $\tau^{*}$ versus target variance $\lambda_{k}$, excluding gray-zone eigenmodes for stability.}
\end{figure}

\section*{B.2.2 MLP-UNet natural image training experiments}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/11e476a8-86be-4d33-b2f0-7738bec1c9ed-23.jpg?height=695&width=1316&top_left_y=945&top_left_x=399}
\captionsetup{labelformat=empty}
\caption{Figure 10: Dynamical alignment onto the covariance eigenframe of data (MLP-UNet, FFHQ32, AFHQ32). Alignment score $\chi$ as function of training step. Alignment score defined as the sum of square of diagonal entries of the rotated sample covariance on the training data eigenframe $U^{T} \tilde{\Sigma}_{\tau} U$, divided by the sum of square of all entries. This quantifies how well the training data eigenframe diagonalizes the generated sample covariance. It will be $\chi=1$ if $U$ is the eigenbasis of $\tilde{\Sigma}_{\tau}$.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/11e476a8-86be-4d33-b2f0-7738bec1c9ed-24.jpg?height=1939&width=1355&top_left_y=259&top_left_x=384}
\captionsetup{labelformat=empty}
\caption{Figure 11: Spectral Learning Dynamics of MLP-UNet (MNIST, CIFAR10, AFHQ32, FFHQ32fixword, random word). A. Generated samples during training. B. Evolution of sample variance $\tilde{\lambda}_{k}(\tau)$ across eigenmodes during training. C. Heatmap of variance trajectories along all eigenmodes, with dots marking mode emergence times $\tau^{*}$ (first-passage time at the geometric mean of initial and final variances). The gray zone ( $0.5-2 \times$ target variance) indicates modes starting too close to their target, causing unreliable $\tau^{*}$ estimates. D. Power-law scaling of $\tau^{*}$ versus target variance $\lambda_{k}$. A separate law was fit for modes with increasing and decreasing variance, excluding the middle gray-zone eigenmodes for stability.}
\end{figure}

\section*{B.2.3 CNN-UNet training experiment}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/11e476a8-86be-4d33-b2f0-7738bec1c9ed-25.jpg?height=416&width=1359&top_left_y=300&top_left_x=384}
\captionsetup{labelformat=empty}
\caption{Figure 12: Spectral Learning Dynamics of CNN-UNet (FFHQ32). (same layout and analysis procedure as main Fig. 3) A. Generated samples during training. B. Evolution of sample variance $\tilde{\lambda}_{k}(\tau)$ across eigenmodes during training. C. Heatmap of variance trajectories along all eigenmodes, with dots marking mode emergence times $\tau^{*}$ (first-passage time at the geometric mean of initial and final variances). The gray zone ( $0.5-2 \times$ target variance) indicates modes starting too close to their target, causing unreliable $\tau^{*}$ estimates. D. Power-law scaling of $\tau^{*}$ versus target variance $\lambda_{k}$, excluding gray-zone eigenmodes for stability.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/11e476a8-86be-4d33-b2f0-7738bec1c9ed-25.jpg?height=710&width=813&top_left_y=1091&top_left_x=655}
\captionsetup{labelformat=empty}
\caption{Figure 13: Dynamical alignment onto the covariance eigenframe of data (CNN-UNet, FFHQ32). Alignment score $r$ as function of training step. Same analysis as Fig 10.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/11e476a8-86be-4d33-b2f0-7738bec1c9ed-26.jpg?height=1299&width=1387&top_left_y=605&top_left_x=369}
\captionsetup{labelformat=empty}
\caption{Figure 14: UNet denoiser can be approximated by linear convolution early in training (CNNUNet, FFHQ32). A. Early in training, the UNet denoiser output can be well approximated by a linear convolutional layer, with a patch size $P$. B. The approximation error as a function of patch size $P$, training time $\tau$ and noise scale $\sigma$. Generally, early in training, the denoiser is very local and linear, well approximated by a linear convolutional layer.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/11e476a8-86be-4d33-b2f0-7738bec1c9ed-27.jpg?height=744&width=1401&top_left_y=251&top_left_x=363}
\captionsetup{labelformat=empty}
\caption{Figure 15: Visualizing the denoiser training dynamics with a fixed image and noise seed (CNNUNet, FFHQ32). $\mathbf{D}(\mathbf{x}+\sigma \mathbf{z}, \sigma)$ as a function of training time $\tau$ and noise scale $\sigma$.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/11e476a8-86be-4d33-b2f0-7738bec1c9ed-27.jpg?height=1236&width=1049&top_left_y=1151&top_left_x=539}
\captionsetup{labelformat=empty}
\caption{Figure 16: Spectral Bias in Whole Image of CNN learning I MNIST Training dynamics of sample (whole image) variance along eigenbasis of training set, normalized by target variance. Upper 0-100 eigen modes, Lower 0-500 eigenmodes.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/11e476a8-86be-4d33-b2f0-7738bec1c9ed-28.jpg?height=1417&width=1255&top_left_y=573&top_left_x=436}
\captionsetup{labelformat=empty}
\caption{Figure 17: Spectral Bias in CNN-Based Diffusion Learning: Variance Dynamics in Image Patches I MNIST ( 32 pixel resolution). Left, Raw variance of generated patches along true eigenbases during training. Right, Scaling relationship between the target variance of eigenmode versus mode emergence time (harmonic mean criterion). Each row corresponds to a different patch size and stride used for extracting patches from images.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/11e476a8-86be-4d33-b2f0-7738bec1c9ed-29.jpg?height=1411&width=1256&top_left_y=577&top_left_x=431}
\captionsetup{labelformat=empty}
\caption{Figure 18: Spectral Bias in CNN-Based Diffusion Learning: Variance Dynamics in Image Patches I CIFAR10 ( 32 pixel resolution). Left, Raw variance of generated patches along true eigenbases during training. Right, Scaling relationship between the target variance of eigenmode versus mode emergence time (harmonic mean criterion). Each row corresponds to a different patch size and stride used for extracting patches from images.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/11e476a8-86be-4d33-b2f0-7738bec1c9ed-30.jpg?height=1401&width=1253&top_left_y=580&top_left_x=436}
\captionsetup{labelformat=empty}
\caption{Figure 19: Spectral Bias in CNN-Based Diffusion Learning: Variance Dynamics in Image Patches I FFHQ ( $\mathbf{6 4}$ pixel resolution). Left, Raw variance of generated patches along true eigenbases during training. Right, Scaling relationship between the target variance of eigenmode versus mode emergence time (harmonic mean criterion). Each row corresponds to a different patch size and stride used for extracting patches from images.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/11e476a8-86be-4d33-b2f0-7738bec1c9ed-31.jpg?height=1399&width=1253&top_left_y=580&top_left_x=436}
\captionsetup{labelformat=empty}
\caption{Figure 20: Spectral Bias in CNN-Based Diffusion Learning: Variance Dynamics in Image Patches I AFHQv2 ( $\mathbf{6 4}$ pixel resolution). Left, Raw variance of generated patches along true eigenbases during training. Right, Scaling relationship between the target variance of eigenmode versus mode emergence time (harmonic mean criterion). Each row corresponds to a different patch size and stride used for extracting patches from images.}
\end{figure}

\section*{C Detailed Derivation: General analysis}

\section*{C. 1 General Property of Linear Regression}

\section*{C.1.1 Gaussian equivalence}

Lemma C.1. For a general linear regression problem, where $\mathbf{x}, \mathbf{y}$ come from an arbitrary joint distribution $p(\mathbf{x}, \mathbf{y})$ with finite moments,
$$
\mathcal{L}=\mathbb{E}_{\mathbf{x}, \mathbf{y}}\|\mathbf{W} \mathbf{x}+\mathbf{b}-\mathbf{y}\|^{2}
$$
then its optimal solution and gradient only depend the first two moments of $\mathbf{x}, \mathbf{y}$,
Proof. Let the error be $\mathbf{e}=\mathbf{W} \mathbf{x}+\mathbf{b}-\mathbf{y}$, then
$$
\begin{aligned}
\nabla_{\mathbf{W}} \mathcal{L} & =\frac{\partial}{\partial \mathbf{W}} \mathbb{E}_{\mathbf{x}, \mathbf{y}}\left[\mathbf{e}^{\top} \mathbf{e}\right] \\
& =2 \mathbb{E}\left[\mathbf{e} \mathbf{x}^{\top}\right] \\
& =2 \mathbb{E}\left[(\mathbf{W} \mathbf{x}+\mathbf{b}-\mathbf{y}) \mathbf{x}^{\top}\right] \\
& =2\left(\mathbf{W} \mathbb{E}\left[\mathbf{x} \mathbf{x}^{\top}\right]+\mathbf{b} \mathbb{E}\left[\mathbf{x}^{\top}\right]-\mathbb{E}\left[\mathbf{y} \mathbf{x}^{\top}\right]\right) \\
& =2\left(\mathbf{W}\left(\Sigma_{x x}+\mu_{x} \mu_{x}^{T}\right)+\mathbf{b} \mu_{x}^{T}-\left(\Sigma_{y x}+\mu_{y} \mu_{x}^{\top}\right)\right) \\
& \left.=2 \mathbf{W}\left(\Sigma_{x x}+\mu_{x} \mu_{x}^{T}\right)+2\left(\mathbf{b}-\mu_{y}\right) \mu_{x}^{T}-2 \Sigma_{y x}\right) \\
& =2\left(\mathbf{W} \Sigma_{x x}-\Sigma_{y x}\right)+2\left(\mathbf{W} \mu_{x}+\mathbf{b}-\mu_{y}\right) \mu_{x}^{T} \\
& =2\left(\mathbf{W} \Sigma_{x x}-\Sigma_{y x}\right)+\nabla_{\mathbf{b}} \mathcal{L} \mu_{x}^{T} \\
& \\
\nabla_{\mathbf{b}} \mathcal{L} & =\frac{\partial}{\partial \mathbf{b}} \mathbb{E}\left[\mathbf{e}^{\top} \mathbf{e}\right] \\
& =2 \mathbb{E}[\mathbf{e}] \\
& =2 \mathbb{E}[\mathbf{W} \mathbf{x}+\mathbf{b}-\mathbf{y}] \\
& =2\left(\mathbf{W} \mu_{x}+\mathbf{b}-\mu_{y}\right)
\end{aligned}
$$

We used the fact that
$$
\begin{aligned}
\mathbb{E}[\mathbf{x}] & =\mu_{x} \\
\mathbb{E}[\mathbf{y}] & =\mu_{y} \\
\mathbb{E}\left[\mathbf{y} \mathbf{x}^{\top}\right] & =\Sigma_{y x}+\mu_{y} \mu_{x}^{\top} \\
\mathbb{E}\left[\mathbf{x} \mathbf{x}^{\top}\right] & =\Sigma_{x x}+\mu_{x} \mu_{x}^{\top}
\end{aligned}
$$

Setting gradient to zero, we get optimal values
$$
\begin{align*}
\mathbf{W}^{*} & =\Sigma_{y x} \Sigma_{x x}^{-1}  \tag{11}\\
\mathbf{b}^{*} & =\mu_{y}-\mathbf{W}^{*} \mu_{x} \tag{12}
\end{align*}
$$

The gradient flow dynamics read
$$
\begin{align*}
\frac{d}{d \tau} \mathbf{W} & =-2 \eta\left(\mathbf{W} \Sigma_{x x}-\Sigma_{y x}\right)-2 \eta \nabla_{\mathbf{b}} \mathcal{L} \mu_{x}^{T}  \tag{13}\\
\frac{d}{d \tau} \mathbf{b} & =-2 \eta\left(\mathbf{W} \mu_{x}+\mathbf{b}-\mu_{y}\right) \tag{14}
\end{align*}
$$

The $\Sigma_{x x}$ determines the gradient flow dynamics and convergence rate of $\mathbf{W}$, while $\Sigma_{x x}^{-1} \Sigma_{y x}$ determines the target level or optimal solution of the regression.

\section*{I Detailed Experimental Procedure}

\section*{I. 1 Computational Resources}

All experiments were conducted on research cluster. Model training was performed on single A100 / H100 GPU. MLP training experiments took $20 \mathrm{mins}-2 \mathrm{hrs}$ while CNN based UNet training experiments took 5-8 hours, using around 20 GB RAM.

Evaluations were also done on single A100 / H100 GPU, with heavy covariance computation done with CUDA and trajectory plotting and fitting on CPU. Covariance computation for generated samples generally took a few minutes.

\section*{I. 2 MLP architecture inspired by UNet}

We used the following custom architecture inspired by UNet in [26] and [59] paper. The basic block is the following
```
class UNetMLPBlock(torch.nn.Module):
    def __init__(self,
        in_features, out_features, emb_features, dropout=0, skip_scale=1, eps=1e-5,
        adaptive_scale=True, init=dict(), init_zero=dict(),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.emb_features = emb_features
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale
        self.norm0 = nn.LayerNorm(in_features, eps=eps)
            #GroupNorm(num_channels=in_features, eps=eps)
        self.fc0 = Linear(in_features=in_features, out_features=out_features, **init)
        self.affine = Linear(in_features=emb_features, out_features=out_features*(2
            if adaptive_scale else 1), **init)
        self.norm1 = nn.LayerNorm(out_features, eps=eps)
            #GroupNorm(num_channels=out_features, eps=eps)
        self.fc1 = Linear(in_features=out_features, out_features=out_features,
            **init_zero)
        self.skip = None
        if out_features != in_features:
            self.skip = Linear(in_features=in_features, out_features=out_features,
                **init)
    def forward(self, x, emb):
        orig = x
        x = self.fc0(F.silu(self.norm0(x)))
        params = self.affine(emb).to(x.dtype) # .unsqueeze(1)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = F.silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = F.silu(self.norm1(x.add_(params)))
        x = self.fc1(F.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale
        return x
```

and the full architecture backbone
```
class UNetBlockStyleMLP_backbone(nn.Module):
    """A time-dependent score-based model."""
    def __init__(self, ndim=2, nlayers=5, nhidden=64, time_embed_dim=64,):
        super().__init__()
        self.embed = GaussianFourierProjection(time_embed_dim, scale=1)
        layers = nn.ModuleList()
        layers.append(UNetMLPBlock(ndim, nhidden, time_embed_dim))
        for _ in range(nlayers-2):
            layers.append(UNetMLPBlock(nhidden, nhidden, time_embed_dim))
        layers.append(nn.Linear(nhidden, ndim))
        self.net = layers
    def forward(self, x, t_enc, cond=None):
        # t_enc : preconditioned version of sigma, usually
        # ln_std_vec = torch.log(std_vec) / 4
        if cond is not None:
            raise NotImplementedError("Conditional training is not implemented")
        t_embed = self.embed(t_enc)
        for layer in self.net[:-1]:
            x = layer(x, t_embed)
        pred = self.net[-1](x)
        return pred
```

```
class EDMPrecondWrapper(nn.Module):
    def __init__(self, model, sigma_data=0.5, sigma_min=0.002, sigma_max=80,
            rho=7.0):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
    def forward(self, X, sigma, cond=None, ):
        sigma[sigma == 0] = self.sigma_min
        ## edm preconditioning for input and output
        ## https://github.com/NVlabs/edm/blob/main/training/networks.py#L632
        # unsqueze sigma to have same dimension as X (which may have 2-4 dim)
        sigma_vec = sigma.view([-1, ] + [1, ] * (X.ndim - 1))
        c_skip = self.sigma_data ** 2 / (sigma_vec ** 2 + self.sigma_data ** 2)
        c_out = sigma_vec * self.sigma_data / (sigma_vec ** 2 + self.sigma_data **
            2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma_vec ** 2).sqrt()
        c_noise = sigma.log() / 4
        model_out = self.model(c_in * X, c_noise, cond=cond)
        return c_skip * X + c_out * model_out
```


This architecture can efficiently learn point cloud distributions. More details about the architecture and training can be found in code supplementary.

\section*{I. 3 EDM Loss Function}

We employ the loss function $\mathcal{L}_{\text {EDM }}$ introduced in the Elucidated Diffusion Model (EDM) paper [26], which is one specific weighting scheme for training diffusion models.
For each data point $\mathbf{x} \in \mathbb{R}^{d}$, the loss is computed as follows. The noise level for each data point is sampled from a log-normal distribution with hyperparameters $P_{\text {mean }}$ and $P_{\text {std }}$ (e.g., $P_{\text {mean }}=-1.2$ and

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/11e476a8-86be-4d33-b2f0-7738bec1c9ed-88.jpg?height=732&width=749&top_left_y=242&top_left_x=689}
\captionsetup{labelformat=empty}
\caption{Figure 23: Example of learning to generate low-dimensional manifold with Song UNet-inspired MLP denoiser.}
\end{figure}
$P_{\text {std }}=1.2$ ). Specifically, the noise level $\sigma$ is sampled via
$$
\sigma=\exp \left(P_{\text {mean }}+P_{\text {std }} \epsilon\right), \quad \epsilon \sim \mathcal{N}(0,1) .
$$

The weighting function per noise scale is defined as:
$$
w(\sigma)=\frac{\sigma^{2}+\sigma_{\mathrm{data}}^{2}}{\left(\sigma \sigma_{\mathrm{data}}\right)^{2}},
$$
with hyperparameter $\sigma_{\text {data }}$ (e.g., $\sigma_{\text {data }}=0.5$ ). The noisy input $\mathbf{y}$ is created by the following,
$$
\mathbf{y}=\mathbf{x}+\sigma \mathbf{n}, \quad \mathbf{n} \sim \mathcal{N}\left(\mathbf{0}, \mathbf{I}_{d}\right),
$$

Let $D_{\theta}(\mathbf{y}, \sigma$, labels $)$ denote the output of the denoising network when given the noisy input $\mathbf{y}$, the noise level $\sigma$, and optional conditioning labels. The EDM loss per data point can be computed as:
$$
\mathcal{L}(\mathbf{x})=w(\sigma) \| D_{\theta}(\mathbf{x}+\sigma \mathbf{n}, \sigma, \text { labels })-\mathbf{x} \|^{2} .
$$

Taking expectation over the data points and noise scales, the overall loss reads
$$
\begin{equation*}
\mathcal{L}_{E D M}=\mathbb{E}_{\mathbf{x} \sim p_{\text {data }}} \mathbb{E}_{\mathbf{n} \sim \mathcal{N}\left(0, \mathbf{I}_{d}\right)} \mathbb{E}_{\sigma}\left[w(\sigma) \| D_{\theta}(\mathbf{x}+\sigma \mathbf{n}, \sigma, \text { labels })-\mathbf{x} \|^{2}\right] \tag{379}
\end{equation*}
$$
```
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
    def __call__(self, net, X, labels=None, ):
        rnd_normal = torch.randn([X.shape[0],] + [1, ] * (X.ndim - 1),
            device=X.device)
        # unsqueeze to match the ndim of X
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        # maybe augment
        n = torch.randn_like(X) * sigma
        D_yn = net(X + n, sigma, cond=labels, )
        loss = weight * ((D_yn - X) ** 2)
        return loss
```


\section*{I. 4 Experiment 1: Diffusion Learning of High-dimensional Gaussian Data}

\section*{I.4.1 Data Generation and Covariance Specification}

We consider learning a score-based generative model on synthetic data drawn from a high-dimensional Gaussian distribution of dimension $d=128,256,512$. Specifically, we first sample a vector of variances
$$
\boldsymbol{\sigma}^{2}=\left(\sigma_{1}^{2}, \sigma_{2}^{2}, \ldots, \sigma_{d}^{2}\right)
$$
where each $\sigma_{i}^{2}$ is drawn from a log-normal distribution (implemented via torch.exp(torch.randn(...))). We then sort them in descending order and normalize these variances to have mean equals 1 to fix the overall scale. Denoting
$$
\mathbf{D}=\operatorname{diag}\left(\sigma_{1}^{2}, \ldots, \sigma_{d}^{2}\right)
$$
we generate a random rotation matrix $\mathbf{R} \in \mathbb{R}^{d \times d}$ by performing a QR decomposition of a matrix of i.i.d. Gaussian entries. This allows us to construct the covariance
$$
\boldsymbol{\Sigma}=\mathbf{R} \mathbf{D} \mathbf{R}^{\top} .
$$

This rotation matrix $\mathbf{R}$ is the eigenbasis of the true covariance matrix. To obtain training samples $\left\{\mathbf{x}_{i}\right\} \subset \mathbb{R}^{d}$, we draw $\mathbf{x}_{i}$ from $\mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$. In practice, we generate a total of 10,000 samples and stack them as pnts. We compute the empirical covariance of the training set, $\boldsymbol{\Sigma}_{\mathrm{emp}}=\operatorname{Cov}(\mathrm{pnts})$, and verify that it is close to the prescribed true covariance $\boldsymbol{\Sigma}$.

\section*{I.4.2 Network Architecture and Training Setup}

We train a multi-layer perceptron (MLP) to approximate the noise conditional score function. The base network, implemented as
model=UNetBlockStyleMLP_backbone(ndim=d, nlayers=5, nhidden=256, time_embed_dim=256)
maps a data vector $\mathbf{x} \in \mathbb{R}^{d}$ and a time embedding $\tau$ to a vector of the same dimension $\mathbb{R}^{d}$. This backbone is then wrapped in an EDM-style preconditioner via:
$$
\text { model_precd }=\text { EDMPrecondWrapper }\left(\operatorname{model}, \sigma_{\mathrm{data}}=0.5, \sigma_{\min }=0.002, \sigma_{\max }=80, \rho=7.0\right),
$$
which standardizes and scales the input according to the EDM framework [26].
We use EDM loss with hyperparameters P_mean $=-1.2, \mathrm{P} \_\mathrm{std}=1.2$, and $\sigma_{\mathrm{data}}=0.5$. We train the model for 5000 steps using mini-batches of size 1024. The Adam optimizer is used with a learning rate $\operatorname{lr}=10^{-4}$. Each training step processes a batch of data from pnts, adds noise with randomized noise scales, and backpropagates through the EDM loss. The loss values at each training steps are recorded.

\section*{I.4.3 Sampling and Trajectory Visualization}

To visualize the sampling evolution, we sample from the diffusion model using the Heun's 2nd order deterministic sampler, starting from $\mathbf{z} \sim \mathcal{N}\left(\mathbf{0}, \mathbf{I}_{d}\right)$
$$
\text { edm_sampler(model, } \left.\mathbf{z}, \text { num_steps }=20, \sigma_{\min }=0.002, \sigma_{\max }=80, \rho=7\right) .
$$

We store these samples in sample_store to track how sampled distribution evolves over training.

\section*{I.4.4 Covariance Evaluation in the True Eigenbasis}

To measure how well the trained model captures the true covariance structure, we compute the sample covariance from the final generated samples, denoted $\widehat{\boldsymbol{\Sigma}}_{\text {sample }}$. We then project $\widehat{\boldsymbol{\Sigma}}_{\text {sample }}$ and the true $\boldsymbol{\Sigma}$ into the eigenbasis of $\boldsymbol{\Sigma}$. Specifically, letting $\mathbf{R}$ be the rotation used above, we compute
$$
\mathbf{R}^{\top} \widehat{\Sigma}_{\text {sample }} \mathbf{R} \quad \text { and } \quad \mathbf{R}^{\top} \boldsymbol{\Sigma} \mathbf{R} .
$$

Since $\boldsymbol{\Sigma}=\mathbf{R} \mathbf{D} \mathbf{R}^{\top}$ is diagonal in that basis, we then compare the diagonal elements of $\mathbf{R}^{\top} \widehat{\boldsymbol{\Sigma}}_{\text {sample }} \mathbf{R}$ with $\operatorname{diag}(\mathbf{D})$. As training proceeds, we track the ratio $\operatorname{diag}\left(\mathbf{R}^{\top} \widehat{\boldsymbol{\Sigma}}_{\text {sample }} \mathbf{R}\right) / \operatorname{diag}(\mathbf{D})$ to observe convergence toward 1 across the spectrum.
All intermediate results, including loss values and sampled trajectories, are stored to disk for later analysis.

\section*{I. 5 Experiment 2: Diffusion Learning of MNIST I MLP}

\section*{I.5.1 Data Preprocessing}

For our second experiment, we apply the same EDM architecture to several natural image datasets: MNIST, CIFAR, AFHQ32, FFHQ32, FFHQ32-fixword, FFHQ32-randomword. All dataset except for MNIST are RGB images with 32 resolution, while MNIST is BW images with 28 resolution. These images were flattened as vectors, (784d for MNIST, 3072 for others) and stacked as pnts matrix. We normalize these intensities from $[0,1]$ to $[-1,1]$ by $\mathbf{x} \mapsto \frac{\mathbf{x}-0.5}{0.5}$. The resulting data tensor pnts is then transferred to GPU memory for training, and we estimate its empirical covariance $\boldsymbol{\Sigma}_{\text {emp }}=\operatorname{Cov}($ pnts $)$ for reference.

\section*{I.5.2 Network Architecture and Training Setup}

Since the natural dataset is higher dimensional than the synthetic data in the previous experiment, we use a deeper MLP network: For MNIST:
model $=$ UNetBlockStyleMLP_backbone(ndim $=784$, nlayers $=8$, nhidden $=1024$, time_embed_dim $=128$ ).
For others
model $=$ UNetBlockStyleMLP_backbone(ndim $=3072$, nlayers $=8$, nhidden $=3072$, time_embed_dim $=128$ ).
We again wrap this MLP in an EDM preconditioner:
$$
\text { model_precd }=\text { EDMPrecondWrapper }\left(\operatorname{model}, \sigma_{\mathrm{data}}=0.5, \sigma_{\min }=0.002, \sigma_{\max }=80, \rho=7.0\right) .
$$

The model is trained using the EDMLoss described in the previous section, with parameters P_mean $= -1.2, \mathrm{P} \_\mathrm{std}=1.2$, and $\sigma_{\text {data }}=0.5$. We set the training hyperparameters to $\mathrm{lr}=10^{-4}$, n_steps $=$ 100000 , and batch_size $=2048$.

\section*{I.5.3 Sampling and Analysis}

As before, we define a callback function sampling_callback_fn that periodically draws i.i.d. Gaussian noise $\mathbf{z} \sim \mathcal{N}\left(\mathbf{0}, \mathbf{I}_{784}\right)$ and applies the EDM sampler to produce generated samples. These intermediate samples are stored in sample_store for later analysis.
In addition, we assess convergence of the mean of the generated samples by computing
$$
\| \mathbb{E}[\text { x_out }]-\mathbb{E}[\text { pnts }] \|^{2},
$$
and we track how this mean-squared error evolves over training steps. We also examine the sample covariance $\widehat{\boldsymbol{\Sigma}}_{\text {sample }}$ of the final outputs, comparing its diagonal in a given eigenbasis to a target spectrum (e.g. the diagonal variances of the training data or a reference covariance).

All trajectories and intermediate statistics are saved to disk for further inspection. In particular, we plot the difference between $\widehat{\boldsymbol{\Sigma}}_{\text {sample }}$ and $\boldsymbol{\Sigma}$ in an eigenbasis to illustrate whether the learned samples capture the underlying covariance structure of the training data.

\section*{I. 6 Experiment 3: Diffusion learning of Image Datasets with EDM-style CNN UNet}

We used model configuration similar to https://github.com/NVlabs/edm but with simplified training code more similar to previous experiments.
For the MNIST dataset, we trained a UNet-based CNN (with four blocks, each containing one layer, no attention, and channel multipliers of $1,2,3$, and 4 ) on MNIST for 50,000 steps using a batch size of 2,048, a learning rate of $10^{-4}, 16$ base model channels, and an evaluation sample size of 5,000.
For the CIFAR-10 dataset, we trained a UNet model (with three blocks, each containing one layer, wide channels of size 128, and attention at resolution 16) for 50,000 steps using a batch size of 512, a learning rate of $10^{-4}$, and an evaluation sample size of 2,000 (evaluated in batches of 1,024 ) with 20 sampling steps.

For the AFHQ, FFHQ (32 pixels) dataset, we used the same UNet architecture and training setup, with four blocks, wide channels of size 128, and attention at resolution 8, trained for 50,000 steps