\title{
The Spacetime of Diffusion Models: An Information Geometry Perspective
}

\section*{A Preprint}

\author{
Rafał Karczewski \\ Aalto University \\ rafal.karczewski@aalto.fi
}

\author{
Markus Heinonen \\ Aalto University \\ markus.o.heinonen@aalto.fi
}

\author{
Alison Pouplin \\ Aalto University \\ alison.pouplin@aalto.fi
}

\author{
Søren Hauberg \\ Technical University of Denmark \\ sohau@dtu.dk
}

\author{
Vikas Garg \\ Aalto University \& YaiYai Ltd \\ vgarg@csail.mit.edu
}

\begin{abstract}
We present a novel geometric perspective on the latent space of diffusion models. We first show that the standard pullback approach, utilizing the deterministic probability flow ODE decoder, is fundamentally flawed. It provably forces geodesics to decode as straight segments in data space, effectively ignoring any intrinsic data geometry beyond the ambient Euclidean space. Complementing this view, diffusion also admits a stochastic decoder via the reverse SDE, which enables an information geometric treatment with the Fisher-Rao metric. However, a choice of $\boldsymbol{x}_{T}$ as the latent representation collapses this metric due to memorylessness. We address this by introducing a latent spacetime $\boldsymbol{z}=\left(\boldsymbol{x}_{t}, t\right)$ that indexes the family of denoising distributions $p\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right)$ across all noise scales, yielding a nontrivial geometric structure. We prove these distributions form an exponential family and derive simulation-free estimators for curve lengths, enabling efficient geodesic computation. The resulting structure induces a principled Diffusion Edit Distance, where geodesics trace minimal sequences of noise and denoise edits between data. We also demonstrate benefits for transition path sampling in molecular systems, including constrained variants such as low-variance transitions and region avoidance. Code is available at: https://github.com/rafalkarczewski/spacetime-geometry.
\end{abstract}

\section*{1 Introduction}

Diffusion models have emerged as a powerful paradigm for generative modeling, demonstrating remarkable success in learning to model and sample data (Yang et al., 2023). While the underlying mathematical frameworks of training and sampling are well-established (Sohl-Dickstein et al., 2015; Kingma et al., 2021; Song et al., 2021; Lu et al., 2022; Holderrieth et al., 2025), analysing how information evolves through the noisy intermediate states $\boldsymbol{x}_{t}$ for $t \in[0, T]$ remains an open question. Our work addresses this by defining and analyzing the geometric structure of diffusion models, which provides a principled framework for understanding their inner workings.

In generative models, a common way to study the intrinsic geometry of the data is to pull back the ambient (Euclidean) metric onto the latent space (Arvanitidis et al., 2018, 2022). Equipped with this pullback metric, shortest paths (i.e., geodesics) in the latent space decode to realistic transitions

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/2cafeaa0-70df-4e35-a8ea-a66a0559b937-01.jpg?height=489&width=762&top_left_y=1860&top_left_x=1121}
\captionsetup{labelformat=empty}
\caption{Figure 1: A geodesic in spacetime is the shortest path between denoising distributions.}
\end{figure}
along data that lie on a lower-dimensional submanifold.

In a diffusion model, a natural choice for the decoder is the reverse ODE $\boldsymbol{x}_{0}\left(\boldsymbol{x}_{T}\right)$, which allows us to derive the pullback geometry of the latents $\boldsymbol{x}_{T}$. Interestingly, we prove that this leads to latent shortest paths always decoding to linear interpolations in data space, which have little practical utility.

We then turn our attention to the decoding distribution $p\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right)$ given by the reverse SDE. We propose an alternative Fisher-Rao geometry, which measures how the denoising distribution $p\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right)$ changes when manipulating the latent $\boldsymbol{x}_{t}$. We introduce the Fisher-Rao metric $\mathbf{G}\left(\boldsymbol{x}_{t}, t\right)$ that varies with both state and time over the latent spacetime $\left(\boldsymbol{x}_{t}, t\right)$ (Fig. 1).

Estimating geodesics in information geometry is usually tractable only for analytic families. Although denoising distributions in diffusion are complex and non-Gaussian, we show that they form an exponential family. This simplifies the geometry and yields a practical method for computing geodesics between any two samples through the spacetime. In the Fisher-Rao setting, curve lengths can be evaluated without running the reverse SDE, which significantly reduces the computational cost.

We demonstrate the utility of the Fisher-Rao geometry in diffusion models in two ways. First, it induces a principled Diffusion Edit Distance on data that admits a clear interpretation: the geodesic between $\boldsymbol{x}^{a}$ and $\boldsymbol{x}^{b}$ traces the minimal sequence of edits, adding just enough noise to forget information specific to $\boldsymbol{x}^{a}$ and then denoising to introduce information specific to $\boldsymbol{x}^{b}$. The resulting length quantifies the total edit cost. Second, spacetime geodesics allow generating transition paths in molecular systems, where we obtain results competitive with specialized state-of-the-art methods and can incorporate constraints such as avoidance of designated regions in data space.

\section*{2 Background on diffusion models}

We assume a data distribution $q$ defined on $\mathbb{R}^{D}$, and the forward process
$$
\begin{equation*}
p\left(\boldsymbol{x}_{t} \mid \boldsymbol{x}_{0}\right)=\mathcal{N}\left(\boldsymbol{x}_{t} \mid \alpha_{t} \boldsymbol{x}_{0}, \sigma_{t}^{2} \boldsymbol{I}\right), \tag{1}
\end{equation*}
$$
which gradually transforms $q$ into pure noise $p_{T} \approx \mathcal{N}\left(\mathbf{0}, \sigma_{T}^{2} \boldsymbol{I}\right)$ at time $T$, where $\alpha_{t}, \sigma_{t}$ define the forward drift $f_{t}$ and diffusion $g_{t}$. There exists a denoising SDE reverse process (Anderson, 1982)
$$
\begin{equation*}
\text { Reverse SDE: } \quad d \boldsymbol{x}=\left(f_{t} \boldsymbol{x}-g_{t}^{2} \nabla \log p_{t}(\boldsymbol{x})\right) d t+g_{t} d \overline{\mathrm{~W}}_{t}, \quad \boldsymbol{x}_{T} \sim p_{T} \text {, } \tag{2}
\end{equation*}
$$
where $p_{t}$ is the marginal distribution of the forward process (Eq. 1) at time $t$, and $\overline{\mathrm{W}}$ is a reverse Wiener process. Somewhat unexpectedly, there exists a deterministic Probability Flow ODE (PF-ODE) with matching marginals (Song et al., 2021):
$$
\begin{equation*}
\text { PF ODE: } \quad d \boldsymbol{x}=\left(f_{t} \boldsymbol{x}-\frac{1}{2} g_{t}^{2} \nabla \log p_{t}(\boldsymbol{x})\right) d t, \quad \quad \boldsymbol{x}_{T} \sim p_{T} \tag{3}
\end{equation*}
$$

Assuming we can approximate the score $\nabla \log p_{t}$ (Karras et al., 2024), we denote by $\boldsymbol{x}_{T} \mapsto \boldsymbol{x}_{0}\left(\boldsymbol{x}_{T}\right)$ the deterministic denoiser of solving the PF-ODE from noise $\boldsymbol{x}_{T}$, while we denote by $p\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right)$ the denoising distributions induced by stochastic sampling of the reverse SDE (Karras et al., 2022).

\section*{3 Riemannian geometry of diffusion models}

Riemannian geometry equips a latent space $\mathcal{Z}$ with a smoothly varying metric tensor $\mathbf{G}(\boldsymbol{z})$ for $\boldsymbol{z} \in \mathcal{Z}$. This metric defines inner products and induces the notions of distance and curve length (Do Carmo and Francis, 1992). Several works have developed diffusion models on top of Riemannian manifolds, such as spheres, tori and hyperboloids (De Bortoli et al., 2022; Huang et al., 2022; Thornton et al., 2022). In this paper, we instead study what kind of Riemannian geometries are implicitly induced by the denoiser within a real vector space $\mathbb{R}^{D}$ (e.g., images).
In Euclidean geometry, the space is flat, with distances given by the length of straight lines connecting points. In Riemannian spaces, the shortest path between two points is no longer straight, but a curved geodesic. A smooth curve $\gamma:[0,1] \rightarrow \mathcal{Z}$ between fixed endpoints $\gamma_{0}, \gamma_{1}$ is a geodesic if it minimizes the length
$$
\begin{equation*}
\ell(\boldsymbol{\gamma})=\int_{0}^{1}\left\|\dot{\boldsymbol{\gamma}}_{s}\right\|_{\mathbf{G}} d s=\int_{0}^{1} \sqrt{\dot{\boldsymbol{\gamma}}_{s}^{T} \mathbf{G}\left(\boldsymbol{\gamma}_{s}\right) \dot{\boldsymbol{\gamma}}_{s}} d s \tag{4}
\end{equation*}
$$
or, equivalently, the energy $\mathcal{E}(\boldsymbol{\gamma})=\frac{1}{2} \int_{0}^{1}\left\|\dot{\gamma}_{s}\right\|_{\mathbf{G}}^{2} d s$.
We introduce two interpretations of Riemannian geometry $\mathbf{G}$ for diffusion models, depending on whether the decoder is deterministic or stochastic. In both cases, we first assume the latent space is the noise space $\boldsymbol{x}_{T}$, and later relax this to cover the entire noisy sample space $\boldsymbol{x}_{t}$.

Deterministic sampler: pullback geometry. Let $\boldsymbol{x}_{T} \mapsto \boldsymbol{x}_{0}\left(\boldsymbol{x}_{T}\right)$ be a deterministic map given by the PF-ODE (Eq. 3) mapping noise to data. We propose the pullback metric (Arvanitidis et al., 2022; Park et al., 2023)
$$
\begin{equation*}
\mathbf{G}_{\mathrm{PB}}\left(\boldsymbol{x}_{T}\right)=\left(\frac{\partial \boldsymbol{x}_{0}}{\partial \boldsymbol{x}_{T}}\right)^{\top}\left(\frac{\partial \boldsymbol{x}_{0}}{\partial \boldsymbol{x}_{T}}\right) \in \mathbb{R}^{D \times D}, \quad \boldsymbol{x}_{0}:=\boldsymbol{x}_{0}\left(\boldsymbol{x}_{T}\right) \in \mathbb{R}^{D} \tag{5}
\end{equation*}
$$
which measures how an infinitesimal noise step $d \boldsymbol{x}_{T}$ changes the decoded sample:
$$
\begin{equation*}
\left\|\boldsymbol{x}_{0}\left(\boldsymbol{x}_{T}+d \boldsymbol{x}_{T}\right)-\boldsymbol{x}_{0}\left(\boldsymbol{x}_{T}\right)\right\|^{2}=d \boldsymbol{x}_{T}^{\top} \mathbf{G}_{\mathrm{PB}}\left(\boldsymbol{x}_{T}\right) d \boldsymbol{x}_{T}+o\left(\left\|d \boldsymbol{x}_{T}\right\|^{2}\right) . \tag{6}
\end{equation*}
$$

Stochastic sampler: information geometry. Alternatively, consider a stochastic decoder that, for each latent $\boldsymbol{x}_{T}$ defines a denoising distribution $p\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{T}\right)$ by solving the Reverse SDE (Eq. 2). We propose the information-geometric viewpoint via the Fisher-Rao metric (Amari, 2016)
$$
\begin{equation*}
\mathbf{G}_{\mathrm{IG}}\left(\boldsymbol{x}_{T}\right)=\mathbb{E}_{\boldsymbol{x}_{0} \sim p\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{T}\right)}\left[\nabla_{\boldsymbol{x}_{T}} \log p\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{T}\right) \nabla_{\boldsymbol{x}_{T}} \log p\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{T}\right)^{\top}\right] \in \mathbb{R}^{D \times D}, \tag{7}
\end{equation*}
$$
which measures how an infinitesimal noise step $d \boldsymbol{x}_{T}$ changes the entire denoising distribution:
$$
\begin{equation*}
\mathrm{KL}\left[p\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{T}\right) \| p\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{T}+d \boldsymbol{x}_{T}\right)\right]=\frac{1}{2} d \boldsymbol{x}_{T}^{\top} \mathbf{G}_{\mathrm{IG}}\left(\boldsymbol{x}_{T}\right) d \boldsymbol{x}_{T}+o\left(\left\|d \boldsymbol{x}_{T}\right\|^{2}\right) . \tag{8}
\end{equation*}
$$

For a helpful tutorial on information geometry, we refer to Mishra et al. (2023).

\section*{4 Pullback geometry collapses in diffusion models}

Both pullback and information geometries are, in principle, applicable. We will first show the pullback geometry has fundamental theoretical limitations in diffusion models, rendering it practically useless.

Assume we estimate a geodesic $\gamma$ in the noise space $\boldsymbol{x}_{T}$ such that its endpoints decode to $\boldsymbol{x}_{0}\left(\gamma_{0}\right)=\boldsymbol{x}^{a}$ and $\boldsymbol{x}_{0}\left(\gamma_{1}\right)=\boldsymbol{x}^{b}$. The pullback energy $\mathcal{E}(\boldsymbol{\gamma})$ (Eq. 4) can be shown to only depend on the decoded curve $\boldsymbol{x}_{0}\left(\boldsymbol{\gamma}_{s}\right)$ in data space (See Appendix B):
$$
\begin{equation*}
\mathcal{E}_{\mathrm{PB}}(\boldsymbol{\gamma})=\frac{1}{2} \int_{0}^{1}\left\|\frac{d}{d s} \boldsymbol{x}_{0}\left(\boldsymbol{\gamma}_{s}\right)\right\|^{2} d s . \tag{9}
\end{equation*}
$$

The unique minimizer is the constant-speed straight line $\boldsymbol{x}_{s}=(1-s) \boldsymbol{x}^{a}+s \boldsymbol{x}^{b}$ in data space. Since the ODE is bijective, this line has a unique latent preimage $\gamma_{s}^{\star}= \boldsymbol{x}_{0}^{-1}\left(\boldsymbol{x}_{s}\right)=\boldsymbol{x}_{T}\left(\boldsymbol{x}_{s}\right)$, which is thus a pullback geodesic, and the energy reduces to Euclidean distance in data space:

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/2cafeaa0-70df-4e35-a8ea-a66a0559b937-03.jpg?height=472&width=833&top_left_y=1271&top_left_x=1052}
\captionsetup{labelformat=empty}
\caption{Figure 2: The pullback geodesics curve in noise space, but decode to straight lines in data space.}
\end{figure}
$$
\begin{equation*}
\mathcal{E}_{\mathrm{PB}}(\boldsymbol{\gamma}):=\frac{1}{2}\left\|\boldsymbol{x}^{a}-\boldsymbol{x}^{b}\right\|^{2} . \tag{10}
\end{equation*}
$$

Hence, all pullback geodesics decode to straight segments, ignoring the curvature of the data manifold and undermining downstream applications (See Fig. 2). The same pathology applies for denoised geodesics in the intermediate space $\boldsymbol{x}_{t}$ as well. The core reason for this is that, in diffusion models, the latent and data spaces have the same dimension. The decoder operates directly in the ambient space and, without further dimensional constraints, it cannot capture the intrinsic structure of the data, even if the data lie on a lower-dimensional submanifold. As a result, the standard pullback metric provides no meaningful geometric information. A formal proof and discussion are in Appendix B.

\section*{5 Information geometry with denoising decoders}

Under the stochastic view, the decoder is the denoising distribution $p\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{T}\right)$ obtained by reversing the diffusion process (Eq. 2). This yields a family of distributions on the data space parametrized with noise vectors $\boldsymbol{x}_{T}$. The information geometry assigns the Fisher-Rao metric to the latent domain, and geodesic energies/lengths are computed as in Section 3.

The latent spacetime. Diffusion models are "memoryless" (Domingo-Enrich et al., 2025):
$$
\begin{equation*}
p\left(\boldsymbol{x}_{T} \mid \boldsymbol{x}_{0}\right) \approx p_{T}\left(\boldsymbol{x}_{T}\right) \Rightarrow p\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{T}\right) \approx q\left(\boldsymbol{x}_{0}\right) . \tag{11}
\end{equation*}
$$

Hence $p\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{T}\right)$ is (approximately) independent of $\boldsymbol{x}_{T}$, implying $\nabla_{\boldsymbol{x}_{T}} \log p\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{T}\right) \approx 0$ and a collapse of the Fisher-Rao metric, $\mathbf{G}_{\mathrm{IG}} \approx \mathbf{0}$ (Eq. 7). Consequently, if we identify the latent space with $\boldsymbol{z}=\boldsymbol{x}_{T}$, all $\boldsymbol{x}_{T}$ become metrically indistinguishable. This could be avoided by choosing $\boldsymbol{z}=\boldsymbol{x}_{t}$ for some $t<T$; however, instead of choosing an arbitrary noise level $t$, we propose to model all noise levels simultaneously by considering points in the ( $D+1$ )-dimensional latent spacetime
$$
\begin{equation*}
\boldsymbol{z}=\left(\boldsymbol{x}_{t}, t\right) \in \mathbb{R}^{D} \times(0, T], \tag{12}
\end{equation*}
$$
which define the family of all denoising distributions $\left\{p\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right)\right\}$ across all noise levels (Fig. 1).
Why include time? The resulting Fisher-Rao metric $\mathbf{G}_{\text {IG }}(\boldsymbol{z})$ varies with state and time, restoring a nontrivial geometry and enabling navigation across noise levels within a unified structure. Identifying clean data with spacetime points $(\boldsymbol{x}, 0)$, for which $p\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{0}=\boldsymbol{x}\right)=\delta_{\boldsymbol{x}}$, lets geodesics connect clean endpoints through noisy intermediates. This yields (i) a principled notion of distance between data as the length of the shortest spacetime path (Diffusion Edit Distance), and (ii) a mechanism for transition-path sampling via spacetime geodesics; both are demonstrated empirically in Section 6.

Tractable energy estimation. Usually, the information-geometric energy of a discretized curve $\boldsymbol{\gamma}=\left\{\boldsymbol{z}_{n}\right\}_{n=0}^{N-1}$ is approximated via the local-KL approximation (Arvanitidis et al., 2022):
$$
\begin{equation*}
\mathcal{E}(\boldsymbol{\gamma}) \approx(N-1) \sum_{n=0}^{N-2} \operatorname{KL}\left[p\left(\cdot \mid \boldsymbol{z}_{n}\right) \| p\left(\cdot \mid \boldsymbol{z}_{n+1}\right)\right], \tag{13}
\end{equation*}
$$
but such KLs are generally intractable, unless $p(\cdot \mid \boldsymbol{z})$ is a simple analytic distribution such as multinomial or Gaussian, which is not the case for denoising distributions $p\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right)$. Nonetheless, we show that in the specific case of the diffusion spacetime, the energy can be tractably estimated.
Proposition 5.1 (Spacetime energy estimation - informal). The energy of discretized spacetime curve $\boldsymbol{\gamma}=\left\{\boldsymbol{z}_{n}\right\}_{n=0}^{N-1}$ with $\boldsymbol{z}_{n}=\left(\boldsymbol{x}_{t_{n}}, t_{n}\right)$ admits an approximation
$$
\begin{equation*}
\mathcal{E}(\gamma) \approx \frac{N-1}{2} \sum_{n=0}^{N-2}\left(\boldsymbol{\eta}\left(\boldsymbol{z}_{n+1}\right)-\boldsymbol{\eta}\left(\boldsymbol{z}_{n}\right)\right)^{\top}\left(\boldsymbol{\mu}\left(\boldsymbol{z}_{n+1}\right)-\boldsymbol{\mu}\left(\boldsymbol{z}_{n}\right)\right), \tag{14}
\end{equation*}
$$
where
$$
\begin{equation*}
\boldsymbol{\eta}\left(\boldsymbol{x}_{t}, t\right)=\left(\frac{\alpha_{t}}{\sigma_{t}^{2}} \boldsymbol{x}_{t},-\frac{\alpha_{t}^{2}}{2 \sigma_{t}^{2}}\right), \quad \boldsymbol{\mu}\left(\boldsymbol{x}_{t}, t\right)=\left(\mathbb{E}\left[\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right], \mathbb{E}\left[\left\|\boldsymbol{x}_{0}\right\|^{2} \mid \boldsymbol{x}_{t}\right]\right) \tag{15}
\end{equation*}
$$

The proof (Appendix C) consists of showing that denoising distributions form an exponential family, which admits a simplified energy formula. In practice, we calculate $\boldsymbol{\mu}\left(\boldsymbol{x}_{t}, t\right)$ with Tweedie's formula over the approximate denoiser $\hat{\boldsymbol{x}}_{0}\left(\boldsymbol{x}_{t}\right)$ (See Appendix C. 2 for details),
$$
\begin{align*}
\mathbb{E}\left[\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right] & \approx \hat{\boldsymbol{x}}_{0}\left(\boldsymbol{x}_{t}\right) \\
\mathbb{E}\left[\left\|\boldsymbol{x}_{0}\right\|^{2} \mid \boldsymbol{x}_{t}\right] & \approx\left\|\hat{\boldsymbol{x}}_{0}\left(\boldsymbol{x}_{t}\right)\right\|^{2}+\frac{\sigma_{t}^{2}}{\alpha_{t}} \operatorname{div}_{\boldsymbol{x}_{t}} \hat{\boldsymbol{x}}_{0}\left(\boldsymbol{x}_{t}\right), \tag{16}
\end{align*}
$$
where both $\hat{\boldsymbol{x}}_{0}$ and $\operatorname{div} \hat{\boldsymbol{x}}_{0}$ are computed efficiently via Hutchinson's trick (Hutchinson, 1989; Grathwohl et al., 2019), enabling the esimation of $\boldsymbol{\mu}\left(\boldsymbol{x}_{t}, t\right)$ with a single Jacobian-vector product (JVP).

Spacetime geodesics are simulation-free: the energy calculation requires only $N$ JVPs of the denoiser $\hat{\boldsymbol{x}}_{0}$ for a curve discretized into $N$ points.

\section*{6 Experiments}

\subsection*{6.1 Sampling trajectories}

We compare the trajectories obtained by solving the PF-ODE $\boldsymbol{x}_{0}\left(\boldsymbol{x}_{T}\right)$ (Eq. 3) with geodesics between the same endpoints $\boldsymbol{x}_{0}, \boldsymbol{x}_{T}$. For a toy example of 1D mixture of Gaussians, we observe the geodesics curving less than the

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/2cafeaa0-70df-4e35-a8ea-a66a0559b937-05.jpg?height=499&width=1647&top_left_y=236&top_left_x=240}
\captionsetup{labelformat=empty}
\caption{Figure 3: PF-ODE paths are similar to energy-minimizing geodesics. Left: Geodesics move in straighter lines than PF-ODE trajectories in 1D toy density. Right: Geodesics are almost indistinguishable to PF-ODE sampling in ImageNet-512 EDM2 model.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/2cafeaa0-70df-4e35-a8ea-a66a0559b937-05.jpg?height=542&width=1601&top_left_y=928&top_left_x=262}
\captionsetup{labelformat=empty}
\caption{Figure 4: Spacetime geodesics between images. Each row shows a geodesic $\gamma$ between clean images. The path passes through noisy states and then denoises, realizing the minimal total edit between endpoints. Its length $\ell(\gamma)$ is the Diffusion Edit Distance (DiffED), which measures how much the denoising distribution changes along the optimal traversal.}
\end{figure}

PF-ODE trajectories in the early sampling (high $t$ ), while being indistinguishable for lower values of $t$ (See Fig. 3 left and Appendix F. 1 for details).

We find only marginal perceptual difference between the PF-ODE sampling trajectories and the geodesics in the EDM2 ImageNet-512 model (Karras et al., 2024). The geodesic appears to generate information slightly earlier, but the difference is minor (See Fig. 3 right, and Appendix F. 2 for details).

We note that spacetime geodesics are not an alternative sampling method since they require knowing the endpoints beforehand. An investigation into whether our framework can be used to improve sampling strategies is an interesting future research direction.

\subsection*{6.2 Diffusion Edit Distance}

The spacetime geometry yields a principled distance on the data space. We identify clean datum $\boldsymbol{x} \in \mathbb{R}^{d}$ with the spacetime point $(\boldsymbol{x}, 0)$, corresponding to the Dirac denoising distribution $\delta_{\boldsymbol{x}}$. Given two points $\boldsymbol{x}^{a}, \boldsymbol{x}^{b}$, we define the Diffusion Edit Distance (DiffED) by
$$
\begin{equation*}
\operatorname{DiffED}\left(\boldsymbol{x}^{a}, \boldsymbol{x}^{b}\right)=\ell(\boldsymbol{\gamma}), \tag{17}
\end{equation*}
$$
where $\gamma$ is the spacetime geodesic between ( $\boldsymbol{x}^{a}, 0$ ) and ( $\boldsymbol{x}^{b}, 0$ ). For numerical stability, we anchor endpoints at a small $t_{\text {min }}>0$ rather than at 0 .

A spacetime geodesic links two clean data points through intermediate noisy states. It can be interpreted as the minimal sequence of edits: add just enough noise to discard information specific to $\boldsymbol{x}^{a}$, then remove noise to introduce information specific to $\boldsymbol{x}^{b}$. The path length is the total edit cost, which is measured by how much the denoising distribution changes along the path. Fig. 4 visualizes the spacetime geodesics: as endpoint similarity decreases, the intermediate points become noisier.

We quantitatively evaluate DiffED on image data. First, we ask whether DiffED correlates with human perception as approximated by Learned Perceptual Image Patch Similarity (LPIPS) (Zhang et al., 2018). We randomly selected 10 classes in the ImageNet dataset and sampled 20 random image pairs for each. We then evaluated the DiffED and LPIPS for each image pair, and found the correlation to be very low at approximately $-7 \%$, suggesting that perceptual similarity and geometric edit cost capture different notions of closeness. We found DiffED to be more closely related to the structural similarity index measure (SSIM) (Wang et al., 2004), which correlates at 53\% with DiffED.

To qualitatively compare different notions of image similarity, we order image pairs by their similarity evaluated with multiple metrics: DiffED, LPIPS, SSIM, and Euclidean. We show the results in Fig. 8.

\subsection*{6.3 Transition path sampling}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/2cafeaa0-70df-4e35-a8ea-a66a0559b937-06.jpg?height=527&width=1649&top_left_y=902&top_left_x=240}
\captionsetup{labelformat=empty}
\caption{Figure 5: Spacetime geodesics enable sampling transition paths between low-energy states. Left: Alanine Dipeptide energy landscape wrt two dihedral angles, with two energy minima $\boldsymbol{x}_{0}^{1}, \boldsymbol{x}_{0}^{2}$. Middle: Spacetime geodesic $\gamma$ connecting $\boldsymbol{x}_{0}^{1}$ and $\boldsymbol{x}_{0}^{2}$. Right: Annealed Langevin transition path samples.}
\end{figure}

Another application of the spacetime geometry is the problem of transition-path sampling (Holdijk et al., 2023; Du et al., 2024; Raja et al., 2025), whose goal is to find probable transition paths between low-energy states. We assume a Boltzmann distribution
$$
\begin{equation*}
q(\boldsymbol{x}) \propto \exp (-U(\boldsymbol{x})), \tag{18}
\end{equation*}
$$
where $U$ is a known energy function, which is a common assumption in molecular dynamics. In this setting, the denoising distribution follows a tractable energy function (See Eq. 57)
$$
\begin{equation*}
p\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right) \propto q\left(\boldsymbol{x}_{0}\right) p\left(\boldsymbol{x}_{t} \mid \boldsymbol{x}_{0}\right) \propto \exp (\underbrace{-U\left(\boldsymbol{x}_{0}\right)-\frac{1}{2} \operatorname{SNR}(t)\left\|\boldsymbol{x}_{0}-\boldsymbol{x}_{t} / \alpha_{t}\right\|^{2}}_{-U\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right)}) . \tag{19}
\end{equation*}
$$

To construct a transition path between two low-energy states $\boldsymbol{x}_{0}^{1}$ and $\boldsymbol{x}_{0}^{2}$, we estimate the spacetime geodesic $\boldsymbol{\gamma}$ between them using a denoiser model $\hat{\boldsymbol{x}}_{0}\left(\boldsymbol{x}_{t}\right) \approx \mathbb{E}\left[\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right]$ with Proposition 5.1, as shown in Fig. 5. At each interpolation point $s \in[0,1]$, the geodesic defines a denoising Boltzmann distribution $p\left(\boldsymbol{x} \mid \boldsymbol{\gamma}_{s}\right)$ where $U\left(\boldsymbol{x} \mid \boldsymbol{\gamma}_{s}\right)$ is the energy at that spacetime location. See Appendix F. 3 for details.

Annealed Langevin Dynamics. To sample transition paths, we use Langevin dynamics
$$
\begin{equation*}
d \boldsymbol{x}=-\nabla_{\boldsymbol{x}} U\left(\boldsymbol{x} \mid \gamma_{s}\right) d t+\sqrt{2} d \mathrm{~W}_{t}, \tag{20}
\end{equation*}
$$
whose stationary distributions are $p\left(\boldsymbol{x} \mid \boldsymbol{\gamma}_{s}\right) \propto \exp \left(-U\left(\boldsymbol{x} \mid \boldsymbol{\gamma}_{s}\right)\right)$ for any $s$. To obtain the trajectories from $\boldsymbol{x}_{0}^{1}$ to $\boldsymbol{x}_{0}^{2}$, we gradually increase $s$ from 0 to 1 using annealed Langevin (Song and Ermon, 2019). After discretizing the geodesic into $N$ points $\gamma_{n}$, we alternate between taking $K$ steps of Eq. 20 conditioned on $\gamma_{n}$ and updating $\gamma_{n} \mapsto \gamma_{n+1}$, as described in Algorithm 1. This approach assumes that $p\left(\boldsymbol{x} \mid \gamma_{n}\right)$ is close to $p\left(\boldsymbol{x} \mid \gamma_{n+1}\right)$, and thus $\boldsymbol{x} \sim p\left(\boldsymbol{x} \mid \boldsymbol{\gamma}_{n}\right)$ is a good starting point to Langevin dynamics conditioned on $\gamma_{n+1}$.

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 1: Spacetime geodesics outperform methods tailored to transition path sampling. Parentheses denote extra energy evaluations used to generate training data for the base diffusion model, which do not scale with the number of generated paths. Baseline details in Appendix G.}
\begin{tabular}{lcc}
\hline & MaxEnergy $(\downarrow)$ & \# Evaluations $(\downarrow)$ \\
\hline Lower Bound & 36.42 & N/A \\
\hline MCMC-fixed-length & $42.54 \pm 7.42$ & 1.29 B \\
MCMC-variable-length & $58.11 \pm 18.51$ & 21.02 M \\
Doob's Lagrangian (Du et al., 2024) & $66.24 \pm 1.01$ & 38.4 M \\
\hline Spacetime geodesic (Ours) & $\mathbf{3 7 . 6 6} \pm 0.61$ & $\mathbf{1 2 8 K}(+\mathbf{1 6 M})$ \\
\hline
\end{tabular}
\end{table}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/2cafeaa0-70df-4e35-a8ea-a66a0559b937-07.jpg?height=396&width=1643&top_left_y=758&top_left_x=242}
\captionsetup{labelformat=empty}
\caption{Figure 6: Transition paths generated with a spacetime geodesic avoid high-energy regions without collapsing to a single path. Compared with MCMC baselines, the spacetime-geodesic method yields transition paths that better avoid high-energy areas, whereas Doob's Lagrangian collapses to generating nearly identical trajectories. Ten sample paths are shown for each method.}
\end{figure}

Alanine dipeptide. We compute a spacetime geodesic connecting two molecular configurations of Alanine Dipeptide, as in Holdijk et al. (2023). In Fig. 5, the energy landscape is visualized over the dihedral angle space, with a neural network used to approximate the potential energy $U$. Using our trained denoiser $\hat{\boldsymbol{x}}_{0}\left(\boldsymbol{x}_{t}\right)$, we estimate the expectation parameter $\boldsymbol{\mu}$, which allows us to compute and visualize a geodesic trajectory through spacetime. Transition paths were generated using Algorithm 1. See Appendix F. 3 for details.

Baselines. We considered Holdijk et al. (2023); Du et al. (2024); Raja et al. (2025) and adopt Doob's Lagrangian (Du et al., 2024); the others were excluded due to reproducibility issues (see Appendix G). We also evaluate two MCMC two-way shooting variants (Brotzakis and Bolhuis, 2016)-uniform point selection with variable or fixed trajectory length-using transition paths from the official Du et al. (2024) code release. For each method we generate 1,000 paths and report mean MaxEnergy (lower is better) and its numerical lower bound $\min _{\gamma} \max _{s} U\left(\gamma_{s}\right)$, along with the number of energy evaluations needed for 1,000 paths. To train a base diffusion model for our method, we generated data using Langevin dynamics ( $16 \mathrm{M}^{1}$ energy evaluations), a one-time cost that does not scale with the number of generated transition paths.

Results. We show in Table 1 that our method outperforms the baselines in the MaxEnergy obtained along the transition paths. It is also considerably closer to the lower bound than to the next best baseline (MCMC-fixed length) while requiring several orders of magnitude fewer energy function evaluations. In Fig. 6, we show a qualitative comparison of transition paths generated with our method and the baselines. Our proposed method shows improved efficiency in avoiding high-energy regions compared to MCMC. In contrast, the Doob's Lagrangian method converged to a suboptimal solution, producing nearly identical transition paths. We discuss this in more detail in Appendix G.

\footnotetext{
${ }^{1} 16 \mathrm{M}$ is the number of energy function evaluations to generate the training set with Langevin dynamics for the base diffusion model. We did not tune this number, and fewer evaluations may yield comparable performance.
}
```
Algorithm 1 Transition Path Sampling with Annealed Langevin Dynamics
Require: $\boldsymbol{x}_{a}, \boldsymbol{x}_{b} \in \mathbb{R}^{D}$ endpoints, $N_{\boldsymbol{\gamma}}>0, T>0, t_{\text {min }}, d t$
    $\gamma \leftarrow \arg \min _{\boldsymbol{\gamma}} \mathcal{E}(\boldsymbol{\gamma}) \quad \triangleright$ Approximate spacetime geodesic connecting $\boldsymbol{x}_{a}$ with $\boldsymbol{x}_{b}$
    $\mathcal{T} \leftarrow\left\{\boldsymbol{x}:=\boldsymbol{x}_{a}\right\} \quad \triangleright$ Initialize chain $\mathcal{T}$ at $\boldsymbol{x}_{a}$
    for $n \in\left\{0, \ldots, N_{\gamma}-1\right\}$ do $\quad \triangleright$ Iterate over the points on the geodesic $\gamma_{n}$
        for $t \in\{1, \ldots, T\}$ do
            $\varepsilon \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I}) \quad$ D Sample Gaussian noise
            $\boldsymbol{x} \leftarrow \boldsymbol{x}-\nabla_{\boldsymbol{x}} U\left(\boldsymbol{x} \mid \gamma_{n}\right) d t+\sqrt{2 d t} \boldsymbol{\varepsilon} \quad \triangleright$ Langevin update
            $\mathcal{T} \leftarrow \mathcal{T} \cup\{\boldsymbol{x}\} \quad \triangleright$ Append state $\boldsymbol{x}$ to chain
        end for
    end for
    return $\mathcal{T} \quad \triangleright$ Return chain
```


\subsection*{6.4 Constrained path sampling}

Suppose we would like to impose additional constraints along the geodesic interpolants. This corresponds to penalized optimization
$$
\begin{equation*}
\min _{\boldsymbol{\gamma}}\left\{\mathcal{E}(\boldsymbol{\gamma})+\lambda \int_{0}^{1} h\left(\boldsymbol{\gamma}_{s}\right) d s, \quad \text { s.t. } \quad \boldsymbol{\gamma}_{0}=\left(\boldsymbol{x}_{0}^{1}, 0\right), \boldsymbol{\gamma}_{1}=\left(\boldsymbol{x}_{0}^{2}, 0\right)\right\}, \tag{21}
\end{equation*}
$$
where $h: \mathbb{R} \times \mathbb{R}^{D} \rightarrow \mathbb{R}$ is some penalty function with $\lambda>0$. We demonstrate the principle by (i) penalizing transition path variance, and (ii) imposing regions to avoid in the data space.

Low-variance transitions. Suppose we want the posterior $p\left(\boldsymbol{x} \mid \boldsymbol{\gamma}_{s}\right)$ to have a low variance. This concentrates the path around a narrower set of plausible states, more repeatable trajectories, albeit at the cost of reduced coverage. By Eq. 53, higher SNR $(t)$ yields lower denoising variance, so we implement this by penalizing low SNR via $h(\boldsymbol{x}, t)= \max (-\log \operatorname{SNR}(t), \rho)$ for some threshold $\rho$.

Avoiding restricted regions. Suppose we want to avoid certain regions in the data space in the transition paths. We encode the region to avoid as a denoising distribution $p\left(\cdot \mid \boldsymbol{z}^{*}\right)$ for some $\boldsymbol{z}^{*}=\left(\boldsymbol{x}_{t}^{*}, t^{*}\right)$ where larger the $t^{*}$, larger the restricted region. We encode the penalty as KL distance between the denoising distributions (See Appendix D for the derivation)
$$
\begin{align*}
\mathrm{KL}\left[p\left(\cdot \mid \boldsymbol{z}^{*}\right) \| p\left(\cdot \mid \boldsymbol{\gamma}_{s}\right)\right] & =\int_{0}^{s}\left(\frac{d}{d u} \boldsymbol{\eta}\left(\boldsymbol{\gamma}_{u}\right)\right)^{\top}\left(\boldsymbol{\mu}\left(\boldsymbol{\gamma}_{u}\right)-\boldsymbol{\mu}\left(\boldsymbol{z}^{*}\right)\right) d u+C  \tag{22}\\
h\left(\boldsymbol{\gamma}_{s}\right) & =\min \left(\rho,-\mathrm{KL}\left[p\left(\cdot \mid \boldsymbol{z}^{*}\right) \| p\left(\cdot \mid \boldsymbol{\gamma}_{s}\right)\right]\right) \tag{23}
\end{align*}
$$

In Fig. 7, we compare spacetime geodesics (unconstrained) with low-variance, and region-avoiding spacetime curves. We visualize both the curves and the corresponding transition paths generated with Algorithm 1. This demonstrates that our framework with the penalized optimization (Eq. 21) can incorporate various preferences on the transition paths.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/2cafeaa0-70df-4e35-a8ea-a66a0559b937-08.jpg?height=364&width=1657&top_left_y=2015&top_left_x=236}
\captionsetup{labelformat=empty}
\caption{Figure 7: Vanilla transition paths can be constrained to have lower variance, or successfully avoid a restricted region $p\left(\cdot \mid z^{*}\right)$. Left: geodesics $\gamma$. Right: transition paths $\mathcal{T}$.}
\end{figure}

\section*{7 Related works}

We review three directions of research related to ours: (i) studies of latent noise in diffusion models, (ii) applications of information geometry in generative modeling, and (iii) geometric formulations for sampling efficiency.

Latent-data geometry. Several works analyze the relation between latent noise $\boldsymbol{x}_{t}$ and data $\boldsymbol{x}_{0}$. Yu et al. (2025) define a geodesic density in diffusion latent space; Park et al. (2023) apply Riemannian geometry to lower-dimensional latent codes; Karczewski et al. (2025) study how noise scaling affects log-densities and perceptual detail. Our work also investigates the $\boldsymbol{x}_{t}$ to $\boldsymbol{x}_{0}$ relationship but (a) uses the Fisher-Rao metric rather than an inverse-density metric, (b) retains the full-dimensional latent space without projection, and (c) analyzes the complete diffusion path across all timesteps.

Information geometry in generative models. Lobashev et al. (2025) introduce the Fisher-Rao metric on families $p(\boldsymbol{x} \mid \theta)$ to study phase-like transitions, where $\theta$ is a low-dimensional variable parametrizing a microstate $\boldsymbol{x}$. In contrast, we place the geometry on diffusion's explicit spacetime coordinates $\boldsymbol{z}=\left(\boldsymbol{x}_{t}, t\right)$, induced by the denoising posterior $p\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right)$.

Geometric approaches to sampling. Two recent works also formulate diffusion models geometrically to improve sampling efficiency. Das et al. (2023) optimize the forward noising process by following the shortest geodesic between $p_{0}$ and $p_{t}$ under the Fisher-Rao metric, assuming $p_{0}\left(\boldsymbol{x}_{0}\right)$ to be Gaussian. Ghimire et al. (2023) model both the forward and reverse processes as Wasserstein gradient flows. Our contribution differs: we use information geometry (not optimal transport), focus on the reverse process (not the forward), and only require $p_{0}$ to admit a density.

\section*{8 Limitations}

Although our framework defines geodesics between any noisy samples, optimizing between nearly clean ones is numerically unstable because their denoising distributions collapse to Dirac deltas, making Fisher-Rao (via local KL) distances effectively infinite. Therefore, consistent with diffusion practice (Song et al., 2021; Lu et al., 2022), we choose endpoints with non-negligible noise for tractable optimization (details in Appendix F).
The proposed distance metric DiffED (Section 6.2) is considerably slower (details in Appendix F.2) than established image similarity metrics such as LPIPS (Zhang et al., 2018), or SSIM (Wang et al., 2004). Exploring a distillation strategy involving training a separate model trained to predict DiffED is a possible future research direction.

\section*{9 Conclusion}

We proposed a novel perspective on the latent space of diffusion models by viewing it as a ( $D+1$ )-dimensional statistical manifold, with the Fisher-Rao metric inducing a geometrical structure. By leveraging the fact that the denoising distributions form an exponential family, we showed that we can tractably estimate geodesics even for high-dimensional image diffusion models. We visualized our methods for image interpolations and demonstrated their utility in molecular transition path sampling.

This work deepens our understanding of the latent space in diffusion models and has the potential to inspire further research, including the development of novel applications of the spacetime geometric framework, such as enhanced sampling techniques.

\section*{Ethics statement}

The use of generative models, especially those capable of producing images and videos, poses considerable risks for misuse. Such technologies have the potential to produce harmful societal effects, primarily through the spread of disinformation, but also by reinforcing harmful stereotypes and implicit biases. In this work, we contribute to a deeper understanding of diffusion models, which currently represent the leading methodology in generative modeling. While this insight may eventually support improvements to these models, thereby increasing the risk of misuse, it is important to note that our research does not introduce any new ethical risks beyond those already associated with generative AI.

We have used Large Language Models to polish writing on a sentence level.

\section*{Acknowledgments}

This work was supported by the Finnish Center for Artificial Intelligence (FCAI) under Flagship R5 (award 15011052). SH was supported by research grants from VILLUM FONDEN (42062), the Novo Nordisk Foundation through the Center for Basic Research in Life Science (NNF20OC0062606), and the European Research Council (ERC) under the European Union's Horizon Programme (grant agreement 101125003). VG acknowledges the support from Saab-WASP (grant 411025), Academy of Finland (grant 342077), and the Jane and Aatos Erkko Foundation (grant 7001703).

\section*{References}

Shun-ichi Amari. Information geometry and its applications. Springer, 2016.
Brian DO Anderson. Reverse-time diffusion equation models. Stochastic Processes and their Applications, 1982.
Georgios Arvanitidis, Lars Kai Hansen, and Søren Hauberg. Latent space oddity: On the curvature of deep generative models. In ICLR, 2018.

Georgios Arvanitidis, Miguel González-Duque, Alison Pouplin, Dimitrios Kalatzis, and Soren Hauberg. Pulling back information geometry. In AISTATS, 2022.

Z Faidon Brotzakis and Peter G Bolhuis. A one-way shooting algorithm for transition path sampling of asymmetric barriers. The Journal of Chemical Physics, 2016.

Ayan Das, Stathi Fotiadis, Anil Batra, Farhang Nabiei, FengTing Liao, Sattar Vakili, Da-shan Shiu, and Alberto Bernacchia. Image generation with shortest path diffusion. In ICML, 2023.

Valentin De Bortoli, Emile Mathieu, Michael Hutchinson, James Thornton, Yee Whye Teh, and Arnaud Doucet. Riemannian score-based generative modelling. In NeurIPS, 2022.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In CVPR, 2009.

Manfredo Perdigao Do Carmo and Flaherty Francis. Riemannian geometry. Springer, 1992.
Carles Domingo-Enrich, Michal Drozdzal, Brian Karrer, and Ricky T. Q. Chen. Adjoint matching: Fine-tuning flow and diffusion generative models with memoryless stochastic optimal control. In ICLR, 2025.

Yuanqi Du, Michael Plainer, Rob Brekelmans, Chenru Duan, Frank Noe, Carla Gomes, Alan Aspuru-Guzik, and Kirill Neklyudov. Doob's Lagrangian: A sample-efficient variational approach to transition path sampling. In NeurIPS, 2024.

Bradley Efron. Tweedie's formula and selection bias. Journal of the American Statistical Association, 106(496): 1602-1614, 2011.

Sandesh Ghimire, Jinyang Liu, Armand Comas, Davin Hill, Aria Masoomi, Octavia Camps, and Jennifer Dy. Geometry of score based generative models. arXiv, 2023.

Will Grathwohl, Ricky TQ Chen, Jesse Bettencourt, and David Duvenaud. Scalable reversible generative models with free-form continuous dynamics. In ICLR, 2019.

Peter Holderrieth, Marton Havasi, Jason Yim, Neta Shaul, Itai Gat, Tommi Jaakkola, Brian Karrer, Ricky TQ Chen, and Yaron Lipman. Generator matching: Generative modeling with arbitrary Markov processes. In ICLR, 2025.

Lars Holdijk, Yuanqi Du, Ferry Hooft, Priyank Jaini, Bernd Ensing, and Max Welling. Stochastic optimal control for collective variable free sampling of molecular transition paths. In NeurIPS, 2023.

Chin-Wei Huang, Milad Aghajohari, Joey Bose, Prakash Panangaden, and Aaron Courville. Riemannian diffusion models. In NeurIPS, 2022.

Michael Hutchinson. A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines. Communications in Statistics - Simulation and Computation, 1989.

Rafał Karczewski, Markus Heinonen, and Vikas Garg. Devil is in the details: Density guidance for detail-aware generation with flow models. In ICML, 2025.

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In NeurIPS, 2022.

Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, and Samuli Laine. Analyzing and improving the training dynamics of diffusion models. In CVPR, 2024.

Diederik Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. In NeurIPS, 2021.
Diederik P Kingma and Ruiqi Gao. Understanding diffusion objectives as the ELBO with simple data augmentation. In NeurIPS, 2023.

Alexander Lobashev, Dmitry Guskov, Maria Larchenko, and Mikhail Tamm. Hessian geometry of latent space in generative models. In ICML, 2025.

Cheng Lu, Kaiwen Zheng, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Maximum likelihood training for score-based diffusion odes by high order denoising score matching. In ICML, 2022.

Chenlin Meng, Yang Song, Wenzhe Li, and Stefano Ermon. Estimating high order gradients of the data distribution by denoising. In NeurIPS, 2021.

Kumar Mishra, Ashok Kumar, and Ting-Kam Leonard Wong. Information geometry for the working information theorist. arXiv, 2023.

OpenMP Architecture Review Board. OpenMP application program interface version 3.0, 2008. URL http://www. openmp.org/mp-documents/spec30.pdf.

Yong-Hyun Park, Mingi Kwon, Jaewoong Choi, Junghyo Jo, and Youngjung Uh. Understanding the latent space of diffusion models through the lens of Riemannian geometry. In NeurIPS, 2023.

Sanjeev Raja, Martin Sipka, Michael Psenka, Tobias Kreiman, Michal Pavelka, and Aditi Krishnapriyan. Actionminimization meets generative modeling: Efficient transition path sampling with the Onsager-Machlup functional. In ICLR Workshop on Generative and Experimental Perspectives for Biomolecular Design, 2025.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, 2015.

Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In NeurIPS, 2019.
Yang Song, Jascha Sohl-Dickstein, Diederik Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In ICLR, 2021.

James Thornton, Michael Hutchinson, Emile Mathieu, Valentin De Bortoli, Yee Whye Teh, and Arnaud Doucet. Riemannian diffusion Schrödinger bridge. In ICML Workshop Continuous Time Methods for Machine Learning, 2022.

Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 2004.

Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Wentao Zhang, Bin Cui, and MingHsuan Yang. Diffusion models: A comprehensive survey of methods and applications. ACM Computing Surveys, 2023.

Qingtao Yu, Jaskirat Singh, Zhaoyuan Yang, Peter Henry Tu, Jing Zhang, Hongdong Li, Richard Hartley, and Dylan Campbell. Probability density geodesics in image diffusion latent space. In CVPR, 2025.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018.

\section*{A Notation}

We denote $\boldsymbol{x}=\left(x^{1}, \ldots, x^{D}\right)^{\top} \in \mathbb{R}^{D}$ a point in $D$-dimensional Euclidean space (a column vector), $\operatorname{Tr}(\boldsymbol{A})=\sum_{i} A_{i i}$ the trace operator of a square matrix $\boldsymbol{A} \in \mathbb{R}^{k \times k}$.

Differential operators. For a scalar function $f: \mathbb{R}^{D} \rightarrow \mathbb{R}, \boldsymbol{x} \mapsto f(\boldsymbol{x}) \in \mathbb{R}$, we denote
$$
\begin{array}{rlr}
\text { gradient: } & \nabla_{\boldsymbol{x}} f(\tilde{\boldsymbol{x}})=\left.\left(\frac{\partial f}{\partial x^{1}}, \ldots, \frac{\partial f}{\partial x^{D}}\right)^{\top}\right|_{\boldsymbol{x}=\tilde{\boldsymbol{x}}} & \in \mathbb{R}^{D} \\
\text { Hessian: } & \nabla_{\boldsymbol{x}}^{2} f(\tilde{\boldsymbol{x}})=\left.\left[\frac{\partial^{2} f}{\partial x^{i} \partial x^{j}}\right]_{i, j}\right|_{\boldsymbol{x}=\tilde{\boldsymbol{x}}} & \in \mathbb{R}^{D \times D} \\
\text { Laplacian: } & \Delta_{\boldsymbol{x}} f(\tilde{\boldsymbol{x}})=\operatorname{Tr}\left(\nabla_{\boldsymbol{x}}^{2} f(\tilde{\boldsymbol{x}})\right)=\left.\sum_{i=1}^{D} \frac{\partial^{2} f}{\partial\left(x^{i}\right)^{2}}\right|_{\boldsymbol{x}=\tilde{\boldsymbol{x}}} & \in \mathbb{R}
\end{array}
$$

For a curve $\gamma:[0,1] \rightarrow \mathbb{R}^{k}, s \mapsto \gamma_{s} \in \mathbb{R}^{k}$ we denote
$$
\text { time derivative: } \quad \dot{\gamma}_{s}=\frac{d}{d s} \gamma_{s} \in \mathbb{R}^{k}
$$

For a vector valued function $f: \mathbb{R}^{k} \rightarrow \mathbb{R}^{m}, \boldsymbol{x} \mapsto\left(f^{1}(\boldsymbol{x}), \ldots, f^{m}(\boldsymbol{x})\right)^{\top} \in \mathbb{R}^{m}$ we denote
$$
\text { Jacobian: } \quad \frac{\partial f(\tilde{\boldsymbol{x}})}{\partial \boldsymbol{x}}=\left.\left[\frac{\partial f^{i}}{\partial x^{j}}\right]_{i, j}\right|_{\boldsymbol{x}=\tilde{\boldsymbol{x}}} \in \mathbb{R}^{m \times k}
$$

When $k=m$, we define
$$
\text { divergence: } \quad \operatorname{div}_{\boldsymbol{x}} f(\tilde{\boldsymbol{x}})=\operatorname{Tr}\left(\frac{\partial f(\tilde{\boldsymbol{x}})}{\partial \boldsymbol{x}}\right)=\left.\sum_{i=1}^{k} \frac{\partial f^{i}}{\partial x^{i}}\right|_{\boldsymbol{x}=\tilde{\boldsymbol{x}}} \in \mathbb{R}
$$

Functions with two arguments. For $f: \mathbb{R}^{k_{1}} \times \mathbb{R}^{k_{2}} \rightarrow \mathbb{R},\left(\boldsymbol{x}_{1}, \boldsymbol{x}_{2}\right) \mapsto f\left(\boldsymbol{x}_{1}, \boldsymbol{x}_{2}\right) \in \mathbb{R}$ we define (analogously w.r.t. second argument)
$$
\text { gradient w.r.t. first argument: } \nabla_{\boldsymbol{x}_{1}} f\left(\tilde{\boldsymbol{x}}_{1}, \tilde{\boldsymbol{x}}_{2}\right)=\left.\left(\frac{\partial f}{\partial x_{1}^{1}}, \ldots, \frac{\partial f}{\partial x_{1}^{k_{1}}}\right)^{\top}\right|_{\left(\boldsymbol{x}_{1}, \boldsymbol{x}_{2}\right)=\left(\tilde{\boldsymbol{x}}_{1}, \tilde{\boldsymbol{x}}_{2}\right)} \in \mathbb{R}^{k_{1}}
$$

For $f: \mathbb{R}^{k_{1}} \times \mathbb{R}^{k_{2}} \rightarrow \mathbb{R}^{m},\left(\boldsymbol{x}_{1}, \boldsymbol{x}_{2}\right) \mapsto\left(f^{1}\left(\boldsymbol{x}_{1}, \boldsymbol{x}_{2}\right), \ldots, f^{m}\left(\boldsymbol{x}_{1}, \boldsymbol{x}_{2}\right)\right)^{\top} \in \mathbb{R}^{m}$ we define (analogously w.r.t. second argument)
$$
\text { Jacobian w.r.t. first argument: } \quad \frac{\partial f\left(\tilde{\boldsymbol{x}}_{1}, \tilde{\boldsymbol{x}}_{2}\right)}{\partial \boldsymbol{x}_{1}}=\left.\left[\frac{\partial f^{i}}{\partial x_{1}^{j}}\right]_{i, j}\right|_{\left(\boldsymbol{x}_{1}, \boldsymbol{x}_{2}\right)=\left(\tilde{\boldsymbol{x}}_{1}, \tilde{\boldsymbol{x}}_{2}\right)} \in \mathbb{R}^{m \times k_{1}}
$$

\section*{B Pullback Geometry in Diffusion models}

Lemma B.1. Let $\mathcal{Z}$ be a latent space, $\mathcal{X}=\mathbb{R}^{d}$ a data space, and $f: \mathcal{Z} \rightarrow \mathcal{X}$ a decoder. Then the length and energy of a curve $\gamma:[0,1] \rightarrow \mathcal{X}$ under the pullback geometry are given by
$$
\begin{align*}
\ell_{\mathrm{PB}}(\boldsymbol{\gamma}) & =\int_{0}^{1}\left\|\frac{d}{d s} f\left(\boldsymbol{\gamma}_{s}\right)\right\| d s  \tag{24}\\
\mathcal{E}_{\mathrm{PB}}(\boldsymbol{\gamma}) & =\frac{1}{2} \int_{0}^{1}\left\|\frac{d}{d s} f\left(\boldsymbol{\gamma}_{s}\right)\right\|^{2} d s \tag{25}
\end{align*}
$$
where $\|\cdot\|$ is the Euclidean norm.
Proof. For a general Riemannian metric $\mathbf{G}$, the length, and energy are given by
$$
\begin{align*}
\ell_{\mathbf{G}}(\boldsymbol{\gamma}) & =\int_{0}^{1}\left\|\dot{\gamma}_{s}\right\|_{\mathbf{G}} d s  \tag{26}\\
\mathcal{E}_{\mathbf{G}}(\boldsymbol{\gamma}) & =\frac{1}{2} \int_{0}^{1}\left\|\dot{\gamma}_{s}\right\|_{\mathbf{G}}^{2} d s \tag{27}
\end{align*}
$$
where $\left\|\dot{\gamma}_{s}\right\|_{\mathbf{G}}^{2}=\dot{\gamma}_{s}^{\top} G\left(\gamma_{s}\right) \dot{\gamma}_{s}$. For $\mathbf{G}=\mathbf{G}_{\mathrm{PB}}$ induced by the decoder $f$, we have
$$
\begin{equation*}
\mathbf{G}_{\mathrm{PB}}(\boldsymbol{z})=\left(\frac{\partial f}{\partial \boldsymbol{z}}(\boldsymbol{z})\right)^{\top}\left(\frac{\partial f}{\partial \boldsymbol{z}}(\boldsymbol{z})\right), \tag{28}
\end{equation*}
$$
which leads to
$$
\begin{align*}
\left\|\dot{\boldsymbol{\gamma}}_{s}\right\|_{\mathbf{G}_{\mathrm{PB}}}^{2} & =\dot{\boldsymbol{\gamma}}_{s}^{\top}\left(\frac{\partial f}{\partial \boldsymbol{z}}\left(\boldsymbol{\gamma}_{s}\right)\right)^{\top}\left(\frac{\partial f}{\partial \boldsymbol{z}}\left(\boldsymbol{\gamma}_{s}\right)\right) \dot{\boldsymbol{\gamma}}_{s}=\left(\frac{\partial f}{\partial \boldsymbol{z}}\left(\boldsymbol{\gamma}_{s}\right) \dot{\boldsymbol{\gamma}}_{s}\right)^{\top}\left(\frac{\partial f}{\partial \boldsymbol{z}}\left(\boldsymbol{\gamma}_{s}\right) \dot{\boldsymbol{\gamma}}_{s}\right)  \tag{29}\\
& \stackrel{(*)}{=}\left(\frac{d}{d s} f\left(\boldsymbol{\gamma}_{s}\right)\right)^{\top}\left(\frac{d}{d s} f\left(\boldsymbol{\gamma}_{s}\right)\right)=\left\|\frac{d}{d s} f\left(\boldsymbol{\gamma}_{s}\right)\right\|^{2}
\end{align*}
$$
where ( $*$ ) follows from the chain rule.
Proposition B. 1 (Pullback geodesics decode to straight lines). Let $\mathcal{Z}$ be a latent space, $\mathcal{X}=\mathbb{R}^{d}$ a data space, and $f: \mathcal{Z} \rightarrow \mathcal{X}$ a bijective decoder. Fix $\boldsymbol{z}^{a}, \boldsymbol{z}^{b} \in \mathcal{Z}$ and write $\boldsymbol{x}^{a}=f\left(\boldsymbol{z}^{a}\right), \boldsymbol{x}^{b}=f\left(\boldsymbol{z}^{b}\right)$. Then any shortest path between $\boldsymbol{z}^{a}$ and $\boldsymbol{z}^{b}$ in the pullback geometry decodes to the straight segment from $\boldsymbol{x}^{a}$ to $\boldsymbol{x}^{b}$.

Proof. Let $\boldsymbol{x}_{s}=(1-s) \boldsymbol{x}^{a}+s \boldsymbol{x}^{b}$ for $s \in[0,1]$. Because $f$ is bijective, define its latent preimage
$$
\boldsymbol{\gamma}_{s}^{\star}=f^{-1}\left(\boldsymbol{x}_{s}\right), \quad s \in[0,1] .
$$

The pullback length of a latent curve $\gamma$ is the Euclidean length of its image (Lemma B.1):
$$
\ell_{\mathrm{PB}}(\boldsymbol{\gamma})=\int_{0}^{1}\left\|\frac{d}{d s} f\left(\boldsymbol{\gamma}_{s}\right)\right\| d s
$$

For $\boldsymbol{\gamma}^{\star}, f\left(\boldsymbol{\gamma}_{s}^{\star}\right)=\boldsymbol{x}_{s}$ has constant velocity $\dot{\boldsymbol{x}}_{s}=\boldsymbol{x}^{b}-\boldsymbol{x}^{a}$, hence
$$
\ell_{\mathrm{PB}}\left(\boldsymbol{\gamma}^{\star}\right)=\int_{0}^{1}\left\|\boldsymbol{x}^{b}-\boldsymbol{x}^{a}\right\| d s=\left\|\boldsymbol{x}^{b}-\boldsymbol{x}^{a}\right\|
$$

For any other smooth latent curve $\bar{\gamma}$ from $\boldsymbol{z}^{a}$ to $\boldsymbol{z}^{b}$, using the triangle inequality:
$$
\ell_{\mathrm{PB}}(\overline{\boldsymbol{\gamma}})=\int_{0}^{1}\left\|\frac{d}{d s} f\left(\overline{\boldsymbol{\gamma}}_{s}\right)\right\| d s \geq\left\|\int_{0}^{1} \frac{d}{d s} f\left(\overline{\boldsymbol{\gamma}}_{s}\right) d s\right\|=\left\|f\left(\boldsymbol{z}^{b}\right)-f\left(\boldsymbol{z}^{a}\right)\right\|=\left\|\boldsymbol{x}^{b}-\boldsymbol{x}^{a}\right\|=\ell_{f}\left(\boldsymbol{\gamma}^{\star}\right)
$$

Therefore $\ell_{\mathrm{PB}}(\bar{\gamma}) \geq \ell_{\mathrm{PB}}\left(\gamma^{\star}\right)$. Hence, any pullback geodesic decodes to the straight segment:
$$
f\left(\boldsymbol{\gamma}_{s}^{\star}\right)=f\left(f^{-1}\left(\boldsymbol{x}_{s}\right)\right)=(1-s) \boldsymbol{x}^{a}+s \boldsymbol{x}^{b} .
$$

In this proposition, we emphasize that any minimizing path in the latent space $\mathcal{Z}$, when measured with the pullback metric, will always decode to a straight line in the data space $\mathcal{X}$. The reason is that the bijective decoder acts only on the ambient coordinates of $\mathcal{X}=\mathbb{R}^{d}$, regardless of whether the actual data lie on a lower-dimensional submanifold. In the denoising diffusion setting, this situation is unavoidable, since the model enforces $\operatorname{dim}(\mathcal{Z})=\operatorname{dim}(\mathcal{X})$. This stands in contrast with prior works using variational autoencoders (Arvanitidis et al., 2018), where latent geodesics live in a lower-dimensional space and can decode to curved trajectories in the ambient space.

Unless the dimension of the latent space is reduced to the intrinsic dimension of the data, the pullback metric carries no meaningful geometric information in the standard denoising diffusion setting, where $\operatorname{dim}(\mathcal{Z})=\operatorname{dim}(\mathcal{X})$.

\section*{C Proof of Proposition 5.1}

In this section, we prove Proposition 5.1. The proof consits of
1. Showing that curve energy $\mathcal{E}$ in exponential families simplifies (Appendix C.1).
2. Showing that the family of denoising distributions forms an exponential family (Appendix C.2).
3. Putting it together (Appendix C.4).

\section*{C. 1 Information geometry in exponential families}

We begin by defining an exponential family of distributions.
Definition C. 1 (Exponential Family). A parametric family of probability distributions $\{p(\cdot \mid \boldsymbol{z})\}$ is called an exponential family if it can be expressed in the form
$$
\begin{equation*}
p(\boldsymbol{x} \mid \boldsymbol{z})=h(\boldsymbol{x}) \exp \left(\boldsymbol{\eta}(\boldsymbol{z})^{\top} T(\boldsymbol{x})-\psi(\boldsymbol{z})\right), \tag{30}
\end{equation*}
$$
with $\boldsymbol{x}$ a random variable modelling the data and $\boldsymbol{z}$ the parameter of the distribution. In addition, $T(\boldsymbol{x})$ is called a sufficient statistic, $\boldsymbol{\eta}(\boldsymbol{z})$ natural (canonical) parameter, $\psi(\boldsymbol{z})$ the log-partition (cumulant) function and $h(\boldsymbol{x})$ is a base measure independent of $\boldsymbol{z}$, and
$$
\begin{equation*}
\boldsymbol{\mu}(\boldsymbol{z})=\mathbb{E}[T(\boldsymbol{x}) \mid \boldsymbol{z}] \tag{31}
\end{equation*}
$$
is the expectation parameter.
In exponential families, the Riemannian metric tensor takes a specific form, which we show now.
Proposition C. 1 (Fisher-Rao metric for an exponential family). Let $\{p(\cdot \mid \boldsymbol{z})\}$ be an exponential family. We denote $\boldsymbol{\eta}(\boldsymbol{z})$ the natural parametrisation, $T(\boldsymbol{x})$ the sufficient statistic and $\boldsymbol{\mu}(\boldsymbol{z})=\mathbb{E}[T(\boldsymbol{x}) \mid \boldsymbol{z}]$ the expectation parameters. The Fisher-Rao metric is given by:
$$
\begin{equation*}
\mathbf{G}_{\mathrm{IG}}(\boldsymbol{z})=\left(\frac{\partial \boldsymbol{\eta}(\boldsymbol{z})}{\partial \boldsymbol{z}}\right)^{\top}\left(\frac{\partial \boldsymbol{\mu}(\boldsymbol{z})}{\partial \boldsymbol{z}}\right) . \tag{32}
\end{equation*}
$$

Proof. For $p(\boldsymbol{x} \mid \boldsymbol{z})=h(\boldsymbol{x}) \exp \left(\boldsymbol{\eta}(\boldsymbol{z})^{\top} T(\boldsymbol{x})-\psi(\boldsymbol{z})\right)$, we have
$$
\begin{equation*}
\nabla_{\boldsymbol{z}} \log p(\boldsymbol{x} \mid \boldsymbol{z})=\nabla_{\boldsymbol{z}} \sum_{k} \eta^{k}(\boldsymbol{z}) T^{k}(\boldsymbol{x})-\nabla_{\boldsymbol{z}} \psi(\boldsymbol{z})=\left(\frac{\partial \boldsymbol{\eta}(\boldsymbol{z})}{\partial \boldsymbol{z}}\right)^{\top} T(\boldsymbol{x})-\nabla_{\boldsymbol{z}} \psi(\boldsymbol{z}) . \tag{33}
\end{equation*}
$$

Note that
$$
\begin{equation*}
\mathbb{E}\left[\nabla_{\boldsymbol{z}} \log p(\boldsymbol{x} \mid \boldsymbol{z}) \mid \boldsymbol{z}\right]=\int p(\boldsymbol{x} \mid \boldsymbol{z}) \nabla_{\boldsymbol{z}} \log p(\boldsymbol{x} \mid \boldsymbol{z}) d \boldsymbol{x}=\int \nabla_{\boldsymbol{z}} p(\boldsymbol{x} \mid \boldsymbol{z}) d \boldsymbol{x}=\nabla_{\boldsymbol{z}} \int p(\boldsymbol{x} \mid \boldsymbol{z}) d \boldsymbol{x}=\mathbf{0} \tag{34}
\end{equation*}
$$

Therefore, by taking the expectation of both sides of Eq. 33, we get
$$
\begin{equation*}
\nabla_{\boldsymbol{z}} \psi(\boldsymbol{z})=\left(\frac{\partial \boldsymbol{\eta}(\boldsymbol{z})}{\partial \boldsymbol{z}}\right)^{\top} \boldsymbol{\mu}(\boldsymbol{z}), \tag{35}
\end{equation*}
$$
where $\boldsymbol{\mu}(\boldsymbol{z})=\mathbb{E}[T(\boldsymbol{x}) \mid \boldsymbol{z}]$. Now we differentiate $j$-th component of both sides of Eq. 34 w.r.t $z^{i}$, and we get
$$
\begin{align*}
0 & =\frac{\partial}{\partial z^{i}} 0=\frac{\partial}{\partial z^{i}} \mathbb{E}\left[\left.\frac{\partial \log p(\boldsymbol{x} \mid \boldsymbol{z})}{\partial z^{j}} \right\rvert\, \boldsymbol{z}\right]=\frac{\partial}{\partial z^{i}} \int p(\boldsymbol{x} \mid \boldsymbol{z}) \frac{\partial \log p(\boldsymbol{x} \mid \boldsymbol{z})}{\partial z^{j}} d \boldsymbol{x} \\
& =\int \frac{\partial p(\boldsymbol{x} \mid \boldsymbol{z})}{\partial z^{i}} \frac{\partial \log p(\boldsymbol{x} \mid \boldsymbol{z})}{\partial z^{j}} d \boldsymbol{x}+\int p(\boldsymbol{x} \mid \boldsymbol{z}) \frac{\partial^{2} \log p(\boldsymbol{x} \mid \boldsymbol{z})}{\partial z^{i} \partial z^{j}} d \boldsymbol{x}  \tag{36}\\
& =\mathbb{E}\left[\left.\frac{\partial \log p(\boldsymbol{x} \mid \boldsymbol{z})}{\partial z^{i}} \frac{\partial \log p(\boldsymbol{x} \mid \boldsymbol{z})}{\partial z^{j}} \right\rvert\, \boldsymbol{z}\right]+\mathbb{E}\left[\left.\frac{\partial^{2} \log p(\boldsymbol{x} \mid \boldsymbol{z})}{\partial z^{i} \partial z^{j}} \right\rvert\, \boldsymbol{z}\right] .
\end{align*}
$$

Therefore
$$
\begin{equation*}
\left[\mathbf{G}_{\mathrm{IG}}(\boldsymbol{z})\right]_{i j}=\mathbb{E}\left[\left.\frac{\partial \log p(\boldsymbol{x} \mid \boldsymbol{z})}{\partial z^{i}} \frac{\partial \log p(\boldsymbol{x} \mid \boldsymbol{z})}{\partial z^{j}} \right\rvert\, \boldsymbol{z}\right]=-\mathbb{E}\left[\left.\frac{\partial^{2} \log p(\boldsymbol{x} \mid \boldsymbol{z})}{\partial z^{i} \partial z^{j}} \right\rvert\, \boldsymbol{z}\right] . \tag{37}
\end{equation*}
$$

Now using Eq. 33, we have
$$
\begin{equation*}
\frac{\partial^{2} \log p(\boldsymbol{x} \mid \boldsymbol{z})}{\partial z^{i} \partial z^{j}}=\frac{\partial}{\partial z^{i}}\left(\sum_{k} \frac{\partial \eta^{k}(\boldsymbol{z})}{\partial z^{j}} T^{k}(\boldsymbol{x})-\frac{\partial \psi(\boldsymbol{z})}{\partial z^{j}}\right)=\sum_{k} \frac{\partial^{2} \eta^{k}(\boldsymbol{z})}{\partial z^{i} \partial z^{j}} T^{k}(\boldsymbol{x})-\frac{\partial^{2} \psi(\boldsymbol{z})}{\partial z^{i} \partial z^{j}} . \tag{38}
\end{equation*}
$$

Therefore, from Eq. 37:
$$
\begin{equation*}
\left[\mathbf{G}_{\mathrm{IG}}(\boldsymbol{z})\right]_{i j}=\frac{\partial^{2} \psi(\boldsymbol{z})}{\partial z^{i} \partial z^{j}}-\sum_{k} \frac{\partial^{2} \eta^{k}(\boldsymbol{z})}{\partial z^{i} \partial z^{j}} \mu^{k}(\boldsymbol{z}) . \tag{39}
\end{equation*}
$$

Now using Eq. 35, we have
$$
\begin{equation*}
\frac{\partial^{2} \psi(\boldsymbol{z})}{\partial z^{i} \partial z^{j}}=\frac{\partial}{\partial z^{j}}\left(\sum_{k} \frac{\partial \eta^{k}(\boldsymbol{z})}{\partial z^{i}} \mu^{k}(\boldsymbol{z})\right)=\sum_{k} \frac{\partial^{2} \eta^{k}(\boldsymbol{z})}{\partial z^{j} \partial z^{i}} \mu^{k}(\boldsymbol{z})+\sum_{k} \frac{\partial \eta^{k}(\boldsymbol{z})}{\partial z^{i}} \frac{\partial \mu^{k}(\boldsymbol{z})}{\partial z^{j}} . \tag{40}
\end{equation*}
$$

Combining (Eq. 39) with (Eq. 40) yields:
$$
\begin{equation*}
\left[\mathbf{G}_{\mathrm{IG}}(\boldsymbol{z})\right]_{i j}=\sum_{k} \frac{\partial \eta^{k}(\boldsymbol{z})}{\partial z^{j}} \frac{\partial \mu^{k}(\boldsymbol{z})}{\partial z^{i}}=\left[\left(\frac{\partial \boldsymbol{\eta}(\boldsymbol{z})}{\partial \boldsymbol{z}}\right)^{\top}\left(\frac{\partial \boldsymbol{\mu}(\boldsymbol{z})}{\partial \boldsymbol{z}}\right)\right]_{i j} . \tag{41}
\end{equation*}
$$

Corollary C. 1 (Energy function for an exponential family). Let $\gamma:[0,1] \rightarrow \mathcal{Z}$ be a smooth curve in the parameter space $\mathcal{Z}$ of an exponential family. Then
$$
\begin{equation*}
\mathcal{E}_{\mathrm{IG}}(\boldsymbol{\gamma})=\frac{1}{2} \int_{0}^{1}\left(\frac{d}{d s} \boldsymbol{\eta}\left(\boldsymbol{\gamma}_{s}\right)\right)^{\top}\left(\frac{d}{d s} \boldsymbol{\mu}\left(\boldsymbol{\gamma}_{s}\right)\right) d s \tag{42}
\end{equation*}
$$

Proof. The energy of $\gamma$ is given by $\mathcal{E}_{\text {IG }}(\gamma)=\frac{1}{2} \int_{0}^{1}\left\|\dot{\gamma}_{s}\right\|_{\text {G }_{\text {IG }}}^{2} d s$. We replace the Riemannian metric $\mathbf{G}_{\text {IG }}$ with the previously obtained expression of the Fisher-Rao metric (Eq. 32).
$$
\begin{aligned}
\mathcal{E}_{\mathrm{IG}}(\boldsymbol{\gamma}) & =\frac{1}{2} \int_{0}^{1}\left\|\dot{\boldsymbol{\gamma}}_{s}\right\|_{\mathbf{G}_{\mathrm{IG}}}^{2} d s=\frac{1}{2} \int_{0}^{1} \dot{\boldsymbol{\gamma}}_{s}^{\top} \mathbf{G}_{\mathrm{IG}}\left(\boldsymbol{\gamma}_{s}\right) \dot{\boldsymbol{\gamma}}_{s} d s \\
& =\frac{1}{2} \int_{0}^{1} \dot{\boldsymbol{\gamma}}_{s}^{\top}\left(\frac{\partial \boldsymbol{\eta}\left(\boldsymbol{\gamma}_{s}\right)}{\partial \boldsymbol{z}}\right)^{\top}\left(\frac{\partial \boldsymbol{\mu}\left(\boldsymbol{\gamma}_{s}\right)}{\partial \boldsymbol{z}}\right) \dot{\boldsymbol{\gamma}}_{s} d s \\
& =\frac{1}{2} \int_{0}^{1}\left(\frac{\partial \boldsymbol{\eta}\left(\boldsymbol{\gamma}_{s}\right)}{\partial \boldsymbol{z}} \dot{\boldsymbol{\gamma}}_{s}\right)^{\top}\left(\frac{\partial \boldsymbol{\mu}\left(\boldsymbol{\gamma}_{s}\right)}{\partial \boldsymbol{z}} \dot{\boldsymbol{\gamma}}_{s}\right) d s \\
& =\frac{1}{2} \int_{0}^{1}\left(\frac{d}{d s} \boldsymbol{\eta}\left(\boldsymbol{\gamma}_{s}\right)\right)^{\top}\left(\frac{d}{d s} \boldsymbol{\mu}\left(\boldsymbol{\gamma}_{s}\right)\right) d s
\end{aligned}
$$

\section*{C. 2 Diffusion denoising distributions are exponential}

A key observation is that the family of denoising distributions $p\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right)$ indexed by both space and time ( $\boldsymbol{x}_{t}, t$ ) is exponential, which we prove now.
Proposition C. 2 (Exponential family of denoising). Let $\boldsymbol{x}_{t}$ be a noisy observation corresponding to diffusion time $t$, as introduced in Eq. 1. Then the denoising distribution can be written as
$$
\begin{equation*}
p\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right)=h\left(\boldsymbol{x}_{0}\right) \exp \left(\boldsymbol{\eta}\left(\boldsymbol{x}_{t}, t\right)^{\top} T\left(\boldsymbol{x}_{0}\right)-\psi\left(\boldsymbol{x}_{t}, t\right)\right), \tag{43}
\end{equation*}
$$
with $h=q$ the data distribution density, $\psi$ the log-partition function, and
$$
\begin{align*}
\boldsymbol{\eta}\left(\boldsymbol{x}_{t}, t\right) & =\left(\frac{\alpha_{t}}{\sigma_{t}^{2}} \boldsymbol{x}_{t},-\frac{\alpha_{t}^{2}}{2 \sigma_{t}^{2}}\right) \quad \text { (natural parameter) }  \tag{44}\\
T\left(\boldsymbol{x}_{0}\right) & =\left(\boldsymbol{x}_{0},\left\|\boldsymbol{x}_{0}\right\|^{2}\right) \quad \text { (sufficient statistic) }  \tag{45}\\
\boldsymbol{\mu}\left(\boldsymbol{x}_{t}, t\right) & =(\underbrace{\mathbb{E}\left[\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right]}_{\text {'space' }}, \underbrace{\frac{\sigma_{t}^{2}}{\alpha_{t}} \operatorname{div}_{\boldsymbol{x}_{t}} \mathbb{E}\left[\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right]+\left\|\mathbb{E}\left[\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right]\right\|^{2}}_{\text {'time': } \mathbb{E}\left[\left\|\boldsymbol{x}_{0}\right\|^{2} \mid \boldsymbol{x}_{t}\right]}), \tag{46}
\end{align*}
$$
which means that the family of denoising distributions $\left\{p\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right)\right\}$ indexed by $\left(\boldsymbol{x}_{t}, t\right)$ is exponential (Definition C.1).
Proof. Step 1: denoising is exponential. The denoising distribution is given by
$$
p\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right)=\frac{p\left(\boldsymbol{x}_{t} \mid \boldsymbol{x}_{0}\right) q\left(\boldsymbol{x}_{0}\right)}{p_{t}\left(\boldsymbol{x}_{t}\right)},
$$
where $q$ is the data distribution, $p\left(\boldsymbol{x}_{t} \mid \boldsymbol{x}_{0}\right)=\mathcal{N}\left(\boldsymbol{x}_{t} \mid \alpha_{t} \boldsymbol{x}_{0}, \sigma_{t}^{2} \boldsymbol{I}\right)$ is the forward density (Eq. 1), and $p_{t}\left(\boldsymbol{x}_{t}\right)= \int p\left(\boldsymbol{x}_{t} \mid \boldsymbol{x}_{0}\right) q\left(\boldsymbol{x}_{0}\right) d \boldsymbol{x}_{0}$ is the marginal distribution at time $t$. Therefore
$$
\begin{align*}
p\left(\boldsymbol{x}_{t} \mid \boldsymbol{x}_{0}\right) & =\frac{1}{\left(2 \pi \sigma_{t}^{2}\right)^{D / 2}} \exp \left(-\frac{\left\|\boldsymbol{x}_{t}-\alpha_{t} \boldsymbol{x}_{0}\right\|^{2}}{2 \sigma_{t}^{2}}\right) \\
& =\frac{1}{\left(2 \pi \sigma_{t}^{2}\right)^{D / 2}} \exp \left(-\frac{\left\|\boldsymbol{x}_{t}\right\|^{2}}{2 \sigma_{t}^{2}}+\frac{\alpha_{t}}{\sigma_{t}^{2}} \boldsymbol{x}_{t}^{\top} \boldsymbol{x}_{0}-\frac{\alpha_{t}^{2}}{2 \sigma_{t}^{2}}\left\|\boldsymbol{x}_{0}\right\|^{2}\right)  \tag{47}\\
& =\exp \left(-\frac{D}{2} \log \left(2 \pi \sigma_{t}^{2}\right)-\frac{\left\|\boldsymbol{x}_{t}\right\|^{2}}{2 \sigma_{t}^{2}}\right) \exp \left(-\frac{\alpha_{t}^{2}}{2 \sigma_{t}^{2}}\left\|\boldsymbol{x}_{0}\right\|^{2}+\frac{\alpha_{t}}{\sigma_{t}^{2}} \boldsymbol{x}_{t}^{\top} \boldsymbol{x}_{0}\right)
\end{align*}
$$

By substituting into the denoising density, we get
$$
\begin{align*}
p\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right) & =q\left(\boldsymbol{x}_{0}\right) \exp \left\{-\frac{\alpha_{t}^{2}}{2 \sigma_{t}^{2}}\left\|\boldsymbol{x}_{0}\right\|^{2}+\frac{\alpha_{t}}{\sigma_{t}^{2}} \boldsymbol{x}_{t}^{\top} \boldsymbol{x}_{0}-\left(\log p_{t}\left(\boldsymbol{x}_{t}\right)+\frac{D}{2} \log \left(2 \pi \sigma_{t}^{2}\right)+\frac{\left\|\boldsymbol{x}_{t}\right\|^{2}}{2 \sigma_{t}^{2}}\right)\right\}  \tag{48}\\
& =h\left(\boldsymbol{x}_{0}\right) \exp \left(\boldsymbol{\eta}\left(\boldsymbol{x}_{t}, t\right)^{\top} T\left(\boldsymbol{x}_{0}\right)-\psi\left(\boldsymbol{x}_{t}, t\right)\right)
\end{align*}
$$
where
$$
\begin{align*}
\boldsymbol{\eta}\left(\boldsymbol{x}_{t}, t\right) & =\left(\frac{\alpha_{t}}{\sigma_{t}^{2}} \boldsymbol{x}_{t},-\frac{\alpha_{t}^{2}}{2 \sigma_{t}^{2}}\right) & & \in \mathbb{R}^{D+1}  \tag{49}\\
T\left(\boldsymbol{x}_{0}\right) & =\left(\boldsymbol{x}_{0},\left\|\boldsymbol{x}_{0}\right\|^{2}\right) & & \in \mathbb{R}^{D+1}  \tag{50}\\
h\left(\boldsymbol{x}_{0}\right) & =q\left(\boldsymbol{x}_{0}\right) & & \in \mathbb{R}  \tag{51}\\
\psi\left(\boldsymbol{x}_{t}, t\right) & =\log p_{t}\left(\boldsymbol{x}_{t}\right)+\frac{D}{2} \log \left(2 \pi \sigma_{t}^{2}\right)+\frac{\left\|\boldsymbol{x}_{t}\right\|^{2}}{2 \sigma_{t}^{2}} & & \in \mathbb{R} \tag{52}
\end{align*}
$$
which proves that the denoising distributions form an exponential family.
Step 2: deriving the expectation parameter $\boldsymbol{\mu}$. The expectation parameter $\boldsymbol{\mu}$ is given by $\boldsymbol{\mu}\left(\boldsymbol{x}_{t}, t\right)=\mathbb{E}\left[T\left(\boldsymbol{x}_{0}\right) \mid \boldsymbol{x}_{t}\right]= \left(\mathbb{E}\left[\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right], \mathbb{E}\left[\left\|\boldsymbol{x}_{0}\right\|^{2} \mid \boldsymbol{x}_{t}\right]\right)$. The denoising covariance is known (Meng et al., 2021):
$$
\begin{equation*}
\operatorname{Cov}\left[\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right]=\frac{\sigma_{t}^{2}}{\alpha_{t}^{2}}\left(\boldsymbol{I}+\sigma_{t}^{2} \nabla_{\boldsymbol{x}_{t}}^{2} \log p_{t}\left(\boldsymbol{x}_{t}\right)\right) \tag{53}
\end{equation*}
$$

Therefore, from the definition of conditional variance, we can deduce the second denoising moment:
$$
\begin{align*}
\mathbb{E}\left[\left\|\boldsymbol{x}_{0}\right\|^{2} \mid \boldsymbol{x}_{t}\right] & =\mathbb{E}\left[\left\|\boldsymbol{x}_{0}-\mathbb{E}\left[\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right]\right\|^{2} \mid \boldsymbol{x}_{t}\right]+\left\|\mathbb{E}\left[\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right]\right\|^{2} \\
& =\operatorname{Tr}\left(\operatorname{Cov}\left[\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right]\right)+\left\|\mathbb{E}\left[\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right]\right\|^{2} \\
& \stackrel{(53)}{=} \frac{\sigma_{t}^{2}}{\alpha_{t}^{2}}\left(D+\sigma_{t}^{2} \Delta \log p_{t}\left(\boldsymbol{x}_{t}\right)\right)+\left\|\mathbb{E}\left[\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right]\right\|^{2}  \tag{54}\\
& =\frac{\sigma_{2}^{2}}{\alpha_{t}} \operatorname{div}_{\boldsymbol{x}_{t}}\left(\frac{\boldsymbol{x}_{t}+\sigma_{t}^{2} \nabla_{\boldsymbol{x}_{t}} \log p_{t}\left(\boldsymbol{x}_{t}\right)}{\alpha_{t}}\right)+\left\|\mathbb{E}\left[\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right]\right\|^{2} \\
& =\frac{\sigma_{t}^{2}}{\alpha_{t}} \operatorname{div}_{\boldsymbol{x}_{t}} \mathbb{E}\left[\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right]+\left\|\mathbb{E}\left[\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right]\right\|^{2}
\end{align*}
$$
where we used the fact that (Efron, 2011)
$$
\begin{equation*}
\mathbb{E}\left[\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right]=\frac{\boldsymbol{x}_{t}+\sigma_{t}^{2} \nabla_{\boldsymbol{x}_{t}} \log p_{t}\left(\boldsymbol{x}_{t}\right)}{\alpha_{t}} . \tag{55}
\end{equation*}
$$

Together, we have
$$
\begin{equation*}
\boldsymbol{\mu}\left(\boldsymbol{x}_{t}, t\right)=\left(\mathbb{E}\left[\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right], \frac{\sigma_{t}^{2}}{\alpha_{t}} \operatorname{div}_{\boldsymbol{x}_{t}} \mathbb{E}\left[\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right]+\left\|\mathbb{E}\left[\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right]\right\|^{2}\right) \tag{56}
\end{equation*}
$$

\section*{C. 3 Boltzmann denoising distributions}

Note that, if the data distribution is Boltzmann, i.e. $q\left(\boldsymbol{x}_{0}\right) \propto \exp \left(-U\left(\boldsymbol{x}_{0}\right)\right)$ for some energy function $U$, we have:
$$
\begin{aligned}
p\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right) & \propto q\left(\boldsymbol{x}_{0}\right) p\left(\boldsymbol{x}_{t} \mid \boldsymbol{x}_{0}\right) \propto \exp \left(-\left(U\left(\boldsymbol{x}_{0}\right)\right) \exp \left(-\frac{\left\|\boldsymbol{x}_{t}-\alpha_{t} \boldsymbol{x}_{0}\right\|^{2}}{2 \sigma_{t}^{2}}\right)\right. \\
& =\exp \left(-U\left(\boldsymbol{x}_{0}\right)-\frac{1}{2} \operatorname{SNR}(t)\left\|\boldsymbol{x}_{0}-\boldsymbol{x}_{t} / \alpha_{t}\right\|^{2}\right)
\end{aligned}
$$

This implies that $p\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right)$ is also a Boltzmann distribution with $p\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right) \propto \exp \left(-U\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right)\right)$ for
$$
\begin{equation*}
U\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right)=U\left(\boldsymbol{x}_{0}\right)+\frac{1}{2} \mathrm{SNR}(t)\left\|\boldsymbol{x}_{0}-\boldsymbol{x}_{t} / \alpha_{t}\right\|^{2} . \tag{57}
\end{equation*}
$$

\section*{C. 4 Putting it together: Proposition 5.1}

The claim of Proposition 5.1 follows from Proposition C. 2 and Corollary C.1.

\section*{D Kullback-Leibler divergence in exponential families}

For any distribution family, the Fisher-Rao metric is the local approximation of the KL divergence, i.e (Arvanitidis et al., 2022):
$$
\mathrm{KL}\left(p\left(\cdot \mid \boldsymbol{z}_{1}\right) \| p\left(\cdot \mid \boldsymbol{z}_{2}\right)\right) \approx \frac{1}{2}\left(\boldsymbol{z}_{1}-\boldsymbol{z}_{2}\right)^{\top} \mathbf{G}_{\mathrm{IG}}\left(\boldsymbol{z}_{1}\right)\left(\boldsymbol{z}_{1}-\boldsymbol{z}_{2}\right) .
$$

In the case of exponential families, we have $\mathbf{G}_{\text {IG }}(\boldsymbol{z})=\left(\frac{\partial \boldsymbol{\eta}(\boldsymbol{z})}{\partial \boldsymbol{z}}\right)^{\top}\left(\frac{\partial \boldsymbol{\mu}(\boldsymbol{z})}{\partial \boldsymbol{z}}\right)$, and thus we can write
$$
\begin{aligned}
\mathrm{KL}\left(p\left(\cdot \mid \boldsymbol{z}_{1}\right) \| p\left(\cdot \mid \boldsymbol{z}_{2}\right)\right) & \approx \frac{1}{2}\left(\boldsymbol{z}_{1}-\boldsymbol{z}_{2}\right)^{\top}\left(\frac{\partial \boldsymbol{\eta}\left(\boldsymbol{z}_{1}\right)}{\partial \boldsymbol{z}}\right)^{\top}\left(\frac{\partial \boldsymbol{\mu}\left(\boldsymbol{z}_{1}\right)}{\partial \boldsymbol{z}}\right)\left(\boldsymbol{z}_{1}-\boldsymbol{z}_{2}\right) \\
& \approx \frac{1}{2}\left(\boldsymbol{\eta}\left(\boldsymbol{z}_{1}\right)-\boldsymbol{\eta}\left(\boldsymbol{z}_{2}\right)\right)^{\top}\left(\boldsymbol{\mu}\left(\boldsymbol{z}_{1}\right)-\boldsymbol{\mu}\left(\boldsymbol{z}_{2}\right)\right)
\end{aligned}
$$

It turns out that the RHS always corresponds to a notion of distribution divergence (not only when $\boldsymbol{z}_{1}$ and $\boldsymbol{z}_{2}$ are close together), namely the symmetrized Kullback-Leibler divergence:
$$
\begin{equation*}
\mathrm{KL}^{\mathrm{S}}(p \| q):=\frac{1}{2}(\mathrm{KL}(p \| q)+\mathrm{KL}(q \| p)) . \tag{58}
\end{equation*}
$$

Lemma D. 1 (KL in exponential families). Let $\mathcal{P}=\{p(\cdot \mid \boldsymbol{z}) \mid \boldsymbol{z} \in \mathcal{Z}\}$ be an exponential family with $p(\boldsymbol{x} \mid \boldsymbol{z})= h(\boldsymbol{x}) \exp \left(\boldsymbol{\eta}(\boldsymbol{z})^{\top} T(\boldsymbol{x})-\psi(\boldsymbol{z})\right)$, and $\boldsymbol{\mu}(\boldsymbol{z})=\mathbb{E}_{\boldsymbol{x} \sim p(\boldsymbol{x} \mid \boldsymbol{z})}[T(\boldsymbol{x})]$. Then
$$
\begin{equation*}
\mathrm{KL}\left(\boldsymbol{z}_{1} \| \boldsymbol{z}_{2}\right)=\left(\boldsymbol{\eta}\left(\boldsymbol{z}_{1}\right)-\boldsymbol{\eta}\left(\boldsymbol{z}_{2}\right)\right)^{\top} \boldsymbol{\mu}\left(\boldsymbol{z}_{1}\right)-\psi\left(\boldsymbol{z}_{1}\right)+\psi\left(\boldsymbol{z}_{2}\right), \tag{59}
\end{equation*}
$$
where we abuse notation and write $\operatorname{KL}\left(\boldsymbol{z}_{1}| | \boldsymbol{z}_{2}\right)$ instead of $\operatorname{KL}\left(p\left(\cdot \mid \boldsymbol{z}_{1}\right)\left|\mid p\left(\cdot \mid \boldsymbol{z}_{2}\right)\right)\right.$.

Proof.
$$
\begin{aligned}
\mathrm{KL}\left(\boldsymbol{z}_{1} \| \boldsymbol{z}_{2}\right) & =\mathbb{E}_{\boldsymbol{x} \sim p\left(\boldsymbol{x} \mid \boldsymbol{z}_{1}\right)}\left[\log p\left(\boldsymbol{x} \mid \boldsymbol{z}_{1}\right)-\log p\left(\boldsymbol{x} \mid \boldsymbol{z}_{2}\right)\right] \\
& =\mathbb{E}_{\boldsymbol{x} \sim p\left(\boldsymbol{x} \mid \boldsymbol{z}_{1}\right)}\left[\boldsymbol{\eta}\left(\boldsymbol{z}_{1}\right)^{\top} T(\boldsymbol{x})-\boldsymbol{\eta}\left(\boldsymbol{z}_{2}\right)^{\top} T(\boldsymbol{x})-\psi\left(\boldsymbol{z}_{1}\right)+\psi\left(\boldsymbol{z}_{2}\right)\right] \\
& =\left(\boldsymbol{\eta}\left(\boldsymbol{z}_{1}\right)-\boldsymbol{\eta}\left(\boldsymbol{z}_{2}\right)\right)^{\top} \mathbb{E}_{\boldsymbol{x} \sim p\left(\boldsymbol{x} \mid \boldsymbol{z}_{1}\right)}[T(\boldsymbol{x})]-\psi\left(\boldsymbol{z}_{1}\right)+\psi\left(\boldsymbol{z}_{2}\right) \\
& =\left(\boldsymbol{\eta}\left(\boldsymbol{z}_{1}\right)-\boldsymbol{\eta}\left(\boldsymbol{z}_{2}\right)\right)^{\top} \boldsymbol{\mu}\left(\boldsymbol{z}_{1}\right)-\psi\left(\boldsymbol{z}_{1}\right)+\psi\left(\boldsymbol{z}_{2}\right) .
\end{aligned}
$$

Lemma D. 2 (Symmetrized KL in exponential families). With assumptions of Lemma D.1, we have
$$
\begin{equation*}
\mathrm{KL}^{\mathrm{S}}\left(\boldsymbol{z}_{1} \| \boldsymbol{z}_{2}\right)=\frac{1}{2}\left(\boldsymbol{\eta}\left(\boldsymbol{z}_{1}\right)-\boldsymbol{\eta}\left(\boldsymbol{z}_{2}\right)\right)^{\top}\left(\boldsymbol{\mu}\left(\boldsymbol{z}_{1}\right)-\boldsymbol{\mu}\left(\boldsymbol{z}_{2}\right)\right) . \tag{60}
\end{equation*}
$$

Proof.
$$
\begin{aligned}
& 2 \mathrm{KL}^{\mathrm{S}}\left(\boldsymbol{z}_{1} \mid \boldsymbol{z}_{2}\right)=\mathrm{KL}\left(\boldsymbol{z}_{1}| | \boldsymbol{z}_{2}\right)+\mathrm{KL}\left(\boldsymbol{z}_{2}| | \boldsymbol{z}_{1}\right) \\
& =\left(\boldsymbol{\eta}\left(\boldsymbol{z}_{1}\right)-\boldsymbol{\eta}\left(\boldsymbol{z}_{2}\right)\right)^{\top} \boldsymbol{\mu}\left(\boldsymbol{z}_{1}\right)-\psi\left(\boldsymbol{z}_{1}\right)+\psi\left(\boldsymbol{z}_{2}\right)+\left(\boldsymbol{\eta}\left(\boldsymbol{z}_{2}\right)-\boldsymbol{\eta}\left(\boldsymbol{z}_{1}\right)\right)^{\top} \boldsymbol{\mu}\left(\boldsymbol{z}_{2}\right)-\psi\left(\boldsymbol{z}_{2}\right)+\psi\left(\boldsymbol{z}_{1}\right) \\
& =\left(\boldsymbol{\eta}\left(\boldsymbol{z}_{1}\right)-\boldsymbol{\eta}\left(\boldsymbol{z}_{2}\right)\right)^{\top}\left(\boldsymbol{\mu}\left(\boldsymbol{z}_{1}\right)-\boldsymbol{\mu}\left(\boldsymbol{z}_{2}\right)\right) .
\end{aligned}
$$ \(\square\)

The formula for KL in Lemma D. 1 is not useful in practice, because it requires knowing $\psi(\boldsymbol{z})$, which can be unknown or expensive to evaluate. However, the gradients with respect to both arguments depend only on $\boldsymbol{\eta}$ and $\boldsymbol{\mu}$.
Lemma D. 3 (KL gradients). With assumptions of Lemma D.1, we have for any $\boldsymbol{z}_{1}, \boldsymbol{z}_{2}$
$$
\begin{align*}
& \nabla_{\boldsymbol{z}_{1}} \operatorname{KL}\left(\boldsymbol{z}_{1} \| \boldsymbol{z}_{2}\right)=\frac{\partial \boldsymbol{\mu}\left(\boldsymbol{z}_{1}\right)^{\top}}{\partial \boldsymbol{z}}\left(\boldsymbol{\eta}\left(\boldsymbol{z}_{1}\right)-\boldsymbol{\eta}\left(\boldsymbol{z}_{2}\right)\right) \\
& \nabla_{\boldsymbol{z}_{2}} \operatorname{KL}\left(\boldsymbol{z}_{1} \| \boldsymbol{z}_{2}\right)=\frac{\partial \boldsymbol{\eta}\left(\boldsymbol{z}_{2}\right)^{\top}}{\partial \boldsymbol{z}}\left(\boldsymbol{\mu}\left(\boldsymbol{z}_{2}\right)-\boldsymbol{\mu}\left(\boldsymbol{z}_{1}\right)\right) \tag{61}
\end{align*}
$$

Proof. The proof is a straightforward calculation using Lemma D. 1 and Eq. 35. We have
$$
\begin{aligned}
\nabla_{\boldsymbol{z}_{1}} \mathrm{KL}\left(\boldsymbol{z}_{1} \| \boldsymbol{z}_{2}\right) & =\nabla_{\boldsymbol{z}_{1}}\left(\left(\boldsymbol{\eta}\left(\boldsymbol{z}_{1}\right)-\boldsymbol{\eta}\left(\boldsymbol{z}_{2}\right)\right)^{\top} \boldsymbol{\mu}\left(\boldsymbol{z}_{1}\right)-\psi\left(\boldsymbol{z}_{1}\right)+\psi\left(\boldsymbol{z}_{2}\right)\right) \\
& =\frac{\partial \boldsymbol{\eta}\left(\boldsymbol{z}_{1}\right)^{\top}}{\partial \boldsymbol{z}} \boldsymbol{\mu}\left(\boldsymbol{z}_{1}\right)+\frac{\partial \boldsymbol{\mu}\left(\boldsymbol{z}_{1}\right)^{\top}}{\partial \boldsymbol{z}}\left(\boldsymbol{\eta}\left(\boldsymbol{z}_{1}\right)-\boldsymbol{\eta}\left(\boldsymbol{z}_{2}\right)\right)-\nabla_{\boldsymbol{z}} \psi\left(\boldsymbol{z}_{1}\right) \\
& \stackrel{(35)}{=} \frac{\partial \boldsymbol{\eta}\left(\boldsymbol{z}_{1}\right)^{\top}}{\partial \boldsymbol{z}} \boldsymbol{\mu}\left(\boldsymbol{z}_{1}\right)+\frac{\partial \boldsymbol{\mu}\left(\boldsymbol{z}_{1}\right)^{\top}}{\partial \boldsymbol{z}}\left(\boldsymbol{\eta}\left(\boldsymbol{z}_{1}\right)-\boldsymbol{\eta}\left(\boldsymbol{z}_{2}\right)\right)-\frac{\partial \boldsymbol{\eta}\left(\boldsymbol{z}_{1}\right)^{\top}}{\partial \boldsymbol{z}} \boldsymbol{\mu}\left(\boldsymbol{z}_{1}\right) \\
& =\frac{\partial \boldsymbol{\mu}\left(\boldsymbol{z}_{1}\right)^{\top}}{\partial \boldsymbol{z}}\left(\boldsymbol{\eta}\left(\boldsymbol{z}_{1}\right)-\boldsymbol{\eta}\left(\boldsymbol{z}_{2}\right)\right)
\end{aligned}
$$
and
$$
\begin{aligned}
\nabla_{\boldsymbol{z}_{2}} \mathrm{KL}\left(\boldsymbol{z}_{1} \| \boldsymbol{z}_{2}\right) & =\nabla_{\boldsymbol{z}_{2}}\left(\left(\boldsymbol{\eta}\left(\boldsymbol{z}_{1}\right)-\boldsymbol{\eta}\left(\boldsymbol{z}_{2}\right)\right)^{\top} \boldsymbol{\mu}\left(\boldsymbol{z}_{1}\right)-\psi\left(\boldsymbol{z}_{1}\right)+\psi\left(\boldsymbol{z}_{2}\right)\right) \\
& \stackrel{(35)}{=}-{\frac{\partial \boldsymbol{\eta}\left(\boldsymbol{z}_{2}\right)^{\top}}{\partial \boldsymbol{z}}}^{\top} \boldsymbol{\mu}\left(\boldsymbol{z}_{1}\right)+{\frac{\partial \boldsymbol{\eta}\left(\boldsymbol{z}_{2}\right)^{\top}}{\partial \boldsymbol{z}}}^{\top} \boldsymbol{\mu}\left(\boldsymbol{z}_{2}\right)=\frac{\partial \boldsymbol{\eta}\left(\boldsymbol{z}_{2}\right)^{\top}}{\partial \boldsymbol{z}}\left(\boldsymbol{\mu}\left(\boldsymbol{z}_{2}\right)-\boldsymbol{\mu}\left(\boldsymbol{z}_{1}\right)\right)
\end{aligned}
$$ \(\square\)

Knowing the gradients allows for estimating the KL divergence along a curve without knowing $\psi$.
Proposition D. 1 (KL along a curve). Let $\gamma:[0,1] \rightarrow \mathcal{Z}$ be a smooth denoising curve, and $z^{*} \in \mathcal{Z}$. Then:
$$
\begin{align*}
& \mathrm{KL}\left(\boldsymbol{\gamma}_{s} \| \boldsymbol{z}^{*}\right)=\mathrm{KL}\left(\boldsymbol{\gamma}_{0} \| \boldsymbol{z}^{*}\right)+\int_{0}^{s}\left(\frac{d}{d u} \boldsymbol{\mu}\left(\boldsymbol{\gamma}_{u}\right)\right)^{\top}\left(\boldsymbol{\eta}\left(\boldsymbol{\gamma}_{u}\right)-\boldsymbol{\eta}\left(\boldsymbol{z}^{*}\right)\right) d u \\
& \mathrm{KL}\left(\boldsymbol{z}^{*} \| \boldsymbol{\gamma}_{s}\right)=\mathrm{KL}\left(\boldsymbol{z}^{*} \| \boldsymbol{\gamma}_{0}\right)+\int_{0}^{s}\left(\frac{d}{d u} \boldsymbol{\eta}\left(\boldsymbol{\gamma}_{u}\right)\right)^{\top}\left(\boldsymbol{\mu}\left(\boldsymbol{\gamma}_{u}\right)-\boldsymbol{\mu}\left(\boldsymbol{z}^{*}\right)\right) d u \tag{62}
\end{align*}
$$

Proof.
$$
\begin{array}{lr}
\operatorname{KL}\left(\boldsymbol{\gamma}_{s} \| \boldsymbol{z}^{*}\right)-\operatorname{KL}\left(\boldsymbol{\gamma}_{0} \| \boldsymbol{z}^{*}\right)= & \text { // Fundamental theorem of calculus } \\
\quad=\int_{0}^{s} \frac{d}{d u}\left(\operatorname{KL}\left(\boldsymbol{\gamma}_{u} \| \boldsymbol{z}^{*}\right)\right) d u & \text { // Chain rule } \\
\quad=\int_{0}^{s} \nabla_{\boldsymbol{z}_{1}} \operatorname{KL}\left(\boldsymbol{\gamma}_{u} \| \boldsymbol{z}^{*}\right)^{\top} \dot{\boldsymbol{\gamma}}_{u} d u & \text { // Lemma D.3 } \\
=\int_{0}^{s}\left(\frac{\partial \boldsymbol{\mu}\left(\boldsymbol{\gamma}_{u}\right)}{\partial \boldsymbol{z}} \dot{\boldsymbol{\gamma}}_{u}\right)^{\top}\left(\boldsymbol{\eta}\left(\boldsymbol{\gamma}_{u}\right)-\boldsymbol{\eta}\left(\boldsymbol{z}^{*}\right)\right) d u & \text { // Chain rule. }
\end{array}
$$

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/2cafeaa0-70df-4e35-a8ea-a66a0559b937-19.jpg?height=717&width=1636&top_left_y=244&top_left_x=244}
\captionsetup{labelformat=empty}
\caption{Figure 8: Comparison of DiffED with other image similarity metrics. Each row corresponds to a different image similarity measure, and images and sorted by their similarity, from most similar (left) to most dissimilar (right). Images shown are 20 random image pairs from class "Space bar".}
\end{figure}

Using the same reasoning we have
$$
\begin{aligned}
\mathrm{KL} & \left(\boldsymbol{z}^{*} \| \boldsymbol{\gamma}_{s}\right)-\mathrm{KL}\left(\boldsymbol{z}^{*} \| \boldsymbol{\gamma}_{0}\right)=\int_{0}^{s} \frac{d}{d u}\left(\mathrm{KL}\left(\boldsymbol{z}^{*} \| \boldsymbol{\gamma}_{u}\right)\right) d u \\
& =\int_{0}^{s} \nabla_{\boldsymbol{z}_{2}} \mathrm{KL}\left(\boldsymbol{z}^{*} \| \gamma_{u}\right)^{\top} \dot{\boldsymbol{\gamma}}_{u} d u \\
& =\int_{0}^{s}\left(\frac{\partial \boldsymbol{\eta}\left(\boldsymbol{\gamma}_{u}\right)}{\partial \boldsymbol{z}} \dot{\boldsymbol{\gamma}}_{u}\right)^{\top}\left(\boldsymbol{\mu}\left(\boldsymbol{\gamma}_{u}\right)-\boldsymbol{\mu}\left(\boldsymbol{z}^{*}\right)\right) d u \\
& =\int_{0}^{s}\left(\frac{d}{d u} \boldsymbol{\eta}\left(\boldsymbol{\gamma}_{u}\right)\right)^{\top}\left(\boldsymbol{\mu}\left(\boldsymbol{\gamma}_{u}\right)-\boldsymbol{\mu}\left(\boldsymbol{z}^{*}\right)\right) d u
\end{aligned}
$$ \(\square\)

\section*{E Qualitative example of DiffED}

In Fig. 8, we compare the Diffusion Edit Distance (DiffED) (Section 6.2) with LPIPS (Zhang et al., 2018), SSIM (Wang et al., 2004), and the Euclidean distance between images. Specifically, for the ImageNet class "Space bar", we generate 20 random image pairs, estimate the similarity of each pair with each method, and rank the pairs according to each method, from the most similar to the most dissimilar.

\section*{F Experimental details}

\section*{F. 1 Toy Gaussian mixture}

For the experiments with a 1D Gaussian mixture (Fig. 1, and Fig. 3 left), we define the data distribution as $p_{0}= \sum_{i=1}^{3} \pi_{i} \mathcal{N}\left(\mu_{i}, \sigma^{2}\right)$ with $\mu_{1}=-2.5, \mu_{2}=0.5, \mu_{3}=2.5, \pi_{1}=0.275, \pi_{2}=0.45, \pi_{3}=0.275$, and $\sigma=0.75$. We specify the forward process (Eq. 1) as Variance-Preserving (Song et al., 2021), i.e. satisfying $\alpha_{t}^{2}+\sigma_{t}^{2}=1$, and assume as $\log$-SNR linear noise schedule, i.e. $\lambda_{t}=\log \operatorname{SNR}(t)=\lambda_{\text {max }}+\left(\lambda_{\text {min }}-\lambda_{\text {max }}\right) t$ for $\lambda_{\text {min }}=-10, \lambda_{\text {max }}=10$. Which implies: $\alpha_{t}^{2}=\operatorname{sigmoid}\left(\lambda_{t}\right), \sigma_{t}^{2}=\operatorname{sigmoid}\left(-\lambda_{t}\right)$.

Since $p_{0}$ is a Gaussian mixture, all marginals $p_{t}$ are also Gaussian mixtures, and training a diffusion model is unnecessary, as the score function $\nabla_{\boldsymbol{x}} \log _{t}(\boldsymbol{x})$ is known analytically. In this example, the data is 1D, and the spacetime is 2D.

To generate Fig. 1 we estimate the geodesic between $\boldsymbol{z}_{1}=(-2.3,0.35)$, and $\boldsymbol{z}_{2}=(2,0.4)$ by parametrizing $\boldsymbol{\gamma}$ with a cubic spline (Arvanitidis et al., 2022) with two nodes, and discretizing it into $N=128$ points and taking 1000 optimization steps with Adam optimizer and learning rate $\eta=0.1$, which takes a few seconds on an M1 CPU.

To generate Fig. 3 left, we generate 3 PF-ODE sampling trajectories starting from $x=1,0,-1$ using an Euler solver with 512 solver steps. We solve only until $t=t_{\min }=0.1$ (as opposed to $t=0$ ), because for $t \approx 0$, the denoising distributions $p\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right)$ become closer to Dirac delta distributions $\delta_{\boldsymbol{x}_{t}}$, which makes the energies very large. For each sampling trajectory, we take the endpoints $\left(x_{1}, 1\right),\left(x_{t_{\text {min }}}, t_{\text {min }}\right)$ and estimate the geodesic between them using Proposition 5.1 with a cubic spline with 10 nodes, discretizing it into 512 points, and taking 2000 gradient steps of AdamW optimizer with learning rate $\eta=0.01$. This takes roughly 10 seconds on an M1 CPU.

\section*{F. 2 Image data}

For all experiments on image data, we use the pretrained EDM2 model trained on ImageNet512 (Karras et al., 2024) (specifically, the edm2-img512-xxl-fid checkpoint), which is a Variance-Exploding model, i.e. $\alpha_{t}=1$, and using the noise schedule $\sigma_{t}=t$. It is a latent diffusion model, using a fixed StabilityVAE (Rombach et al., 2022) as the encoder/decoder.

Image interpolations. To interpolate between to images, we encode them with StabilityVAE to obtain two latent codes $\boldsymbol{x}_{0}^{1}, \boldsymbol{x}_{0}^{2}$, and encode them both with PF-ODE (Eq. 3) from $t=0$ to $t=t_{\text {min }}=0.368$, corresponding to $\log \operatorname{SNR}\left(t_{\text {min }}\right)=2$. This is to avoid very high values of energy for $t \approx 0$. We then optimize the geodesic between $\left(\boldsymbol{x}_{t_{\text {min }}}^{1}, t_{\text {min }}\right)$ and $\left(\boldsymbol{x}_{t_{\text {min }}}^{2}, t_{\text {min }}\right)$ by parametrizing it with a cubic spline with 8 nodes, and minimizing the energy defined in Proposition 5.1 using AdamW optimizer with learning rate $\eta=0.1$. The curve is discretized into 16 points, and optimized for 200 gradient steps, which takes roughly 6 minutes on an A100 NVIDIA GPU per interpolation image pair.
Note that in our experiments, we used the largest release model edm2-img512-xxl-fid. The image interpolation time can be reduced to roughly a minute by considering the smallest model version edm2-img512-xs-fid.

PF-ODE sampling trajectories. To generate PF-ODE sampling trajectories, we use the 2nd order Heun solver (Karras et al., 2022) with 64 steps, and solve from $t=80$ to $t_{\text {min }}=0.135$ corresponding to $\log \operatorname{SNR}\left(t_{\text {min }}\right)=4$. This is to avoid instabilities for small $t$. We parametrize the geodesic directly with the entire sampling trajectory $\gamma_{t}=\left(\boldsymbol{x}_{t}, t\right)$ for $t=T, \ldots, t_{\min }$, where the $t$ schedule corresponds to EDM2 model's sampling schedule.
We then fix the endpoints of the trajectory, and optimize the intermediate points using AdamW optimizer with learning rate $\eta=0.0001$ (larger learning rates lead to NaN values) and take 600 optimization steps. This procedure took roughly 2 hours on an A100 NVIDIA GPU per a single sampling trajectory.
To visualize intermediate noisy images at diffusion time $t$, we rescale them with $\frac{\sigma_{\text {data }}}{\sqrt{\sigma_{\text {data }}^{2}+\sigma_{t}^{2}}}$ before decoding with the VAE deocoder, to avoid unrealistic color values, where we set $\sigma_{\mathrm{data}}=0.5$ as in Karras et al. (2022).

\section*{F. 3 Molecular data}

Approximating the base energy function with a neural network. We follow Holdijk et al. (2023) and represent the energy function of Alanine Dipeptide in the space of two dihedral angles $\phi, \psi \in[-\pi, \pi)$. We use the code provided by the authors at github.com/LarsHoldijk/SOCTransitionPaths, which estimates the energy $U(\phi, \psi)$. However, even though the values of the energy $U$ looked reasonably, we found that the provided implementation of $\frac{\partial U}{\partial \phi}$, and $\frac{\partial U}{\partial \psi}$ yielded unstable results due to discontinuities.

Instead, we trained an auxiliary feedforward neural network $U_{\theta}$ to approximate $U$. We parametrized with two hidden layers of size 64 with SiLU activation functions, and trained it on a uniformly discretized grid $[-\pi, \pi] \times[-\pi, \pi]$ into 16384 points. We trained the model with mean squared error for 8192 steps using Adam optimizer with a learning rate $\eta=0.001$ until the model converged to an average loss of $\approx 1.5$. This took approximately two and a half minutes on an M1 CPU. In the subsequent experiments, we estimate $\nabla_{\boldsymbol{x}} U(\boldsymbol{x})$ with automatic differentiation on the trained auxiliary model.

Generating samples from the energy landscape. To generate samples from the data distribution $p_{0}\left(\boldsymbol{x}_{0}\right) \propto \exp \left(-U\left(\boldsymbol{x}_{0}\right)\right)$, we initialize the samples uniformly on the $[-\pi, \pi] \times[-\pi, \pi]$ grid, and use Langevin dynamics
$$
\begin{equation*}
d \boldsymbol{x}=-\nabla_{\boldsymbol{x}} U(\boldsymbol{x}) d t+\sqrt{2} d W_{t} \tag{63}
\end{equation*}
$$
with the Euler-Maruyama solver for $d t=0.001$ and $N=1000$ steps.

Training a diffusion model on the energy landscape. To estimate the spacetime geodesics, we need a denoiser network approximating the denoising mean $\hat{\boldsymbol{x}}_{0}\left(\boldsymbol{x}_{t}, t\right) \approx \mathbb{E}\left[\boldsymbol{x}_{0} \mid \boldsymbol{x}_{t}\right]$. We parametrize the denoiser network with
```
from ddpm import MLP
model = MLP(
    hidden_size=128,
    hidden_layers=3,
    emb_size=128,
    time_emb="sinusoidal",
    input_emb="sinusoidal"
)
```

using the TinyDiffusion implementation github.com/tanelp/tiny-diffusion. We trained the model using the weighted denoising loss: $w\left(\lambda_{t}\right)\left\|\hat{\boldsymbol{x}}_{0}\left(\boldsymbol{x}_{t}, t\right)-\boldsymbol{x}_{0}\right\|^{2}$ with a weight function $w\left(\lambda_{t}\right)=\sqrt{\operatorname{sigmoid}\left(\lambda_{t}+2\right)}$ and an adaptive noise schedule (Kingma and Gao, 2023). We train the model for 4000 steps using the AdamW optimizer with learning rate $\eta=0.001$, which took roughly 1 minute on an M1 CPU.

Spacetime geodesics. With a trained denoiser $\hat{\boldsymbol{x}}_{0}\left(\boldsymbol{x}_{t}, t\right)$, we can estimate the expectation parameter $\boldsymbol{\mu}(\mathrm{Eq}$. 16) and thus curves energies in the spacetime geometry (Proposition 5.1).
In Section 6.3, we want to interpolate between two low-energy states: $\boldsymbol{x}_{0}^{1}=(-2.55,2.7)$ and $\boldsymbol{x}_{0}^{2}=(0.95,-0.4)$. To avoid instabilities for $t \approx 0$, we represent them on the spacetime manifold as $\boldsymbol{z}_{1}=\left(-2.55,2.7, t_{\text {min }}\right)$, and $\boldsymbol{z}_{2}=\left(0.95,-0.4, t_{\text {min }}\right)$, where $\log \operatorname{SNR}\left(t_{\text {min }}\right)=7$. We then approximate the geodesic between them, by parametrizing $\gamma$ as a cubic spline with 10 nodes and fixed endpoints $\gamma_{0}=\boldsymbol{z}_{1}$, and $\gamma_{1}=\boldsymbol{z}_{2}$ and discretize it into 512 points. We then optimize it by minimizing Proposition 5.1 with the Adam optimizer with learning rate $\eta=0.1$ and take 10000 optimization steps, which takes roughly 6 minutes on an M1 CPU.

Annealed Langevin dynamics. To generate transition paths, we use Annealed Langevin dynamics (Algorithm 1) with the geodesic discretized into $N=512$ points, $K=128$ Langevin steps for each point on the geodesic $\gamma$, and use $d t=0.0001$, i.e., requiring 65536 evaluations of the gradient of the auxiliary energy function. We generate 8 independent paths in parallel, which takes roughly 27 seconds on an M1 CPU.

Constrained transition paths. Constrained transition paths were also parametrized with cubic splines with 10 nodes, but discretized into 1024 points.

For the low-variance transition paths, we chose the threshold $\rho=3$, and $\lambda=0$ for the first 1200 optimization steps, and $\lambda$ linearly increasing from 0 to 100 for the last 3800 optimization steps, for the total of 5000 optimization steps with the Adam optimizer with a learning $\eta=0.01$. This took just under 6 minutes on an M1 CPU.

For the region-avoiding transition paths, we encode the restricted region with $\boldsymbol{z}^{*}=\left(-0.8,-0.1, t^{*}\right)$ with $\log \operatorname{SNR}\left(t^{*}\right)=4$, and combine two penalty functions: $h_{1}$ is the low-variance penalty described above, but with $\rho_{1}=3.75$ threshold, and $h_{2}$ is the KL penalty with $\rho_{2}=-4350$ threshold. We define $\lambda_{1}$ as in the low-variance transitions, and fix $\lambda_{2}=1$. The optimization was performed with Adam optimizer, learning rate $\eta=0.1$, and ran for 4000 steps for a runtime of just under 5 minutes on an M1 CPU.

The reason we include the low-variance penalty in the region-avoiding experiment is because $\operatorname{KL}\left(p\left(\cdot \mid z^{*}\right) \| p\left(\cdot \mid \gamma_{s}\right)\right)$ can trivially be increased by simply increasing entropy of $p\left(\cdot \mid \gamma_{s}\right)$ which would not result in avoiding the region defined by $p\left(\cdot \mid z^{*}\right)$.

\section*{G Note on transition path sampling baselines}

For transition path experiments performed in Section 6.3, we considered Holdijk et al. (2023); Du et al. (2024); Raja et al. (2025) as baselines. However, we encountered reproducibility issues. Specifically
- Holdijk et al. (2023) released the implementation: github.com/LarsHoldijk/SOCTransitionPaths. However, it does not appear to be supported. Several issues in the repository highlight failures to reproduce results, which have remained unresolved for more than a year.
- Raja et al. (2025) released the implementation: github.com/ASK-Berkeley/OM-TPS. However, it does not contain the code for the alanine dipeptide experiments, and the authors did not respond to a request to release it.
- Du et al. (2024) released the implementation: github.com/plainerman/Variational-Doob that we were able to use. However, we obtained results significantly worse than those reported in the original publication. We have contacted the authors, who acknowledged our question but did not provide guidance on how to resolve the issue.

For Doob's Lagrangian (Du et al., 2024), we experimented with: different numbers of epochs, different numbers of Gaussians, first vs second order ODE, MLP vs spline, and internal vs external coordinates. We reported the results of the configuration that was the best. Many configurations either diverged completely (returned NaN values) or collapsed to completely straight transition paths, oblivious to the underlying energy landscape. These issues persisted even after switching to double precision (as advised in the official code repository).

\section*{H Expectation parameter estimation code}
```
import jax
import jax.random as jr
import jax.numpy as jnp
def f(x, t, key): # Implemenation of the expected denoising
    pass
def sigma_and_alpha(t): # Depends on the choice of SDE and noise schedule
    pass
def mu(x, t, key):
    model_key, eps_key = jr.split(key, 2)
    eps = jr.rademacher(eps_key, (x.size,), dtype=jnp.float32)
    def pred_fn(x_):
        return f(x_, t, key=model_key)
    f_pred, f_grad = jax.jvp(pred_f, (x,), (eps,))
    div = jnp.sum(f_grad * eps)
    sigma, alpha = sigma_and_alpha(t)
    return sigma**2/alpha * div + jnp.sum(f_pred ** 2), f_pred
```


Listing 1: JAX Implementation of $\boldsymbol{\mu}$ estimation

\section*{I Licences}
- EDM2 model (Karras et al., 2024): Creative Commons BY-NC-SA 4.0 license
- ImageNet dataset (Deng et al., 2009): Custom non-commercial license
- SDVAE model (Rombach et al., 2022): CreativeML Open RAIL++-M license
- OpenM++ (OpenMP Architecture Review Board, 2008): MIT License