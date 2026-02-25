\title{
When and why PINNs fail to train: A neural tangent kernel perspective
}

\author{
Sifan Wang ${ }^{\mathrm{a}}$, Xinling Yu ${ }^{\mathrm{a}}$, Paris Perdikaris ${ }^{\mathrm{b}, *}$ \\ ${ }^{\mathrm{a}}$ Graduate Group in Applied Mathematics and Computational Science, University of Pennsylvania, Philadelphia, PA 19104, United States of America \\ ${ }^{\mathrm{b}}$ Department of Mechanichal Engineering and Applied Mechanics, University of Pennsylvania, Philadelphia, PA 19104, United States of America
}

\section*{ARTICLE INFO}

\section*{Article history:}

Available online 11 October 2021

\section*{Keywords:}

Physics-informed neural networks
Spectral bias
Multi-task learning
Gradient descent
Scientific machine learning

\begin{abstract}
Physics-informed neural networks (PINNs) have lately received great attention thanks to their flexibility in tackling a wide range of forward and inverse problems involving partial differential equations. However, despite their noticeable empirical success, little is known about how such constrained neural networks behave during their training via gradient descent. More importantly, even less is known about why such models sometimes fail to train at all. In this work, we aim to investigate these questions through the lens of the Neural Tangent Kernel (NTK); a kernel that captures the behavior of fully-connected neural networks in the infinite width limit during training via gradient descent. Specifically, we derive the NTK of PINNs and prove that, under appropriate conditions, it converges to a deterministic kernel that stays constant during training in the infinite-width limit. This allows us to analyze the training dynamics of PINNs through the lens of their limiting NTK and find a remarkable discrepancy in the convergence rate of the different loss components contributing to the total training error. To address this fundamental pathology, we propose a novel gradient descent algorithm that utilizes the eigenvalues of the NTK to adaptively calibrate the convergence rate of the total training error. Finally, we perform a series of numerical experiments to verify the correctness of our theory and the practical effectiveness of the proposed algorithms. The data and code accompanying this manuscript are publicly available at https://github.com/PredictiveIntelligenceLab/PINNsNTK.
\end{abstract}
© 2021 Elsevier Inc. All rights reserved.

\section*{1. Introduction}

Thanks to the approximation capabilities of neural networks, physics-informed neural networks (PINNs) have already led to a series of remarkable results across a range of problems in computational science and engineering, including fluids mechanics [1-5], bio-engineering [6,7], meta-material design [8-10], free boundary problems [11], Bayesian networks and uncertainty quantification [12-16], high-dimensional PDEs [17,18], stochastic differential equations [19], fractional differential equations [20,21], and beyond [22-25]. However, PINNs using fully connected architectures often fail to achieve stable training and produce accurate predictions, especially when the underlying PDE solutions contain high-frequencies or multiscale features [26,13,27]. Recent work by Wang et al. [28] attributed this pathological behavior to multi-scale interactions

\footnotetext{
* Corresponding author.

E-mail addresses: sifanw@sas.upenn.edu (S. Wang), xlyu@sas.upenn.edu (X. Yu), pgp@seas.upenn.edu (P. Perdikaris).
}
between different terms in the PINNs loss function, ultimately leading to stiffness in the gradient flow dynamics, which, consequently, introduces stringent stability requirements on the learning rate. To mitigate this pathology, Wang et al. [28] proposed an empirical learning-rate annealing scheme that utilizes the back-propagated gradient statistics during training to adaptively assign importance weights to different terms in a PINNs loss function, with the goal of balancing the magnitudes of the back-propagated gradients. Although this approach was demonstrated to produce significant and consistent improvements in the trainability and accuracy of PINNs, the fundamental reasons behind the practical difficulties of training fully-connected PINNs still remain unclear [26].

Parallel to the development of PINNs, recent investigations have shed light into the representation shortcomings and training deficiencies of fully-connected neural networks. Specifically, it has been shown that conventional fully-connected architectures - such as the ones typically used in PINNs - suffer from "spectral bias" and are incapable of learning functions with high frequencies, both in theory and in practice [29-32]. These observations are rigorously grounded by the newly developed neural tangent kernel theory [33,34] that, by exploring the connection between deep neural networks and kernel regression methods, elucidates the training dynamics of deep learning models. Specifically, the original work of Jacot et al. [33] proved that, at initialization, fully-connected networks are equivalent to Gaussian processes in the infinite-width limit [35-37], while the evolution of a infinite-width network during training can also be described by another kernel, the so-called Neural Tangent Kernel (NTK) [33]. Remarkably, this function-space viewpoint allows us then to rigorously analyze the training convergence of deep neural networks by examining the spectral properties of their limiting NTK [33,34,32].

Drawing motivation from the aforementioned developments, this work sets sail into investigating the training dynamics of PINNs. To this end, we rigorously study fully-connected PINNs models through the lens of their neural tangent kernel, and produce novel insights into when and why such models can be effectively trained, or not. Specifically, our main contributions can be summarized into the following points:
- We prove that fully-connected PINNs converge to Gaussian processes at the infinite width limit for linear PDEs.
- We derive the neural tangent kernel (NTK) of PINNs and prove that, under suitable assumptions, it converges to a deterministic kernel and remains constant during training via gradient descent with an infinitesimally small learning rate.
- We show how the convergence rate of the total training error of a PINNs model can be analyzed in terms of the spectrum of its NTK at initialization.
- We show that fully-connected PINNs not only suffer from spectral bias, but also from a remarkable discrepancy of convergence rate in the different components of their loss function.
- We propose a novel adaptive training strategy for resolving this pathological convergence behavior, and significantly enhance the trainability and predictive accuracy of PINNs.

Taken together, these developments provide a novel path into analyzing the convergence of PINNs, and enable the design of novel training algorithms that can significantly improve their trainability, accuracy and robustness.

This paper is organized as follows. In section 2, we present a brief overview of fully-connected neural networks and their behavior in the infinite-width limit following the original formulation of Jacot et al. [33]. Next, we derive the NTK of PINNs in a general setting and prove that, under suitable assumptions, it converges to a deterministic kernel and remains constant during training via gradient descent with an infinitesimally small learning rate, see section 3, 4. Furthermore, in section 5 we analyze the training dynamics of PINNs, demonstrate that PINNs models suffer from spectral bias, and then propose a novel algorithm to improve PINNs' performance in practice. Finally we carry out a series of numerical experiments to verify the developed NTK theory and validate the effectiveness of the proposed algorithm.

\section*{2. Infinitely wide neural networks}

In this section, we revisit the definition of fully-connected neural networks and investigate their behavior under the infinite-width limit. Let us start by formally defining the forward pass of a scalar valued fully-connected network with $L$ hidden layers, with the input and output dimensions denoted as $d_{0}=d$, and $d_{L+1}=1$, respectively. For inputs $\boldsymbol{x} \in \mathbb{R}^{d}$ we also denote the input layer of the network as $\boldsymbol{f}^{(0)}(\boldsymbol{x})=\boldsymbol{x}$ for convenience. Then a fully-connected neural network with $L$ hidden layers is defined recursively as
$$
\begin{align*}
& \boldsymbol{g}^{(h)}(\boldsymbol{x})=\frac{1}{\sqrt{d_{h}}} \boldsymbol{W}^{(h)} \cdot \boldsymbol{f}^{(h)}+\boldsymbol{b}^{(h)} \in \mathbb{R}^{d_{h+1}},  \tag{2.1}\\
& \boldsymbol{f}^{(h+1)}(\boldsymbol{x})=\sigma\left(\mathbf{g}^{(h)}(\boldsymbol{x})\right), \tag{2.2}
\end{align*}
$$
for $h=0,1, \ldots, L-1$, where $\boldsymbol{W}^{(h)} \in \mathbb{R}^{d_{h+1} \times d_{h}}$ are weight matrices and $\boldsymbol{b}^{(h)} \in \mathbb{R}^{d_{h+1}}$ are bias vectors in the $h$-th hidden layer, and $\sigma: \mathbb{R} \rightarrow \mathbb{R}$ is a coordinate-wise smooth activation function. The final output of the neural network is given by
$$
\begin{equation*}
f(\boldsymbol{x} ; \boldsymbol{\theta})=\frac{1}{\sqrt{d_{L}}} \boldsymbol{W}_{L}^{(L)} \cdot \boldsymbol{f}^{(L)}+\boldsymbol{b}^{(L)}, \tag{2.3}
\end{equation*}
$$
where $\boldsymbol{W}^{(L)} \in \mathbb{R}^{1 \times d_{L}}$ and $\boldsymbol{b}^{(L)} \in \mathbb{R}$ are the weight and bias parameters of the last layer. Here, $\boldsymbol{\theta}=\left\{\boldsymbol{W}^{(0)}, \boldsymbol{b}^{(0)}, \ldots, \boldsymbol{W}^{(L)}, \boldsymbol{b}^{(L)}\right\}$ represents all parameters of the neural network. Such a parameterization is known as the "NTK parameterization" following the original work of Jacot et al. [33]. We remark that the scaling factors $\frac{1}{\sqrt{d_{h}}}$ are key to obtaining a consistent asymptotic behavior of neural networks as the widths of the hidden layers $d_{1}, d_{2}, \ldots, d_{h}$ grow to infinity.

We initialize all the weights and biases to be independent and identically distributed (i.i.d.) as standard normal distribution $\mathcal{N}(0,1)$ random variables, and consider the sequential limit of hidden widths $d_{1}, d_{2}, \ldots, d_{L} \rightarrow \infty$. As described in [33,36,35], all coordinates of $\boldsymbol{f}^{(h)}$ at each hidden layer asymptotically converge to an i.i.d. centered Gaussian process with covariance function $\Sigma^{h-1}: \mathbb{R}^{d_{h-1}} \times \mathbb{R}^{d_{h-1}} \rightarrow \mathbb{R}$ defined recursively as
$$
\begin{align*}
\Sigma^{(0)}\left(\boldsymbol{x}, \boldsymbol{x}^{\prime}\right) & =\boldsymbol{x}^{T} \boldsymbol{x}^{\prime}+1, \\
\Lambda^{(h)}\left(\boldsymbol{x}, \boldsymbol{x}^{\prime}\right) & =\left(\begin{array}{cc}
\Sigma^{(h-1)}(\boldsymbol{x}, \boldsymbol{x}) & \Sigma^{(h-1)}\left(\boldsymbol{x}, \boldsymbol{x}^{\prime}\right) \\
\Sigma^{(h-1)}\left(\boldsymbol{x}^{\prime}, \boldsymbol{x}\right) & \Sigma^{(h-1)}\left(\boldsymbol{x}^{\prime}, \boldsymbol{x}^{\prime}\right)
\end{array}\right) \in \mathbb{R}^{2 \times 2},  \tag{2.4}\\
\Sigma^{(h)}\left(\boldsymbol{x}, \boldsymbol{x}^{\prime}\right) & =\underset{(u, v) \sim \mathcal{N}\left(0, \Lambda^{(h)}\right)}{\mathbb{E}}[\sigma(u) \sigma(v)]+1,
\end{align*}
$$
for $h=1,2, \ldots, L$.
To introduce the neural tangent kernel (NTK), we also need to define
$$
\begin{equation*}
\dot{\Sigma}^{(h)}\left(\boldsymbol{x}, \boldsymbol{x}^{\prime}\right)=\underset{(u, v) \sim \mathcal{N}\left(0, \Lambda^{(h)}\right)}{\mathbb{E}}[\dot{\sigma}(u) \dot{\sigma}(v)] \tag{2.5}
\end{equation*}
$$
where $\dot{\sigma}$ denotes the derivative of the activation function $\sigma$.
Following the derivation of [33,38], the neural tangent kernel can be generally defined at any training time $t$, as the neural network parameters $\boldsymbol{\theta}(t)$ are changing during model training by gradient descent. This definition takes the form
$$
\begin{equation*}
\operatorname{Ker}_{t}\left(\boldsymbol{x}, \boldsymbol{x}^{\prime}\right)=\left\langle\frac{\partial f(\boldsymbol{x} ; \boldsymbol{\theta}(t))}{\partial \boldsymbol{\theta}}, \frac{\partial f\left(\boldsymbol{x}^{\prime} ; \boldsymbol{\theta}(t)\right)}{\partial \theta}\right\rangle=\sum_{\theta \in \boldsymbol{\theta}} \frac{\partial f(\boldsymbol{x} ; \theta(t))}{\partial \theta} \frac{\partial f\left(\boldsymbol{x}^{\prime} ; \theta(t)\right)}{\partial \theta}, \tag{2.6}
\end{equation*}
$$
where $\langle\cdot, \cdot\rangle$ denotes an inner product over $\boldsymbol{\theta}$. This kernel converges in probability to a deterministic kernel $\Theta^{(L)}\left(\boldsymbol{x}, \boldsymbol{x}^{\prime}\right)$ at random initialization as the width of hidden layers goes to infinity [33]. Specifically,
$$
\begin{equation*}
\lim _{d_{L} \rightarrow \infty} \cdots \lim _{d_{1} \rightarrow \infty} \operatorname{Ker}_{0}\left(\boldsymbol{x}, \boldsymbol{x}^{\prime}\right)=\lim _{d_{L} \rightarrow \infty} \cdots \lim _{d_{1} \rightarrow \infty}\left\langle\frac{\partial f(\boldsymbol{x} ; \boldsymbol{\theta}(0))}{\partial \boldsymbol{\theta}}, \frac{\partial f\left(\boldsymbol{x}^{\prime} ; \boldsymbol{\theta}(0)\right)}{\partial \boldsymbol{\theta}}\right\rangle=\Theta^{(L)}\left(\boldsymbol{x}, \boldsymbol{x}^{\prime}\right) . \tag{2.7}
\end{equation*}
$$

Here $\Theta^{(L)}\left(\boldsymbol{x}, \boldsymbol{x}^{\prime}\right)$ denotes the limiting kernel function of a $L$-layer fully-connected neural network, which is defined by
$$
\begin{equation*}
\Theta^{(L)}\left(\boldsymbol{x}, \boldsymbol{x}^{\prime}\right)=\sum_{h=1}^{L+1}\left(\Sigma^{(h-1)}\left(\boldsymbol{x}, \boldsymbol{x}^{\prime}\right) \cdot \prod_{h^{\prime}=h}^{L+1} \dot{\Sigma}^{\left(h^{\prime}\right)}\left(\boldsymbol{x}, \boldsymbol{x}^{\prime}\right)\right) \tag{2.8}
\end{equation*}
$$
where $\dot{\Sigma}^{(L+1)}\left(\boldsymbol{x}, \boldsymbol{x}^{\prime}\right)=1$ for convenience. Moreover, Jacot et al. [33] proved that, under some suitable conditions and training time $T$ fixed, $\operatorname{Ker}_{t}$ converges to $\Theta^{(L)}$ for all $0 \leq t \leq T$, as the width goes to infinity. As a consequence, a properly randomly initialized and sufficiently wide deep neural network trained by gradient descent is equivalent to a kernel regression with a deterministic kernel.

\section*{3. Physics-informed neural networks (PINNs)}

In this section, we study physics-informed neural networks (PINNs) and their corresponding neural tangent kernels. To this end, we consider the following well-posed partial differential equation (PDE) defined on a bounded domain $\Omega \subset \mathbb{R}^{d}$
$$
\begin{align*}
& \mathcal{N}[u](\boldsymbol{x})=f(\boldsymbol{x}), \quad x \in \Omega  \tag{3.1}\\
& u(\boldsymbol{x})=g(\boldsymbol{x}), \quad x \in \partial \Omega \tag{3.2}
\end{align*}
$$
where $\mathcal{N}$ denotes differential operator and $u(\boldsymbol{x}): \bar{\Omega} \rightarrow \mathbb{R}$ is the unknown solution with $\boldsymbol{x}=\left(x_{1}, x_{2}, \cdots, x_{d}\right)$. Here we remark that for time-dependent problems, we consider time $t$ as an additional coordinate in $\boldsymbol{x}$ and $\Omega$ denotes the spatio-temporal domain. Then, the initial condition can be simply treated as a special type of Dirichlet boundary condition and included in equation (3.2).

Following the original work of Raissi et al. [39], we assume that the latent solution $u(\boldsymbol{x})$ can be approximated by a deep neural network $u(\boldsymbol{x}, \boldsymbol{\theta})$ with parameters $\boldsymbol{\theta}$, where $\boldsymbol{\theta}$ is a collection of all the parameters in the network. We can then define the PDE residual $r(\boldsymbol{x} ; \boldsymbol{\theta})$ as
$$
\begin{equation*}
r(\boldsymbol{x} ; \boldsymbol{\theta}):=\mathcal{N}[u](\boldsymbol{x} ; \boldsymbol{\theta})-f(\boldsymbol{x}) \tag{3.3}
\end{equation*}
$$

Note that the parameters of $u(\boldsymbol{x} ; \boldsymbol{\theta})$ can be "learned" by minimizing the following composite loss function
$$
\begin{equation*}
\mathcal{L}(\boldsymbol{\theta})=\mathcal{L}_{b}(\boldsymbol{\theta})+\mathcal{L}_{r}(\boldsymbol{\theta}) \tag{3.4}
\end{equation*}
$$
where
$$
\begin{align*}
& \mathcal{L}_{b}(\boldsymbol{\theta})=\frac{1}{2} \sum_{i=1}^{N_{b}}\left|u\left(\boldsymbol{x}_{b}^{i} ; \boldsymbol{\theta}\right)-g\left(\boldsymbol{x}_{b}^{i}\right)\right|^{2}  \tag{3.5}\\
& \mathcal{L}_{r}(\boldsymbol{\theta})=\frac{1}{2} \sum_{i=1}^{N_{r}}\left|r\left(\boldsymbol{x}_{r}^{i} ; \boldsymbol{\theta}\right)\right|^{2} \tag{3.6}
\end{align*}
$$

Here $N_{b}$ and $N_{r}$ denote the batch sizes for the training data $\left\{\boldsymbol{x}_{b}^{i}, g\left(\boldsymbol{x}_{b}^{i}\right)\right\}_{i=1}^{N_{b}}$ and $\left\{\boldsymbol{x}_{r}^{i}, f\left(\boldsymbol{x}_{b}^{i}\right)\right\}_{i=1}^{N_{r}}$ respectively, which can be randomly sampled at each iteration of a gradient descent algorithm.

\subsection*{3.1. Neural tangent kernel theory for PINNs}

In this section we derive the neural tangent kernel of a physics-informed neural network. To this end, consider minimizing the loss function (3.4) by gradient descent with an infinitesimally small learning rate, yielding the continuous-time gradient flow system
$$
\begin{equation*}
\frac{d \boldsymbol{\theta}}{d t}=-\nabla \mathcal{L}(\boldsymbol{\theta}) \tag{3.7}
\end{equation*}
$$
and let $u(t)=u\left(\boldsymbol{x}_{b} ; \boldsymbol{\theta}(t)\right)=\left\{u\left(\boldsymbol{x}_{b}^{i} ; \boldsymbol{\theta}(t)\right)\right\}_{i=1}^{N_{b}}$ and $\mathcal{N}[u](t)=\mathcal{N}[u]\left(\boldsymbol{x}_{r} ; \boldsymbol{\theta}(t)\right)=\left\{\mathcal{N}[u]\left(\boldsymbol{x}_{r} ; \boldsymbol{\theta}(t)\right)\right\}_{i=1}^{N_{r}}$. Then the following lemma characterizes how $u(t)$ and $\mathcal{N}[u](t)$ evolve during training by gradient descent.

Lemma 3.1. Given the data points $\left\{\boldsymbol{x}_{b}^{i}, g\left(\boldsymbol{x}_{b}^{i}\right)\right\}_{i=1}^{N_{b}},\left\{\boldsymbol{x}_{r}^{i}, f\left(\boldsymbol{x}_{r}^{i}\right)\right\}_{i=1}^{N_{r}}$ and the gradient flow (3.7), $u(t)$ and $\mathcal{N}[u](t)$ obey the following evolution
$$
\left[\begin{array}{c}
\frac{d u\left(\boldsymbol{x}_{b} ; \boldsymbol{\theta}(t)\right)}{d t}  \tag{3.8}\\
\frac{d \mathcal{N}[u]\left(\boldsymbol{x}_{r} ; \boldsymbol{\theta}(t)\right)}{d t}
\end{array}\right]=-\left[\begin{array}{cc}
\boldsymbol{K}_{u u}(t) & \boldsymbol{K}_{u r}(t) \\
\boldsymbol{K}_{r u}(t) & \boldsymbol{K}_{r r}(t)
\end{array}\right] \cdot\left[\begin{array}{c}
u\left(\boldsymbol{x}_{b} ; \boldsymbol{\theta}(t)\right)-g\left(\boldsymbol{x}_{b}\right) \\
\mathcal{N}[u]\left(\boldsymbol{x}_{r} ; \boldsymbol{\theta}(t)\right)-f\left(\boldsymbol{x}_{r}\right)
\end{array}\right],
$$
where $\boldsymbol{K}_{r u}(t)=\boldsymbol{K}_{u r}^{T}(t)$ and $\boldsymbol{K}_{u u}(t) \in \mathbb{R}^{N_{b} \times N_{b}}, \boldsymbol{K}_{u r}(t) \in \mathbb{R}^{N_{b} \times N_{r}}$, and $\boldsymbol{K}_{r r}(t) \in \mathbb{R}^{N_{r} \times N_{r}}$ whose ( $i, j$ )-th entry is given by
$$
\begin{align*}
\left(\boldsymbol{K}_{u u}\right)_{i j}(t) & =\left\langle\frac{d u\left(\boldsymbol{x}_{b}^{i} ; \boldsymbol{\theta}(t)\right)}{d \boldsymbol{\theta}}, \frac{d u\left(\boldsymbol{x}_{b}^{j} ; \boldsymbol{\theta}(t)\right)}{d \boldsymbol{\theta}}\right\rangle \\
\left(\boldsymbol{K}_{u r}\right)_{i j}(t) & =\left\langle\frac{d u\left(\boldsymbol{x}_{b}^{i} ; \boldsymbol{\theta}(t)\right)}{d \boldsymbol{\theta}}, \frac{d \mathcal{N}[u]\left(\boldsymbol{x}_{r}^{j} ; \boldsymbol{\theta}(t)\right)}{d \boldsymbol{\theta}}\right\rangle  \tag{3.9}\\
\left(\boldsymbol{K}_{r r}\right)_{i j}(t) & =\left\langle\frac{d \mathcal{N}[u]\left(\boldsymbol{x}_{r}^{i} ; \boldsymbol{\theta}(t)\right)}{d \boldsymbol{\theta}}, \frac{d \mathcal{N}[u]\left(\boldsymbol{x}_{r}^{j} ; \boldsymbol{\theta}(t)\right)}{d \boldsymbol{\theta}}\right\rangle
\end{align*}
$$

Proof. The proof of Lemma 3.1 is given in Appendix A.
Remark 3.2. $\langle\cdot, \cdot\rangle$ here denotes the inner product over all neural network parameters in $\boldsymbol{\theta}$. For example,
$$
\left(\boldsymbol{K}_{u u}\right)_{i j}(t)=\sum_{\theta \in \boldsymbol{\theta}} \frac{d u\left(\boldsymbol{x}_{b}^{i} ; \boldsymbol{\theta}(t)\right)}{d \theta} \cdot \frac{d u\left(\boldsymbol{x}_{b}^{j} ; \boldsymbol{\theta}(t)\right)}{d \theta}
$$

Remark 3.3. We will denote the matrix $\left[\begin{array}{ll}\boldsymbol{K}_{u u}(t) & \boldsymbol{K}_{u r}(t) \\ \boldsymbol{K}_{r u}(t) & \boldsymbol{K}_{r r}(t)\end{array}\right]$ by $\boldsymbol{K}(t)$ in the following sections. It is easy to see that $\boldsymbol{K}_{u u}(t), \boldsymbol{K}_{r r}(t)$ and $\boldsymbol{K}(t)$ are all positive semi-definite matrices. Indeed, let $\boldsymbol{J}_{u}(t)$ and $\boldsymbol{J}_{r}(t)$ be the Jacobian matrices of $u(t)$ and $\mathcal{N}[u](t)$ with respect to $\boldsymbol{\theta}$ respectively. Then, we can observe that
$$
\boldsymbol{K}_{u u}(t)=\boldsymbol{J}_{u}(t) \boldsymbol{J}_{u}^{T}(t), \quad \boldsymbol{K}_{r r}(t)=\boldsymbol{J}_{r}(t) \boldsymbol{J}_{r}^{T}(t), \quad \boldsymbol{K}(t)=\left[\begin{array}{c}
\boldsymbol{J}_{u}(t) \\
\boldsymbol{J}_{r}(t)
\end{array}\right]\left[\boldsymbol{J}_{u}^{T}(t), \boldsymbol{J}_{r}^{T}(t)\right]
$$

Remark 3.4. It is worth pointing out that equation (3.8) holds for any differential operator $\mathcal{L}$ and any neural network architecture.

The statement of Lemma 3.1 involves the matrix $\boldsymbol{K}(t)$, which we call the neural tangent kernel of a physics-informed neural network (NTK of PINNS). Recall that an infinitely wide neural network is a Gaussian process, and its NTK remains constant during training [33]. Now two natural questions arise: how does the PDE residual behave in the infinite width limit? Does the NTK of PINNs exhibit similar behavior as the standard NTK? If so, what is the expression of the corresponding kernel? In the next subsections, we will answer these questions and show that, in the infinite-width limit, the NTK of PINNs indeed converges to a deterministic kernel at initialization and then remains constant during training.

\section*{4. Analyzing the training dynamics of PINNs through the lens of their NTK}

To simplify the proof and understand the key ideas clearly, we confine ourselves to a simple model problem using a fully-connected neural network with one hidden layer. To this end, we consider a one-dimensional Poisson equation as our model problem. Let $\Omega$ be a bounded open interval in $\mathbb{R}$. The partial differential equation is summarized as follows
$$
\begin{align*}
& u_{x x}(x)=f(x), \quad \forall x \in \Omega  \tag{4.1}\\
& u(x)=g(x), \quad x \in \partial \Omega
\end{align*}
$$

We proceed by approximating the solution $u(x)$ by a fully-connected neural network denoted by $u(x, \boldsymbol{\theta})$ with one hidden layer. Now we define the network explicitly:
$$
\begin{equation*}
u(x ; \boldsymbol{\theta})=\frac{1}{\sqrt{N}} \boldsymbol{W}^{(1)} \cdot \sigma\left(\boldsymbol{W}^{(0)} x+\boldsymbol{b}^{(0)}\right)+\boldsymbol{b}^{(1)} \tag{4.2}
\end{equation*}
$$
where $\boldsymbol{W}^{(0)} \in \mathbb{R}^{N \times 1}, \boldsymbol{W}^{(1)} \in \mathbb{R}^{1 \times N}$ are weights, $\boldsymbol{b}^{(0)} \in \mathbb{R}^{N}, \boldsymbol{b}^{(1)} \in \mathbb{R}^{1}$ are biases $\boldsymbol{\theta}=\left(\boldsymbol{W}^{(0)}, \boldsymbol{W}^{(1)}, \boldsymbol{b}^{(0)}, \boldsymbol{b}^{(1)}\right)$ represents all parameters in the network, and $\sigma$ is a smooth activation function. Then it is straightforward to show that
$$
\begin{equation*}
u_{x x}(x ; \boldsymbol{\theta})=\frac{1}{\sqrt{N}} \boldsymbol{W}^{(1)} \cdot\left[\ddot{\sigma}\left(\boldsymbol{W}^{(0)} x+\boldsymbol{b}^{(0)}\right) \odot \boldsymbol{W}^{(0)} \odot \boldsymbol{W}^{(0)}\right] \tag{4.3}
\end{equation*}
$$
where ⋅ denotes point-wise multiplication and $\ddot{\sigma}$ denotes the second order derivative of the activation function $\sigma$.
We initialize all the weights and biases to be i.i.d. $\mathcal{N}(0,1)$ random variables. Based on our presentation in section 2 , we already know that, in the infinite width limit, $u(x ; \boldsymbol{\theta})$ is a centered Gaussian process with covariance function $\Sigma^{(1)}\left(x, x^{\prime}\right)$ at initialization, which is defined in equation (2.4). The following theorem reveals that $u_{x x}(x ; \boldsymbol{\theta})$ converges in distribution to another centered Gaussian process with a covariance function $\Sigma_{x x}^{(1)}$ under the same limit.

Theorem 4.1. Assume that the activation function $\sigma$ is smooth and has a bounded second order derivative $\ddot{\sigma}$. Then for a fully-connected neural network of one hidden layer at initialization,
$$
\begin{align*}
& u(x ; \boldsymbol{\theta}) \xrightarrow{\mathcal{D}} \mathcal{G} \mathcal{P}\left(0, \Sigma^{(1)}\left(x, x^{\prime}\right)\right)  \tag{4.4}\\
& u_{x x}(x ; \boldsymbol{\theta}) \xrightarrow{\mathcal{D}} \mathcal{G} \mathcal{P}\left(0, \Sigma_{x x}^{(1)}\left(x, x^{\prime}\right)\right), \tag{4.5}
\end{align*}
$$
as $N \rightarrow \infty$, where $\mathcal{D}$ means convergence in distribution and
$$
\begin{equation*}
\Sigma_{x x}^{(1)}\left(x, x^{\prime}\right)=\underset{u, v \sim \mathcal{N}(0,1)}{\mathbb{E}}\left[u^{4} \ddot{\sigma}(u x+v) \ddot{\sigma}\left(u x^{\prime}+v\right)\right] \tag{4.6}
\end{equation*}
$$

Proof. The proof can be found in Appendix B.
Remark 4.2. By induction, the proof of Theorem 4.1 can be extended to differential operators of any order and fullyconnected neural networks with multiple hidden layers. Observe that a linear combination of Gaussian processes is still a Gaussian process. Therefore, Theorem 4.1 can be generalized to any linear partial differential operator under appropriate regularity conditions.

As an immediate corollary, a sufficiently wide physics-informed neural network for model problem (4.1) induces a joint Gaussian process (GP) between the function values and the PDE residual at initialization, indicating a PINNs-GP correspondence for linear PDEs.

The next question we investigate is whether the NTK of PINNs behaves similarly as the NTK of standard neural networks. The next theorem proves that indeed the kernel $\boldsymbol{K}(0)$ converges in probability to a certain deterministic kernel matrix as the width of the network goes to infinity.

Theorem 4.3. For a physics-informed network with one hidden layer at initialization, and in the limit as the layer's width $N \rightarrow \infty$, the NTK $\mathbf{K}(t)$ of the PINNs model defined in equation (3.9) converges in probability to a deterministic limiting kernel, i.e.,
$$
\boldsymbol{K}(0)=\left[\begin{array}{ll}
\boldsymbol{K}_{u u}(0) & \boldsymbol{K}_{u r}(0)  \tag{4.7}\\
\boldsymbol{K}_{r u}(0) & \boldsymbol{K}_{r r}(0)
\end{array}\right] \rightarrow\left[\begin{array}{cc}
\Theta_{u u}^{(1)} & \Theta_{u r}^{(1)} \\
\Theta_{r u}^{(1)} & \Theta_{r r}^{(1)}
\end{array}\right]:=\boldsymbol{K}^{*},
$$
where the explicit expression of $\mathbf{K}^{*}$ is provided in appendix $C$.
Proof. The proof can be found in Appendix C.
Our second key result is that the NTK of PINNs stays asymptotically constant during training, i.e. $\boldsymbol{K}(t) \approx \boldsymbol{K}(0)$ for all $t$. To state and prove the theorem rigorously, we may assume that all parameters and the loss function do not blow up and are uniformly bounded during training. The first two assumptions are both reasonable and practical, otherwise one would obtain unstable and divergent training processes. In addition, the activation has to be 4 -th order smooth and all its derivatives are bounded. The last assumption is not a strong restriction since it is satisfied by most of the activation functions commonly used for PINNs such as sigmoids, hyperbolic tangents, sine functions, etc.

Theorem 4.4. For the model problem (4.1) with a fully-connected neural network of one hidden layer, consider minimizing the loss function (3.4) by gradient descent with an infinitesimally small learning rate. For any $T>0$ satisfying the following assumptions:
(i) there exists a constant $C>0$ such that all parameters of the network is uniformly bounded for $t \in T$, i.e.
$$
\sup _{t \in[0, T]}\|\boldsymbol{\theta}(t)\|_{\infty} \leq C
$$
where $C$ does not depend on $N$.
(ii) there exists a constant $C>0$ such that
$$
\begin{aligned}
& \int_{0}^{T}\left|\sum_{i=1}^{N_{b}}\left(u\left(x_{b}^{i} ; \boldsymbol{\theta}(\tau)\right)-g\left(x_{b}^{i}\right)\right)\right| d \tau \leq C \\
& \int_{0}^{T}\left|\sum_{i=1}^{N_{r}}\left(u_{x x}\left(x_{r}^{i} ; \boldsymbol{\theta}(\tau)\right)-f\left(x_{r}^{i}\right)\right)\right| d \tau \leq C
\end{aligned}
$$
(iii) the activation function $\sigma$ is smooth and $\left|\sigma^{(k)}\right| \leq C$ for $k=0,1,2,3,4$, where $\sigma^{(k)}$ denotes $k$-th order derivative of $\sigma$.

Then we have
$$
\begin{equation*}
\lim _{N \rightarrow \infty} \sup _{t \in[0, T]}\|\boldsymbol{K}(t)-\boldsymbol{K}(0)\|_{2}=0 \tag{4.8}
\end{equation*}
$$
where $\boldsymbol{K}(t)$ is the corresponding NTK of PINNs.
Proof. The proof can be found in Appendix D.
Here we provide some intuition behind the proof. The crucial observation is that all parameters of the network change little during training (see Lemma D. 2 in the Appendix). By intuition, for sufficient wide neural networks, any slight movement of weights would contribute to a non-negligible change in the network output. As a result, the gradients of the outputs $u(x, \boldsymbol{\theta})$ and $u_{x x}(x, \boldsymbol{\theta})$ with respect to parameters barely change (see Lemma D.4), and, therefore, the kernel remains almost static during training.

Combining Theorem 4.3 and Theorem 4.4 we may conclude that, for the model problem of equation (4.1), we have
$$
\boldsymbol{K}(t) \approx \boldsymbol{K}(0) \approx \boldsymbol{K}^{*}, \quad \forall t,
$$
from which (and equation (3.8)) we immediately obtain
$$
\left[\begin{array}{c}
\frac{d u\left(\boldsymbol{x}_{b} ; \boldsymbol{\theta}(t)\right)}{d t}  \tag{4.9}\\
\frac{d u_{x x}\left(\boldsymbol{x}_{r} ; \boldsymbol{\theta}(t)\right)}{d t}
\end{array}\right]=-\boldsymbol{K}(t) \cdot\left[\begin{array}{c}
u\left(\boldsymbol{x}_{b} ; \boldsymbol{\theta}(t)\right)-g\left(\boldsymbol{x}_{b}\right) \\
u_{x x}\left(\boldsymbol{x}_{r} ; \boldsymbol{\theta}(t)\right)-f\left(\boldsymbol{x}_{r}\right)
\end{array}\right] \approx-\boldsymbol{K}^{*}\left[\begin{array}{c}
u\left(\boldsymbol{x}_{b} ; \boldsymbol{\theta}(t)\right)-g\left(\boldsymbol{x}_{b}\right) \\
u_{x x}\left(\boldsymbol{x}_{r} ; \boldsymbol{\theta}(t)\right)-f\left(\boldsymbol{x}_{r}\right) .
\end{array}\right] .
$$

Note that if the matrix $\boldsymbol{K}^{*}$ is invertible, then according to [40,31], the network's outputs $u(x, \boldsymbol{\theta})$ and $u_{x x}(x, \boldsymbol{\theta})$ can be approximated for any arbitrary test data $\boldsymbol{x}_{\text {test }}$ after $t$ steps of gradient descent as
$$
\left[\begin{array}{c}
u\left(\boldsymbol{x}_{\text {test }} ; \boldsymbol{\theta}(t)\right)  \tag{4.10}\\
u_{x x}\left(\boldsymbol{x}_{\text {test }} ; \boldsymbol{\theta}(t)\right)
\end{array}\right] \approx \boldsymbol{K}_{\text {test }}^{*}\left(\boldsymbol{K}^{*}\right)^{-1}\left(I-e^{-\boldsymbol{K}^{*} t}\right) \cdot\left[\begin{array}{l}
g\left(\boldsymbol{x}_{b}\right) \\
f\left(\boldsymbol{x}_{r}\right)
\end{array}\right],
$$
where $\boldsymbol{K}_{\text {test }}$ is the NTK matrix between all points in $\boldsymbol{x}_{\text {test }}$ and all training data. Letting $t \rightarrow \infty$, we obtain
$$
\left[\begin{array}{c}
u\left(\boldsymbol{x}_{\text {test }} ; \boldsymbol{\theta}(\infty)\right) \\
u_{x x}\left(\boldsymbol{x}_{\text {test }} ; \boldsymbol{\theta}(\infty)\right)
\end{array}\right] \approx \boldsymbol{K}_{\text {test }}^{*}\left(\boldsymbol{K}^{*}\right)^{-1} \cdot\left[\begin{array}{l}
g\left(\boldsymbol{x}_{b}\right) \\
f\left(\boldsymbol{x}_{r}\right)
\end{array}\right] .
$$

This implies that, under the assumption that $\boldsymbol{K}^{*}$ is invertible, an infinitely wide physics-informed neural network for model problem (4.1) is also equivalent to a kernel regression. However, from the authors' experience, the NTK of PINNs is always degenerate (see Figs. 2c, 3c) which means that we cannot invert the kernel matrix and thus be able to casually perform kernel regression predictions in practice.

\section*{5. Spectral bias in physics-informed neural networks}

In this section, we will utilize the developed theory to investigate whether physics-informed neural networks are spectrally biased. The term "spectral bias" [29,41,32] refers to a well known pathology that prevents deep fully-connected networks from learning high-frequency functions.

Since the NTK of PINNs barely changes during training, we may rewrite equation (4.9) as
$$
\left[\begin{array}{c}
\frac{d u\left(\boldsymbol{x}_{b} ; \boldsymbol{\theta}(t)\right)}{d t}  \tag{5.1}\\
\frac{d u_{x X}\left(\boldsymbol{x}_{r} ; \boldsymbol{\theta}(t)\right)}{d t}
\end{array}\right] \approx-\boldsymbol{K}(0)\left[\begin{array}{c}
u\left(\boldsymbol{x}_{b} ; \boldsymbol{\theta}(t)\right)-g\left(\boldsymbol{x}_{b}\right) \\
u_{x x}\left(\boldsymbol{x}_{r} ; \boldsymbol{\theta}(t)\right)-f\left(\boldsymbol{x}_{r}\right)
\end{array}\right],
$$
which leads to
$$
\left[\begin{array}{c}
u\left(\boldsymbol{x}_{b} ; \boldsymbol{\theta}(t)\right)  \tag{5.2}\\
u_{x x}\left(\boldsymbol{x}_{r} ; \boldsymbol{\theta}(t)\right)
\end{array}\right] \approx\left(I-e^{-\boldsymbol{K}(0) t}\right) \cdot\left[\begin{array}{l}
g\left(\boldsymbol{x}_{b}\right) \\
f\left(\boldsymbol{x}_{r}\right)
\end{array}\right] .
$$

As mentioned in Remark 3.3, the NTK of PINNs is also positive semi-definite. So we can take its spectral decomposition $\boldsymbol{K}(0)=\boldsymbol{Q}^{T} \boldsymbol{\Lambda} \boldsymbol{Q}$, where $\boldsymbol{Q}$ is an orthogonal matrix and $\boldsymbol{\Lambda}$ is a diagonal matrix whose entries are the eigenvalues $\lambda_{i} \geq 0$ of $\boldsymbol{K}(0)$. Consequently, the training error is given by
$$
\begin{aligned}
{\left[\begin{array}{c}
u\left(\boldsymbol{x}_{b} ; \boldsymbol{\theta}(t)\right) \\
u_{x x}\left(\boldsymbol{x}_{r} ; \boldsymbol{\theta}(t)\right)
\end{array}\right]-\left[\begin{array}{l}
g\left(\boldsymbol{x}_{b}\right) \\
f\left(\boldsymbol{x}_{r}\right)
\end{array}\right] } & \approx\left(I-e^{-\boldsymbol{K}(0) t}\right) \cdot\left[\begin{array}{l}
g\left(\boldsymbol{x}_{b}\right) \\
f\left(\boldsymbol{x}_{r}\right)
\end{array}\right]-\left[\begin{array}{l}
g\left(\boldsymbol{x}_{b}\right) \\
f\left(\boldsymbol{x}_{r}\right)
\end{array}\right] \\
& \approx-\boldsymbol{Q}^{T} e^{-\boldsymbol{\Lambda} t} \boldsymbol{Q} \cdot\left[\begin{array}{l}
g\left(\boldsymbol{x}_{b}\right) \\
f\left(\boldsymbol{x}_{r}\right)
\end{array}\right],
\end{aligned}
$$
which is equivalent to
$$
\boldsymbol{Q}\left(\left[\begin{array}{c}
u\left(\boldsymbol{x}_{b} ; \boldsymbol{\theta}(t)\right)  \tag{5.3}\\
u_{x x}\left(\boldsymbol{x}_{r} ; \boldsymbol{\theta}(t)\right)
\end{array}\right]-\left[\begin{array}{l}
g\left(\boldsymbol{x}_{b}\right) \\
f\left(\boldsymbol{x}_{r}\right)
\end{array}\right]\right) \approx-e^{-\boldsymbol{\Lambda} t} \boldsymbol{Q} \cdot\left[\begin{array}{l}
g\left(\boldsymbol{x}_{b}\right) \\
f\left(\boldsymbol{x}_{r}\right)
\end{array}\right] .
$$

This implies that the $i$-th component of the left hand side in equation (5.3) will decay approximately at the rate $e^{-\lambda_{i} t}$. In other words, the eigenvalues of the kernel characterize how fast the absolute training error decreases. Particularly, components of the target function that correspond to kernel eigenvectors with larger eigenvalues will be learned faster. For fully-connected networks, the eigenvectors corresponding to higher eigenvalues of the NTK matrix generally exhibit lower frequencies [29,42,32]. From Fig. 1, one can observe that the eigenvalues of the NTK of PINNs decay rapidly. This results in extremely slow convergence to the high-frequency components of the target function. Thus we may conclude that PINNs suffer from the spectral bias either.

More generally, the NTK of PINNs after $t$ steps of gradient descent is given by
$$
\boldsymbol{K}(t)=\left[\begin{array}{ll}
\boldsymbol{K}_{u u}(t) & \boldsymbol{K}_{u r}(t) \\
\boldsymbol{K}_{r u}(t) & \boldsymbol{K}_{r r}(t)
\end{array}\right]=\left[\begin{array}{l}
\boldsymbol{J}_{u}(t) \\
\boldsymbol{J}_{r}(t)
\end{array}\right]\left[\boldsymbol{J}_{u}^{T}(t), \boldsymbol{J}_{r}^{T}(t)\right]=\boldsymbol{J}(t) \boldsymbol{J}^{T}(t) .
$$

It follows that
$$
\begin{aligned}
\sum_{i=1}^{N_{b}+N_{r}} \lambda_{i}(t) & =\operatorname{Tr}(\boldsymbol{K}(t))=\operatorname{Tr}\left(\boldsymbol{J}(t) \boldsymbol{J}^{T}(t)\right)=\operatorname{Tr}\left(\boldsymbol{J}^{T}(t) \boldsymbol{J}(t)\right) \\
& =\operatorname{Tr}\left(\boldsymbol{J}_{u}^{T}(t) \boldsymbol{J}_{u}(t)+\boldsymbol{J}_{r}^{T}(t) \boldsymbol{J}_{r}(t)\right)=\operatorname{Tr}\left(\boldsymbol{J}_{u}(t) \boldsymbol{J}_{u}^{T}(t)\right)+\operatorname{Tr}\left(\boldsymbol{J}_{r}(t) \boldsymbol{J}_{r}^{T}(t)\right) \\
& =\sum_{i=1}^{N_{b}} \lambda_{i}^{u u}(t)+\sum_{i=1}^{N_{r}} \lambda_{i}^{r r}(t)
\end{aligned}
$$

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/898f93b4-fd47-41e4-87e3-9a7fc6bc993f-08.jpg?height=402&width=1309&top_left_y=185&top_left_x=283}
\captionsetup{labelformat=empty}
\caption{Fig. 1. Model problem (1D Poisson equation): The eigenvalues of $\boldsymbol{K}, \boldsymbol{K}_{u u}$ and $\boldsymbol{K}_{r r}$ at initialization in descending order for different fabricated solutions $u(x)= \sin (a \pi x)$ where $a=1,2,4$.}
\end{figure}
where $\lambda_{i}(t), \lambda_{i}^{u u}(t)$ and $\lambda_{i}^{r r}(t)$ denote the eigenvalues of $\boldsymbol{K}(t), \boldsymbol{K}_{u u}(t)$ and $\boldsymbol{K}_{r r}(t)$, respectively. This reveals that the overall convergence rate of the total training error is characterized by the eigenvalues of $\boldsymbol{K}_{u u}$ and $\boldsymbol{K}_{r r}$ together. Meanwhile, the separate training error of $u\left(\boldsymbol{x}_{b} ; \boldsymbol{\theta}\right)$ and $u_{x x}\left(\boldsymbol{x}_{r} ; \boldsymbol{\theta}\right)$ is determined by the eigenvalues of $\boldsymbol{K}_{u u}$ and $\boldsymbol{K}_{r r}$, respectively. The above observation motivates us to give the following definition.

Definition 5.1. For a positive semi-definite kernel matrix $\boldsymbol{K} \in \mathbb{R}^{n \times n}$, the average convergence rate $c$ is defined as the mean of all its eigenvalues $\lambda_{i}$ 's, i.e.
$$
\begin{equation*}
c=\frac{\sum_{i=1}^{n} \lambda_{i}}{n}=\frac{\operatorname{Tr}(K)}{n} . \tag{5.4}
\end{equation*}
$$

In particular, for any two kernel matrices $\mathbf{K}_{1}, \mathbf{K}_{2}$ with average convergence rate $c_{1}$ and $c_{2}$ respectively, we say that $\mathbf{K}_{1}$ dominates $\boldsymbol{K}_{2}$ if $c_{1} \gg c_{2}$.

As a concrete example, we train a fully-connected neural network with one hidden layer and 100 neurons to solve the model Problem 7.1 with a fabricated solution $u(x)=\sin (a \pi x)$ for different frequency amplitudes $a$. Fig. 1 shows all eigenvalues of $\boldsymbol{K}, \boldsymbol{K}_{u u}$ and $\boldsymbol{K}_{r r}$ at initialization in descending order. As with conventional deep fully-connected networks, the eigenvalues of the PINNs' NTK decay rapidly and most of the eigenvalues are near zero. Moreover, the distribution of eigenvalues of $\boldsymbol{K}$ looks similar for different frequency functions (different $a$ ), which may heuristically explain that PINNs tend to learn all frequencies almost simultaneously, as observed in Lu et al. [43].

Another key observation here is that the eigenvalues of $\boldsymbol{K}_{r r}$ are much greater than $\boldsymbol{K}_{u u}$, namely $\boldsymbol{K}_{r r}$ dominates $\boldsymbol{K}_{u u}$ by Definition 5.1. As a consequence, the PDE residual converges much faster than fitting the PDE boundary conditions, which may prevent the network from approximating the correct solution. From the authors' experience, high frequency functions typically lead to high eigenvalues in $\boldsymbol{K}_{r r}$, but in some cases $\boldsymbol{K}_{u u}$ can dominate $\boldsymbol{K}_{r r}$. We believe that such a discrepancy between $\boldsymbol{K}_{u u}$ and $\boldsymbol{K}_{r r}$ is one of the key fundamental reasons behind why PINNs can often fail to train and yield accurate predictions. In light of this evidence, in the next section, we describe a practical technique to address this pathology by appropriately assigning weights to the different terms in a PINNs loss function.

\section*{6. Practical insights}

In this section, we consider general PDEs of the form (3.1) - (3.2) by leveraging the NTK theory we developed for PINNs. We approximate the latent solution $u(\boldsymbol{x})$ by a fully-connected neural network $u(\boldsymbol{x}, \boldsymbol{\theta})$ with multiple hidden layers, and train its parameters $\boldsymbol{\theta}$ by minimizing the following composite loss function
$$
\begin{align*}
\mathcal{L}(\boldsymbol{\theta}) & =\mathcal{L}_{b}(\boldsymbol{\theta})+\mathcal{L}_{r}(\boldsymbol{\theta})  \tag{6.1}\\
& =\frac{\lambda_{b}}{2 N_{b}} \sum_{i=1}^{N_{b}}\left|u\left(\boldsymbol{x}_{b}^{i} ; \boldsymbol{\theta}\right)-g\left(\boldsymbol{x}_{b}^{i}\right)\right|^{2}+\frac{\lambda_{r}}{2 N_{r}} \sum_{i=1}^{N_{r}}\left|r\left(\boldsymbol{x}_{r}^{i} ; \boldsymbol{\theta}\right)\right|^{2}, \tag{6.2}
\end{align*}
$$
where $\lambda_{b}$ and $\lambda_{r}$ are some hyper-parameters which may be tuned manually or automatically by utilizing the backpropagated gradient statistics during training [28]. Here, the training data $\left\{\boldsymbol{x}_{b}^{i}, g\left(\boldsymbol{x}_{b}^{i}\right)\right\}_{i=1}^{N_{b}}$ and $\left\{\boldsymbol{x}_{r}^{i}, f\left(\boldsymbol{x}_{b}^{i}\right)\right\}$ may correspond to the full data-batch or mini-batches that are randomly sampled at each iteration of gradient descent.

Similar to the proof of Lemma 3.1, we can derive the dynamics of the outputs $u(\boldsymbol{x}, \boldsymbol{\theta})$ and $u(\boldsymbol{x}, \boldsymbol{\theta})$ corresponding to the above loss function as
$$
\left[\begin{array}{c}
\frac{d u\left(\boldsymbol{x}_{b} ; \boldsymbol{\theta}(t)\right)}{d t}  \tag{6.3}\\
\frac{d \mathcal{N}[u]\left(\boldsymbol{x}_{r} ; \boldsymbol{\theta}(t)\right)}{d t}
\end{array}\right]=-\left[\begin{array}{cc}
\frac{\lambda_{b}}{N_{b}} \boldsymbol{K}_{u u}(t) & \frac{\lambda_{r}}{N_{r}} \boldsymbol{K}_{u r}(t) \\
\frac{\lambda_{b}}{N_{b}} \boldsymbol{K}_{r u}(t) & \frac{\lambda_{r}}{N_{r}} \boldsymbol{K}_{r r}(t)
\end{array}\right] \cdot\left[\begin{array}{c}
u\left(\boldsymbol{x}_{b} ; \boldsymbol{\theta}(t)\right)-g\left(\boldsymbol{x}_{b}\right) \\
\mathcal{N}[u]\left(\boldsymbol{x}_{r} ; \boldsymbol{\theta}(t)\right)-f\left(\boldsymbol{x}_{r}\right)
\end{array}\right]
$$
$$
:=-\widetilde{\boldsymbol{K}}(t) \cdot\left[\begin{array}{c}
u\left(\boldsymbol{x}_{b} ; \boldsymbol{\theta}(t)\right)-g\left(\boldsymbol{x}_{b}\right)  \tag{6.4}\\
\mathcal{N}[u]\left(\boldsymbol{x}_{r} ; \boldsymbol{\theta}(t)\right)-f\left(\boldsymbol{x}_{r}\right)
\end{array}\right],
$$
where $\boldsymbol{K}_{u u}, \boldsymbol{K}_{u r}$ and $\boldsymbol{K}_{r r}$ are defined to be the same as in equation (3.9). From simple stability analysis of a gradient descent (i.e. forward Euler [44]) discretization of above ODE system, the maximum learning rate should be less than or equal to $2 / \lambda_{\max }(\widetilde{\boldsymbol{K}}(t))$. Also note that an alternative mechanism for controlling stability is to increase the batch size, which effectively corresponds to decreasing the learning rate. Recall that the current setup in the main theorems put forth in this work holds for the model problem in equation (4.1) and fully-connected networks of one hidden layer with an NTK parameterization. This implies that, for general nonlinear PDEs, the NTK of PINNs may not remain fixed during training. Nevertheless, as mentioned in Remark 3.4, we emphasize that, given an infinitesimal learning rate, equation (6.3) holds for any network architecture and any differential operator. Similarly, the singular values of NTK $\widetilde{\boldsymbol{K}}(t)$ determine the convergence rate of the training error using singular value decomposition, since $\widetilde{\boldsymbol{K}}(t)$ may not necessarily be semi-positive definite. Therefore, we can still understand the training dynamics of PINNs by tracking their NTK $\widetilde{\boldsymbol{K}}(t)$ during training, even for general nonlinear PDE problems.

A key observation here is that the magnitude of $\lambda_{b}, \lambda_{r}$, as well as the size of mini-batch would have a crucial impact on the singular values of $\widetilde{\boldsymbol{K}}(t)$, and, thus, the convergence rate of the training error of $u\left(\boldsymbol{x}_{b} ; \boldsymbol{\theta}\right)$ and $\mathcal{N}[u]\left(\boldsymbol{x}_{r} ; \boldsymbol{\theta}\right)$. For instance, if we increase $\lambda_{b}$ and fix the batch size $N_{b}, N_{r}$ and the weight $\lambda_{r}$, then this will improve the convergence rate of $u\left(\boldsymbol{x}_{b}, \boldsymbol{\theta}\right)$. Furthermore, in the sense of convergence rate, changing the weights $\lambda_{b}$ or $\lambda_{r}$ is equivalent to changing the corresponding batch size $N_{b}, N_{r}$. Based on these observations, we can overcome the discrepancy between $\boldsymbol{K}_{u u}$ and $\boldsymbol{K}_{r r}$ discussed in section 5 by calibrating the weights or batch size such that each component of $u\left(\boldsymbol{x}_{r} ; \boldsymbol{\theta}\right)$ and $\mathcal{N}[u]\left(\boldsymbol{x}_{r} ; \boldsymbol{\theta}\right)$ has similar convergence rate in magnitude. Since manipulating the batch size may involve extra computational costs (e.g., it may result to prohibitively very large batches), here we fix the batch size and just consider adjusting the weights $\lambda_{b}$ or $\lambda_{r}$ according to Algorithm 1 .
```
Algorithm 1: Adaptive weights for physics-informed neural networks.
    Consider a physics-informed neural network $u(\boldsymbol{x} ; \boldsymbol{\theta})$ with parameters $\boldsymbol{\theta}$ and a loss function
            $\mathcal{L}(\boldsymbol{\theta}):=\lambda_{b} \mathcal{L}_{b}(\boldsymbol{\theta})+\lambda_{r} \mathcal{L}_{r}(\boldsymbol{\theta})$,
    where $\mathcal{L}_{r}(\boldsymbol{\theta})$ denotes the PDE residual loss and $\mathcal{L}_{r}(\boldsymbol{\theta})$ corresponds to boundary conditions. Initialize the weights $\lambda_{b}, \lambda_{r}$ to 1 and use $S$ steps of a
    gradient descent algorithm to update the parameters $\boldsymbol{\theta}$ as:
    for $n=1, \ldots, S$ do
        (a) Compute $\lambda_{b}$ and $\lambda_{r}$ by
                $\lambda_{b}=\frac{\sum_{i=1}^{N_{r}+N_{b}} \lambda_{i}(n)}{\sum_{i=1}^{N_{b}} \lambda_{i}^{u u}(n)}=\frac{\operatorname{Tr}(\boldsymbol{K}(n))}{\operatorname{Tr}\left(\boldsymbol{K}_{u u}(n)\right)}$
                $\lambda_{r}=\frac{\sum_{i=1}^{N_{r}+N_{b}} \lambda_{i}(n)}{\sum_{i=1}^{N_{r}} \lambda_{i}^{r r}(n)}=\frac{\operatorname{Tr}(\boldsymbol{K}(n))}{\operatorname{Tr}\left(\boldsymbol{K}_{r r}(n)\right)}$
```

```
        where $\lambda_{i}(n), \lambda_{i}^{u u}(n)$ and $\lambda_{i}^{r r}(n)$ are eigenvalues of $\boldsymbol{K}(n), \boldsymbol{K}_{u u}(n), \boldsymbol{K}_{r r}(n)$ at $n$-th iteration.
        (b) Update the parameters $\boldsymbol{\theta}$ via gradient descent
                $\boldsymbol{\theta}_{n+1}=\boldsymbol{\theta}_{n}-\eta \nabla_{\boldsymbol{\theta}} \mathcal{L}\left(\boldsymbol{\theta}_{n}\right)$
    end
```


First we remark that the updates in equations (6.5) and (6.6) can either take place at every iteration of the gradient descent loop, or at a frequency specified by the user (e.g., every 10 gradient descent steps). To compute the sum of eigenvalues, it suffices to compute the trace of the corresponding NTK matrices, which can save some computational resources. Besides, we point out that the computation of the NTK $\boldsymbol{K}(t)$ is associated with the training data points fed to the network at each iteration, which means that the values of the kernel are not necessarily same at each iteration. However, if we assume that all training data points are sampled from the same distribution and the change of NTK at each iteration is negligible, then the computed kernel should be approximately equal up to a permutation matrix. As a result, the change of eigenvalues of $\boldsymbol{K}(t)$ at each iteration is also negligible and thus the training process of Algorithm 1 should be stable. In section 7.2, we performed detailed numerical experiments to validate the effectiveness of the proposed algorithm.

Here we also note that, in previous work, Wang et al. introduced an alternative empirical approach for automatically tuning the weights $\lambda_{b}$ or $\lambda_{r}$ with the goal of balancing the magnitudes of the back-propagated gradients originating from different terms in a PINNs loss function. While effective in practice, this approach lacked any theoretical justification and did not provide a deeper insight into the training dynamics of PINNs. In contrast, the approach presented here follows naturally from the NTK theory derived in section 4, and aims to trace and tackle the pathological convergence behavior of PINNs at its root.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/898f93b4-fd47-41e4-87e3-9a7fc6bc993f-10.jpg?height=482&width=1492&top_left_y=181&top_left_x=197}
\captionsetup{labelformat=empty}
\caption{Fig. 2. Model Problem 7.1 (1D Poisson equation): (a) (b) The relative change of parameters $\boldsymbol{\theta}$ and the NTK of PINNs $\boldsymbol{K}$ obtained by training a fully-connected neural network with one hidden layer and different widths $(10,100,500)$ via 10,000 iterations of full-batch gradient descent with a learning rate of $10^{-5}$. (c) The eigenvalues of the NTK $\boldsymbol{K}$ at initialization and at the last step ( $\left(n=10^{4}\right)$ of training a width $=500$ fully-connected neural network.}
\end{figure}

\section*{7. Numerical experiments}

In this section, we provide a series of numerical studies that aim to validate our theory or access the performance of the proposed algorithm against the standard PINNs [27] for inferring the solution of PDEs. Throughout numerical experiments we will approximate the latent variables by fully-connected neural networks with NTK parameterization (2.3) and hyperbolic tangent activation functions. All networks are trained using standard stochastic gradient descent, unless otherwise specified. Finally, all results presented in this section can be reproduced using our publicly available code https:// github.com/PredictiveIntelligenceLab/PINNsNTK.

\subsection*{7.1. Convergence of the NTK of PINNs}

As our first numerical example, we still focus on the model problem (4.1) and verify the convergence of the PINNs' NTK. Specifically, we set $\Omega$ to be the unit interval $[0,1]$ and fabricate the exact solution to this problem taking the form $u(x)=\sin (\pi x)$. The corresponding $f$ and $g$ are given by
$$
\begin{aligned}
& f(x)=-\pi^{2} \sin (\pi x), \quad x \in[0,1] \\
& g(x)=0, \quad x=0,1 .
\end{aligned}
$$

We proceed by approximating the latent solution $u(x)$ by a fully-connected neural network $u(x ; \boldsymbol{\theta})$ of one hidden layer with NTK parameterization (see equation (2.3)), and a hyperbolic tangent activation function. The corresponding loss function is given by
$$
\begin{align*}
\mathcal{L}(\boldsymbol{\theta}) & =\mathcal{L}_{b}(\boldsymbol{\theta})+\mathcal{L}_{r}(\boldsymbol{\theta})  \tag{7.1}\\
& =\frac{1}{2} \sum_{i=1}^{N_{b}}\left|u\left(\boldsymbol{x}_{b}^{i} ; \boldsymbol{\theta}\right)-g\left(\boldsymbol{x}_{b}^{i}\right)\right|^{2}+\frac{1}{2} \sum_{i=1}^{N_{r}}\left|u_{x x}\left(\boldsymbol{x}_{r}^{i} ; \boldsymbol{\theta}\right)-f\left(\boldsymbol{x}_{r}^{i}\right)\right|^{2} \tag{7.2}
\end{align*}
$$

Here we choose $N_{b}=N_{r}=100$ and the collocation points $\left\{x_{r}^{i}\right\}_{i=1}^{N_{r}}$ are uniformly spaced in the unit interval. To monitor the change of the NTK $\boldsymbol{K}(t)$ for this PINN model, we train the network for different widths and for 10,000 iterations by minimizing the loss function given above using standard full-batch gradient descent with a learning rate of $10^{-5}$. Here we remark that, in order to keep the gradient descent dynamics (3.8) steady, the learning rate should be less than $2 / \lambda_{\max }$, where $\lambda_{\text {max }}$ denotes the maximum eigenvalue of $\mathbf{K}(t)$.

Fig. 2a and 2b present the relative change in the norm of network's weights and NTK (starting from a random initialization) during training. As it can be seen, the change of both the weights and the NTK tends to zero as the width of the network grows to infinity, which is consistent with Lemma D. 2 and Theorem 4.4. Moreover, we know that convergence in a matrix norm implies convergence in eigenvalues, and eigenvalues characterize the properties of a given matrix. To this end, we compute and monitor all eigenvalues of $\boldsymbol{K}(\mathrm{t})$ of the network for width $=500$ at initialization and after 10,000 steps of gradient and plot them in descending order in Fig. 2c. As expected, we see that all eigenvalues barely change for these two snapshots. Based on these observations, we may conclude that the NTK of PINNs with one hidden layer stays almost fixed during training.

However, PINNs of multiple hidden layers are not covered by our theory at the moment. Out of interest, we also investigate the relative change of weights, kernel, as well as the kernel's eigenvalues for a fully-connected network with three hidden layers (see Fig. 3). We can observe that the change in both the weights and the NTK behaves almost identical to the

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/898f93b4-fd47-41e4-87e3-9a7fc6bc993f-11.jpg?height=482&width=1496&top_left_y=181&top_left_x=195}
\captionsetup{labelformat=empty}
\caption{Fig. 3. Model Problem 7.1 (1D Poisson equation): (a) (b) The relative change of parameters $\boldsymbol{\theta}$ and the NTK of PINNs $\boldsymbol{K}$ obtained by training a fully-connected neural network with three hidden layers and different widths $(10,100,500)$ via 10,000 iterations of full-batch gradient descent with a learning rate of $10^{-5}$. (c) The eigenvalues of the NTK $\boldsymbol{K}$ at initialization and at the last step $\left(n=10^{4}\right)$ of training a width $=500$ fully-connected neural network.}
\end{figure}
case of a fully-connected network with one hidden layer shown in Fig. 2. Therefore we may conjecture that, for any linear or even nonlinear PDEs, the NTK of PINNs converges to a deterministic kernel and remains constant during training in the infinite width limit.

\subsection*{7.2. Adaptive training for PINNs}

In this section, we aim to validate the developed theory and examine the effectiveness of the proposed adaptive training algorithm on the model problem of equation 7.1. To this end, we consider a fabricated exact solution of the form $u(x)= \sin (4 \pi x)$, inducing a corresponding forcing term $f$ and Dirichlet boundary condition $g$ given by
$$
\begin{aligned}
& f(x)=-16 \pi^{2} \sin (4 \pi x), \quad x \in[0,1] \\
& g(x)=0, \quad x=0,1 .
\end{aligned}
$$

We proceed by approximating the latent solution $u(x)$ by a fully-connected neural network with one hidden layer and width set to 100 . Recall from Theorem 4.3 and Theorem 4.4, that the NTK barely changes during training. This implies that the weights $\lambda_{b}, \lambda_{r}$ are determined by NTK at initialization and thus they can be regarded as fixed weights during training. Moreover, from Fig. 1, we already know that $\boldsymbol{K}_{r r}$ dominates $\boldsymbol{K}_{u u}$ for this example. Therefore, the updating rule for hyper-parameters $\lambda_{b}, \lambda_{r}$ at $t$ step of gradient descent can be reduced to
$$
\begin{align*}
& \lambda_{b}=\frac{\sum_{i=1}^{N_{b}+N_{r}} \lambda_{i}(t)}{\sum_{i=1}^{N_{b}} \lambda_{i}^{u u}(t)} \approx \frac{\sum_{i=1}^{N_{r}} \lambda_{i}^{r r}(t)}{\sum_{i=1}^{N_{b}} \lambda_{i}^{u u}(t)} \approx \frac{\operatorname{Tr}\left(\boldsymbol{K}_{r r}(0)\right)}{\operatorname{Tr}\left(\boldsymbol{K}_{u u}(0)\right)}  \tag{7.3}\\
& \lambda_{r}=\frac{\sum_{i=1}^{N_{b}+N_{r}} \lambda_{i}(t)}{\sum_{i=1}^{N_{r}} \lambda_{i}^{r r}(t)} \approx 1 \tag{7.4}
\end{align*}
$$

We proceed by training the network via full-batch gradient descent with a learning rate of $10^{-5}$ to minimize the following loss function
$$
\begin{aligned}
\mathcal{L}(\boldsymbol{\theta}) & =\mathcal{L}_{b}(\boldsymbol{\theta})+\mathcal{L}_{r}(\boldsymbol{\theta}) \\
& =\frac{\lambda_{b}}{2 N_{b}} \sum_{i=1}^{N_{b}}\left|u\left(\boldsymbol{x}_{b}^{i} ; \boldsymbol{\theta}\right)-g\left(\boldsymbol{x}_{b}^{i}\right)\right|^{2}+\frac{\lambda_{r}}{2 N_{r}} \sum_{i=1}^{N_{r}}\left|u_{x x}\left(\boldsymbol{x}_{r}^{i} ; \boldsymbol{\theta}\right)-f\left(\boldsymbol{x}_{r}^{i}\right)\right|^{2},
\end{aligned}
$$
where the batch sizes are $N_{b}=N_{r}=100, \lambda_{r}=1$ and the computed $\lambda_{b} \approx 100$.
A comparison of predicted solution $u(x)$ between the original PINNs ( $\lambda_{b}=\lambda_{r}=1$ ) and PINNs with adaptive weights after 40,000 iterations are shown in Fig. 4. It can be observed that the proposed algorithm yields a much more accurate predicted solution and improves the relative $L^{2}$ error by about two orders of magnitude. Furthermore, we also investigate how the predicted performance of PINNs depends on the choice of different weights in the loss function. To this end, we fix $\lambda_{r}=1$ and train the same network, but now we manually tune $\lambda_{b}$. Fig. 5 presents a visual assessment of relative $L^{2}$ errors of predicted solutions for different $\lambda_{b} \in[1,500]$ averaged over ten independent trials. One can see that the relative $L^{2}$ error decreases rapidly to a local minimum as $\lambda_{b}$ increases from 1 to about 100 and then shows oscillations as $\lambda_{b}$ continues to

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/898f93b4-fd47-41e4-87e3-9a7fc6bc993f-12.jpg?height=1183&width=1312&top_left_y=169&top_left_x=282}
\captionsetup{labelformat=empty}
\caption{Fig. 4. Model Problem 7.2 (1D Poisson equation): (a) The predicted solution against the exact solution obtained by training a fully-connected neural network of one hidden layer with width $=100$ via 40,000 iterations of full-batch gradient descent with a learning rate of $10^{-5}$. The relative $L^{2}$ error is $2.40 e-01$. (b) The predicted solution against the exact solution obtained by training the same neural network using fixed weights $\lambda_{b}=100, \lambda_{r}=1$ via 40 , 000 iterations of full-batch gradient descent with a learning rate of $10^{-5}$. The relative $L^{2}$ error is $1.63 e-03$.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/898f93b4-fd47-41e4-87e3-9a7fc6bc993f-12.jpg?height=517&width=650&top_left_y=1573&top_left_x=616}
\captionsetup{labelformat=empty}
\caption{Fig. 5. Model problem of equation 7.2 (1D Poisson equation): The relative $L^{2}$ error of predicted solutions averaged over 10 independent trials by training a fully-connected neural network of one hidden layer with width $=100$ using different fixed weights $\lambda_{b} \in[1,500]$ for 40,000 gradient descent iterations.}
\end{figure}
increase. Moreover, a large magnitude of $\lambda_{b}$ seems to lead to a large standard deviation in the $L^{2}$ error, which may be due to the imaginary eigenvalues of the indefinite kernel $\widetilde{\boldsymbol{K}}$ resulting in an unstable training process. This empirical simulation study confirms that the weights $\lambda_{r}=1$ and $\lambda_{b}$ suggested by our theoretical analysis based on analyzing the NTK spectrum are robust and closely agree with the optimal weights obtained via manual hyper-parameter tuning.

\subsection*{7.3. One-dimensional wave equation}

As our last example, we present a study that demonstrates the effectiveness of Algorithm 1 in a practical problem for which conventional PINNs models face severe diffuculties. To this end, we consider a one-dimensional wave equation in the domain $\Omega=[0,1] \times[0,1]$ taking the form
$$
\begin{align*}
& u_{t t}(x, t)-4 u_{x x}(x, t)=0, \quad(x, t) \in(0,1) \times(0,1)  \tag{7.5}\\
& u(0, t)=u(1, t)=0, \quad t \in[0,1]  \tag{7.6}\\
& u(x, 0)=\sin (\pi x)+\frac{1}{2} \sin (4 \pi x), \quad x \in[0,1]  \tag{7.7}\\
& u_{t}(x, 0)=0, \quad x \in[0,1] \tag{7.8}
\end{align*}
$$

First, by d'Alembert's formula [45], the solution $u(x, t)$ is given by
$$
\begin{equation*}
u(x, t)=\sin (\pi x) \cos (2 \pi t)+\frac{1}{2} \sin (4 \pi x) \cos (8 \pi t) \tag{7.9}
\end{equation*}
$$

Here we treat the temporal coordinate $t$ as an additional spatial coordinate in $\boldsymbol{x}$ and then the initial condition (7.7) can be included in the boundary condition (7.6), namely
$$
u(\boldsymbol{x})=g(\boldsymbol{x}), \quad x \in \partial \Omega
$$

Now we approximate the solution $u$ by a 5 -layer deep fully-connected network $u(\boldsymbol{x}, \boldsymbol{\theta})$ with 500 neurons per hidden layer, where $\boldsymbol{x}=(x, t)$. Then we can formulate a "physics-informed" loss function by
$$
\begin{align*}
\mathcal{L}(\boldsymbol{\theta}) & =\lambda_{u} \mathcal{L}_{u}(\boldsymbol{\theta})+\lambda_{u_{t}} \mathcal{L}_{u_{t}}(\boldsymbol{\theta})+\lambda_{r} \mathcal{L}_{r}(\boldsymbol{\theta})  \tag{7.10}\\
& =\frac{\lambda_{u}}{2 N_{u}} \sum_{i=1}^{N_{u}}\left|u\left(\boldsymbol{x}_{u}^{i} ; \boldsymbol{\theta}\right)-g\left(\boldsymbol{x}_{u}^{i}\right)\right|^{2}+\frac{\lambda_{u_{t}}}{2 N_{u_{t}}} \sum_{i=1}^{N_{u_{t}}}\left|u_{t}\left(\boldsymbol{x}_{u_{t}}^{i} ; \boldsymbol{\theta}\right)\right|^{2}+\frac{\lambda_{r}}{2 N_{r}} \sum_{i=1}^{N_{r}}\left|\mathcal{N} u\left(\boldsymbol{x}_{r}^{i} ; \boldsymbol{\theta}\right)\right|^{2}, \tag{7.11}
\end{align*}
$$
where the hyper-parameters $\lambda_{u}, \lambda_{u_{t}}, \lambda_{r}$ are initialized to 1 , the batch sizes are set to $N_{u}=N_{u_{t}}=N_{r}=300$, and $\mathcal{N}= \partial_{t t}-4 \partial_{x x}$. Here all training data are uniformly sampling inside the computational domain at each gradient descent iteration. The network $u(\boldsymbol{x} ; \boldsymbol{\theta})$ is initialized using the standard Glorot scheme [46] and then trained by minimizing the above loss function via stochastic gradient descent using the Adam optimizer with default settings [47]. Fig. 6a provides a comparison between the predicted solution against the ground truth obtained after 80,000 training iterations. Clearly the original PINN model fails to approximate the ground truth solution and the relative $L^{2}$ error is above $40 \%$.

To explore the reason behind PINN's failure for this example, we compute its NTK and track it during training. Similar to the proof of Lemma 3.1, the corresponding NTK can be derived from the loss function (7.10)
$$
\left[\begin{array}{c}
\frac{d u\left(\boldsymbol{x}_{u} ; \boldsymbol{\theta}(t)\right)}{d t}  \tag{7.12}\\
\frac{d u_{t}\left(\boldsymbol{x}_{u_{t}} ; \boldsymbol{\theta}(t)\right)}{d t} \\
\frac{d \mathcal{N}[u]\left(\boldsymbol{x}_{r} ; \boldsymbol{\theta}(t)\right)}{d t}
\end{array}\right]:=\widetilde{\boldsymbol{K}}(t) \cdot\left[\begin{array}{c}
u\left(\boldsymbol{x}_{b} ; \boldsymbol{\theta}(t)\right)-g\left(\boldsymbol{x}_{b}\right) \\
u_{t}\left(\boldsymbol{x}_{u_{t}} ; \boldsymbol{\theta}(t)\right) \\
\mathcal{N}[u]\left(\boldsymbol{x}_{r} ; \boldsymbol{\theta}(t)\right)
\end{array}\right],
$$
where
$$
\begin{aligned}
& \widetilde{\boldsymbol{K}}(t)=\left[\begin{array}{c}
\frac{\lambda_{u}}{N_{u}} \boldsymbol{J}_{u}(t) \\
\frac{\lambda_{u_{u}}}{N_{u_{t}}} \boldsymbol{J}_{u_{t}}(t) \\
\frac{\lambda_{r}}{N_{r}} \boldsymbol{J}_{r}(t)
\end{array}\right] \cdot\left[\boldsymbol{J}_{u}^{T}(t), \boldsymbol{J}_{u_{t}}^{T}(t), \boldsymbol{J}_{r}^{T}(t)\right] \\
& {\left[\boldsymbol{K}_{u}(t)\right]_{i j}=\left[\boldsymbol{J}_{u}(t) \boldsymbol{J}_{u}^{T}(t)\right]_{i j}=\left\langle\frac{d u\left(\boldsymbol{x}_{u}^{i} ; \boldsymbol{\theta}(t)\right)}{d \boldsymbol{\theta}}, \frac{d u\left(\boldsymbol{x}_{u}^{j} ; \boldsymbol{\theta}(t)\right)}{d \boldsymbol{\theta}}\right\rangle} \\
& {\left[\boldsymbol{K}_{u_{t}}(t)\right]_{i j}=\left[\boldsymbol{J}_{u_{t}}(t) \boldsymbol{J}_{u_{t}}^{T}(t)\right]_{i j}=\left\langle\frac{d u_{t}\left(\boldsymbol{x}_{u_{t}}^{i} ; \boldsymbol{\theta}(t)\right)}{d \boldsymbol{\theta}}, \frac{d u\left(\boldsymbol{x}_{u_{t}}^{j} ; \boldsymbol{\theta}(t)\right)}{d \boldsymbol{\theta}}\right\rangle} \\
& {\left[\boldsymbol{K}_{r}(t)\right]_{i j}=\left[\boldsymbol{J}_{r}(t) \boldsymbol{J}_{r}^{T}(t)\right]_{i j}=\left\langle\frac{d \mathcal{N}[u]\left(\boldsymbol{x}_{r}^{i} ; \boldsymbol{\theta}(t)\right)}{d \boldsymbol{\theta}}, \frac{d \mathcal{N}[u]\left(\boldsymbol{x}_{r}^{j} ; \boldsymbol{\theta}(t)\right)}{d \boldsymbol{\theta}}\right\rangle}
\end{aligned}
$$

A visual assessment of the eigenvalues of $\boldsymbol{K}_{u}, \boldsymbol{K}_{u_{t}}$ and $\boldsymbol{K}_{r}$ at initialization and the last step of gradient descent are presented in Fig. 7. It can be observed that the NTK does not remain fixed and all eigenvalues move "outward" in the beginning of the training, and then remain almost static such that $\boldsymbol{K}_{r}$ and $\boldsymbol{K}_{u_{t}}$ dominate $\boldsymbol{K}_{u}$ during training. Consequently, the components

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/898f93b4-fd47-41e4-87e3-9a7fc6bc993f-14.jpg?height=998&width=1642&top_left_y=168&top_left_x=127}
\captionsetup{labelformat=empty}
\caption{Fig. 6. One-dimensional wave equation: (a) The predicted solution versus the exact solution by training a fully-connected neural network with five hidden layers and 500 neurons per layer using the Adam optimizer with default settings [47] after 80,000 iterations. The relative $L^{2}$ error is $4.518 e-01$. (b) The predicted solution versus the exact solution by training the same network using Algorithm 1 after 80,000 iterations. The relative $L^{2}$ error is $1.728 e-03$. (For interpretation of the colors in the figure(s), the reader is referred to the web version of this article.)}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/898f93b4-fd47-41e4-87e3-9a7fc6bc993f-14.jpg?height=470&width=1612&top_left_y=1358&top_left_x=137}
\captionsetup{labelformat=empty}
\caption{Fig. 7. One-dimensional wave equation: Eigenvalues of $\boldsymbol{K}_{u}, \boldsymbol{K}_{u_{t}}$ and $\boldsymbol{K}_{r}$ at different snapshots during training, sorted in descending order.}
\end{figure}
of $u_{t}\left(\boldsymbol{x}_{u_{t}} ; \boldsymbol{\theta}\right)$ and $\mathcal{N}[u]\left(\boldsymbol{x}_{r} ; \boldsymbol{\theta}\right)$ converge much faster than the loss of boundary conditions, and, therefore, introduce a severe discrepancy in the convergence rate of each different term in the loss, causing this standard PINNs model to collapse. To verify our hypothesis, we also train the same network using Algorithm 1 with the following generalized updating rule for hyper-parameters $\lambda_{u}, \lambda_{u_{t}}$ and $\lambda_{r}$
$$
\begin{align*}
\lambda_{u} & =\frac{\operatorname{Tr}\left(\boldsymbol{K}_{u}\right)+\operatorname{Tr}\left(\boldsymbol{K}_{u_{t}}\right)+\operatorname{Tr}\left(\boldsymbol{K}_{r}\right)}{\operatorname{Tr}\left(\boldsymbol{K}_{u}\right)}  \tag{7.13}\\
\lambda_{u_{t}} & =\frac{\operatorname{Tr}\left(\boldsymbol{K}_{u}\right)+\operatorname{Tr}\left(\boldsymbol{K}_{u_{t}}\right)+\operatorname{Tr}\left(\boldsymbol{K}_{r}\right)}{\operatorname{Tr}\left(\boldsymbol{K}_{u_{t}}\right)}  \tag{7.14}\\
\lambda_{r} & =\frac{\operatorname{Tr}\left(\boldsymbol{K}_{u}\right)+\operatorname{Tr}\left(\boldsymbol{K}_{u_{t}}\right)+\operatorname{Tr}\left(\boldsymbol{K}_{r}\right)}{\operatorname{Tr}\left(\boldsymbol{K}_{r}\right)} . \tag{7.15}
\end{align*}
$$

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/898f93b4-fd47-41e4-87e3-9a7fc6bc993f-15.jpg?height=561&width=1196&top_left_y=177&top_left_x=342}
\captionsetup{labelformat=empty}
\caption{Fig. 8. One-dimensional wave equation: (a) The evolution of hyper-parameters $\lambda_{u}, \lambda_{u_{t}}$ and $\lambda_{r}$ during training of a five-layer deep fully-connected neural network with 500 neurons per layer using Algorithm 1. (b) The eigenvalues of $\boldsymbol{K}_{u}, \boldsymbol{K}_{u_{t}}, \boldsymbol{K}_{r}$ and $\lambda_{u} \boldsymbol{K}_{u}, \lambda_{u_{t}} \boldsymbol{K}_{u_{t}}, \lambda_{r} \boldsymbol{K}_{r}$ at last step of training.}
\end{figure}

In particular, we update these weights every 1,000 training iterations, hence the extra computational costs compared to a standard PINNs approach is negligible. The results of this experiment are shown in Fig. 6b, from which one can easily see that the predicted solution obtained using the proposed adaptive training scheme achieves excellent agreement with the ground truth and the relative $L^{2}$ error is $1.73 e-3$. To quantify the effect of the hyper-parameters $\lambda_{u}, \lambda_{u_{t}}$ and $\lambda_{r}$ on the NTK, we also compare the eigenvalues of $\boldsymbol{K}_{u}, \boldsymbol{K}_{u_{t}}$ and $\boldsymbol{K}_{r}$ multiplied with or without the hyper-parameters at last step of gradient descent. As it can be seen in Fig. 8b, the discrepancy of the convergence rate of different components in total training errors is considerably resolved. Furthermore, Fig. 8a presents the change of weights during training and we can see that $\lambda_{u}, \lambda_{u_{t}}$ increase rapidly and then remain almost fixed while $\lambda_{r}$ is near 1 for all time. So we may conclude that the overall training process using Algorithm 1 is stable.

\section*{8. Discussion}

This work has produced a novel theoretical understanding of physics-informed neural networks by deriving and analyzing their limiting neural tangent kernel. Specifically, we first show that infinitely wide physics-informed neural networks under the NTK parameterization converge to Gaussian processes. Furthermore, we derive NTK of PINNs and show that, under suitable assumptions, it converges to a deterministic kernel and barely changes during training as the width of the network grows to infinity. To provide further insight, we analyze the training dynamics of fully-connected PINNs through the lens of their NTK and show that not only they suffer from spectral bias, but they also exhibit a discrepancy in the convergence rate among the different loss components contributing to the total training error. To resolve this discrepancy, we propose a novel algorithm such that the coefficients of different terms in a PINNs' loss function can be dynamically updated according to balance the average convergence rate of different components in the total training error. Finally, we carry out a series of numerical experiments to verify our theory and validate the effectiveness of the proposed algorithms.

Although this work takes an essential step towards understanding PINNs and their training dynamics, there are many open questions worth exploring. Can the proposed NTK theory for PINNs be extended fully-connected networks with multiple hidden layers, nonlinear equations, as well as the neural network architectures such as convolutional neural networks, residual networks, etc.? To which extend these architecture suffer from spectral bias or exhibit similar discrepancies in their convergence rate? In a parallel thrust, it is well-known that PINNs perform much better for inverse problems than for forward problems, such as the ones considered in this work. Can we incorporate the current theory to analyze inverse problems and explain they are better suited to PINNs? Moreover, going beyond vanilla gradient descent dynamics, how do the training dynamics of PINNs and their corresponding NTK evolve via gradient descent with momentum (e.g. Adam [47])? In practice, despite some improvements in the performance of PINNs brought by assigning appropriate weights to the loss function, we emphasize that such methods cannot change the distribution of eigenvalues of the NTK and, thus, cannot directly resolve spectral bias. Apart from this, assigning weights may result in indefinite kernels which can have imaginary eigenvalues and thus yield unstable training processes. Therefore, can we come up with better methodologies to resolve spectral bias using specialized network architectures, loss functions, etc.? In a broad sense, PINNs can be regarded as a special multi-task learning problem in which a neural network is asked to simultaneously fit the observed data and minimize a PDE residual. It is then natural to ask: does the proposed NTK theory hold for general multi-task learning problems? Can other heuristic methods for multi-task learning [48-50] be analyzed and improved under this setting? We believe that answering these questions not only paves a new way to better understand PINNs and its training dynamics, but also opens a new door for developing scientific machine learning algorithms with provable convergence guarantees, as needed for many critical applications in computational science and engineering.

\section*{CRediT authorship contribution statement}

Sifan Wang: Conceptualization, Methodology, Implementation, Numerical experiments, Visualization, Draft Preparation.
Xinling Yu: Numerical experiments.
Paris Perdikaris: Conceptualization, Methodology, Supervision, Funding, Writing-Reviewing and Editing.

\section*{Declaration of competing interest}

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

\section*{Acknowledgements}

This work received support from DOE grant DE-SC0019116, AFOSR grant FA9550-20-1-0060, and DOE-ARPA grant DEAR0001201.

\section*{Appendix A. Proof of Lemma 3.1}

Proof. Recall that for given training data $\left\{\boldsymbol{x}_{b}^{i}, \boldsymbol{g}\left(\boldsymbol{x}_{b}^{i}\right)\right\}_{i=1}^{N_{b}},\left\{\boldsymbol{x}_{r}^{i}, f\left(\boldsymbol{x}_{r}^{i}\right)\right\}_{i=1}^{N_{r}}$, the loss function is given by
$$
\begin{aligned}
\mathcal{L}(\boldsymbol{\theta}) & =\mathcal{L}_{b}(\boldsymbol{\theta})+\mathcal{L}_{r}(\boldsymbol{\theta}) \\
& =\frac{1}{2} \sum_{i=1}^{N_{b}}\left|u\left(\boldsymbol{x}_{b}^{i} ; \boldsymbol{\theta}\right)-g\left(\boldsymbol{x}_{b}^{i}\right)\right|^{2}+\frac{1}{2} \sum_{i=1}^{N_{r}}\left|r\left(\boldsymbol{x}_{r}^{i} ; \boldsymbol{\theta}\right)\right|^{2} .
\end{aligned}
$$

Now let us consider the corresponding gradient flow
$$
\frac{d \boldsymbol{\theta}}{d t}=-\nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta})=-\left[\sum_{i=1}^{N_{b}}\left(u\left(\boldsymbol{x}_{b}^{i} ; \boldsymbol{\theta}(t)\right)-g\left(\boldsymbol{x}_{b}^{i}\right)\right) \frac{\partial u}{\partial \boldsymbol{\theta}}\left(\boldsymbol{x}_{b}^{i} ; \boldsymbol{\theta}(t)\right)+\sum_{i=1}^{N_{r}}\left(\mathcal{N}[u]\left(\boldsymbol{x}_{r}^{i} ; \boldsymbol{\theta}(t)\right)-f\left(\boldsymbol{x}_{r}^{i}\right)\right) \frac{\partial \mathcal{N}[u]}{\partial \boldsymbol{\theta}}\left(\boldsymbol{x}_{r}^{i} ; \boldsymbol{\theta}(t)\right)\right] .
$$

It follows that for $0 \leq j \leq N_{b}$,
$$
\begin{aligned}
& \frac{d u\left(\boldsymbol{x}_{b}^{j} ; \boldsymbol{\theta}(t)\right)}{d t}=\frac{d u\left(\boldsymbol{x}_{b}^{j} ; \boldsymbol{\theta}(t)\right)^{T}}{d \boldsymbol{\theta}} \cdot \frac{d \boldsymbol{\theta}}{d t} \\
& \left.=-\frac{d u\left(\boldsymbol{x}_{b}^{j} ; \boldsymbol{\theta}(t)\right)}{d \boldsymbol{\theta}}\right)^{T} \cdot\left[\sum_{i=1}^{N_{b}}\left(u\left(\boldsymbol{x}_{b}^{i} ; \boldsymbol{\theta}(t)\right)-g\left(\boldsymbol{x}_{b}^{i}\right)\right) \frac{\partial u}{\partial \boldsymbol{\theta}}\left(\boldsymbol{x}_{b}^{i} ; \boldsymbol{\theta}(t)\right)+\sum_{i=1}^{N_{r}}\left(\mathcal{L}\left(\boldsymbol{x}_{r}^{i} ; \boldsymbol{\theta}(t)\right)-f\left(\boldsymbol{x}_{r}^{i}\right)\right) \frac{\partial \mathcal{N}[u]}{\partial \boldsymbol{\theta}}\left(\boldsymbol{x}_{r}^{i} ; \boldsymbol{\theta}(t)\right)\right] \\
& =-\sum_{i=1}^{N_{b}}\left(u\left(\boldsymbol{x}_{b}^{i} ; \boldsymbol{\theta}\right)-g\left(\boldsymbol{x}_{b}^{i}\right)\right)\left\langle\frac{d u\left(\boldsymbol{x}_{b}^{i} ; \boldsymbol{\theta}(t)\right)}{d \boldsymbol{\theta}}, \frac{d u\left(\boldsymbol{x}_{b}^{j} ; \boldsymbol{\theta}(t)\right)}{d \boldsymbol{\theta}}\right\rangle \\
& -\sum_{i=1}^{N_{r}}\left(\mathcal{N}[u]\left(\boldsymbol{x}_{r}^{i} ; \boldsymbol{\theta}\right)-f\left(\boldsymbol{x}_{r}^{i}\right)\right)\left\langle\frac{\mathcal{N}[u]\left(\boldsymbol{x}_{r}^{i} ; \boldsymbol{\theta}(t)\right)}{d \boldsymbol{\theta}}, \frac{d u\left(\boldsymbol{x}_{b}^{j} ; \boldsymbol{\theta}(t)\right)}{d \boldsymbol{\theta}}\right\rangle
\end{aligned}
$$

Similarly,
$$
\begin{aligned}
& \frac{d \mathcal{N}[u]\left(\boldsymbol{x}_{r}^{j} ; \boldsymbol{\theta}(t)\right)}{d t}=\frac{d \mathcal{N}[u]\left(\boldsymbol{x}_{r}^{j} ; \boldsymbol{\theta}(t)\right)^{T}}{d \boldsymbol{\theta}} \cdot \frac{d \boldsymbol{\theta}}{d t} \\
& =\frac{d \mathcal{N}[u]\left(\boldsymbol{x}_{r}^{j} ; \boldsymbol{\theta}(t)\right)^{T}}{d \boldsymbol{\theta}} \cdot\left[\sum_{i=1}^{N_{b}}\left(u\left(\boldsymbol{x}_{b}^{i} ; \boldsymbol{\theta}(t)\right)-g\left(\boldsymbol{x}_{b}^{i}\right)\right) \frac{\partial u}{\partial \boldsymbol{\theta}}\left(\boldsymbol{x}_{b}^{i} ; \boldsymbol{\theta}(t)\right)+\sum_{i=1}^{N_{r}}\left(\mathcal{L}\left(\boldsymbol{x}_{r}^{i} ; \boldsymbol{\theta}(t)\right)-f\left(\boldsymbol{x}_{r}^{i}\right)\right) \frac{\partial \mathcal{N}[u]}{\partial \boldsymbol{\theta}}\left(\boldsymbol{x}_{r}^{i} ; \boldsymbol{\theta}(t)\right)\right] \\
& =-\sum_{i=1}^{N_{b}}\left(u\left(\boldsymbol{x}_{b}^{i} ; \boldsymbol{\theta}\right)-g\left(\boldsymbol{x}_{b}^{i}\right)\right)\left\langle\frac{d u\left(\boldsymbol{x}_{b}^{i} ; \boldsymbol{\theta}(t)\right)}{d \boldsymbol{\theta}}, \frac{d \mathcal{N}[u]\left(\boldsymbol{x}_{b}^{j} ; \boldsymbol{\theta}(t)\right)}{d \boldsymbol{\theta}}\right\rangle \\
& -\sum_{i=1}^{N_{r}}\left(\mathcal{N}[u]\left(\boldsymbol{x}_{r}^{i} ; \boldsymbol{\theta}\right)-f\left(\boldsymbol{x}_{r}^{i}\right)\right)\left\langle\frac{\mathcal{N}[u]\left(\boldsymbol{x}_{r}^{i} ; \boldsymbol{\theta}(t)\right)}{d \boldsymbol{\theta}}, \frac{d \mathcal{N}[u]\left(\boldsymbol{x}_{r}^{j} ; \boldsymbol{\theta}(t)\right)}{d \boldsymbol{\theta}}\right\rangle
\end{aligned}
$$

Then we can rewrite the above equations as
$$
\begin{align*}
& \frac{d u\left(\boldsymbol{x}_{b} ; \boldsymbol{\theta}(t)\right)}{d t}=-\boldsymbol{K}_{u u}(t) \cdot\left(u\left(\boldsymbol{x}_{b} ; \boldsymbol{\theta}(t)\right)-g\left(\boldsymbol{x}_{b}\right)\right)-\boldsymbol{K}_{u r}(t) \cdot\left(\mathcal{N}[u]\left(\boldsymbol{x}_{r} ; \boldsymbol{\theta}(t)\right)-f\left(\boldsymbol{x}_{r}\right)\right)  \tag{A.1}\\
& \frac{d \mathcal{N}[u]\left(\boldsymbol{x}_{r} ; \boldsymbol{\theta}(t)\right)}{d t}=-\boldsymbol{K}_{r u}(t) \cdot\left(u\left(\boldsymbol{x}_{b} ; \boldsymbol{\theta}(t)\right)-g\left(\boldsymbol{x}_{b}\right)\right)-\boldsymbol{K}_{r r}(t) \cdot\left(\mathcal{N}[u]\left(\boldsymbol{x}_{r} ; \boldsymbol{\theta}(t)\right)-f\left(\boldsymbol{x}_{r}\right)\right) \tag{A.2}
\end{align*}
$$
where $\boldsymbol{K}_{r u}(t)=\boldsymbol{K}_{u r}^{T}(t)$ and the ( $i, j$ )-th entries are given by
$$
\begin{aligned}
\left(\boldsymbol{K}_{u u}\right)_{i j}(t) & =\left\langle\frac{d u\left(\boldsymbol{x}_{b}^{i} ; \boldsymbol{\theta}(t)\right)}{d \boldsymbol{\theta}}, \frac{d u\left(\boldsymbol{x}_{b}^{j} ; \boldsymbol{\theta}(t)\right)}{d \boldsymbol{\theta}}\right\rangle \\
\left(\boldsymbol{K}_{u r}\right)_{i j}(t) & =\left\langle\frac{d u\left(\boldsymbol{x}_{b}^{i} ; \boldsymbol{\theta}(t)\right)}{d \boldsymbol{\theta}}, \frac{d \mathcal{N}[u]\left(\boldsymbol{x}_{r}^{j} ; \boldsymbol{\theta}(t)\right)}{d \boldsymbol{\theta}}\right\rangle \\
\left(\boldsymbol{K}_{r r}\right)_{i j}(t) & =\left\langle\frac{d \mathcal{N}[u]\left(\boldsymbol{x}_{r}^{i} ; \boldsymbol{\theta}(t)\right)}{d \boldsymbol{\theta}}, \frac{d \mathcal{N}[u]\left(\boldsymbol{x}_{r}^{j} ; \boldsymbol{\theta}(t)\right)}{d \boldsymbol{\theta}}\right\rangle
\end{aligned}
$$

\section*{Appendix B. Proof of Theorem 4.1}

Proof. Recall equation (4.3) and that all weights and biases are initialized by independent standard Gaussian distributions. Then by the central limit theorem, we have
$$
\begin{aligned}
u_{x x}(x ; \boldsymbol{\theta}) & =\frac{1}{\sqrt{N}} \boldsymbol{W}^{(1)} \cdot\left[\ddot{\sigma}\left(\boldsymbol{W}^{(0)} x+\boldsymbol{b}^{(0)}\right) \odot \boldsymbol{W}^{(0)} \odot \boldsymbol{W}^{(0)}\right] \\
& =\frac{1}{\sqrt{N}} \sum_{k=1}^{N} \boldsymbol{W}_{k}^{(1)} \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right)\left(\boldsymbol{W}_{k}^{(0)}\right)^{2} \\
& \xrightarrow{\mathcal{D}} \mathcal{N}(0, \Sigma(x)) \triangleq Y(x)
\end{aligned}
$$
as $N \rightarrow \infty$, where $\mathcal{D}$ denotes convergence in distribution and $Y(x)$ is a centered Gaussian random variable with covariance
$$
\Sigma(x)=\operatorname{Var}\left[\boldsymbol{W}_{k}^{(1)} \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right)\left(\boldsymbol{W}_{k}^{(0)}\right)^{2}\right] .
$$

Since $\ddot{\sigma}$ is bounded, we may assume that $|\ddot{\sigma}| \leq C$. Then we have
$$
\begin{aligned}
\sup _{N} \mathbb{E}\left[\left|u_{x x}(x ; \theta)\right|^{2}\right] & =\sup _{N} \mathbb{E}\left[\mathbb{E}\left[\left|u_{x x}(x ; \theta)\right|^{2} \mid \boldsymbol{W}^{(0)}, \boldsymbol{b}^{(0)}\right]\right] \\
& =\sup _{N} \mathbb{E}\left[\frac{1}{N} \sum_{k=1}^{N}\left(\boldsymbol{W}_{k}^{(0)}\right)^{4}\left(\ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right)\right)^{2}\right] \\
& \leq C \mathbb{E}\left[\left(\boldsymbol{W}_{k}^{(0)}\right)^{4}\right]<\infty .
\end{aligned}
$$

This implies that $u_{x x}(x ; \boldsymbol{\theta})$ is uniformly integrable with respect to $N$. Now for any given point $x, x^{\prime}$, we have
$$
\begin{aligned}
\Sigma_{x x}^{(1)}\left(x, x^{\prime}\right) & \left.\triangleq \mathbb{E}\left[Y(x) Y\left(x^{\prime}\right)\right]=\lim _{N \rightarrow \infty} \mathbb{E}\left[u_{x x}(x, \boldsymbol{\theta}) u_{x x}\left(x^{\prime}, \boldsymbol{\theta}\right)\right)\right] \\
& =\lim _{N \rightarrow \infty} \mathbb{E}\left[\mathbb{E}\left[u_{x x}(x, \boldsymbol{\theta}) u_{x x}\left(x^{\prime}, \boldsymbol{\theta}\right) \mid \boldsymbol{W}^{(0)}, \boldsymbol{b}^{(0)}\right]\right] \\
& =\lim _{N \rightarrow \infty} \mathbb{E}\left[\frac{1}{N}\left(\ddot{\sigma}\left(\boldsymbol{W}^{(0)} x+\boldsymbol{b}^{(0)}\right) \odot \boldsymbol{W}^{(0)} \odot \boldsymbol{W}^{(0)}\right)^{T}\left(\ddot{\sigma}\left(\boldsymbol{W}^{(0)} x^{\prime}+\boldsymbol{b}^{(0)}\right) \odot \boldsymbol{W}^{(0)} \odot \boldsymbol{W}^{(0)}\right)\right] \\
& =\lim _{N \rightarrow \infty} \mathbb{E}\left[\frac{1}{N} \sum_{k=1}^{N}\left(\boldsymbol{W}_{k}^{(0)}\right)^{4} \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x^{\prime}+\boldsymbol{b}_{k}^{(0)}\right)\right] \\
& =\underset{u, v \sim \mathcal{N}(0,1)}{\mathbb{E}}\left[u^{4} \ddot{\sigma}(u x+v) \ddot{\sigma}\left(u x^{\prime}+v\right)\right]
\end{aligned}
$$

This concludes the proof.

\section*{Appendix C. Proof of Theorem 4.3}

Proof. To warm up, we first compute $\boldsymbol{K}_{u u}(0)$ and its infinite width limit, which is already covered in [33]. By the definition of $\boldsymbol{K}_{u u}(0)$, for any two given input $x, x^{\prime}$ we have
$$
\boldsymbol{K}_{u u}(0)=\left\langle\frac{d u(x ; \boldsymbol{\theta}(0))}{d \boldsymbol{\theta}}, \frac{d u\left(x^{\prime}, \boldsymbol{\theta}(0)\right)}{d \boldsymbol{\theta}}\right\rangle
$$

Recall that
$$
u(x ; \boldsymbol{\theta})=\frac{1}{\sqrt{N}} \boldsymbol{W}^{(1)} \cdot \sigma\left(\boldsymbol{W}^{(0)} x+\boldsymbol{b}^{(0)}\right)+\boldsymbol{b}^{(1)}=\frac{1}{\sqrt{N}} \sum_{k=1}^{N} \boldsymbol{W}_{k}^{(1)} \sigma\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right)+\boldsymbol{b}^{(1)},
$$
and $\boldsymbol{\theta}=\left(\boldsymbol{W}^{(0)}, \boldsymbol{W}^{(1)}, \boldsymbol{b}^{(0)}, \boldsymbol{b}^{(1)}\right)$. Then we have
$$
\begin{aligned}
\frac{\partial u(x ; \boldsymbol{\theta})}{\partial \boldsymbol{W}_{k}^{(0)}} & =\frac{1}{\sqrt{N}} \boldsymbol{W}_{k}^{(1)} \dot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) x \\
\frac{\partial u(x ; \boldsymbol{\theta})}{\partial \boldsymbol{W}_{k}^{(1)}} & =\frac{1}{\sqrt{N}} \sigma\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) \\
\frac{\partial u(x ; \boldsymbol{\theta})}{\partial \boldsymbol{b}_{k}^{(0)}} & =\frac{1}{\sqrt{N}} \boldsymbol{W}_{k}^{(1)} \dot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) \\
\frac{\partial u(x ; \boldsymbol{\theta})}{\partial \boldsymbol{b}^{(1)}} & =1 .
\end{aligned}
$$

Then by the law of large numbers we have
$$
\begin{aligned}
\sum_{k=1}^{N} \frac{\partial u(x ; \boldsymbol{\theta})}{\partial \boldsymbol{W}_{k}^{(0)}} \frac{\partial u\left(x^{\prime} ; \boldsymbol{\theta}\right)}{\partial \boldsymbol{W}_{k}^{(0)}} & =\frac{1}{N} \sum_{k=1}^{N}\left[\boldsymbol{W}_{k}^{(1)} \dot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) x\right] \cdot\left[\boldsymbol{W}_{k}^{(1)} \dot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x^{\prime}+\boldsymbol{b}_{K}^{(0)}\right) x^{\prime}\right] \\
& =\frac{1}{N}\left(\sum_{k=1}^{N}\left(\boldsymbol{W}_{k}^{(1)}\right)^{2} \dot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) \dot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x^{\prime}+\boldsymbol{b}_{k}^{(0)}\right)\right)\left(x x^{\prime}\right) \\
& \xrightarrow{\mathcal{P}} \mathbb{E}\left[\left(\boldsymbol{W}_{k}^{(1)}\right)^{2} \dot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) \dot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x^{\prime}+\boldsymbol{b}_{k}^{(0)}\right)\right]\left(x x^{\prime}\right) \\
& =\mathbb{E}\left[\left(\boldsymbol{W}_{k}^{(1)}\right)^{2}\right] \mathbb{E}\left[\dot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) \dot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x^{\prime}+\boldsymbol{b}_{k}^{(0)}\right)\right]\left(x x^{\prime}\right)=\dot{\Sigma}^{(1)}\left(x, x^{\prime}\right)\left(x x^{\prime}\right)
\end{aligned}
$$
as $N \rightarrow \infty$, where $\dot{\Sigma}^{(1)}\left(x, x^{\prime}\right)$ is defined in equation (2.5).
Moreover,
$$
\begin{aligned}
\sum_{k=1}^{N} \frac{\partial u(x ; \boldsymbol{\theta})}{\partial \boldsymbol{W}_{k}^{(1)}} \frac{\partial u\left(x^{\prime} ; \boldsymbol{\theta}\right)}{\partial \boldsymbol{W}_{k}^{(1)}} & =\frac{1}{N} \sum_{k=1}^{N} \sigma\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{0}\right) \sigma\left(\boldsymbol{W}_{k}^{(0)} x^{\prime}+\boldsymbol{b}_{k}^{(0)}\right) \\
& \xrightarrow{\mathcal{P}} \mathbb{E}\left[\sigma\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) \sigma\left(\boldsymbol{W}_{k}^{(0)} x^{\prime}+\boldsymbol{b}_{k}^{(0)}\right)\right]=\Sigma^{(1)}\left(x, x^{\prime}\right),
\end{aligned}
$$
as $N \rightarrow \infty$, where $\Sigma^{(1)}\left(x, x^{\prime}\right)$ is defined in equation (2.4).
Also,
$$
\begin{aligned}
\sum_{k=1}^{N} \frac{\partial u(x ; \boldsymbol{\theta})}{\partial \boldsymbol{b}_{k}^{(0)}} \frac{\partial u\left(x^{\prime} ; \boldsymbol{\theta}\right)}{\partial \boldsymbol{b}_{k}^{(0)}} & =\frac{1}{N} \sum_{k=1}^{N}\left[\left(\boldsymbol{W}_{k}^{(1)}\right)^{2} \dot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) \dot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x^{\prime}+\boldsymbol{b}_{k}^{(0)}\right)\right] \\
& \xrightarrow{\mathcal{P}} \mathbb{E}\left[\dot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) \dot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x^{\prime}+\boldsymbol{b}_{k}^{(0)}\right)\right]=\dot{\Sigma}^{(1)}\left(x, x^{\prime}\right)
\end{aligned}
$$

Then plugging all these together we obtain
$$
\begin{aligned}
\boldsymbol{K}_{u u}(0) & =\left\langle\frac{d u(x ; \boldsymbol{\theta}(0))}{d \boldsymbol{\theta}}, \frac{d u\left(x^{\prime}, \boldsymbol{\theta}(0)\right)}{d \boldsymbol{\theta}}\right\rangle \\
& =\sum_{l=0}^{1} \sum_{k=1}^{N} \frac{\partial u(x ; \boldsymbol{\theta})}{\partial \boldsymbol{W}_{k}^{(l)}} \frac{\partial u\left(x^{\prime} ; \boldsymbol{\theta}\right)}{\partial \boldsymbol{W}_{k}^{(l)}}+\sum_{k=1}^{N} \frac{\partial u(x ; \boldsymbol{\theta})}{\partial \boldsymbol{b}_{k}^{(0)}} \frac{\partial u\left(x^{\prime} ; \boldsymbol{\theta}\right)}{\partial \boldsymbol{b}_{k}^{(0)}}+\frac{\partial u(x ; \boldsymbol{\theta})}{\partial \boldsymbol{b}^{(1)}} \frac{\partial u\left(x^{\prime} ; \boldsymbol{\theta}\right)}{\partial \boldsymbol{b}^{(1)}}
\end{aligned}
$$
$$
\xrightarrow{\mathcal{P}} \dot{\Sigma}^{(1)}\left(x, x^{\prime}\right)\left(x x^{\prime}\right)+\Sigma^{(1)}\left(x, x^{\prime}\right)+\dot{\Sigma}^{(1)}\left(x, x^{\prime}\right)+1 \triangleq \Theta_{u u}^{(1)},
$$
as $N \rightarrow \infty$. This formula is also consistent with equation (2.8).
Next, we compute $\boldsymbol{K}_{r r}(0)$. To this end, recall that
$$
u_{x x}(x ; \boldsymbol{\theta})=\frac{1}{\sqrt{N}} \boldsymbol{W}^{(1)} \cdot\left[\ddot{\sigma}\left(\boldsymbol{W}^{(0)} x+\boldsymbol{b}^{(0)}\right) \odot \boldsymbol{W}^{(0)} \odot \boldsymbol{W}^{(0)}\right]=\frac{1}{\sqrt{N}} \sum_{k=1}^{N} \boldsymbol{W}_{k}^{(1)}\left(\boldsymbol{W}_{k}^{(0)}\right)^{2} \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right)
$$

It is then easy to compute that
$$
\begin{aligned}
\frac{\partial u_{x x}(x ; \boldsymbol{\theta})}{\partial \boldsymbol{W}_{k}^{(0)}} & =\frac{1}{\sqrt{N}} \boldsymbol{W}_{k}^{(1)} \boldsymbol{W}_{k}^{(0)}\left[\boldsymbol{W}_{k}^{(0)} \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) x+2 \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right)\right] \\
\frac{\partial u_{x x}(x ; \boldsymbol{\theta})}{\partial \boldsymbol{W}_{k}^{(1)}} & =\frac{1}{\sqrt{N}}\left(\boldsymbol{W}_{k}^{(0)}\right)^{2} \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) \\
\frac{\partial u_{x x}(x ; \boldsymbol{\theta})}{\partial \boldsymbol{b}_{k}^{(0)}} & =\frac{1}{\sqrt{N}} \boldsymbol{W}_{k}^{(1)}\left(\boldsymbol{W}_{k}^{(0)}\right)^{2} \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right)
\end{aligned}
$$
where $\dddot{\sigma}$ denotes third order derivative of $\sigma$. Then we have
$$
\begin{aligned}
\sum_{k=1}^{N} \frac{\partial u_{x x}(x ; \boldsymbol{\theta})}{\partial \boldsymbol{W}_{k}^{(0)}} \frac{\partial u_{x x}\left(x^{\prime} ; \boldsymbol{\theta}\right)}{\partial \boldsymbol{W}_{k}^{(0)}} & =\frac{1}{N} \sum_{k=1}^{N}\left(\boldsymbol{W}_{k}^{(1)} \boldsymbol{W}_{k}^{(0)}\right)^{2}\left(\left[\boldsymbol{W}_{k}^{(0)} \dddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) x+2 \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right)\right]\right) \\
& \cdot\left(\left[\boldsymbol{W}_{k}^{(0)} \dddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x^{\prime}+\boldsymbol{b}_{k}^{(0)}\right) x^{\prime}+2 \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x^{\prime}+\boldsymbol{b}_{k}^{(0)}\right)\right]\right) \\
& =I_{1}+I_{2}+I_{3}+I_{4}
\end{aligned}
$$
where
$$
\begin{aligned}
& I_{1}=\frac{1}{N} \sum_{k=1}^{N}\left(\boldsymbol{W}_{k}^{(1)}\right)^{2}\left(\boldsymbol{W}_{k}^{(0)}\right)^{4} \dddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) x \cdot \dddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x^{\prime}+\boldsymbol{b}_{k}^{(0)}\right) x^{\prime} \\
& I_{2}=\frac{2}{N} \sum_{k=1}^{N}\left(\boldsymbol{W}_{k}^{(1)}\right)^{2}\left(\boldsymbol{W}_{k}^{(0)}\right)^{3} \dddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) \cdot \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x^{\prime}+\boldsymbol{b}_{k}^{(0)}\right) x \\
& I_{3}=\frac{2}{N} \sum_{k=1}^{N}\left(\boldsymbol{W}_{k}^{(1)}\right)^{2}\left(\boldsymbol{W}_{k}^{(0)}\right)^{3} \dddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x^{\prime}+\boldsymbol{b}_{k}^{(0)}\right) \cdot \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) x^{\prime} \\
& I_{4}=\frac{4}{N} \sum_{k=1}^{N}\left(\boldsymbol{W}_{k}^{(1)} \boldsymbol{W}_{k}^{(0)}\right)^{2} \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) \cdot \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x^{\prime}+\boldsymbol{b}_{k}^{(0)}\right)
\end{aligned}
$$

By the law of large numbers, letting $N \rightarrow \infty$ gives
$$
\begin{aligned}
& I_{1} \xrightarrow{\mathcal{P}} \mathbb{E}\left[\left(\boldsymbol{W}_{k}^{(0)}\right)^{4} \dddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) \cdot \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x^{\prime}+\boldsymbol{b}_{k}^{(0)}\right)\right] x x^{\prime}:=J_{1} \\
& I_{2} \xrightarrow{\mathcal{P}} \mathbb{E}\left[\left(\boldsymbol{W}_{k}^{(0)}\right)^{3} \dddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) \cdot \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x^{\prime}+\boldsymbol{b}_{k}^{(0)}\right)\right] x:=J_{2} \\
& I_{3} \xrightarrow{\mathcal{P}} \mathbb{E}\left[\left(\boldsymbol{W}_{k}^{(0)}\right)^{3} \dddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x^{\prime}+\boldsymbol{b}_{k}^{(0)}\right) \cdot \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right)\right] x^{\prime}:=J_{3} \\
& I_{4} \xrightarrow{\mathcal{P}} \mathbb{E}\left[\left(\boldsymbol{W}_{k}^{(0)}\right)^{2} \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) \cdot \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x^{\prime}+\boldsymbol{b}_{k}^{(0)}\right)\right]:=J_{4}
\end{aligned}
$$

In conclusion we have
$$
\sum_{k=1}^{N} \frac{\partial u_{x x}(x ; \boldsymbol{\theta})}{\partial \boldsymbol{W}_{k}^{(0)}} \frac{\partial u_{x x}\left(x^{\prime} ; \boldsymbol{\theta}\right)}{\partial \boldsymbol{W}_{k}^{(0)}} \xrightarrow{\mathcal{P}} J_{1}+J_{2}+J_{3}+J_{4}:=A_{r r}
$$

Moreover,
$$
\begin{aligned}
\sum_{k=1}^{N} \frac{\partial u_{x x}(x ; \boldsymbol{\theta})}{\partial \boldsymbol{W}_{k}^{(1)}} \frac{\partial u_{x x}\left(x^{\prime} ; \boldsymbol{\theta}\right)}{\partial \boldsymbol{W}_{k}^{(1)}} & =\frac{1}{N} \sum_{k=1}^{N}\left(\boldsymbol{W}_{k}^{(0)}\right)^{4} \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) \cdot \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) \\
& \xrightarrow{\mathcal{P}} \mathbb{E}\left[\left(\boldsymbol{W}_{k}^{(0)}\right)^{4} \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) \cdot \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right)\right]:=B_{r r}
\end{aligned}
$$
and
$$
\begin{aligned}
\sum_{k=1}^{N} \frac{\partial u_{x x}(x ; \boldsymbol{\theta})}{\partial \boldsymbol{b}_{k}^{(0)}} \frac{\partial u_{x x}\left(x^{\prime} ; \boldsymbol{\theta}\right)}{\partial \boldsymbol{b}_{k}^{(0)}} & =\frac{1}{N} \sum_{k=1}^{N}\left(\boldsymbol{W}_{k}^{(1)}\right)^{2}\left(\boldsymbol{W}_{k}^{(0)}\right)^{4}\left[\dddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) \cdot \dddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x^{\prime}+\boldsymbol{b}_{k}^{(0)}\right)\right] \\
& \xrightarrow{\mathcal{P}} \mathbb{E}\left[\left(\boldsymbol{W}_{k}^{(0)}\right)^{4}\left(\dddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) \cdot \dddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x^{\prime}+\boldsymbol{b}_{k}^{(0)}\right)\right)\right]:=C_{r r}
\end{aligned}
$$

Now, recall that
$$
\boldsymbol{K}_{r r}(0)=\left\langle\frac{d u_{x x}(\boldsymbol{x} ; \boldsymbol{\theta}(0))}{d \boldsymbol{\theta}}, \frac{d u_{x x}\left(\boldsymbol{x}^{\prime} ; \boldsymbol{\theta}(0)\right)}{d \boldsymbol{\theta}}\right\rangle
$$

Thus we can conclude that as $N \rightarrow \infty$,
$$
\boldsymbol{K}_{r r}(0) \xrightarrow{\mathcal{P}} A_{r r}+B_{r r}+C_{r r}:=\Theta_{r r}\left(x, x^{\prime}\right)
$$

Finally, recall that $\boldsymbol{K}_{u r}\left(x, x^{\prime}\right)=\boldsymbol{K}_{r u}\left(x^{\prime}, x\right)$. So it suffices to compute $\boldsymbol{K}_{u r}\left(x, x^{\prime}\right)$ and its limit. To this end, recall that
$$
\boldsymbol{K}_{u r}\left(x, x^{\prime}\right)=\left\langle\frac{d u(\boldsymbol{x} ; \boldsymbol{\theta}(t))}{d \boldsymbol{\theta}}, \frac{d u_{x x}\left(\boldsymbol{x}^{\prime} ; \boldsymbol{\theta}(t)\right)}{d \boldsymbol{\theta}}\right\rangle .
$$

Then letting $N \rightarrow \infty$ gives
$$
\begin{aligned}
& \sum_{k=1}^{N} \frac{\partial u(x ; \boldsymbol{\theta})}{\partial \boldsymbol{W}_{k}^{(0)}} \frac{\partial u_{x x}\left(x^{\prime} ; \boldsymbol{\theta}\right)}{\partial \boldsymbol{W}_{k}^{(0)}} \\
& =\frac{1}{N} \sum_{k=1}^{N}\left(\boldsymbol{W}_{k}^{(1)}\right)^{2} \boldsymbol{W}_{k}^{(0)}\left[\dot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) x\right] \cdot\left[\boldsymbol{W}_{k}^{(0)} \dddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x^{\prime}+\boldsymbol{b}_{k}^{(0)}\right) x^{\prime}+2 \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x^{\prime}+\boldsymbol{b}_{k}^{(0)}\right)\right] \\
& \xrightarrow{\mathcal{P}} \mathbb{E}\left[\left(\boldsymbol{W}_{k}^{(0)}\right)^{2} \dot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) \cdot \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x^{\prime}+\boldsymbol{b}_{k}^{(0)}\right)\right]\left(x x^{\prime}\right)+2 \mathbb{E}\left[\boldsymbol{W}_{k}^{(0)} \dot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) \cdot \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x^{\prime}+\boldsymbol{b}_{k}^{(0)}\right)\right] x \\
& :=A_{u r}
\end{aligned}
$$
and
$$
\begin{aligned}
\sum_{k=1}^{N} \frac{\partial u(x ; \boldsymbol{\theta})}{\partial \boldsymbol{W}_{k}^{(1)}} \frac{\partial u_{x x}\left(x^{\prime} ; \boldsymbol{\theta}\right)}{\partial \boldsymbol{W}_{k}^{(1)}} & =\frac{1}{N} \sum_{k=1}^{N}\left[\left(\boldsymbol{W}_{k}^{(0)}\right)^{2} \sigma\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) \cdot \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right)\right] \\
& \xrightarrow{\mathcal{P}} \mathbb{E}\left[\left(\boldsymbol{W}_{k}^{(0)}\right)^{2} \sigma\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) \cdot \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right)\right]:=B_{u r}
\end{aligned}
$$
and
$$
\begin{aligned}
\sum_{k=1}^{N} \frac{\partial u(x ; \boldsymbol{\theta})}{\partial \boldsymbol{b}_{k}^{(0)}} \frac{\partial u_{x x}\left(x^{\prime} ; \boldsymbol{\theta}\right)}{\partial \boldsymbol{b}_{k}^{(0)}} & =\frac{1}{N} \sum_{k=1}^{N}\left[\boldsymbol{W}_{k}^{(1)} \boldsymbol{W}_{k}^{(0)}\right]^{2} \cdot\left[\dot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) \cdot \dddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x^{\prime}+\boldsymbol{b}_{k}^{(0)}\right)\right] \\
& \xrightarrow{\mathcal{P}} \mathbb{E}\left[\left(\boldsymbol{W}_{k}^{(0)}\right)^{2} \cdot\left(\dot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) \cdot \dddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x^{\prime}+\boldsymbol{b}_{k}^{(0)}\right)\right)^{2}\right]:=C_{u r}
\end{aligned}
$$

As a result, we obtain
$$
\boldsymbol{K}_{u r}\left(x, x^{\prime}\right)=\left\langle\frac{d u(\boldsymbol{x} ; \boldsymbol{\theta}(t))}{d \boldsymbol{\theta}}, \frac{d u_{x x}\left(\boldsymbol{x}^{\prime} ; \boldsymbol{\theta}(t)\right)}{d \boldsymbol{\theta}}\right\rangle \xrightarrow{\mathcal{P}} A_{u r}+B_{u r}+C_{u r}: \Theta_{u r}^{(1)}
$$
as $N \rightarrow \infty$. This concludes the proof.

\section*{Appendix D. Proof of Theorem 4.4}

Before we prove the main theorem, we need to prove a series of lemmas.
Lemma D.1. Under the setting of Theorem 4.4, for $l=0,1$, we have
$$
\begin{aligned}
& \sup _{t \in[0, T]}\left\|\frac{\partial u}{\partial \boldsymbol{W}^{(l)}}\right\|_{\infty}=\mathcal{O}\left(\frac{1}{\sqrt{N}}\right) \\
& \sup _{t \in[0, T]}\left\|\frac{\partial u_{x x}}{\partial \boldsymbol{W}^{(l)}}\right\|_{\infty}=\mathcal{O}\left(\frac{1}{\sqrt{N}}\right) \\
& \sup _{t \in[0, T]}\left\|\frac{\partial u}{\partial \boldsymbol{b}^{(0)}}\right\|_{\infty}=\mathcal{O}\left(\frac{1}{\sqrt{N}}\right) \\
& \sup _{t \in[0, T]}\left\|\frac{\partial u_{x x}}{\partial \boldsymbol{b}^{(0)}}\right\|_{\infty}=\mathcal{O}\left(\frac{1}{\sqrt{N}}\right)
\end{aligned}
$$

Proof. For the given model problem, recall that
$$
u(x ; \boldsymbol{\theta})=\frac{1}{\sqrt{N}} \boldsymbol{W}^{(1)} \sigma\left(\boldsymbol{W}^{(0)}(t) x+\boldsymbol{b}^{(0)}\right)+\boldsymbol{b}^{(1)}
$$
and
$$
\begin{aligned}
\frac{\partial u(x ; \boldsymbol{\theta})}{\partial \boldsymbol{W}_{k}^{(0)}} & =\frac{1}{\sqrt{N}} \boldsymbol{W}_{k}^{(1)} \dot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) x \\
\frac{\partial u(x ; \boldsymbol{\theta})}{\partial \boldsymbol{W}_{k}^{(1)}} & =\frac{1}{\sqrt{N}} \sigma\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) \\
\frac{\partial u(x ; \boldsymbol{\theta})}{\partial \boldsymbol{b}_{k}^{(0)}} & =\frac{1}{\sqrt{N}} \boldsymbol{W}_{k}^{(1)} \dot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right)
\end{aligned}
$$

Then by assumptions (i), (ii), and given that $\Omega$ is bounded, we have
$$
\begin{aligned}
& \sup _{t \in[0, T]}\left\|\frac{\partial u}{\partial \boldsymbol{W}^{(l)}}\right\|_{\infty} \leq \frac{C}{\sqrt{N}}, \quad l=0,1 . \\
& \sup _{t \in[0, T]}\left\|\frac{\partial u}{\partial \boldsymbol{b}^{(0)}}\right\|_{\infty} \leq \frac{C}{\sqrt{N}} .
\end{aligned}
$$

Also,
$$
u_{x x}(x ; \boldsymbol{\theta})=\frac{1}{\sqrt{N}} \boldsymbol{W}^{(1)} \cdot\left[\ddot{\sigma}\left(\boldsymbol{W}^{(0)} x+\boldsymbol{b}^{(0)}\right) \odot \boldsymbol{W}^{(0)} \odot \boldsymbol{W}^{(0)}\right]=\frac{1}{\sqrt{N}} \sum_{k=1}^{N} \boldsymbol{W}_{k}^{(1)}\left(\boldsymbol{W}_{k}^{(0)}\right)^{2} \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right)
$$
and
$$
\begin{aligned}
\frac{\partial u_{x x}(x ; \boldsymbol{\theta})}{\partial \boldsymbol{W}_{k}^{(0)}} & =\frac{1}{\sqrt{N}} \boldsymbol{W}_{k}^{(1)} \boldsymbol{W}_{k}^{(0)}\left[\boldsymbol{W}_{k}^{(0)} \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) x+2 \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right)\right] \\
\frac{\partial u_{x x}(x ; \boldsymbol{\theta})}{\partial \boldsymbol{W}_{k}^{(1)}} & =\frac{1}{\sqrt{N}}\left(\boldsymbol{W}_{k}^{(0)}\right)^{2} \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) \\
\frac{\partial u_{x x}(x ; \boldsymbol{\theta})}{\partial \boldsymbol{b}_{k}^{(0)}} & =\frac{1}{\sqrt{N}} \boldsymbol{W}_{k}^{(1)}\left(\boldsymbol{W}_{k}^{(0)}\right)^{2} \dddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right)
\end{aligned}
$$

Again, using assumptions (i), (ii) gives
$$
\begin{aligned}
& \sup _{t \in[0, T]}\left\|\frac{\partial u_{x x}}{\partial \boldsymbol{W}^{(l)}}\right\|_{\infty} \leq \frac{C^{4}}{\sqrt{N}}, \quad l=0,1 \\
& \sup _{t \in[0, T]}\left\|\frac{\partial u_{x x}}{\partial \boldsymbol{b}^{(0)}}\right\|_{\infty} \leq \frac{C^{4}}{\sqrt{N}}
\end{aligned}
$$

This completes the proof.

Lemma D.2. Under the setting of Theorem 4.4, we have
$$
\begin{align*}
& \lim _{N \rightarrow \infty} \sup _{t \in[0, T]}\left\|\frac{1}{\sqrt{N}}\left(\boldsymbol{W}^{(l)}(t)-\boldsymbol{W}^{(l)}(0)\right)\right\|_{2}=0, \quad l=0,1  \tag{D.1}\\
& \lim _{N \rightarrow \infty} \sup _{t \in[0, T]}\left\|\frac{1}{\sqrt{N}}\left(\boldsymbol{b}^{(0)}(t)-\boldsymbol{b}^{(0)}(0)\right)\right\|_{2}=0 \tag{D.2}
\end{align*}
$$

Proof. Recall that the loss function for the model problem (4.1) is given by
$$
\mathcal{L}(\boldsymbol{\theta})=\mathcal{L}_{b}(\boldsymbol{\theta})+\mathcal{L}_{r}(\boldsymbol{\theta})=\frac{1}{2} \sum_{i=1}^{N_{b}}\left|u\left(x_{b}^{i} ; \boldsymbol{\theta}\right)-g\left(x_{b}^{i}\right)\right|^{2}+\frac{1}{2} \sum_{i=1}^{N_{r}}\left|u_{x x}\left(x_{r}^{i} ; \boldsymbol{\theta}\right)-f\left(x_{r}^{i}\right)\right|^{2}
$$

Consider minimizing the loss function $\mathcal{L}(\boldsymbol{\theta})$ by gradient descent with an infinitesimally small learning rate:
$$
\frac{d \boldsymbol{\theta}}{d t}=-\nabla \mathcal{L}(\boldsymbol{\theta})
$$

This implies that
$$
\begin{aligned}
\frac{d \boldsymbol{W}^{(l)}}{d t} & =-\frac{\partial \mathcal{L}(\boldsymbol{\theta})}{\partial \boldsymbol{W}^{(l)}}, \quad l=0,1 \\
\frac{d \boldsymbol{b}^{(0)}}{d t} & =-\frac{\partial \mathcal{L}(\boldsymbol{\theta})}{\partial \boldsymbol{b}^{(0)}}
\end{aligned}
$$

Then we have
$$
\begin{aligned}
& \left\|\frac{1}{\sqrt{N}}\left(\boldsymbol{W}^{(l)}(t)-\boldsymbol{W}^{(l)}(0)\right)\right\|_{2}=\left\|\frac{1}{\sqrt{N}} \int_{0}^{t} \frac{d \boldsymbol{W}^{(l)}(\tau)}{d \tau} d \tau\right\|_{2}=\left\|\frac{1}{\sqrt{N}} \int_{0}^{t} \frac{\partial \mathcal{L}(\boldsymbol{\theta}(\tau))}{\partial \boldsymbol{W}^{(l)}} d \tau\right\|_{2} \\
& =\left\|\frac{1}{\sqrt{N}} \int_{0}^{t}\left[\sum_{i=1}^{N_{b}}\left(u\left(x_{b}^{i} ; \boldsymbol{\theta}(\tau)\right)-g\left(x_{b}^{i}\right)\right) \frac{\partial u}{\partial \boldsymbol{W}^{(l)}}\left(x_{b}^{i} ; \boldsymbol{\theta}(\tau)\right)+\sum_{i=1}^{N_{r}}\left(u_{x x}\left(x_{r}^{i} ; \boldsymbol{\theta}(\tau)\right)-f\left(x_{r}^{i}\right)\right) \frac{\partial u_{x x}}{\partial \boldsymbol{W}^{(l)}}\left(x_{r}^{i} ; \boldsymbol{\theta}(\tau)\right)\right] d \tau\right\|_{2} \\
& \leq I_{1}^{(l)}+I_{2}^{(l)}
\end{aligned}
$$
where
$$
\begin{aligned}
& I_{1}^{(l)}=\left\|\frac{1}{\sqrt{N}} \int_{0}^{t}\left[\sum_{i=1}^{N_{b}}\left(u\left(x_{b}^{i} ; \boldsymbol{\theta}(\tau)\right)-g\left(x_{b}^{i}\right)\right) \frac{\partial u}{\partial \boldsymbol{W}^{(l)}}\left(x_{b}^{i} ; \boldsymbol{\theta}(\tau)\right)\right] d \tau\right\|_{2} \\
& I_{2}^{(l)}=\left\|\frac{1}{\sqrt{N}} \int_{0}^{t}\left[\sum_{i=1}^{N_{r}}\left(u_{x x}\left(x_{r}^{i} ; \boldsymbol{\theta}(\tau)\right)-f\left(x_{r}^{i}\right)\right) \frac{\partial u_{x x}}{\partial \boldsymbol{W}^{(l)}}\left(x_{r}^{i} ; \boldsymbol{\theta}(\tau)\right)\right] d \tau\right\|_{2}
\end{aligned}
$$

We first process to estimate $I_{1}^{(l)}$ as
$$
\begin{aligned}
I_{1}^{(l)} & \leq \frac{1}{\sqrt{N}} \int_{0}^{t}\left\|\left[\sum_{i=1}^{N_{b}}\left(u\left(x_{b}^{i} ; \boldsymbol{\theta}(\tau)\right)-g\left(x_{b}^{i}\right)\right) \frac{\partial u}{\partial \boldsymbol{W}^{(l)}}\left(x_{b}^{i} ; \boldsymbol{\theta}(\tau)\right)\right]\right\|_{2} d \tau \\
& =\frac{1}{\sqrt{N}} \int_{0}^{t} \sqrt{\sum_{k=1}^{N}\left(\sum_{i=1}^{N_{b}}\left(u\left(x_{b}^{i} ; \boldsymbol{\theta}(\tau)\right)-g\left(x_{b}^{i}\right)\right) \frac{\partial u}{\partial \boldsymbol{W}_{k}^{(l)}}\left(x_{b}^{i} ; \boldsymbol{\theta}(\tau)\right)\right)^{2} d \tau} \\
& \leq \frac{1}{\sqrt{N}} \int_{0}^{T}\left\|\frac{\partial u}{\partial \boldsymbol{W}^{(l)}}\left(x_{b}^{i} ; \boldsymbol{\theta}(\tau)\right)\right\|_{\infty} \sqrt{\sum_{k=1}^{N}\left(\sum_{i=1}^{N_{b}}\left(u\left(x_{b}^{i} ; \boldsymbol{\theta}(\tau)\right)-g\left(x_{b}^{i}\right)\right)\right)^{2}} d \tau
\end{aligned}
$$
$$
=\frac{1}{\sqrt{N}} \int_{0}^{T} \sqrt{N}\left\|\frac{\partial u}{\partial \boldsymbol{W}^{(l)}}\left(x_{b}^{i} ; \boldsymbol{\theta}(\tau)\right)\right\|_{\infty} \cdot\left|\sum_{i=1}^{N_{b}}\left(u\left(x_{b}^{i} ; \boldsymbol{\theta}(\tau)\right)-g\left(x_{b}^{i}\right)\right)\right| d \tau .
$$

Thus, by assumptions and Lemma D.1, for $l=0,1$ we have
$$
\begin{aligned}
\sup _{t \in[0, T]} I_{1}^{(l)} & =\sup _{t \in[0, T]} \int_{0}^{T}\left\|\frac{\partial u}{\partial \boldsymbol{W}^{(l)}}\left(x_{b}^{i} ; \boldsymbol{\theta}(\tau)\right)\right\|_{\infty} \cdot\left|\sum_{i=1}^{N_{b}}\left(u\left(x_{b}^{i} ; \boldsymbol{\theta}(\tau)\right)-g\left(x_{b}^{i}\right)\right)\right| d \tau \\
& \leq \frac{C}{\sqrt{N}} \longrightarrow 0, \quad \text { as } N \longrightarrow \infty
\end{aligned}
$$

Similarly,
$$
\begin{aligned}
\sup _{t \in[0, T]} I_{2}^{(l)} \leq \sup _{t \in[0, T]} & \leq \frac{1}{\sqrt{N}} \int_{0}^{T}\left\|\frac{\partial u}{\partial \boldsymbol{W}^{(l)}}\left(x_{b}^{i} ; \boldsymbol{\theta}(\tau)\right)\right\|_{\infty} \sqrt{\sum_{k=1}^{N}\left(\sum_{i=1}^{N_{b}}\left(u\left(x_{b}^{i} ; \boldsymbol{\theta}(\tau)\right)-g\left(x_{b}^{i}\right)\right)\right)^{2}} d \tau \\
& =\frac{1}{\sqrt{N}} \int_{0}^{T} \sqrt{N}\left\|\frac{\partial u_{x x}}{\partial \boldsymbol{W}^{(l)}}\left(x_{r}^{i} ; \boldsymbol{\theta}(\tau)\right)\right\|_{\infty} \cdot\left|\sum_{i=1}^{N_{r}}\left(u_{x x}\left(x_{r}^{i} ; \boldsymbol{\theta}(\tau)\right)-f\left(x_{r}^{i}\right)\right)\right| d \tau \\
& \leq \frac{C^{4}}{\sqrt{N}} \longrightarrow 0, \quad \text { as } N \longrightarrow \infty
\end{aligned}
$$

Plugging these together, we obtain
$$
\lim _{N \rightarrow \infty} \sup _{t \in[0, T]}\left\|\frac{1}{\sqrt{N}}\left(\boldsymbol{W}^{(l)}(t)-\boldsymbol{W}^{(l)}(0)\right)\right\|_{2} \leq \lim _{N \rightarrow \infty} \sup _{t \in[0, T]} I_{1}^{(l)}+I_{2}^{(l)}=0
$$
for $l=1,2$. Similarly, applying the same strategy to $\boldsymbol{b}^{(0)}$ we can show
$$
\lim _{N \rightarrow \infty} \sup _{t \in[0, T]}\left\|\frac{1}{\sqrt{N}}\left(\boldsymbol{b}^{(0)}(t)-\boldsymbol{b}^{(0)}(0)\right)\right\|_{2}=0
$$

This concludes the proof.

Lemma D.3. Under the setting of Theorem 4.4, we have
$$
\begin{equation*}
\lim _{N \rightarrow \infty} \sup _{t \in[0, T]}\left\|\frac{1}{\sqrt{N}}\left(\sigma^{(k)}\left(\boldsymbol{W}^{(0)}(t) x+\boldsymbol{b}^{(0)}(t)\right)-\sigma^{(k)}\left(\boldsymbol{W}^{(0)}(t) x+\boldsymbol{b}^{(0)}(0)\right)\right)\right\|_{2}=0 \tag{D.3}
\end{equation*}
$$
for $k=0,1,2,3$, where $\sigma^{(k)}$ denotes the $k$-th order derivative of $\sigma$.

Proof. By the mean-value theorem for vector-valued function and Lemma D.2, there exists $\xi$
$$
\begin{aligned}
& \left\|\frac{1}{\sqrt{N}}\left(\sigma^{(k)}\left(\boldsymbol{W}^{(0)}(t) x+\boldsymbol{b}^{(0)}(t)\right)-\sigma^{(k)}\left(\boldsymbol{W}^{(0)}(0) x+\boldsymbol{b}^{(0)}(0)\right)\right)\right\|_{2} \\
& \leq\left\|\sigma^{(k+1)}(\xi)\right\|\left\|\frac{1}{\sqrt{N}}\left(\boldsymbol{W}^{(0)}(t) x+\boldsymbol{b}^{(0)}(t)-\boldsymbol{W}^{(0)}(0) x+\boldsymbol{b}^{(0)}(0)\right)\right\|_{2} \\
& \leq C\left\|\frac{1}{\sqrt{N}}\left(\boldsymbol{W}^{(0)}(t)-\boldsymbol{W}^{(0)}(0)\right)\right\|_{2}+C\left\|\frac{1}{\sqrt{N}}\left(\boldsymbol{b}^{(0)}(t)-\boldsymbol{b}^{(0)}(0)\right)\right\|_{2} \\
& \longrightarrow 0
\end{aligned}
$$
as $N \rightarrow \infty$. Here we use the assumption that $\sigma^{(k)}$ is bounded for $k=0,1,2,3,4$. This concludes the proof.

Lemma D.4. Under the setting of Theorem 4.4, we have
$$
\begin{align*}
& \lim _{N \rightarrow \infty} \sup _{t \in[0, T]}\left\|\frac{\partial u(x ; \boldsymbol{\theta}(t))}{\partial \boldsymbol{\theta}}-\frac{\partial u(x ; \boldsymbol{\theta}(0))}{\partial \boldsymbol{\theta}}\right\|_{2}  \tag{D.4}\\
& \lim _{N \rightarrow \infty} \sup _{t \in[0, T]}\left\|\frac{\partial u_{x x}(x ; \boldsymbol{\theta}(t))}{\partial \boldsymbol{\theta}}-\frac{\partial u_{x x}(x ; \boldsymbol{\theta}(0))}{\partial \boldsymbol{\theta}}\right\|_{2} . \tag{D.5}
\end{align*}
$$

Proof. Recall that
$$
\begin{aligned}
\frac{\partial u(x ; \boldsymbol{\theta})}{\partial \boldsymbol{W}_{k}^{(0)}} & =\frac{1}{\sqrt{N}} \boldsymbol{W}_{k}^{(1)} \dot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) x \\
\frac{\partial u(x ; \boldsymbol{\theta})}{\partial \boldsymbol{W}_{k}^{(1)}} & =\frac{1}{\sqrt{N}} \sigma\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) \\
\frac{\partial u(x ; \boldsymbol{\theta})}{\partial \boldsymbol{b}_{k}^{(0)}} & =\frac{1}{\sqrt{N}} \boldsymbol{W}_{k}^{(1)} \dot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) \\
\frac{\partial u(x ; \boldsymbol{\theta})}{\partial \boldsymbol{b}^{(1)}} & =1 .
\end{aligned}
$$

To simplify notation, let us define
$$
\begin{aligned}
& \boldsymbol{A}(t)=\left[\boldsymbol{W}^{(1)}(t)\right]^{T} \\
& \boldsymbol{B}(t)=\dot{\sigma}\left(\boldsymbol{W}^{(0)}(t) x+\boldsymbol{b}^{(0)}(t)\right) x
\end{aligned}
$$

Then by assumption (i) Lemma D. 2 D.3, we have
$$
\begin{aligned}
& \sup _{t \in[0, T]}\left\|\frac{\partial u(x ; \boldsymbol{\theta}(t))}{\partial \boldsymbol{W}^{(0)}}-\frac{\partial u(x ; \boldsymbol{\theta}(0))}{\partial \boldsymbol{W}^{(0)}}\right\|_{2} \\
& =\sup _{t \in[0, T]}\left\|\frac{1}{\sqrt{N}}(\boldsymbol{A}(t) \odot \boldsymbol{B}(t)-\boldsymbol{A}(0) \odot \boldsymbol{B}(0))\right\|_{2} \\
& \leq \sup _{t \in[0, T]}\left\|\frac{1}{\sqrt{N}}(\boldsymbol{A}(t)-\boldsymbol{A}(0)) \odot \boldsymbol{B}(t)\right\|_{2}+\left\|\frac{1}{\sqrt{N}} \boldsymbol{A}(0) \odot(\boldsymbol{B}(t)-\boldsymbol{B}(0))\right\|_{2} \\
& \leq \sup _{t \in[0, T]}\|\boldsymbol{B}(t)\|_{\infty}\left\|\frac{1}{\sqrt{N}}(\boldsymbol{A}(t)-\boldsymbol{A}(0))\right\|_{2}+\sup _{t \in[0, T]}\|\boldsymbol{A}(0)\|_{\infty}\left\|\frac{1}{\sqrt{N}}(\boldsymbol{B}(t)-\boldsymbol{B}(0))\right\|_{2} \\
& \longrightarrow 0,
\end{aligned}
$$
as $N \rightarrow \infty$. Here $\odot$ denotes point-wise multiplication.
Similarly, we can show that
$$
\begin{aligned}
& \lim _{N \rightarrow \infty} \sup _{t \in[0, T]}\left\|\frac{\partial u(x ; \boldsymbol{\theta}(t))}{\partial \boldsymbol{W}^{(1)}}-\frac{\partial u(x ; \boldsymbol{\theta}(0))}{\partial \boldsymbol{W}^{(1)}}\right\|_{2}=0, \\
& \lim _{N \rightarrow \infty} \sup _{t \in[0, T]}\left\|\frac{\partial u(x ; \boldsymbol{\theta}(t))}{\partial \boldsymbol{b}^{(0)}}-\frac{\partial u(x ; \boldsymbol{\theta}(0))}{\partial \boldsymbol{b}^{(0)}}\right\|_{2}=0 .
\end{aligned}
$$

Thus, we conclude that
$$
\lim _{N \rightarrow \infty} \sup _{t \in[0, T]}\left\|\frac{\partial u(x ; \boldsymbol{\theta}(t))}{\partial \boldsymbol{\theta}}-\frac{\partial u(x ; \boldsymbol{\theta}(0))}{\partial \boldsymbol{\theta}}\right\|_{2}=0 .
$$

Now for $u_{x x}$, we know that
$$
\begin{aligned}
\frac{\partial u_{x x}(x ; \boldsymbol{\theta})}{\partial \boldsymbol{W}_{k}^{(0)}} & =\frac{1}{\sqrt{N}} \boldsymbol{W}_{k}^{(1)} \boldsymbol{W}_{k}^{(0)}\left[\boldsymbol{W}_{k}^{(0)} \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) x+2 \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right)\right] \\
\frac{\partial u_{x x}(x ; \boldsymbol{\theta})}{\partial \boldsymbol{W}_{k}^{(1)}} & =\frac{1}{\sqrt{N}}\left(\boldsymbol{W}_{k}^{(0)}\right)^{2} \ddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right) \\
\frac{\partial u_{x x}(x ; \boldsymbol{\theta})}{\partial \boldsymbol{b}_{k}^{(0)}} & =\frac{1}{\sqrt{N}} \boldsymbol{W}_{k}^{(1)}\left(\boldsymbol{W}_{k}^{(0)}\right)^{2} \dddot{\sigma}\left(\boldsymbol{W}_{k}^{(0)} x+\boldsymbol{b}_{k}^{(0)}\right)
\end{aligned}
$$

Then for $\boldsymbol{W}^{(0)}$, again we define
$$
\begin{aligned}
& \boldsymbol{A}(t)=\left[\boldsymbol{W}^{(1)}\right]^{T} \\
& \boldsymbol{B}(t)=\boldsymbol{W}^{(0)} \\
& \boldsymbol{C}(t)=\dddot{\sigma}\left(\boldsymbol{W}^{(0)} x+\boldsymbol{b}^{(0)}\right) x \\
& \boldsymbol{D}(t)=2 \ddot{\sigma}\left(\boldsymbol{W}^{(0)} x+\boldsymbol{b}^{(0)}\right) .
\end{aligned}
$$

Then,
$$
\begin{aligned}
& \sup _{t \in[0, T]}\left\|\frac{\partial u_{x x}(x ; \boldsymbol{\theta}(t))}{\partial \boldsymbol{W}^{(0)}}-\frac{\partial u_{x x}(x ; \boldsymbol{\theta}(0))}{\partial \boldsymbol{W}^{(0)}}\right\|_{2} \\
& =\sup _{t \in[0, T]}\left\|\frac{1}{\sqrt{N}}(\boldsymbol{A}(t) \odot \boldsymbol{B}(t) \odot[\boldsymbol{B}(t) \odot \boldsymbol{C}(t)+\boldsymbol{D}(t)]-\boldsymbol{A}(0) \odot \boldsymbol{B}(0) \odot[\boldsymbol{B}(0) \odot \boldsymbol{C}(0)+\boldsymbol{D}(0)])\right\|_{2} \\
& \leq \sup _{t \in[0, T]}\left\|\frac{1}{\sqrt{N}}(\boldsymbol{A}(t) \odot \boldsymbol{B}(t) \odot \boldsymbol{B}(t) \odot \boldsymbol{C}(t)-\boldsymbol{A}(0) \odot \boldsymbol{B}(0) \odot \boldsymbol{B}(0) \odot \boldsymbol{C}(0))\right\|_{2} \\
& +\sup _{t \in[0, T]}\left\|\frac{1}{\sqrt{N}}(\boldsymbol{A}(t) \odot \boldsymbol{B}(t) \odot \boldsymbol{D}(t)-\boldsymbol{A}(0) \odot \boldsymbol{B}(0) \odot \boldsymbol{D}(0))\right\|_{2} \\
& :=I_{1}+I_{2}
\end{aligned}
$$

For $I_{1}$, we have
$$
\begin{aligned}
& \sup _{t \in[0, T]}\left\|\frac{1}{\sqrt{N}}(\boldsymbol{A}(t) \odot \boldsymbol{B}(t) \odot \boldsymbol{B}(t) \odot \boldsymbol{C}(t)-\boldsymbol{A}(0) \odot \boldsymbol{B}(0) \odot \boldsymbol{B}(0) \odot \boldsymbol{C}(0))\right\|_{2} \\
& \leq \sup _{t \in[0, T]}\left\|\frac{1}{\sqrt{N}}([\boldsymbol{A}(t)-\boldsymbol{A}(0)] \odot \boldsymbol{B}(t) \odot \boldsymbol{B}(t) \odot \boldsymbol{C}(t))\right\|_{2} \\
& +\sup _{t \in[0, T]}\left\|\frac{1}{\sqrt{N}}(\boldsymbol{A}(0) \odot[\boldsymbol{B}(t) \odot \boldsymbol{B}(t) \odot \boldsymbol{C}(t)-\boldsymbol{B}(0) \odot \boldsymbol{B}(0) \odot \boldsymbol{C}(0)])\right\|_{2} \\
& \leq \sup _{t \in[0, T]}\|\boldsymbol{B}(t)\|_{\infty}^{2}\|\boldsymbol{C}(t)\|_{\infty}\left\|\frac{1}{\sqrt{N}}(\boldsymbol{A}(t)-\boldsymbol{A}(0))\right\|_{2} \\
& +\sup _{t \in[0, T]}\|\boldsymbol{A}(0)\|_{\infty}\left\|\frac{1}{\sqrt{N}}(\boldsymbol{B}(t) \odot \boldsymbol{B}(t) \odot \boldsymbol{C}(t)-\boldsymbol{B}(0) \odot \boldsymbol{B}(0) \odot \boldsymbol{C}(0))\right\|_{2} \\
& \lesssim \sup _{t \in[0, T]}\left\|\frac{1}{\sqrt{N}}(\boldsymbol{A}(t)-\boldsymbol{A}(0))\right\|_{2}+\sup _{t \in[0, T]}\left\|\frac{1}{\sqrt{N}}(\boldsymbol{B}(t) \odot \boldsymbol{B}(t) \odot \boldsymbol{C}(t)-\boldsymbol{B}(0) \odot \boldsymbol{B}(0) \odot \boldsymbol{C}(0))\right\|_{2} \\
& \cdots \\
& \lesssim \sup _{t \in[0, T]}\left\|\frac{1}{\sqrt{N}}(\boldsymbol{A}(t)-\boldsymbol{A}(0))\right\|_{2}+\sup _{t \in[0, T]}\left\|\frac{1}{\sqrt{N}}(\boldsymbol{B}(t)-\boldsymbol{B}(0))\right\|_{2} \\
& +\sup _{t \in[0, T]}\left\|\frac{1}{\sqrt{N}}(\boldsymbol{C}(t)-\boldsymbol{C}(0))\right\|_{2}+\sup _{t \in[0, T]}\left\|\frac{1}{\sqrt{N}}(\boldsymbol{D}(t)-\boldsymbol{D}(0))\right\|_{2} \\
& \longrightarrow 0,
\end{aligned}
$$
as $N \rightarrow \infty$. We can use the same strategy to $I_{2}$ as well as $\boldsymbol{W}^{(1)}$ and $\boldsymbol{b}^{(0)}$. As a consequence, we conclude
$$
\lim _{N \rightarrow \infty} \sup _{t \in[0, T]}\left\|\frac{\partial u_{x x}(x ; \boldsymbol{\theta}(t))}{\partial \boldsymbol{\theta}}-\frac{\partial u_{x x}(x ; \boldsymbol{\theta}(0))}{\partial \boldsymbol{\theta}}\right\|_{2}=0 .
$$

This concludes the proof.
With these lemmas, now we can prove our main Theorem 4.4
Proof of Theorem 4.4. For a given data set $\left\{x_{b}^{i}, g\left(x_{b}^{i}\right)\right\}_{i=1}^{N_{b}},\left\{x_{r}^{i}, f\left(x_{r}^{i}\right)\right\}_{i=1}^{N_{r}}$, let $\boldsymbol{J}_{u}(t)$ and $\boldsymbol{J}_{r}(t)$ be the Jacobian matrix of $u\left(x_{b} ; \boldsymbol{\theta}(t)\right)$ and $u_{x x}\left(x_{r} ; \boldsymbol{\theta}\right)$ with respect to $\boldsymbol{\theta}$, respectively,
$$
\boldsymbol{J}_{u}(t)=\left(\frac{\partial u\left(x_{b}^{i} ; \boldsymbol{\theta}(t)\right)}{\partial \boldsymbol{\theta}_{j}}\right), \boldsymbol{J}_{r}(t)=\left(\frac{\partial u_{x x}\left(x_{r}^{i} ; \boldsymbol{\theta}(t)\right)}{\partial \boldsymbol{\theta}_{j}}\right)
$$

Note that
$$
\boldsymbol{K}(t)=\left[\begin{array}{l}
\boldsymbol{J}_{u}(t) \\
\boldsymbol{J}_{r}(t)
\end{array}\right]\left[\boldsymbol{J}_{u}^{T}(t), \boldsymbol{J}_{r}^{T}(t)\right]:=\boldsymbol{J}(t) \boldsymbol{J}^{T}(t)
$$

This implies that
$$
\begin{aligned}
\|\boldsymbol{K}(t)-\boldsymbol{K}(0)\|_{2} & =\left\|\boldsymbol{J}(t) \boldsymbol{J}^{T}(t)-\boldsymbol{J}(0) \boldsymbol{J}^{T}(0)\right\|_{2} \\
& \leq\left\|\boldsymbol{J}(t)\left[\boldsymbol{J}^{T}(t)-\boldsymbol{J}^{T}(0)\right]\right\|_{2}+\left\|[\boldsymbol{J}(t)-\boldsymbol{J}(0)] \boldsymbol{J}^{T}(0)\right\|_{2} \\
& \leq\|\boldsymbol{J}(t)\|_{2}\|\boldsymbol{J}(t)-\boldsymbol{J}(0)\|_{2}+\|\boldsymbol{J}(t)-\boldsymbol{J}(0)\|_{2}\|\boldsymbol{J}(0)\|_{2}
\end{aligned}
$$

By Lemma D.1, it is easy to show that $\|\boldsymbol{J}(t)\|_{2}$ is bounded. So it now suffices to show that
$$
\begin{align*}
& \sup _{t \in[0, T]}\left\|\boldsymbol{J}_{u}(t)-\boldsymbol{J}_{u}(0)\right\|_{F} \rightarrow 0  \tag{D.6}\\
& \sup _{t \in[0, T]}\left\|\boldsymbol{J}_{r}(t)-\boldsymbol{J}_{r}(0)\right\|_{F} \rightarrow 0 \tag{D.7}
\end{align*}
$$
as $N \rightarrow \infty$. Since the training data is finite, it suffices to consider just two inputs $x, x^{\prime}$. By the Cauchy-Schwartz inequality, we obtain
$$
\begin{aligned}
& \left|\left\langle\frac{\partial u(x ; \boldsymbol{\theta}(t))}{\partial \boldsymbol{\theta}}, \frac{\partial u\left(x^{\prime} ; \boldsymbol{\theta}(t)\right)}{\partial \boldsymbol{\theta}}\right\rangle-\left\langle\frac{\partial u(x ; \boldsymbol{\theta}(0))}{\partial \boldsymbol{\theta}}, \frac{\partial u\left(x^{\prime} ; \boldsymbol{\theta}(0)\right)}{\partial \boldsymbol{\theta}}\right\rangle\right| \\
& \leq\left|\left\langle\frac{\partial u(x ; \boldsymbol{\theta}(t))}{\partial \boldsymbol{\theta}}, \frac{\partial u\left(x^{\prime} ; \boldsymbol{\theta}(t)\right)}{\partial \boldsymbol{\theta}}-\frac{\partial u\left(x^{\prime} ; \boldsymbol{\theta}(0)\right)}{\partial \boldsymbol{\theta}}\right\rangle\right|+\left|\left\langle\frac{\partial u(x ; \boldsymbol{\theta}(t))}{\partial \boldsymbol{\theta}}-\frac{\partial u(x ; \boldsymbol{\theta}(0))}{\partial \boldsymbol{\theta}}, \frac{\partial u\left(x^{\prime} ; \boldsymbol{\theta}(0)\right)}{\partial \boldsymbol{\theta}}\right\rangle\right| \\
& \leq\left\|\frac{\partial u(x ; \boldsymbol{\theta}(t))}{\partial \boldsymbol{\theta}}\right\|_{2}\left\|\frac{\partial u\left(x^{\prime} ; \boldsymbol{\theta}(t)\right)}{\partial \boldsymbol{\theta}}-\frac{\partial u\left(x^{\prime} ; \boldsymbol{\theta}(0)\right)}{\partial \boldsymbol{\theta}}\right\|_{2}+\left\|\frac{\partial u(x ; \boldsymbol{\theta}(t))}{\partial \boldsymbol{\theta}}-\frac{\partial u(x ; \boldsymbol{\theta}(0))}{\partial \boldsymbol{\theta}}\right\|_{2}\left\|\frac{\partial u\left(x^{\prime} ; \boldsymbol{\theta}(0)\right)}{\partial \boldsymbol{\theta}}\right\|_{2} .
\end{aligned}
$$

From Lemma D.4, we have $\left\|\frac{\partial u(x ; \boldsymbol{\theta}(t))}{\partial \boldsymbol{\theta}}\right\|_{2}$ is uniformly bounded for $t \in[0, T]$. Then using Lemma D. 4 again gives
$$
\begin{aligned}
& \sup _{t \in[0, T]}\left|\left\langle\frac{\partial u(x ; \boldsymbol{\theta}(t))}{\partial \boldsymbol{\theta}}, \frac{\partial u\left(x^{\prime} ; \boldsymbol{\theta}(t)\right)}{\partial \boldsymbol{\theta}}\right\rangle-\left\langle\frac{\partial u(x ; \boldsymbol{\theta}(0))}{\partial \boldsymbol{\theta}}, \frac{\partial u\left(x^{\prime} ; \boldsymbol{\theta}(0)\right)}{\partial \boldsymbol{\theta}}\right\rangle\right| \\
& \leq C \sup _{t \in[0, T]}\left\|\frac{\partial u\left(x^{\prime} ; \boldsymbol{\theta}(t)\right)}{\partial \boldsymbol{\theta}}-\frac{\partial u\left(x^{\prime} ; \boldsymbol{\theta}(0)\right)}{\partial \boldsymbol{\theta}}\right\|_{2}+C \sup _{t \in[0, T]}\left\|\frac{\partial u(x ; \boldsymbol{\theta}(t))}{\partial \boldsymbol{\theta}}-\frac{\partial u(x ; \boldsymbol{\theta}(0))}{\partial \boldsymbol{\theta}}\right\|_{2} \\
& \longrightarrow 0
\end{aligned}
$$
as $N \rightarrow \infty$. This implies that
$$
\lim _{N \rightarrow \infty} \sup _{t \in[0, T]}\left\|\boldsymbol{J}_{u}(t)-\boldsymbol{J}_{u}(0)\right\|_{2}=0
$$

Similarly, we can repeat this calculation for $\boldsymbol{J}_{r}$, i.e.,
$$
\begin{aligned}
& \sup _{t \in[0, T]}\left|\left\langle\frac{\partial u_{x x}(x ; \boldsymbol{\theta}(t))}{\partial \boldsymbol{\theta}}, \frac{\partial u_{x x}\left(x^{\prime} ; \boldsymbol{\theta}(t)\right)}{\partial \boldsymbol{\theta}}\right\rangle-\left\langle\frac{\partial u_{x x}(x ; \boldsymbol{\theta}(0))}{\partial \boldsymbol{\theta}}, \frac{\partial u_{x x}\left(x^{\prime} ; \boldsymbol{\theta}(0)\right)}{\partial \boldsymbol{\theta}}\right\rangle\right| \\
& \leq \sup _{t \in[0, T]}\left\|\frac{\partial u_{x x}(x ; \boldsymbol{\theta}(t))}{\partial \boldsymbol{\theta}}\right\|_{2}\left\|\frac{\partial u_{x x}\left(x^{\prime} ; \boldsymbol{\theta}(t)\right)}{\partial \boldsymbol{\theta}}-\frac{\partial u_{x x}\left(x^{\prime} ; \boldsymbol{\theta}(0)\right)}{\partial \boldsymbol{\theta}}\right\|_{2} \\
& +\sup _{t \in[0, T]}\left\|\frac{\partial u_{x x}(x ; \boldsymbol{\theta}(t))}{\partial \boldsymbol{\theta}}-\frac{\partial u_{x x}(x ; \boldsymbol{\theta}(0))}{\partial \boldsymbol{\theta}}\right\|_{2}\left\|\frac{\partial u_{x x}\left(x^{\prime} ; \boldsymbol{\theta}(0)\right)}{\partial \boldsymbol{\theta}}\right\|_{2} \\
& \leq C \sup _{t \in[0, T]}\left\|\frac{\partial u_{x x}\left(x^{\prime} ; \boldsymbol{\theta}(t)\right)}{\partial \boldsymbol{\theta}}-\frac{\partial u_{x x}\left(x^{\prime} ; \boldsymbol{\theta}(0)\right)}{\partial \boldsymbol{\theta}}\right\|_{2}+C \sup _{t \in[0, T]}\left\|\frac{\partial u_{x x}(x ; \boldsymbol{\theta}(t))}{\partial \boldsymbol{\theta}}-\frac{\partial u_{x x}(x ; \boldsymbol{\theta}(0))}{\partial \boldsymbol{\theta}}\right\|_{2} \\
& \longrightarrow 0
\end{aligned}
$$
as $N \rightarrow \infty$. Hence, we get
$$
\lim _{N \rightarrow \infty} \sup _{t \in[0, T]}\left\|\boldsymbol{J}_{r}(t)-\boldsymbol{J}_{r}(0)\right\|_{2}=0
$$
and thus we conclude that
$$
\lim _{N \rightarrow \infty} \sup _{t \in[0, T]}\|\boldsymbol{K}(t)-\boldsymbol{K}(0)\|_{2}=0
$$

\section*{This concludes the proof.}

\section*{References}
[1] Maziar Raissi, Alireza Yazdani, George Em Karniadakis, Hidden fluid mechanics: learning velocity and pressure fields from flow visualizations, Science 367 (6481) (2020) 1026-1030.
[2] Luning Sun, Han Gao, Shaowu Pan, Jian-Xun Wang, Surrogate modeling for fluid flows based on physics-constrained deep learning without simulation data, Comput. Methods Appl. Mech. Eng. 361 (2020) 112732.
[3] Maziar Raissi, Hessam Babaee, Peyman Givi, Deep learning of turbulent scalar mixing, Phys. Rev. Fluids 4 (12) (2019) 124501.
[4] Maziar Raissi, Zhicheng Wang, Michael S. Triantafyllou, George Em Karniadakis, Deep learning of vortex-induced vibrations, J. Fluid Mech. 861 (2019) 119-137.
[5] Xiaowei Jin, Shengze Cai, Hui Li, George Em Karniadakis, NSFnets (Navier-Stokes flow nets): physics-informed neural networks for the incompressible Navier-Stokes equations, arXiv preprint arXiv:2003.06496, 2020.
[6] Francisco Sahli Costabal, Yibo Yang, Paris Perdikaris, Daniel E. Hurtado, Ellen Kuhl, Physics-informed neural networks for cardiac activation mapping, Front. Phys. 8 (2020) 42.
[7] Georgios Kissas, Yibo Yang, Eileen Hwuang, Walter R. Witschey, John A. Detre, Paris Perdikaris, Machine learning in cardiovascular flows modeling: predicting arterial blood pressure from non-invasive 4D flow MRI data using physics-informed neural networks, Comput. Methods Appl. Mech. Eng. 358 (2020) 112623.
[8] Zhiwei Fang, Justin Zhan, Deep physical informed neural networks for metamaterial design, IEEE Access 8 (2019) 24506-24513.
[9] Dehao Liu, Yan Wang, Multi-fidelity physics-constrained neural network and its application in materials modeling, J. Mech. Des. 141 (12) (2019).
[10] Yuyao Chen, Lu Lu, George Em Karniadakis, Luca Dal Negro, Physics-informed neural networks for inverse problems in nano-optics and metamaterials, Opt. Express 28 (8) (2020) 11618-11633.
[11] Sifan Wang, Paris Perdikaris, Deep learning of free boundary and Stefan problems, arXiv preprint, arXiv:2006.05311, 2020.
[12] Yibo Yang, Paris Perdikaris, Adversarial uncertainty quantification in physics-informed neural networks, J. Comput. Phys. 394 (2019) 136-152.
[13] Yinhao Zhu, Nicholas Zabaras, Phaedon-Stelios Koutsourelakis, Paris Perdikaris, Physics-constrained deep learning for high-dimensional surrogate modeling and uncertainty quantification without labeled data, J. Comput. Phys. 394 (2019) 56-81.
[14] Yibo Yang, Paris Perdikaris, Physics-informed deep generative models, arXiv preprint arXiv:1812.03511, 2018.
[15] Luning Sun, Jian-Xun Wang, Physics-constrained Bayesian neural network for fluid flow reconstruction with sparse and noisy data, arXiv preprint arXiv:2001.05542, 2020.
[16] Liu Yang, Xuhui Meng, George Em Karniadakis, B-PINNs: Bayesian physics-informed neural networks for forward and inverse PDE problems with noisy data, arXiv preprint arXiv:2003.06097, 2020.
[17] Justin Sirignano, Konstantinos Spiliopoulos, DGM: a deep learning algorithm for solving partial differential equations, J. Comput. Phys. 375 (2018) 1339-1364.
[18] Jiequn Han, Arnulf Jentzen, E. Weinan, Solving high-dimensional partial differential equations using deep learning, Proc. Natl. Acad. Sci. USA 115 (34) (2018) 8505-8510.
[19] Dongkun Zhang, Ling Guo, George Em Karniadakis, Learning in modal space: solving time-dependent stochastic PDEs using physics-informed neural networks, SIAM J. Sci. Comput. 42 (2) (2020) A639-A665.
[20] Guofei Pang, Lu Lu, George Em Karniadakis, fPINNs: fractional physics-informed neural networks, SIAM J. Sci. Comput. 41 (4) (2019) A2603-A2626.
[21] Guofei Pang, Marta D'Elia, Michael Parks, George E. Karniadakis, nPINNs: nonlocal physics-informed neural networks for a parametrized nonlocal universal Laplacian operator. Algorithms and applications, arXiv preprint arXiv:2004.04276, 2020.
[22] Alexandre M. Tartakovsky, Carlos Ortiz Marrero, Paris Perdikaris, Guzel D. Tartakovsky, David Barajas-Solano, Learning parameters and constitutive relationships with physics informed deep neural networks, arXiv preprint arXiv:1808.03398, 2018.
[23] Lu Lu, Pengzhan Jin, George Em Karniadakis, DeepONet: learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators, arXiv preprint, arXiv:1910.03193, 2019.
[24] A.M. Tartakovsky, C. Ortiz Marrero, Paris Perdikaris, G.D. Tartakovsky, D. Barajas-Solano, Physics-informed deep neural networks for learning parameters and constitutive relationships in subsurface flow problems, Water Resour. Res. 56 (5) (2020) e2019WR026731.
[25] Yeonjong Shin, Jerome Darbon, George Em Karniadakis, On the convergence and generalization of physics informed neural networks, arXiv preprint arXiv:2004.01806, 2020.
[26] Hamdi A. Tchelepi, Olga Fuks, Limitations of physics informed machine learning for nonlinear two-phase transport in porous media, J. Mach. Learn. Model. Comput. 1 (1) (2020).
[27] Maziar Raissi, Deep hidden physics models: deep learning of nonlinear partial differential equations, J. Mach. Learn. Res. 19 (1) (2018) 932-955.
[28] Sifan Wang, Yujun Teng, Paris Perdikaris, Understanding and mitigating gradient pathologies in physics-informed neural networks, arXiv preprint arXiv:2001.04536, 2020.
[29] Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred Hamprecht, Yoshua Bengio, Aaron Courville, On the spectral bias of neural networks, in: International Conference on Machine Learning, 2019, pp. 5301-5310.
[30] Yuan Cao, Zhiying Fang, Yue Wu, Ding-Xuan Zhou, Quanquan Gu, Towards understanding the spectral bias of deep learning, arXiv preprint arXiv: 1912.01198, 2019.
[31] Matthew Tancik, Pratul P. Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan T. Barron, Ren Ng, Fourier features let networks learn high frequency functions in low dimensional domains, arXiv preprint arXiv:2006.10739, 2020.
[32] Ronen Basri, Meirav Galun, Amnon Geifman, David Jacobs, Yoni Kasten, Shira Kritchman, Frequency bias in neural networks for input of non-uniform density, arXiv preprint arXiv:2003.04560, 2020.
[33] Arthur Jacot, Franck Gabriel, Clément Hongler, Neural tangent kernel: convergence and generalization in neural networks, in: Advances in Neural Information Processing Systems, 2018, pp. 8571-8580.
[34] Greg Yang, Scaling limits of wide neural networks with weight sharing: Gaussian process behavior, gradient independence, and neural tangent kernel derivation, arXiv preprint arXiv:1902.04760, 2019.
[35] Alexander G. de, G. Matthews, Mark Rowland, Jiri Hron, Richard E. Turner, Zoubin Ghahramani, Gaussian process behaviour in wide deep neural networks, arXiv preprint arXiv:1804.11271, 2018.
[36] Jaehoon Lee, Yasaman Bahri, Roman Novak, Samuel S. Schoenholz, Jeffrey Pennington, Jascha Sohl-Dickstein, Deep neural networks as Gaussian processes, arXiv preprint arXiv:1711.00165, 2017.
[37] David J.C. MacKay, Introduction to Gaussian Processes, NATO ASI Series F Computer and Systems Sciences, vol. 168, 1998, pp. 133-166.
[38] Sanjeev Arora, Simon S. Du, Wei Hu, Zhiyuan Li, Russ R. Salakhutdinov, Ruosong Wang, On exact computation with an infinitely wide neural net, in: Advances in Neural Information Processing Systems, 2019, pp. 8141-8150.
[39] Maziar Raissi, Paris Perdikaris, George E. Karniadakis, Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, J. Comput. Phys. 378 (2019) 686-707.
[40] Jaehoon Lee, Lechao Xiao, Samuel Schoenholz, Yasaman Bahri, Roman Novak, Jascha Sohl-Dickstein, Jeffrey Pennington, Wide neural networks of any depth evolve as linear models under gradient descent, in: Advances in Neural Information Processing Systems, 2019, pp. 8572-8583.
[41] Zhi-Qin John Xu, Yaoyu Zhang, Tao Luo, Yanyang Xiao, Zheng Ma, Frequency principle: Fourier analysis sheds light on deep neural networks, arXiv preprint arXiv:1901.06523, 2019.
[42] Basri Ronen, David Jacobs, Yoni Kasten, Shira Kritchman, The convergence rate of neural networks for learned functions of different frequencies, in: Advances in Neural Information Processing Systems, 2019, pp. 4761-4771.
[43] Lu Lu, Xuhui Meng, Zhiping Mao, George E. Karniadakis, DeepXDE: a deep learning library for solving differential equations, arXiv preprint arXiv: 1907.04502, 2019.
[44] Parviz Moin, Fundamentals of Engineering Numerical Analysis, Cambridge University Press, 2010.
[45] L.C. Evans, American Mathematical Society. Partial Differential Equations, Graduate Studies in Mathematics, American Mathematical Society, 1998.
[46] Xavier Glorot, Yoshua Bengio, Understanding the difficulty of training deep feedforward neural networks, in: Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, 2010, pp. 249-256.
[47] Diederik P. Kingma, Jimmy Ba Adam, A method for stochastic optimization, arXiv preprint arXiv:1412.6980, 2014.
[48] Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, Andrew Rabinovich, GradNorm: gradient normalization for adaptive loss balancing in deep multitask networks, in: International Conference on Machine Learning, PMLR, 2018, pp. 794-803.
[49] Haowen Xu, Hao Zhang, Zhiting Hu, Xiaodan Liang, Ruslan Salakhutdinov, Eric Xing, AutoLoss: learning discrete schedules for alternate optimization, arXiv preprint arXiv:1810.02442, 2018.
[50] A. Ali Heydari, Craig A. Thompson, Asif Mehmood, SoftAdapt: techniques for adaptive loss weighting of neural networks with multi-part loss functions, arXiv preprint arXiv:1912.12355, 2019.