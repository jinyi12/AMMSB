\title{
Stochastic modeling of mesoscopic elasticity random field
}

\author{
V-P. Tran ${ }^{\mathrm{a}, \mathrm{b}}$, J. Guilleminot ${ }^{\mathrm{b}, *}$, S. Brisard ${ }^{\mathrm{a}}$, K. Sab ${ }^{\mathrm{a}}$ \\ ${ }^{\mathrm{a}}$ Université Paris-Est, Laboratoire Navier (UMR 8205 CNRS, ENPC, IFSTTAR), Marne-la-Vallée 77455, France \\ ${ }^{\mathrm{b}}$ Université Paris-Est, Laboratoire Modélisation et Simulation Multi Echelle (MSME UMR 8208 CNRS), 5 Boulevard Descartes, Champs-sur-Marne, Marne-la-Vallée 77454, France
}

\section*{ARTICLE INFO}

\section*{Article history:}

Received 15 April 2015
Revised 16 September 2015
Available online 24 October 2015

\section*{Keywords:}

Numerical homogenization
MaxEnt principle
Probabilistic model

\begin{abstract}
In the homogenization setting, the effective properties of a heterogeneous material can be retrieved from the solution of the so-called corrector problem. In some cases of practical interest, obtaining such a solution remains a challenging computational task requiring an extremely fine discretization of microstructural features. In this context, Bignonnet et al. recently proposed a framework where smooth mesoscopic elasticity random fields are defined through a filtering procedure. In this work, we investigate the capabilities of informationtheoretic random field models to accurately represent such mesoscopic elasticity fields. The aim is to substantially reduce the homogenization cost through the use of coarser discretizations while solving mesoscale corrector problems. The analysis is performed on a simple but non-trivial model microstructure. First of all, we recall the theoretical background related to the filtering and multiscale frameworks, and subsequently characterize some statistical properties of the filtered stiffness field. Based on these properties, we further introduce a random field model and address its calibration through statistical estimators and the maximum likelihood principle. Finally, the validation of the model is discussed by comparing some quantities of interest that are obtained either from numerical experiments on the underlying random microstructure or from model-based simulations. It is shown that for the case under study, the information-theoretic model can be calibrated with a limited set of realizations and still allows for accurate predictions of the effective properties.
\end{abstract}
© 2015 Elsevier Ltd. All rights reserved.

\section*{1. Introduction}

In the homogenization setting, the homogenized (effective) properties of a heterogeneous material can be retrieved from the solution to the so-called corrector problem formulated over a truncated domain, named the representative volume element (RVE in short), with adequate (e.g. periodic) boundary conditions. In some cases, the resolution of the corrector problem remains a challenging computational task requiring an extremely fine mesh so as to capture the details

\footnotetext{
* Corresponding author. Tel.: +33 160957789.

E-mail addresses: vinh-phuc.tran@enpc.fr (V-P. Tran), johann.guilleminot@u-pem.fr (J. Guilleminot), sebastien.brisard@ifsttar.fr (S. Brisard), karam.sab@enpc.fr (K. Sab).
}
of all microstructural heterogeneities. This notably includes the case of heterogeneous materials made up of constitutive phases exhibiting high contrasts in their mechanical properties, as well as the case of microstructures exhibiting multiple, not well-separated characteristic length scales (here, this latter will be referred to as the case of non-separated scales). The computational effort is even more pronounced in a probabilistic setting (that is, when the underlying microstructure is random), where the corrector problem has to be resolved over numerous realizations.

Several contributions have been proposed to overcome such a challenge by means of various methods, including (in a non-exhaustive manner) local-global upscaling (Farmer, 2002), the use of the Multiscale Finite Element Method (MsFEM) (Hou and Wu, 1997) or the definition of a filtering
framework (Bignonnet et al., 2014, see the references therein). These techniques basically resort to the definition of smooth mesoscopic elasticity random fields through a two-step homogenization procedure. The corrector problem is subsequently resolved on the mesoscale domain using a coarse spatial discretization. Note that in the case of nonseparated scales, alternative definitions of such mesoscopic random fields can be found in Baxter and Graham (2000), Baxter et al. (2001) and Graham and Baxter (2001), as well as in Ostoja-Starzewski (1998) and Sena et al. (2013).

In the present work, we investigate the capabilities of information-theoretic random field models to accurately represent mesoscale elasticity fields. For illustration purposes, the analysis is performed on a model microstructure, namely a linear elastic matrix reinforced by bi-disperse spherical stiff heterogeneities. From the perspective of numerical homogenization, the ultimate aim is to use the aforementioned probabilistic representations as surrogate models that can be calibrated on a limited number of realizations, thus allowing for computational savings in the prediction of effective properties for random microstructures. Such an approach assumes that the probabilistic model can reproduce the fundamental statistics that significantly impact the macroscopic properties, and that the mesoscale realizations on which the model is calibrated do not introduce a bias with respect to these properties. The second condition motivates the use of the filtering framework proposed in Bignonnet et al. (2014), since the latter specifically exhibits a consistency property as regards the effective properties (see Section 2.1 for a discussion).

This paper is organized as follows. Section 2 is devoted to the filtering and homogenization frameworks. The theoretical background is first recalled, and the model microstructure is introduced. A statistical characterization of the mesoscale elasticity fields is next presented and used in order to construct a suitable probabilistic representation. The definition of the latter within the framework of information theory is specifically addressed in Section 3. The calibration of the model is then performed by using either statistical estimators or the maximum likelihood principle. Finally, the validation of the model is discussed by comparing some quantities of interest, such as the induced mesoscale stress field or the macroscopic homogenized properties, that are obtained either from numerical experiments on the underlying random microstructure or from model-based simulations.

\section*{2. Monte Carlo simulation of filtered mesoscopic stiffness tensor}

\subsection*{2.1. Overview of the filtering framework}

This part briefly recalls the theoretical filtering framework for periodic homogenization introduced by Bignonnet et al. (2014). This framework is particularly relevant to microstructures which exhibit multiple, not well-separated, characteristic length-scales. The structure obtained after the filtering of a microstructure at an intermediate scale will be called a mesostructure. We start by reformulating the corrector problem on a unit cell in periodic homogenization. The following notations will be used throughout this manuscript. Let $\Omega \subset \mathbb{R}^{d}(d=2,3)$ be the unit cell of a random heterogeneous
medium $\Omega=[0, L]^{d}$, and $\mathbf{C}(\mathbf{x})$ the local stiffness at point $\mathbf{x} \in \Omega$. It is known that the macroscopic (homogenized) properties of such periodic, heterogeneous materials can be derived from the solution to the following corrector problem:
$$
\begin{equation*}
\forall \mathbf{x} \in \Omega, \quad \nabla_{\mathbf{x}} \cdot \boldsymbol{\sigma}=\mathbf{0}, \tag{1a}
\end{equation*}
$$
$$
\begin{equation*}
\boldsymbol{\sigma}(\mathbf{x})=\mathbf{C}(\mathbf{x}): \varepsilon(\mathbf{x}), \tag{1b}
\end{equation*}
$$
$$
\begin{equation*}
\varepsilon(\mathbf{x})=\nabla_{\mathbf{x}}^{\mathrm{s}} \mathbf{u}, \tag{1c}
\end{equation*}
$$
$$
\begin{equation*}
\mathbf{u}(\mathbf{x})=\mathbf{E} \cdot \mathbf{x}+\mathbf{u}^{\mathrm{per}}(\mathbf{x}), \tag{1d}
\end{equation*}
$$
where $\boldsymbol{\varepsilon}(\mathbf{x})$ is the local strain derived from displacement $\mathbf{u}(\mathbf{x}), \boldsymbol{\sigma}(\mathbf{x})$ denotes the local stress and $\mathbf{E}$ is the macroscopic strain subject to the unit cell. The periodic fluctuation part of the displacement field about its macroscopic counterpart E . $\mathbf{x}$ is denoted as $\mathbf{u}^{\text {per }}(\mathbf{x})$.

By definition, the effective stiffness $\mathbf{C}^{\text {eff }}$ relates the macroscopic strain $\bar{\varepsilon}$ to the macroscopic stress $\overline{\boldsymbol{\sigma}}$ :
$$
\begin{equation*}
\overline{\boldsymbol{\sigma}}=\mathbf{C}^{\text {eff }}: \bar{\varepsilon}, \tag{2}
\end{equation*}
$$
in which $\overline{\boldsymbol{\sigma}}$ and $\bar{\varepsilon}$ denote the volume averages of the local stress and strain fields over the unit cell, respectively. Eq. (1d) ensures that $\bar{\varepsilon}=\mathbf{E}$. Because of the linearity of problem defined by Eqs. (1), the local strain $\boldsymbol{\varepsilon}(\mathbf{x})$ depends linearly on the macroscopic strain $\mathbf{E}$. This linear relationship is expressed through the strain localization tensor $\mathbf{A}(\mathbf{x})$ :
$$
\begin{equation*}
\varepsilon(\mathbf{x})=\mathbf{A}(\mathbf{x}): \mathbf{E} . \tag{3}
\end{equation*}
$$

Suppose now that Eqs. (1) are resolved, so that the local strain $\boldsymbol{\varepsilon}(\mathbf{x})$ and the local stress $\boldsymbol{\sigma}(\mathbf{x})$ are known. The so-called mesoscopic strain and mesoscopic stress fields are then obtained by convolution with a compactly-supported kernel $\rho$ :
$$
\begin{equation*}
\tilde{\boldsymbol{\sigma}}(\mathbf{x})=\int_{\mathbb{R}^{d}} \rho(\mathbf{x}-\mathbf{y}) \boldsymbol{\sigma}(\mathbf{y}) \mathrm{d} V_{\mathbf{y}}=(\rho * \boldsymbol{\sigma})(\mathbf{x}), \tag{4a}
\end{equation*}
$$
$$
\begin{equation*}
\tilde{\varepsilon}(\mathbf{x})=\int_{\mathbb{R}^{d}} \rho(\mathbf{x}-\mathbf{y}) \varepsilon(\mathbf{y}) \mathrm{d} V_{\mathbf{y}}=(\rho * \varepsilon)(\mathbf{x}) \tag{4b}
\end{equation*}
$$
where the kernel $\rho$ satisfies the normalization property
$$
\begin{equation*}
\int_{\mathbb{R}^{d}} \rho(\mathbf{x}) \mathrm{d} V_{\mathbf{x}}=1 . \tag{5}
\end{equation*}
$$

Introducing the fourth-order tensor $\mathbf{\tilde { \mathbf { C } }}$ defined as follows:
$$
\begin{equation*}
\tilde{\mathbf{C}}(\mathbf{x})=(\rho *(\mathbf{C}: \mathbf{A})(\mathbf{x})):(\rho * \mathbf{A})^{-1}(\mathbf{x}), \tag{6}
\end{equation*}
$$
it can readily be shown (Bignonnet et al., 2014) that $\tilde{\boldsymbol{\sigma}}(\mathbf{x})= \tilde{\mathbf{C}}(\mathbf{x}): \tilde{\varepsilon}(\mathbf{x})$. Therefore, $\tilde{\mathbf{C}}$ can be seen as the stiffness of the (filtered) mesostructure. It should be noticed, however, that this tensor does not exhibit the major symmetry. Note further that from a theoretical point of view, different admissible kernels could be selected in order to define the mesoscopic strain and stress fields. However, noticing that the kernel actually defines the mesoscopic scale for the field under consideration (and then, the probabilistic properties of the random field thus defined), and given the linear relation between the filtered quantities $\tilde{\boldsymbol{\sigma}}$ and $\tilde{\varepsilon}$, there is no a priori physical reason to select different kernels for the definition of the filtered stress and strain fields. The following equations then hold:
$$
\begin{equation*}
\forall \mathbf{x} \in \Omega, \quad \nabla_{\mathbf{x}} \cdot \tilde{\boldsymbol{\sigma}}=\mathbf{0}, \tag{7a}
\end{equation*}
$$
$\tilde{\boldsymbol{\sigma}}(\mathbf{x})=\tilde{\mathbf{C}}(\mathbf{x}): \tilde{\varepsilon}(\mathbf{x})$,
$\tilde{\varepsilon}(\mathbf{x})=\nabla_{\mathbf{x}}^{\mathrm{s}} \tilde{\mathbf{u}}$,
$\tilde{\mathbf{u}}(\mathbf{x})=\mathbf{E} \cdot \mathbf{x}+\tilde{\mathbf{u}}^{\mathrm{per}}(\mathbf{x})$.

Consequently, the mesoscopic stress and strain fields are the solution to the corrector problem involving the mesoscopic stiffness. Observing that convolution with a normalized kernel does not affect volume average, Bignonnet et al. (2014) concluded that the microstructure $\mathbf{C}$ and the mesostructure $\tilde{\mathbf{C}}$ have the same homogenized properties. More precisely, one has
$\overline{\tilde{\boldsymbol{\sigma}}}=\mathbf{C}^{\text {eff }}: \overline{\tilde{\varepsilon}}$,
where $\mathbf{C}^{\text {eff }}$ is defined by Eq. (2). It follows that the homogenized properties of the initial microstructure can readily be obtained by solving the corrector problem given by Eqs. (7), defined on the filtered microstructure. In practice, the latter problem turns out to be more tractable from a computational standpoint than the original one (which is defined at the microscale), since the mesoscopic elasticity field is smoother than the microscopic one and can therefore be discretized on a coarser grid/mesh. It is worth noticing that by varying the parameters involved in the kernel $\rho$, the filtering framework allows for a continuous description of the local stiffness ranging from the microscale up to the macroscale.

In this work, the above filtering framework is used as the mechanical solver within Monte Carlo simulations on random microstructures. A probabilistic representation of the resulting mesoscale elasticity random fields is then introduced and subsequently used as a surrogate model that can be calibrated on a limited number of realizations. In the next section, the model microstructure, as well as the computation of the mesoscopic and macroscopic properties, is presented. In addition, a statistical characterization of the filtered elasticity field is provided. Note that in the remainder of this paper, the set of generated microstructures and mesostructures will be referred to as the numerical experiments.

\subsection*{2.2. Generation of numerical experiments}

\subsection*{2.2.1. Generation of microstructures}

Here, a bidisperse assembly of spherical inclusions is selected as a model microstructure. The diameter of the smallest inclusions is $D / 3$, where $D$ denotes the diameter of the largest inclusions. The size of the periodic domain is $L=6 D$. The volume fraction of the largest (resp. smallest) inclusions is $20 \%$ (resp. $10 \%$ ). The microstructures are generated by means of a standard Monte Carlo simulation for an assembly of hard spheres (Allen and Tildesley, 1987), starting from initial configurations generated by random sequential addition (Torquato, 2002). In order to ensure convergence of the statistical estimators used here and there, a set of $N_{\text {exp }}=700$ independent realizations (indexed by $\left\{\theta_{i}\right\}_{i=1}^{N_{\text {exp }}}$ ) is generated.

\subsection*{2.2.2. Computation of the macroscopic stiffness tensor}

The determination of the mesoscopic stiffness tensor requires the local stress and strain fields (see Eqs. (4) and (8)). In this work, we resort to FFT-based homogenization
methods, initially introduced in Moulinec and Suquet (1994); 1998). More precisely, a variational form of this method (Brisard and Dormieux, 2010; 2012) is used to solve the corrector problem defined by Eqs. (1) and (7). The microstructures are discretized on a Cartesian grid of $128 \times 128 \times 128$ voxels. Both phases are isotropic linear elastic. The matrix and inclusions have the same Poisson's ratio $v=0.2$, and the shear modulus of the inclusions is taken as $\mu_{\mathrm{i}}=1000 \mu_{\mathrm{m}}$, where $\mu_{\mathrm{m}}$ denotes the shear modulus of the matrix. It should be noted that the elastic contrast was fixed purposely at a high value in order to illustrate the model robustness. The reference material to be used in the aforementioned numerical scheme is chosen to be softer than all phases, with $\mu_{0}=0.9 \mu_{\mathrm{m}}$ and $\nu_{0}=0.2$.

\subsection*{2.2.3. Computation of the mesoscopic stiffness tensor}

The mesoscopic stiffness tensor is computed through Eq. (6), making use of a truncated Gaussian filter:
$$
\begin{equation*}
\rho(\mathbf{x})=\alpha \exp \left(-\frac{1}{2} \frac{\|\mathbf{x}\|^{2}}{\gamma^{2}}\right), \text { if }\left|x_{i}\right| \leqslant H / 2, \forall i \in[[1, d]], \tag{9}
\end{equation*}
$$
where $\gamma=H / 6$ and $\alpha$ is chosen in order to ensure that Eq. (5) holds. It should be noticed that despite the anisotropic truncation, the above kernel is practically isotropic (since the ratio $\gamma / H$ is taken small enough). It induces a mass concentration at the origin (which is physically consistent with the definition of the mesoscopic fields). Realizations of microscopic and associated mesoscopic elasticity random field are shown in Fig. 1, and the effect of progressive smoothing induced by the filter parameter $H$ is clearly shown in Fig. 2.

Unless stated otherwise, the value $H=3 D$ will be used from now on.

\subsection*{2.3. Statistical characterization of mesoscopic stiffness tensor}

This section is devoted to the characterization of a few statistical properties for the filtered elasticity, based on the numerical experiments. In accordance with the multiscale framework (in a periodic setting), the mesoscopic elasticity random field $\left\{\tilde{\mathbf{C}}(\mathbf{x}, \theta), \mathbf{x} \in \mathbb{R}^{d}\right\}$ is identified with the periodized version, in the almost sure sense, of $\{\tilde{\mathbf{C}}(\mathbf{x}, \theta), \mathbf{x} \in \Omega\}$. Note that in the sequel, such an extension will be used with no specific notation for any decomposition of the elasticity random field.

\subsection*{2.3.1. Degree of asymmetry}

As reported in Bignonnet et al. (2014), it turns out that the mesoscopic stiffness tensor is not generally symmetric. In order to quantify the degree of asymmetry, let $\left\{\epsilon_{\text {asym }}(\mathbf{x}, \theta), \mathbf{x} \in\right. \Omega\}$ be the $\mathbb{R}^{+}$-valued random field measuring the normalized distance, for all $\mathbf{x}$ fixed in $\Omega$, between $\tilde{\mathbf{C}}(\mathbf{x}, \theta)$ and its symmetric counterpart:
$\forall \mathbf{x} \in \Omega, \quad \epsilon_{\text {asym }}(\mathbf{x}, \theta)=\frac{\left\|\tilde{\mathbf{C}}(\mathbf{x}, \theta)-\tilde{\mathbf{C}}^{\mathrm{T}}(\mathbf{x}, \theta)\right\|_{F}}{2\|\tilde{\mathbf{C}}(\mathbf{x}, \theta)\|_{F}}$,
where $C_{i j k l}^{\mathrm{T}}=C_{k l i j}$. The mean field $\mathbf{x} \mapsto\left\langle\epsilon_{\text {asym }}(\mathbf{x})\right\rangle$ of the above random field is then determined through the

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/f8994420-ef58-4ead-9ba2-9433d12e1c47-04.jpg?height=533&width=1182&top_left_y=183&top_left_x=362}
\captionsetup{labelformat=empty}
\caption{Fig. 1. One realization of microscopic elasticity random field $\left\{C_{1111}(\mathbf{x}, \theta), \mathbf{x} \in \Omega\right\}$ (left) and associated filtered random field $\left\{\tilde{C}_{1111}(\mathbf{x}, \theta), \mathbf{x} \in \Omega\right\}$ with $H=2 D$ (right).}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/f8994420-ef58-4ead-9ba2-9433d12e1c47-04.jpg?height=1147&width=1244&top_left_y=822&top_left_x=330}
\captionsetup{labelformat=empty}
\caption{Fig. 2. One realization (2D slice) of microscopic elasticity random field $\left\{C_{1111}(\mathbf{x}, \theta), \mathbf{x} \in \Omega\right\}$ (top-left) and associated mesoscale random fields $\left\{\tilde{C}_{1111}(\mathbf{x}, \theta), \mathbf{x} \in \Omega\right\}$ obtained for $H=2 D$ (top-right), $H=2.5 D$ (bottom-left) and $H=3 D$ (bottom-right). The same color scale was used for all above-mentioned mesoscale random fields to highlight the evolution of the contrast.}
\end{figure}
following estimator:
$$
\begin{equation*}
\forall \mathbf{x} \in \Omega, \quad\left\langle\epsilon_{\mathrm{asym}}(\mathbf{x})\right\rangle=\frac{1}{N_{\mathrm{exp}}} \sum_{i=1}^{N_{\mathrm{exp}}} \epsilon_{\mathrm{asym}}\left(\mathbf{x}, \theta_{i}\right) . \tag{11}
\end{equation*}
$$

The convergence of the statistical estimator at the center of the domain $\Omega$ is shown in Fig. 3 . It is further found that the
relative error is very small at all points of $\Omega$, both in mean and variance.

\subsection*{2.3.2. Material symmetry}

Here, we define the isotropic projection $\left\{\tilde{\mathbf{C}}^{\text {iso }}(\mathbf{x}, \theta), \mathbf{x} \in\right. \Omega\}$ of the mesoscale random field and investigate whether or not it can constitute a reasonable approximation for

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/f8994420-ef58-4ead-9ba2-9433d12e1c47-05.jpg?height=567&width=779&top_left_y=185&top_left_x=122}
\captionsetup{labelformat=empty}
\caption{Fig. 3. Convergence of the statistical estimator for the mean field of $\left\{\epsilon_{\text {asym }}(\mathbf{x}, \theta), \mathbf{x} \in \Omega\right\}$ at the center of domain $\Omega$.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/f8994420-ef58-4ead-9ba2-9433d12e1c47-05.jpg?height=581&width=785&top_left_y=862&top_left_x=120}
\captionsetup{labelformat=empty}
\caption{Fig. 4. Statistical estimator of distance field at the center of domain $\Omega$.}
\end{figure}
the probabilistic modeling. Specifically, and following e.g. Moakher and Norris (2006), the isotropic mesoscale random field is first defined through a euclidean projection:
$$
\begin{align*}
& \forall \mathbf{x} \in \Omega \\
& \tilde{\mathbf{C}}^{\text {iso }}(\mathbf{x}, \theta)=\mathcal{P}^{\text {iso }}(\tilde{\mathbf{C}}(\mathbf{x}, \theta))=\underset{\mathbf{M} \in \mathbb{M}_{n}^{\text {iso }}(\mathbb{R})}{\operatorname{argmin}}\|\tilde{\mathbf{C}}(\mathbf{x}, \theta)-\mathbf{M}\|_{\mathrm{F}}, \tag{12}
\end{align*}
$$
where $\mathbb{M}_{n}^{\text {iso }}(\mathbb{R})$ is the set of all isotropic tensors. In order to quantify the error induced by this approximation, let us introduce the normalized distance random field $\{d(\mathbf{x}, \theta), \mathbf{x} \in \Omega\}$ (see Guilleminot and Soize, 2012):
$$
\begin{equation*}
\forall \mathbf{x} \in \Omega, \quad d(\mathbf{x}, \theta)=\frac{\left\|\tilde{\mathbf{C}}(\mathbf{x}, \theta)-\mathbf{C}^{\mathrm{iso}}(\mathbf{x}, \theta)\right\|_{\mathrm{F}}}{\|\tilde{\mathbf{C}}(\mathbf{x}, \theta)\|_{\mathrm{F}}} . \tag{13}
\end{equation*}
$$

Let $\mathbf{x} \mapsto\langle d(\mathbf{x})\rangle$ be the mean function of the above random field. The convergence of the statistical estimator of $\langle d(\mathbf{x})\rangle$ at the center of domain $\Omega$ is shown in Fig. 4. Furthermore, the graph of the mean function $\left(x_{2}, x_{3}\right) \mapsto\left\langle d\left(L / 2, x_{2}, x_{3}\right)\right\rangle$ is shown in Fig. 5. It is seen on these graphs that the relative error is fairly small (with a maximum value of $3.3 \%$ approximately) in mean and exhibits contained fluctuations, hence showing that the realizations of the mesoscopic stiffness are almost

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/f8994420-ef58-4ead-9ba2-9433d12e1c47-05.jpg?height=617&width=776&top_left_y=187&top_left_x=975}
\captionsetup{labelformat=empty}
\caption{Fig. 5. Plot of mean function $\left(x_{2}, x_{3}\right) \mapsto\left\langle d\left(L / 2, x_{2}, x_{3}\right)\right\rangle$.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/f8994420-ef58-4ead-9ba2-9433d12e1c47-05.jpg?height=595&width=780&top_left_y=898&top_left_x=973}
\captionsetup{labelformat=empty}
\caption{Fig. 6. Max of mean function $\mathbf{x} \mapsto\langle d(\mathbf{x})\rangle$ over $\Omega$.}
\end{figure}
isotropic. Finally, the influence of the ratio $H / D$ on the level of isotropy is illustrated in Fig. 6. As expected, it is seen that the stochastic residual decreases, in mean, as the ratio H/D gets larger.

\subsection*{2.3.3. Mean value and correlation structure of mesoscopic bulk and shear moduli random fields}

The mean function $\mathbf{x} \mapsto\langle\tilde{\mathbf{C}}(\mathbf{x})\rangle$ of the mesoscopic stiffness random field is determined as:
$$
\begin{equation*}
\forall \mathbf{x} \in \Omega, \quad\langle\tilde{\mathbf{C}}(\mathbf{x})\rangle=\frac{1}{N_{\exp }} \sum_{i=1}^{N_{\exp }} \tilde{\mathbf{C}}\left(\mathbf{x}, \theta_{i}\right) . \tag{14}
\end{equation*}
$$

The graph of mean function $\left(x_{2}, x_{3}\right) \mapsto\left\langle\tilde{C}_{1111}\left(L / 2, x_{2}, x_{3}\right)\right\rangle$ is shown in Fig. 7 and may suggest that the statistical mean field $\mathbf{x} \mapsto\langle\tilde{\mathbf{C}}(\mathbf{x})\rangle$ fluctuates over $\Omega$.

It should however be noticed that the range of fluctuations falls within the confidence interval at $99 \%$, so that the observed spatial fluctuations are likely explained by finite-sampling. The spatial fluctuations of mean function $\mathbf{x} \mapsto\langle\tilde{\mathbf{C}}(\mathbf{x})\rangle$ are further characterized by the parameter $\delta_{\langle\tilde{\mathbf{C}}(\mathbf{x})\rangle}$

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/f8994420-ef58-4ead-9ba2-9433d12e1c47-06.jpg?height=543&width=668&top_left_y=189&top_left_x=193}
\captionsetup{labelformat=empty}
\caption{Fig. 7. Plot of mean function $\left(x_{2}, x_{3}\right) \mapsto \mu_{\mathrm{m}}^{-1}\left\langle\tilde{C}_{1111}\left(L / 2, x_{2}, x_{3}\right)\right\rangle$.}
\end{figure}
defined as
$$
\begin{equation*}
\delta_{\langle\tilde{\mathbf{C}}(\mathbf{x})\rangle}=\left\{\left(N_{p}^{-1} \sum_{i=1}^{N_{p}}\left\|\left\langle\tilde{\mathbf{C}}\left(\mathbf{x}_{i}\right)\right\rangle-\overline{\langle\tilde{\mathbf{C}}\rangle}\right\|_{F}^{2}\right)\|\langle\overline{\tilde{\mathbf{C}}}\rangle\|_{F}^{-2}\right\}^{1 / 2}, \tag{15}
\end{equation*}
$$
where $N_{\mathrm{p}}$ is the number of voxels in domain $\Omega\left(N_{\mathrm{p}}=128^{3}\right)$ and $\langle\tilde{\mathbf{C}}\rangle$ denotes the spatial average of mean field $\mathbf{x} \mapsto\langle\tilde{\mathbf{C}}(\mathbf{x})\rangle$ :

The value of $\delta_{\langle\tilde{\mathbf{C}}(\mathbf{x})\rangle}$ is found to be reasonably small $(0.58 \%)$. For these reasons, and taking into account the statistical homogeneity of the underlying microstructure, it is assumed from now on that the mean function does not depend on $\mathbf{x}$.

Furthermore, the normalized distance between $\overline{\langle\tilde{\mathbf{C}}\rangle}$ and its isotropic projection $\mathcal{P}^{\text {iso }}(\langle\overline{\widetilde{\mathbf{C}}}\rangle)$,
$$
\begin{equation*}
d\left(\overline{\langle\tilde{\mathbf{C}}\rangle}, \mathcal{P}^{\text {iso }}(\overline{\langle\tilde{\mathbf{C}}\rangle})\right)=\frac{\left\|\overline{\langle\tilde{\mathbf{C}}\rangle}-\mathcal{P}^{\text {iso }}(\widehat{\mathbf{C}})\right\|_{\mathrm{F}}}{\|\overline{\langle\tilde{\mathbf{C}}\rangle}\|_{\mathrm{F}}}, \tag{17}
\end{equation*}
$$
is equal to $0.255 \%$, showing that the mean value of the mesoscopic stiffness tensor is almost isotropic. The mean value is therefore approximated by using the projected mean values for both the bulk and shear moduli:
$$
\begin{equation*}
\langle\tilde{\kappa}\rangle=2.35 \mu_{\mathrm{m}}, \quad\langle\tilde{\mu}\rangle=1.78 \mu_{\mathrm{m}} . \tag{18}
\end{equation*}
$$

The coefficients of variation (defined as the standard deviation to mean ratio) of random bulk and shear moduli are found to be:
$$
\begin{equation*}
\delta_{\tilde{\kappa}} \approx 13.9 \%, \quad \delta_{\tilde{\mu}} \approx 14.4 \% . \tag{19}
\end{equation*}
$$

Let $\boldsymbol{\tau} \mapsto \mathcal{R}_{\tilde{\kappa}}(\boldsymbol{\tau})$ be the normalized covariance function of the bulk random field, defined for all $\boldsymbol{\tau}$ in $\mathbb{R}^{d}$ by
$$
\begin{equation*}
\mathcal{R}_{\tilde{\kappa}}(\boldsymbol{\tau})=\frac{\mathbb{E}\{[\tilde{\kappa}(\mathbf{x}+\boldsymbol{\tau}, \theta)-\langle\tilde{\kappa}(\mathbf{x}+\boldsymbol{\tau}, \theta)\rangle][\tilde{\kappa}(\mathbf{x}, \theta)-\langle\tilde{\kappa}(\mathbf{x}, \theta)\rangle]\}}{\left(\left[\mathbb{E}\left\{\tilde{\kappa}(\mathbf{x}+\boldsymbol{\tau}, \theta)^{2}\right\}-\langle\tilde{\kappa}(\mathbf{x}+\boldsymbol{\tau}, \theta)\rangle^{2}\right]\left[\mathbb{E}\left\{\tilde{\kappa}(\mathbf{x}, \theta)^{2}\right\}-\langle\tilde{\kappa}(\mathbf{x}, \theta)\rangle^{2}\right]\right)^{1 / 2}} . \tag{20}
\end{equation*}
$$
$$
\frac{1}{\mu_{\mathrm{m}}} \overline{\langle\tilde{\mathbf{C}}\rangle}=\left[\begin{array}{cccccc}
4.740 & 1.154 & 1.154 & 0 & 0 & -0.001  \tag{16}\\
1.154 & 4.742 & 1.154 & 0.001 & 0 & -0.001 \\
1.154 & 1.154 & 4.740 & 0 & 0.001 & 0 \\
0 & 0.001 & 0 & 3.561 & 0 & 0 \\
0 & 0 & 0.001 & 0 & 3.562 & 0.001 \\
0.001 & -0.001 & 0 & 0 & 0.001 & 3.562
\end{array}\right]
$$

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/f8994420-ef58-4ead-9ba2-9433d12e1c47-06.jpg?height=606&width=778&top_left_y=1714&top_left_x=137}
\captionsetup{labelformat=empty}
\caption{Fig. 8. Plot of the covariance functions $\tau / D \mapsto \mathcal{R}_{\tilde{k}}\left(\tau \mathbf{e}_{i}\right), 1 \leqslant i \leqslant 3$, for the bulk modulus random field.}
\end{figure}

Owing to statistical homogeneity, the right hand side of the above equation is in fact independent on the observation point $\mathbf{x}$. A similar notation is used for the normalized covariance function associated with the shear modulus random field $\{\tilde{\mu}(\mathbf{x}, \theta), \mathbf{x} \in \Omega\}$. The plot of the covariance function $\mathcal{R}_{\tilde{\kappa}}$ in the three directions defined by the canonical basis of $\mathbb{R}^{3}$ is shown in Fig. 8. It is seen that the above function is almost identical regardless of the direction, as expected from the statistical homogeneity and isotropy of the underlying microstructure (note that the retained Gaussian filter also introduces an isotropic smoothing). An interesting and unexpected result is that the normalized covariance functions are almost identical for the bulk and shear moduli, as shown in Fig. 9. Note that this feature may be explained by the fact that the two materials that were considered while generating the numerical database (see Section 2.2.2) share the same Poisson ratio.

\section*{3. Stochastic modeling}

\subsection*{3.1. Methodology}

This section is devoted to the construction of a probabilistic model for the mesoscopic elasticity random field $\{\tilde{\mathbf{C}}(\mathbf{x}, \theta), \mathbf{x} \in \Omega\}$. Following the statistical characterization, it is assumed that:
- for all $\mathbf{x}$ fixed in $\Omega$, the random matrix $\tilde{\mathbf{C}}(\mathbf{x}, \theta)$ is symmetric a.s.;
- the random field $\{\tilde{\mathbf{C}}(\mathbf{x}, \theta), \mathbf{x} \in \Omega\}$ is isotropic a.s. and is therefore completely defined by the associated bulk and

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/f8994420-ef58-4ead-9ba2-9433d12e1c47-07.jpg?height=607&width=789&top_left_y=185&top_left_x=116}
\captionsetup{labelformat=empty}
\caption{Fig. 9. Plot of the covariance functions $\tau / D \mapsto \mathcal{R}_{\tilde{\kappa}}\left(\tau \mathbf{e}_{2}\right)$ and $\tau / D \mapsto \mathcal{R}_{\tilde{\mu}}\left(\tau \mathbf{e}_{2}\right)$.}
\end{figure}
shear moduli random fields, denoted by $\{\tilde{\kappa}(\mathbf{x}, \theta), \mathbf{x} \in \Omega\}$ and $\{\tilde{\mu}(\mathbf{x}, \theta), \mathbf{x} \in \Omega\}$ respectively, such that
$$
\begin{equation*}
\forall \mathbf{x} \in \Omega, \quad \tilde{\mathbf{C}}(\mathbf{x}, \theta)=d \tilde{\kappa}(\mathbf{x}, \theta) \mathbf{J}+2 \tilde{\mu}(\mathbf{x}, \theta) \mathbf{K} \tag{21}
\end{equation*}
$$
with ( $\mathbf{J}, \mathbf{K}$ ) the classical (deterministic) tensor basis of $\mathbb{M}_{n}^{\text {iso }}(\mathbb{R})(n=3,6)$ (Walpole, 1984) and $d$ the physical dimension ( $d=2,3$ );
- the bulk and shear moduli random fields are statistically independent (a discussion for the random matrix case can be found in Guilleminot and Soize (2013a)) and exhibit the same $L$-periodic correlation structure.

In addition, it is assumed that the random fields $\{\tilde{\kappa}(\mathbf{x}, \theta), \mathbf{x} \in \Omega\}$ and $\{\tilde{\mu}(\mathbf{x}, \theta), \mathbf{x} \in \Omega\}$ are bounded from below by the associated matrix properties:
$$
\begin{equation*}
\tilde{\kappa}(\mathbf{x}, \theta)>\kappa_{\mathrm{m}}, \quad \tilde{\mu}(\mathbf{x}, \theta)>\mu_{\mathrm{m}} \text { a.s. } \tag{22}
\end{equation*}
$$
for all $\mathbf{x}$ in $\Omega$, with $\kappa_{\mathrm{m}}$ and $\mu_{\mathrm{m}}$ the bulk and shear moduli of the matrix phase. Eq. (22) appears as a reasonable assumption, given that the retained value of parameter $H$ (see Eq. (9)) is large in the simulations (recall that the heterogeneities are stiffer than the isotropic matrix phase). Note that the strict inequalities mean that the lower bounds must then be reached with a null probability. Consequently, let us introduce two auxiliary $\mathbb{R}_{*}^{+}$-valued random fields $\left\{\kappa^{\prime}(\mathbf{x}, \theta), \mathbf{x}\right. \in \Omega\}$ and $\left\{\mu^{\prime}(\mathbf{x}, \theta), \mathbf{x} \in \Omega\right\}$ such that $\forall \mathbf{x} \in \Omega$ :
$$
\begin{equation*}
\tilde{\kappa}(\mathbf{x}, \theta)=\kappa^{\prime}(\mathbf{x}, \theta)+\kappa_{\mathrm{m}}, \quad \tilde{\mu}(\mathbf{x}, \theta)=\mu^{\prime}(\mathbf{x}, \theta)+\mu_{\mathrm{m}} . \tag{23}
\end{equation*}
$$

Below, we address the construction of stochastic representations for random fields $\left\{\kappa^{\prime}(\mathbf{x}, \theta), \mathbf{x} \in \Omega\right\}$ and $\left\{\mu^{\prime}(\mathbf{x}, \theta)\right.$, $\mathbf{x} \in \Omega\}$. The methodology, pioneered in the seminal work of Soize (2006), proceeds along the following two main steps. First, a stochastic model for random variable $\kappa^{\prime}(\mathbf{x}, \theta)$ (resp. $\mu^{\prime}(\mathbf{x}, \theta)$ ), $\mathbf{x}$ being fixed in $\Omega$, is defined by invoking Jaynes' maximum entropy principle (Jaynes, 1957a,b) (see Section 3.2). Here, it is assumed that the available information on each random variable is independent of $\mathbf{x}$, so that the probability density functions $p_{\kappa^{\prime}(\mathbf{x}, \theta)}$ and $p_{\mu^{\prime}(\mathbf{x}, \theta)}$ defining $\kappa^{\prime}(\mathbf{x}, \theta)$ and $\mu^{\prime}(\mathbf{x}, \theta)$ do not depend on $\mathbf{x}$ either. Note that this assumption can be readily relaxed at the expense
of notational complexity. Second, the bulk and shear moduli random fields are defined through measurable nonlinear transformations of two underlying $\mathbb{R}$-valued Gaussian random fields such that $\left\{\kappa^{\prime}(\mathbf{x}, \theta), \mathbf{x} \in \Omega\right\}$ and $\left\{\mu^{\prime}(\mathbf{x}, \theta), \mathbf{x} \in \Omega\right\}$ respectively admit $p_{K^{\prime}(\mathbf{x}, \theta)}$ and $p_{\mu^{\prime}(\mathbf{x}, \theta)}$ as their first-order marginal probability density functions. This construction is detailed in Section 3.3.

\subsection*{3.2. Construction of an information-theoretic model for the probability density functions $p_{\kappa^{\prime}(\mathbf{x}, \theta)}$ and $p_{\mu^{\prime}(\mathbf{x}, \theta)}$}

Let $\mathbf{x}$ be fixed in $\Omega$. As mentioned above, probabilistic models for random variables $\kappa^{\prime}(\mathbf{x}, \theta)$ and $\mu^{\prime}(\mathbf{x}, \theta)$ can be constructed within the framework of information theory and more precisely, by having recourse to the maximum entropy (MaxEnt) principle (see Jaynes, 1957a,b). Since the methodology proceeds similarly for the two random moduli, technical aspects are detailed below for the bulk modulus $\kappa^{\prime}(\mathbf{x}, \theta)$ only - the result for the shear modulus is provided at the end of the section. The MaxEnt principle relies on the maximization of Shannon's differential entropy (Shannon, 1948) and then states that among all the probability density functions satisfying some constraints (stated in the form of mathematical expectations) related to a given amount of available information, the most objective probability density function (that is, the one that introduces the most reduced modeling bias) is the one that maximizes the above entropy (as a relative measure of uncertainties).

In this work, we assume that the $\mathbb{R}_{*}^{+}$-valued random variable $\kappa^{\prime}(\mathbf{x}, \theta)$ satisfies the following constraints:
$$
\begin{equation*}
\mathbb{E}\left\{\kappa^{\prime}(\mathbf{x}, \theta)\right\}=\left\langle\kappa^{\prime}\right\rangle, \tag{24}
\end{equation*}
$$
$$
\begin{equation*}
\mathbb{E}\left\{\ln \left(\kappa^{\prime}(\mathbf{x}, \theta)\right)\right\}=\vartheta, \quad|\vartheta|<+\infty, \tag{25}
\end{equation*}
$$
where $\left\langle\kappa^{\prime}(\mathbf{x}, \theta)\right\rangle=\langle\tilde{\kappa}(\mathbf{x}, \theta)\rangle-\kappa_{\mathrm{m}}$. The constraint stated by Eq. (24) allows for the prescription of the mean value, whereas Eq. (25) implies (under some given assumptions that will be made more precise later) that $\kappa^{\prime}(\mathbf{x}, \theta)$ and $\kappa^{\prime}(\mathbf{x}, \theta)^{-1}$ are second-order random variables. By using the calculus of variations, it can then be shown that the probability density function $p_{\kappa^{\prime}(\mathbf{x}, \theta)}$ takes the form
$$
\begin{equation*}
p_{\kappa^{\prime}(\mathbf{x}, \theta)}(k)=1_{\mathbb{R}_{*}^{+}}(k) \times c \times k^{-\lambda_{1}} \exp \left(-\lambda_{2} k\right), \tag{26}
\end{equation*}
$$
where $1_{\mathbb{R}_{*}^{+}}$is the indicator function of $\mathbb{R}_{*}^{+}, c$ is a real positive normalization constant and ( $\lambda_{1}, \lambda_{2}$ ) is a couple of Lagrange multipliers such that the above constraints are fulfilled. Owing to a change of parameters, it can be easily proven that the above p.d.f. can be written as
$$
\begin{equation*}
p_{\kappa^{\prime}(\mathbf{x}, \theta)}(k)=1_{\mathbb{R}_{*}^{+}}(k) \times c \times\left(\frac{k}{\left\langle\kappa^{\prime}\right\rangle}\right)^{1 / \delta_{\kappa^{\prime}}^{2}-1} \exp \left(-\frac{k}{\left\langle\kappa^{\prime}\right\rangle \delta_{\kappa^{\prime}}^{2}}\right), \tag{27}
\end{equation*}
$$
where $\delta_{K^{\prime}}$ is the coefficient of variation of bulk modulus $\kappa^{\prime}(\mathbf{x}, \theta)$ (see Ta et al. (2010) for similar results, as well as Guilleminot and Soize (2013a,b) for random matrix and random field models exhibiting other material symmetry properties). The normalization constant $c$ is then given by
$$
\begin{equation*}
c=\frac{1}{\left\langle\kappa^{\prime}\right\rangle}\left(\frac{1}{\delta_{\kappa^{\prime}}^{2}}\right)^{1 / \delta_{\kappa^{\prime}}^{2}} \frac{1}{\Gamma\left(1 / \delta_{\kappa^{\prime}}^{2}\right)}, \tag{28}
\end{equation*}
$$
with $\Gamma$ the Gamma function. Note that the condition $\delta_{K^{\prime}}< 1 / \sqrt{2}$ must hold in order to ensure the finiteness of the second-order moments for $\kappa^{\prime}(\mathbf{x}, \theta)$ and $\kappa^{\prime}(\mathbf{x}, \theta)^{-1}$ (see Soize, 2000). It can be deduced that $\kappa^{\prime}(\mathbf{x}, \theta)$ is distributed according to a Gamma distribution with shape parameter $1 / \delta_{\kappa^{\prime}}^{2}$ and scale parameter $\left\langle\kappa^{\prime}\right\rangle \delta_{\kappa^{\prime}}^{2}$ (with $\lambda_{1}=1-1 / \delta_{\kappa^{\prime}}^{2}$ and $\lambda_{2}= 1 /\left(\left\langle\kappa^{\prime}\right\rangle \delta_{\kappa^{\prime}}^{2}\right)$ ).

Similarly, it can be shown that the random shear modulus $\mu^{\prime}(\mathbf{x}, \theta)$ follows a Gamma distribution defined by shape parameter $1 / \delta_{\mu^{\prime}}^{2}$ and scale parameter $\left\langle\mu^{\prime}\right\rangle \delta_{\mu^{\prime}}^{2}$. Additional comments related to the above construction for the first-order marginal probability distribution are listed below.
- First, it is worthwhile to note that the use of an isotropic approximation is motivated by the statistical characterization detailed in Section 2.3.2. The latter shows that the stochastic residual (defined by Eq. (13)) exhibits a negligible mean and a small variance, regardless of the location under consideration.
- Second, and while the recourse to an isotropic model alleviates the computational cost, it is by no means a limitation of the overall methodology that can readily be extended to any symmetry class (see Guilleminot and Soize (2013b) for the construction of the stochastic model for any symmetry class).
- Third, the nature of the statistical dependence essentially depends on the information that is plugged into the principle of maximum entropy. In other words, changing the constraints by adding, for instance, information on cross-correlation would yield another dependence structure. However, getting converged statistical estimators for high-order moments generally requires a large amount of data, which is not the framework retained in this study. Rather, the model is here tailored in order to allow for a calibration through an underdetermined inverse problem - hence the consideration of minimal mathematical requirements only ${ }^{1}$ - and relies in part on imposing the finiteness of the second order moment for the stiffness tensor (this property is then equivalently imposed on the bulk and shear modulus, with no information on crosscorrelation between these parameters, since $\mathbf{J}$ and $\mathbf{K}$ are orthogonal projectors).

\subsection*{3.3. Definition of the mesoscale bulk and shear moduli random fields}

Following Sections 3.1 and 3.2, the bulk and shear moduli random fields can be readily defined through the following local measurable nonlinear transformations:
$$
\begin{align*}
& \forall \mathbf{x} \in \Omega, \quad \kappa^{\prime}(\mathbf{x}, \theta)=\mathrm{F}_{\mathcal{G}_{\kappa^{\prime}}}^{-1}\left(\Phi\left(\Xi_{\kappa^{\prime}}(\mathbf{x}, \theta)\right)\right) \\
& \mu^{\prime}(\mathbf{x}, \theta)=\mathrm{F}_{\mathcal{G}_{\mu^{\prime}}}^{-1}\left(\Phi\left(\Xi_{\mu^{\prime}}(\mathbf{x}, \theta)\right)\right) \tag{29}
\end{align*}
$$
where
- $\mathrm{F}_{\mathcal{G}^{\prime}}^{-1}$ (resp. $\mathrm{F}_{\mathcal{G}_{\mu^{\prime}}}^{-1}$ ) is the Gamma inverse cumulative distribution function with shape parameter $1 / \delta_{\kappa^{\prime}}^{2}\left(\right.$ resp. $\left.1 / \delta_{\mu^{\prime}}^{2}\right)$ and scale parameter $\left\langle\kappa^{\prime}\right\rangle \delta_{\kappa^{\prime}}^{2}$ (resp. $\left\langle\mu^{\prime}\right\rangle \delta_{\mu^{\prime}}^{2}$ );

\footnotetext{
${ }^{1}$ These properties allow one to prove the existence and uniqueness of a second-order solution for the associated stochastic boundary value problem.
}
- $\Phi$ is the cumulative distribution function of the standard normal distribution;
- $\left\{\Xi_{\kappa^{\prime}}(\mathbf{x}, \theta), \mathbf{x} \in \mathbb{R}^{d}\right\}$ and $\left\{\Xi_{\mu^{\prime}}(\mathbf{x}, \theta), \mathbf{x} \in \mathbb{R}^{d}\right\}$ are centered Gaussian random fields such that for all $\mathbf{x}$ in $\Omega$ and for all $\mathbf{y}$ in $\mathbb{Z}^{d}$ :
$$
\begin{align*}
& \Xi_{\kappa^{\prime}}(\mathbf{x}+L \mathbf{y}, \theta)=\Xi_{\kappa^{\prime}}(\mathbf{x}, \theta) \text { and } \\
& \Xi_{\mu^{\prime}}(\mathbf{x}+L \mathbf{y}, \theta)=\Xi_{\mu^{\prime}}(\mathbf{x}, \theta) \text { a.s. } \tag{30}
\end{align*}
$$

Moreover, and for $(\mathbf{x}, \mathbf{y}) \in \Omega \times \Omega$, the associated normalized correlation functions $\mathcal{R}_{\Xi_{\kappa^{\prime}}}$ and $\mathcal{R}_{\Xi_{\mu^{\prime}}}$ are such that
$$
\begin{equation*}
\mathcal{R}_{\Xi_{\kappa^{\prime}}}(\mathbf{x}, \mathbf{y})=\mathbb{E}\left\{\Xi_{\kappa^{\prime}}(\mathbf{x}) \Xi_{\kappa^{\prime}}(\mathbf{y})\right\}, \quad \mathcal{R}_{\Xi_{\kappa^{\prime}}}(\mathbf{x}, \mathbf{x})=1 \tag{31}
\end{equation*}
$$
and
$\mathcal{R}_{\Xi_{\mu^{\prime}}}(\mathbf{x}, \mathbf{y})=\mathbb{E}\left\{\Xi_{\mu^{\prime}}(\mathbf{x}) \Xi_{\mu^{\prime}}(\mathbf{y})\right\}, \quad \mathcal{R}_{\Xi_{\mu^{\prime}}}(\mathbf{x}, \mathbf{x})=1$.

Each of the above correlation function is assumed to have a separable structure, namely
$$
\begin{equation*}
\mathcal{R}_{\Xi_{\kappa^{\prime}}}(\mathbf{x}, \mathbf{y})=\mathcal{R}_{\Xi_{\mu^{\prime}}}(\mathbf{x}, \mathbf{y})=\prod_{k=1}^{d} r\left(\left|x_{k}-y_{k}\right|\right) \tag{33}
\end{equation*}
$$
for all ( $\mathbf{x}, \mathbf{y}$ ) in $\Omega \times \Omega$. From the statistical characterization on the numerical database, it is further assumed that each correlation function depends on a single parameter denoted by $\alpha$, no matter the Gaussian field or the direction involved. In addition, the one-dimensional correlation function $r$ is here chosen as the so-called periodic correlation function (Rasmussen and Williams, 2005):
$\forall \tau \in[0, L], \quad r(\tau)=\exp \left(-\frac{2}{\alpha^{2}} \sin ^{2}\left(\frac{\pi \tau}{L}\right)\right)$.

The above function can be recovered by introducing polar coordinates in a two-dimensional squared exponential correlation function, so that it satisfies all mathematical requirements in accordance with the framework of periodic homogenization. Finally, it should be noticed that a general representation could be obtained by considering expansions of the correlation functions in Fourier series: such expansions would, however, require an extensive database - hence justifying the alternative path that is followed in this work.

It is worth noticing that due to the nonlinear mapping defined by Eq. (29), the correlation functions of the bulk and shear moduli random fields do not coincide with those of the associated Gaussian random fields. However, numerical evidences show that the nonlinear transformations generally induce similar shapes for the correlation functions of the image random fields, with a slight modification of the correlation length. In practice, parameter $\alpha$ must then be calibrated by solving an inverse problem, so that the resulting random fields $\left\{\kappa^{\prime}(\mathbf{x}, \theta), \mathbf{x} \in \Omega\right\}$ and $\left\{\mu^{\prime}(\mathbf{x}, \theta), \mathbf{x} \in \Omega\right\}$ exhibit the target correlation structure. Note that in a more general context, different correlation lengths can be readily introduced in the previous formulation, hence allowing for the modeling of bulk and shear moduli random fields exhibiting different correlation lengths in some given directions. The identification task for the parameters of the probabilistic representation is now addressed in the following section.

\subsection*{3.4. Calibration of the probabilistic model}

\subsection*{3.4.1. Identification through statistical estimators}

From the previous section, it follows that the stochastic model depends on the parameters defining the first-order marginal distributions, as well as on parameter $\alpha$ (which controls the correlation lengths of the fields). The former can be readily obtained by using statistical estimators on the simulated fields. Specifically, it is found that
$$
\begin{equation*}
\left\langle\kappa^{\prime}\right\rangle=1.02 \mu_{m}, \quad\left\langle\mu^{\prime}\right\rangle=0.78 \mu_{m} \tag{35}
\end{equation*}
$$
and
$$
\begin{equation*}
\delta_{K^{\prime}}=0.321, \quad \delta_{\mu^{\prime}}=0.329 . \tag{36}
\end{equation*}
$$

We denote by $\tau \mapsto \mathcal{R}_{\kappa^{\prime}}^{\bmod }\left(\tau \mathbf{e}_{i} ; \alpha\right)$ (see Eq. (20); note that the dependence on $\alpha$ is made explicit for subsequent mathematical consistency) the correlation function for the bulk modulus random field defined by the stochastic model, whereas $\tau \mapsto \mathcal{R}_{K^{\prime}}^{\text {data }}\left(\tau \mathbf{e}_{i}\right)$ is the correlation function for the bulk modulus that is estimated from the numerical experiments - similar notations are used for the shear modulus random fields. The optimal value $\alpha^{\text {opt }}$ for the correlation parameter involved in Eq. (34) is then defined as
$$
\begin{equation*}
\alpha^{\mathrm{opt}}=\underset{\alpha \in] 0,+\infty[ }{\operatorname{argmin}} \mathcal{J}(\alpha), \tag{37}
\end{equation*}
$$
where the cost function $\mathcal{J}$ is given by:
$$
\begin{equation*}
\mathcal{J}(\alpha)=\sum_{i}^{d} \int_{0}^{L}\left|\mathcal{R}_{\kappa^{\prime}}^{\mathrm{mod}}\left(\tau \mathbf{e}_{i} ; \alpha\right)-\mathcal{R}_{\kappa^{\prime}}^{\mathrm{data}}\left(\tau \mathbf{e}_{i}\right)\right| d \tau . \tag{38}
\end{equation*}
$$

Note that $\mathcal{J}$ only involves the bulk modulus, since the latter and the shear modulus random fields exhibit the same correlation functions, both in the numerical experiments on the microstructure and in the model-based simulations. Here, the optimization problem defined by Eq. (37) is solved with the Newton method; the plot of cost function $\alpha \mapsto \mathcal{J}(\alpha)$ is shown in Fig. 10. The optimized value is found to be $\alpha^{\text {opt }}=$ 0.74.

\subsection*{3.4.2. Identification through the maximum likelihood method}

In this section, we address the calibration of the random field model by invoking the maximum likelihood principle,

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/f8994420-ef58-4ead-9ba2-9433d12e1c47-09.jpg?height=600&width=779&top_left_y=1750&top_left_x=122}
\captionsetup{labelformat=empty}
\caption{Fig. 10. Plot of the cost function $\alpha \mapsto \mathcal{J}(\alpha)$ defined by Eq. (38).}
\end{figure}
with the aim to substantially reduce the number of corrector problems to be solved. The first-order marginal distributions are first calibrated as follows. Let us consider the random bulk modulus, and define the optimal parameters as
$$
\begin{equation*}
\left(\left\langle\kappa^{\prime}\right\rangle, \delta_{\kappa^{\prime}}\right)=\underset{(m, \delta) \in \mathbb{V}}{\arg \max } \ln (\mathcal{L}(m, \delta)), \tag{39}
\end{equation*}
$$
where $\left.\mathbb{V}=\mathbb{R}_{+}^{*} \times\right] 0,1 / \sqrt{2}[$ and $\mathcal{L}$ denotes the following likelihood function:
$$
\begin{equation*}
\mathcal{L}(m, \delta)=\prod_{i=1}^{N_{\exp }} p\left(\kappa^{\prime \text { data }}\left(\theta_{i}\right) ; m, \delta\right), \tag{40}
\end{equation*}
$$
with $p(\cdot ; m, \delta)$ the probability density function of the Gamma distribution with mean $m$ and coefficient of variation $\delta$ (see Eq. (27)), and $\left\{\kappa^{\prime \text { data }}\left(\theta_{i}\right)\right\}_{1 \leqslant i \leqslant N_{\text {exp }}}$ a set of digitally generated experimental realizations obtained either from a single realization of the field (by sampling points that are sufficiently far apart from each other) and from several realizations. The parameters for the first-order probability density function associated with the random shear moduli are similarly calibrated by using the digitally generated experimental realizations $\left\{\mu^{\prime \text { data }}\left(\theta_{i}\right)\right\}_{1 \leqslant i \leqslant N_{\text {exp }}}$. The graphs of the above cost function for each modulus are shown in Fig. 11 for two values of $N_{\text {exp }}$.The optimal parameters calibrated by the maximum likelihood principle are found to be as follows:
$$
\begin{align*}
& \left\langle\kappa^{\prime}\right\rangle=1.02 \mu_{m}, \quad \delta_{\kappa^{\prime}}=0.323 ; \quad\left\langle\mu^{\prime}\right\rangle=0.78 \mu_{m} \\
& \delta_{\mu^{\prime}}=0.332 . \tag{41}
\end{align*}
$$

These values are reasonably close to the ones obtained by using classical statistical estimators (see Section 3.4.1). It is worth noticing that although the cost function involved in Eq. (39) exhibits slightly different shapes as $N_{\text {exp }}$ varies in $\{10,50,150,200,700\}$, the maximum value turns out to be reached at the same point, hence providing a robust estimator for the parameters under consideration. In order to proceed with the calibration of the parameter $\alpha$ involved in the description of the correlation structure of the underlying Gaussian random field, let us consider a set of points $\left\{\tilde{\mathbf{x}}^{(i)}\right\}_{1 \leqslant i \leqslant N_{p}}$ in $\Omega$ and introduce the $N_{p}$-valued random vector
$$
\begin{equation*}
\boldsymbol{\kappa}^{\prime}=\left(\kappa^{\prime}\left(\tilde{\mathbf{x}}^{(i)}\right), \ldots, \kappa^{\prime}\left(\tilde{\mathbf{x}}^{\left(N_{p}\right)}\right)\right) . \tag{42}
\end{equation*}
$$

In practice, the above set of points may be selected and optimized, for a given value of $N_{p}$, in accordance with the expected structure of correlation. In this work, where the underlying microstructure is statistically isotropic, the points are randomly placed and such that for all $1 \leq i \leq N_{p}$ and 1 $\leq j \leq 3, \tilde{x}_{j}^{(i)}$ corresponds to the absolute value of an independent realization of a Gaussian random variable with a null mean and a standard deviation equal to $L / 6$. Note that the probability distribution of $\boldsymbol{\kappa}^{\prime}$ is unknown. Let $p_{\boldsymbol{\kappa}^{\prime \text { model }}}(\cdot ; \alpha)$ be the probability density function of $\boldsymbol{\kappa}^{\prime}$ estimated from realizations of the probabilistic model for the bulk modulus random field, with the first-order marginal defined by the parameters given above (hence, the explicit dependence on $\alpha$ ). An optimal value is then sought as
$$
\begin{equation*}
\alpha^{\mathrm{opt}}=\underset{\alpha>0}{\arg \max } \ln \left(\mathcal{L}^{*}(\alpha)\right), \tag{43}
\end{equation*}
$$

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/f8994420-ef58-4ead-9ba2-9433d12e1c47-10.jpg?height=1423&width=998&top_left_y=183&top_left_x=439}
\captionsetup{labelformat=empty}
\caption{Fig. 11. Plot of the cost function $\ln (\mathcal{L}(m, \delta))$ (see Eq. (39) for $N_{\exp }=50$ (top) and $N_{\exp }=100$ (bottom).}
\end{figure}
with
$$
\begin{equation*}
\mathcal{L}^{*}(\alpha)=\prod_{i=1}^{N_{\exp }} p_{\boldsymbol{\kappa}^{\prime \operatorname{model}}}\left(\boldsymbol{\kappa}^{\prime \text { data }}\left(\theta_{i}\right) ; \alpha\right) \tag{44}
\end{equation*}
$$
the multidimensional likelihood function. Note that the cost function does not involve information on the shear modulus, since the latter exhibits, in the present case, the same correlation characteristics. The plot of cost function $\alpha \mapsto \mathcal{L}^{*}(\alpha)$ is shown in Fig. 12 for several values of $N_{\text {exp }}$. An optimal value is found as $\alpha^{\mathrm{opt}}=0.85$. It is seen that this estimation on correlation parameter $\alpha$ is also robust with respect to the number of realizations, so that the corrector problem has now to be solved on a limited number of realizations. It should be noted that the underlying Gaussian random fields are generated through a Krylov iterative method (Chow and Saad, 2014). The choice of the stopping criterion parameter in this

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/f8994420-ef58-4ead-9ba2-9433d12e1c47-10.jpg?height=600&width=783&top_left_y=1748&top_left_x=985}
\captionsetup{labelformat=empty}
\caption{Fig. 12. Plot of the cost function $\alpha \mapsto \ln \left(\mathcal{L}^{*}(\alpha)\right)$.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/f8994420-ef58-4ead-9ba2-9433d12e1c47-11.jpg?height=613&width=787&top_left_y=185&top_left_x=118}
\captionsetup{labelformat=empty}
\caption{Fig. 13. Comparison between the probability density functions of mesoscopic bulk moduli obtained from the numerical experiments and from model-based simulations.}
\end{figure}
method may explain the discrepancy between the above estimation and the one obtained from the statistical estimator.

\subsection*{3.5. Validation of the information-theoretic probabilistic model}

This section is concerned with the validation of the probabilistic model calibrated in Section 3.4.1. The relevance of model-based predictions is first discussed on the basis of mesoscale quantities of interest in Section 3.5.1, whereas a comparison for homogenized effective properties is provided in Section 3.5.2.

\subsection*{3.5.1. Validation on mesoscale quantities of interest}

Here, we investigate to what extent the probabilistic model can represent some quantities of interest defined at mesoscale. The probability density functions of bulk and shear moduli at a given point (recall that these quantities are invariant under translation in $\mathbb{R}^{d}$ ) are shown in Figs. 13 and 14, both for the numerical experiments and for the modelbased simulations.

A good match is observed, regardless of the modulus under investigation and no matter the calibration strategy (in the present case, the parameters obtained from statistical estimators or from the maximum likelihood principle coincide). Next, the numerical experiments and model-based correlation functions are compared in Fig. 15 for the bulk modulus random field.

It is seen that the correlation function calibrated by using the statistical estimator reproduces almost perfectly the numerical experiments curve, whereas a small discrepancy with the one based on the likelihood estimator is observed. In the next section, we finally address the validation task in terms of effective properties.

\subsection*{3.5.2. Validation based on effective properties}

Below, we compare the statistical averages (over 100 realizations) of effective properties as obtained, either by solving a set of mesoscale corrector problems involving the calibrated stochastic model, or by directly solving the

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/f8994420-ef58-4ead-9ba2-9433d12e1c47-11.jpg?height=615&width=788&top_left_y=185&top_left_x=967}
\captionsetup{labelformat=empty}
\caption{Fig. 14. Comparison between the probability density functions of mesoscopic shear moduli obtained from the numerical experiments and from model-based simulations.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/f8994420-ef58-4ead-9ba2-9433d12e1c47-11.jpg?height=604&width=782&top_left_y=942&top_left_x=971}
\captionsetup{labelformat=empty}
\caption{Fig. 15. Estimated covariance function of the random bulk modulus along the $x_{1}$-axis: comparison between numerical experiments and model-based simulations.}
\end{figure}

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 1
Comparison between the effectives moduli derived from the numerical experiments and the probabilistic model (the $99 \%$ confidence interval is reported).}
\begin{tabular}{|l|l|l|l|}
\hline & Numerical experiments & Model SE & Model MLE \\
\hline $\left\langle\kappa^{\prime}\right\rangle / \mu_{\mathrm{m}}$ & & 1.02 & 1.02 \\
\hline $\left\langle\mu^{\prime}\right\rangle / \mu_{\mathrm{m}}$ & & 0.78 & 0.78 \\
\hline $\delta_{K^{\prime}}$ & & 0.321 & 0.323 \\
\hline $\delta_{\mu^{\prime}}$ & & 0.329 & 0.332 \\
\hline $\alpha$ & & 0.74 & 0.85 \\
\hline $\kappa^{\text {eff }} / \mu_{\mathrm{m}}$ & $2.326 \pm 0.001$ & $2.331 \pm 0.015$ & $2.314 \pm 0.087$ \\
\hline $\mu^{\text {eff }} / \mu_{\mathrm{m}}$ & $1.767 \pm 0.001$ & $1.762 \pm 0.012$ & $1.748 \pm 0.068$ \\
\hline
\end{tabular}
\end{table}
corrector problems defined at the microscale. The results (within the $99 \%$ confidence interval) are summarized in Table 1. It is found that the effective properties computed from the model are in very good agreement with the ones derived from the numerical experiments.

\section*{4. Conclusion}

In this work, we have investigated the capabilities of information-theoretic random field models to accurately represent mesoscopic elasticity fields obtained through the filtering procedure proposed in (Bignonnet et al., 2014). For illustration purposes, the analysis has been performed on a simple but non-trivial model microstructure. We first introduced a random field model and addressed the calibration task by using either statistical estimators or the maximum likelihood principle. A comparison on some quantities of interest, such as the effective properties, was subsequently carried out. It is shown that for the case under study, the information-theoretic model can be calibrated on a limited set of realizations and still allows for accurate predictions of the effective properties, hence allowing for substantial computational savings in numerical homogenization.

Whereas the capabilities of the model to mimic the mesoscale features strongly depend on the statistical characteristics of the underlying microstructure, it should be noticed that more elaborated random field models have been proposed elsewhere (see e.g. Guilleminot and Soize, 2013b) in order to handle more complex morphologies. The extension of these techniques to nonlinear/inelastic behaviors is worth investigating and raises many difficulties due to the large number of internal variables involved in the constitutive models. In regard of filtering technique, the definition of mesoscopic stress (strain) field remains consistent with macroscopic stress (strain) regardless of the non-linearity. For elastoplastic composites, a promising technique is the Nonuniform Transformation Field Analysis technique (NTFA) (Michel and Suquet, 2003), for instance. The idea behind NTFA is that the anelastic strains can be expressed by a finite number of modes. If the modes are well chosen, the number of internal variables is low and the results still accurate. However, extension of the filtering framework to such behaviors remains unclear. Besides, the construction of suitable probabilistic models is still an open question.

\section*{Acknowledgments}

The authors would like to thank the two anonymous reviewers for their valuable comments on this paper. This work has benefited from a French Government grant managed by ANR within the frame of the national program Investments for the Future ANR-11-LABX-022-01. The work of J. Guilleminot was supported by the French National Research Agency (ANR) (MOSAIC project, ANR-12-JS09-0001-01).

\section*{References}

Allen, M.P., Tildesley, D.J., 1987. Computer Simulation of Liquids. Oxford University Press, Oxford.
Baxter, S., Graham, L., 2000. Characterization of random composites using moving-window technique. J. Eng. Mech. 126 (4), 389-397. doi:10.1061/ (ASCE)0733-9399(2000)126:4(389).
Baxter, S., Hossain, M., Graham, L., 2001. Micromechanics based random material property fields for particulate reinforced composites. Int. J. Solids Struct. 38, 9209-9220. doi:10.1016/S0020-7683(01)00076-2.

Bignonnet, F., Sab, K., Dormieux, L., Brisard, S., Bisson, A., 2014. Macroscopically consistent non-local modeling of heterogeneous media. Comput. Method. Appl. Mech. 278, 218-238. doi:10.1016/j.cma.2014.05.014.
Brisard, S., Dormieux, L., 2010. FFT-based methods for the mechanics of composites: a general variational framework. Comp. Mater. Sci. 49 (3), 663671. doi:10.1016/j.commatsci.2010.06.009.

Brisard, S., Dormieux, L., 2012. Combining Galerkin approximation techniques with the principle of Hashin and Shtrikman to derive a new FFTbased numerical method for the homogenization of composites. Comput. Method. Appl. Mech. 217-220, 197-212. doi:10.1016/j.cma.2012.01. 003.

Chow, E., Saad, Y., 2014. Preconditioned Krylov subspace methods for sampling multivariate gaussian distributions. SIAM J. Sci. Comput. 36 (2), A588-A608. doi:10.1137/130920587.
Farmer, C.L., 2002. Upscaling: a review. Int. J. Numer. Methods Fluids 40 (12), 63-78. doi:10.1002/fld.267.

Graham, L., Baxter, S., 2001. Simulation of local material properties based on moving-window GMC. Probab. Eng. Mech. 16, 295-305. doi:10.1016/ S0266-8920(01)00022-4.
Guilleminot, J., Soize, C., 2012. Stochastic modeling of anisotropy in multiscale analysis of heterogeneous materials: a comprehensive overview on random matrix approaches. Mech. Mater. 44, 35-46. doi:10.1016/j. mechmat.2011.06.003.
Guilleminot, J., Soize, C., 2013a. On the statistical dependence for the components of random elasticity tensors exhibiting material symmetry properties. J. Elast. 111 (2), 109-130. doi:10.1007/s10659-012-9396-z.
Guilleminot, J., Soize, C., 2013b. Stochastic model and generator for random fields with symmetry properties: application to the mesoscopic modeling of elastic random media. SIAM Multiscale Model. Simul. 11 (3), 840870. doi:10.1137/120898346.

Hou, T.Y., Wu, X.-H., 1997. A multiscale finite element method for elliptic problems in composite materials and porous media. J. Comput. Phys. 134 (1), 169-189. doi:10.1006/jcph.1997.5682.

Jaynes, E.T., 1957a. Information theory and statistical mechanics. Phys. Rev. 106, 620-630. doi:10.1103/PhysRev.106.620.
Jaynes, E.T., 1957b. Information theory and statistical mechanics. II. Phys. Rev. 108, 171-190. doi:10.1103/PhysRev.108.171.
Michel, J., Suquet, P., 2003. Nonuniform transformation field analysis. Int. J. Solids Struct. 40 (25), 6937-6955. http://dx.doi.org/10.1016/ S0020-7683(03)00346-9. Special issue in Honor of George J. Dvorak.
Moakher, M., Norris, A.N., 2006. The closest elastic tensor of arbitrary symmetry to an elasticity tensor of lower symmetry. J. Elast. 215-263. doi:10. 1007/s10659-006-9082-0.
Moulinec, H., Suquet, P., 1994. A fast numerical method for computing the linear and nonlinear properties of composites. C. R. Acad. Sci. IIB: Mech. 318 (11), 1417-1423.
Moulinec, H., Suquet, P., 1998. A numerical method for computing the overall response of nonlinear composites with complex microstructure. Comput. Method. Appl. Mech. 157 (1-2), 69-94.
Ostoja-Starzewski, M., 1998. Random field models of heterogeneous materials. Int. J. Solids Struct. 35, 2429-2455. doi:10.1016/S0020-7683(97) 00144-3.
Rasmussen, C.E., Williams, C.K.I., 2005. Gaussian processes for machine learning. Adaptive Computation and Machine Learning. The MIT Press doi:10.1007/b99676.
Sena, M., Ostoja-Starzewski, M., Costa, L., 2013. Stiffness tensor random fields through upscaling of planar random materials. Probab. Eng. Mech. 34, 131-156. doi:10.1016/j.probengmech.2013.08.008.
Shannon, C.E., 1948. A mathematical theory of communication. Bell Syst. Tech. J. 27, 379-423,623-656. doi:10.1002/j.1538-7305.1948.tb00917.x.
Soize, C., 2000. A nonparametric model of random uncertainties for reduced matrix models in structural dynamics. Probab. Eng. Mech. 15 (3), 277294. doi:10.1016/S0266-8920(99)00028-4.

Soize, C., 2006. Non-gaussian positive-definite matrix-valued random fields for elliptic stochastic partial differential operators. Comput. Method. Appl. M. 195 (1-3), 26-64. doi:10.1016/j.cma.2004.12.014.
Ta, Q.A., Clouteau, D., Cottereau, R., 2010. Modeling of random anisotropic elastic media and impact on wave propagation. Eur. J. Comput. Mech. 19 (1-3), 241-253. doi:10.3166/ejcm.19.241-253.
Torquato, S., 2002. Random Heterogeneous Materials. Springer, New York.
Walpole, L.J., 1984. Fourth-rank tensors of the thirty-two crystal classes: multiplication tables. Proc. R. Soc. Lond. A 391 (1800), pp.149-179. doi:10.1098/rspa.1984.0008.