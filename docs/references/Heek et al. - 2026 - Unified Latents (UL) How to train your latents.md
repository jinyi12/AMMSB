\title{
Unified Latents (UL): How to train your latents
}

\author{
Jonathan Heek ${ }^{1}$, Emiel Hoogeboom ${ }^{1}$, Thomas Mensink ${ }^{1}$ and Tim Salimans ${ }^{1}$ \\ ${ }^{1}$ Google DeepMind Amsterdam
}

We present Unified Latents (UL), a framework for learning latent representations that are jointly regularized by a diffusion prior and decoded by a diffusion model. By linking the encoder's output noise to the prior's minimum noise level, we obtain a simple training objective that provides a tight upper bound on the latent bitrate. On ImageNet-512, our approach achieves competitive FID of 1.4, with high reconstruction quality (PSNR) while requiring fewer training FLOPs than models trained on Stable Diffusion latents. On Kinetics-600, we set a new state-of-the-art FVD of 1.3.

\section*{1. Introduction}

Diffusion models have become remarkably successful for image, video, and audio generation. An important factor in this success has been latent representations, compact encodings that allow diffusion models to scale to higher resolutions more efficiently.

However, it remains unclear how best to learn such latents. The original Latent Diffusion Model (Rombach et al., 2022) uses a VAE-style KL penalty between the latent distribution and a standard Gaussian. Since the decoder lacks a likelihood-based loss, the weight of the KL term must be set manually, making it difficult to reason about the information content of the latents.

Recently, works have focused on getting semantic representations from either pretrained networks (e.g. from DINO) or heavily regularized autoencoders. These latents are usually easier to learn due to their lower information density and obtain impressive FIDs. However, high frequency information typically gets lost, which can be seen by worse PSNRs or heavy reconstruction artifacts.

Simply put, there is a trade-off between the information content of the latent, and the reconstruction quality of the output. If the structure of the latent is easier to learn, this generally leads to better generation performance. The easier to learn a latent is while retaining its information density, the better the resulting generation quality.

In theory, even a single unregularized latent channel could encode an arbitrary amount of information. In practice, the actual information is limited by machine precision and encoder smoothness. The number of latent channels therefore determines the information capacity: fewer channels yield easier-to-model latents at the cost of reconstruction quality, while more channels enable near-perfect reconstruction but require greater modeling capacity. This paper shows how to navigate this trade-off systematically.

The key question we address is: How should latents be regularized when they will subsequently be modeled by a diffusion model? Our answer: by co-training a diffusion prior on them. This approach, which we call Unified Latents, rests on three key ideas:
- Encode latents with a fixed amount of Gaussian noise.
- Align the prior diffusion model with the minimum noise level. As a consequence the KL term reduces to a simple weighted MSE over noise levels.
- Use a reweighted elbo loss (sigmoid weighting) for the decoder.

These components work together to train latents that are simultaneously encoded, regularized, and

\footnotetext{
Corresponding author(s): \{jheek, emielh, mensink, salimans\}@google.com
© 2026 Google DeepMind. All rights reserved
}
modeled using diffusion. This provides an interpretable bound on the bits in the latents, and simple hyper-parameters to control the reconstruction-modelling tradeoff.

\section*{2. Background}

Variational AutoEncoders Variational inference provides a principled approach to learning latent representations. Given images $\boldsymbol{x}$ that we wish to model, we can derive the Evidence Lower Bound (ELBO) on the log-likelihood when using a latent variable $\boldsymbol{z}_{0}$ :
$$
\begin{equation*}
-\log p_{\theta}(\boldsymbol{x}) \leq \mathbb{E}_{\boldsymbol{z}_{0} \sim p_{\theta}\left(\boldsymbol{z}_{0} \mid \boldsymbol{x}\right)}[-\underbrace{\log p_{\theta}\left(\boldsymbol{x} \mid \boldsymbol{z}_{0}\right)}_{\text {decoder }}]+\operatorname{KL}[\underbrace{p_{\theta}\left(\boldsymbol{z}_{0} \mid \boldsymbol{x}\right)}_{\text {encoder }} \mid \underbrace{p_{\theta}\left(\boldsymbol{z}_{0}\right)}_{\text {prior }}]=L(\boldsymbol{x}), \tag{1}
\end{equation*}
$$

In this work both the decoder $p_{\theta}\left(\boldsymbol{x} \mid \boldsymbol{z}_{0}\right)$ and the prior $p_{\theta}\left(\boldsymbol{z}_{0}\right)$ will be learned with a diffusion model.

Diffusion Models Diffusion models can be used to model arbitrary continuous distributions. A diffusion model learns to revert a gradual destruction process which enables compression, likelihood estimation, and sampling from the distribution of interest.

Consider a data distribution $q(x)$ and a destruction process $\boldsymbol{x}_{t}=\alpha(t) x+\sigma(t) \epsilon$ with $\epsilon \sim \mathcal{N}(0,1)$. The level of destruction is defined by the logsnr schedule $\lambda(t)=\log \left(\alpha_{t}^{2} / \sigma_{t}^{2}\right)$. Additionally, we use $\alpha_{t}^{2}+\sigma_{t}^{2}=1$ for convenience. A learned model predicts the clean data $\hat{\boldsymbol{x}}\left(\boldsymbol{z}_{t}, \theta\right)$. One can show (Ho et al., 2020; Kingma et al., 2021) that the information required to encode a sample $p\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}\right)$ using the diffusion model $p\left(\boldsymbol{x}_{0}\right)$ is
$$
\begin{align*}
\operatorname{KL}\left[p\left(\boldsymbol{x}_{0} \mid \boldsymbol{x}\right) \mid p\left(\boldsymbol{x}_{0}\right)\right] & \leq \operatorname{KL}\left[p\left(\boldsymbol{x}_{0}, \ldots, \boldsymbol{x}_{1} \mid \boldsymbol{x}\right) \mid p_{\theta}\left(\boldsymbol{x}_{0}, \ldots, \boldsymbol{x}_{1}\right)\right] \\
& =\mathbb{E}_{t \sim \mathcal{U}(0,1)}\left[-\frac{\mathrm{d} \lambda(t)}{\mathrm{d} t} \frac{\exp \lambda(t)}{2} w\left(\lambda_{t}\right)\left\|\boldsymbol{x}-\hat{\boldsymbol{x}}\left(\boldsymbol{x}_{t}, \theta\right)\right\|^{2}\right]+\operatorname{KL}\left[p\left(\boldsymbol{x}_{1} \mid \boldsymbol{x}\right) \mid p\left(\boldsymbol{x}_{1}\right)\right], \tag{2}
\end{align*}
$$
where $w\left(\lambda_{t}\right)=1$ is required for the bound to hold. However, in standard diffusion models a more image-quality friendly weighting is chosen such as $w\left(\lambda_{t}\right)=\operatorname{sigmoid}\left(\lambda_{t}-b\right)$. This re-weighted ELBO formulation has the added benefit that the weighting is invariant to the choice of schedule $\lambda(t)$ (Kingma \& Gao, 2023).

Note that although the destruction process is applied to a clean data point $\boldsymbol{x}$, it only models up to the minimal noise level $\boldsymbol{x}_{0}$. This will be important later, as our prior model will output slightly noisy latents $\boldsymbol{z}_{0}$. If the VAE encoder does not predict a single encoding but a distribution $p(z \mid x)$ the above ELBO is insufficient. Prior work has generalized the KL for the more complex case of arbitrary encoder distribution (Vahdat et al., 2021).

Weighting diffusion ELBOs The above mentioned weighting means that diffusion models offer a unique perspective on log-likelihood optimization. Their loss is decomposed over noise levels. This can be used for example to down-weight the loss contributions of imperceptible high frequency details. For a latent prior however, the encoder could abuse this by encoding information at the most discounted noise levels. Therefore, a VAE with a diffusion prior should use unweighted ELBO loss $w\left(\lambda_{z}(t)\right)=1$.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/f31373f8-0b83-4f59-ac12-b02ffd8edde1-03.jpg?height=686&width=1534&top_left_y=299&top_left_x=260}
\captionsetup{labelformat=empty}
\caption{Figure 1 | Schematic overview of our model, include the Encoder $\left(E_{\theta}\right)$, the prior latent diffusion model $\left(P_{\theta}\right)$, and the diffusion decoder model ( $D_{\theta}$ ).}
\end{figure}

\section*{3. Unified Latent Diffusion}

This section describes how to train Unified Latents. The first section covers the latent encoding which is regularized by a diffusion prior using Eq. 1. Secondly, we describe how to design a diffusion decoder which models the reconstruction term $\log p_{\theta}\left(\boldsymbol{x} \mid \boldsymbol{z}_{0}\right)$. Lastly, we describe the second stage of training where the encoder and decoder are frozen and a new model is trained on the latents. An overview of inputs and outputs during training is visualized in Figure 1.
```
Algorithm 1 Training Unified Latents
    Sample $\boldsymbol{x} \sim p_{\text {data }}$
    Encode the data $z_{\text {clean }}=E(\boldsymbol{x}, \theta)$
    Sample $t \sim \mathcal{U}(0,1), \epsilon \sim \mathcal{N}(0, \mathrm{I})$
    $\boldsymbol{z}_{t}=\alpha_{z}(t) \boldsymbol{z}_{\text {clean }}+\sigma_{z}(t) \boldsymbol{\epsilon}$
    Compute prior loss $\mathcal{L}_{z}(\theta)=-\frac{\mathrm{d} \lambda_{z}(t)}{\mathrm{d} t} \frac{\exp \lambda_{z}(t)}{2}\left\|\boldsymbol{z}_{\text {clean }}-\hat{\boldsymbol{z}}\left(\boldsymbol{z}_{t}, \theta\right)\right\|^{2}+\operatorname{KL}\left[p\left(\boldsymbol{z}_{1} \mid \boldsymbol{x}\right) \mid p\left(\boldsymbol{z}_{1}\right)\right]$
    Sample $t \sim \mathcal{U}(0,1), \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I}), \boldsymbol{\epsilon}_{z} \sim \mathcal{N}(0, \mathbf{I})$
    $\boldsymbol{z}_{0}=\alpha_{z}(0) \boldsymbol{z}_{\text {clean }}+\sigma_{z}(0) \boldsymbol{\epsilon}_{z}$
    $\boldsymbol{x}_{t}=\alpha_{x}(t) x+\sigma_{x}(t) \boldsymbol{\epsilon}$
    Compute decoder loss $\mathcal{L}_{x}(\theta)=\frac{\mathrm{d} \lambda_{x}(t)}{\mathrm{d} t} \frac{\exp \lambda_{x}(t)}{2} w\left(\lambda_{x}(t)\right)\left\|\boldsymbol{x}-\hat{\boldsymbol{x}}\left(\boldsymbol{x}_{t}, \boldsymbol{z}_{0}, \theta\right)\right\|^{2}$
    Optimize $\mathcal{L}(\theta)=\mathcal{L}_{z}(\theta)+\mathcal{L}_{x}(\theta)$
```

```
Algorithm 2 Sampling Unified Latents
    Sample $\boldsymbol{z}_{1} \sim \mathcal{N}(0, \mathbf{I})$
    Sample $\boldsymbol{z}_{0} \sim p_{\theta}\left(\boldsymbol{z}_{0} \mid \boldsymbol{z}_{1}\right)$ from diffusion base model
    Sample $\boldsymbol{x}_{1} \sim \mathcal{N}(0, \mathbf{I})$
    Sample $\boldsymbol{x} \sim p_{\theta}\left(\boldsymbol{x} \mid \boldsymbol{z}_{0}, \boldsymbol{x}_{1}\right)$ from diffusion decoder model
```


\subsection*{3.1. Encoding and Prior: Linking encoding noise and diffusion precision}

A key design decision is how much precision to use when encoding the latent. In principle, a continuous variable can encode infinite bits, and floating-point representations typically support 16-32 bits-

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/f31373f8-0b83-4f59-ac12-b02ffd8edde1-04.jpg?height=339&width=1132&top_left_y=294&top_left_x=479}
\captionsetup{labelformat=empty}
\caption{Figure 2 | Unified Latents overview. An image $\boldsymbol{x}$ is encoded to $\boldsymbol{z}_{\text {clean }}$. A diffusion prior models the path from pure noise $\boldsymbol{z}_{1}$ to a slightly noisy latent $\boldsymbol{z}_{0}$. This $\boldsymbol{z}_{0}$ is then used by a diffusion decoder to reconstruct the image. The prior thus measures and regularizes the information content of $z_{0}$.}
\end{figure}
though encoder and decoder smoothness make it practically impossible to utilize this capacity in full. In standard VAEs, encoder noise limits information content; for example, Stable Diffusion learns this noise level. We take a different approach: we explicitly link the encoder noise to the maximum precision of the prior diffusion model.

Let $\boldsymbol{z}_{\text {clean }}=E(\boldsymbol{x}, \theta)$ denote the deterministic latent encoding. The encoder should approximate the posterior $p(z \mid x)$, which is typically parameterized by a flexible distribution. However, following Vahdat et al. (2021), we find that learning a flexible encoder distribution causes instability.

We propose a simpler approach: the encoder predicts a single deterministic latent $z_{\text {clean }}$, which is then forward-noised to time $t=0$. We use a final $\log$-SNR of $\lambda(0)=5$, defining $p\left(z_{0} \mid z_{\text {clean }}\right)=\mathcal{N}\left(\alpha_{0} z_{\text {clean }}, \sigma_{0}\right)$. For a variance-preserving noise schedule, this corresponds to $\alpha_{0}=\sqrt{\operatorname{sigmoid}(+5)} \approx 1.0$ and $\sigma_{0}= \sqrt{\operatorname{sigmoid}(-5)} \approx 0.08$. The KL term for the VAE loss is then:
$$
\begin{align*}
\mathrm{KL}\left[p\left(\boldsymbol{z}_{0} \mid \boldsymbol{x}\right) \mid p_{\theta}\left(\boldsymbol{z}_{0}\right)\right] & \leq \operatorname{KL}\left[p\left(\boldsymbol{z}_{0}, \ldots, \boldsymbol{z}_{1} \mid \boldsymbol{x}\right) \mid p_{\theta}\left(\boldsymbol{z}_{0}, \ldots, \boldsymbol{z}_{1}\right)\right] \\
& =\mathbb{E}_{t}\left[-\frac{\mathrm{d} \lambda_{z}(t)}{\mathrm{d} t} \frac{\exp \lambda_{z}(t)}{2} w\left(\lambda_{z}(t)\right)| | \boldsymbol{z}_{\text {clean }}-\hat{\boldsymbol{z}}\left(\boldsymbol{z}_{t}, \theta\right) \|^{2}\right]+\operatorname{KL}\left[p\left(\boldsymbol{z}_{1} \mid \boldsymbol{x}\right) \mid \mathcal{N}(0, \mathcal{I})\right] . \tag{3}
\end{align*}
$$

Thus, the latent $z_{0}$ is sampled using a learned mean and fixed diagonal noise.

\subsection*{3.2. Decoding: A diffusion decoder}

The decoder is also a diffusion model, but operating in the image space with $\boldsymbol{x}_{t}=\alpha_{t} \boldsymbol{x}+\sigma_{t} \boldsymbol{\epsilon}$. The reconstruction loss can be written as:
$$
\begin{align*}
-\log p_{\theta}\left(\boldsymbol{x} \mid \boldsymbol{z}_{0}\right) & \leq \operatorname{KL}\left[p\left(\boldsymbol{x}_{0}, \ldots, \boldsymbol{x}_{1} \mid \boldsymbol{x}\right) \mid p_{\theta}\left(\boldsymbol{x}_{0}, \ldots, \boldsymbol{x}_{1} \mid \boldsymbol{z}_{0}\right)\right] \\
& =\mathbb{E}_{t \sim \mathcal{U}(0,1)}\left[\frac{\mathrm{d} \lambda_{x}(t)}{\mathrm{d} t} \frac{\exp \lambda_{x}(t)}{2} w_{x}\left(\lambda_{x}(t)\right)\left\|\boldsymbol{x}-\hat{\boldsymbol{x}}\left(\boldsymbol{x}_{t}, \boldsymbol{z}_{0}, \theta\right)\right\|^{2}\right] \tag{4}
\end{align*}
$$

The key distinction is that the decoder network $D_{\theta}=\hat{\boldsymbol{x}}\left(\boldsymbol{x}_{t}, \boldsymbol{z}_{0}, \theta\right)$ conditions on both the noisy data $\boldsymbol{x}_{t}$ and the latent $\boldsymbol{z}_{0}$. Since the decoder does not affect $\boldsymbol{x}$, the prior term $\operatorname{KL}\left[p\left(\boldsymbol{x}_{1} \mid x\right) \| \mathcal{N}(0, \mathcal{I})\right]$ can be ignored from the loss.

Decoder weighting and loss factor In contrast with the prior, the decoder loss can be a re-weighted ELBO. By discounting low noise levels, high frequency features will always be modelled by the decoder because the cost per bit of information is lower. For example, in many experiments we use the sigmoid loss (Hoogeboom et al., 2024; Kingma \& Gao, 2023), $w(\lambda(t))=\operatorname{sigmoid}(\lambda(t)-b)$.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/f31373f8-0b83-4f59-ac12-b02ffd8edde1-05.jpg?height=460&width=719&top_left_y=319&top_left_x=664}
\captionsetup{labelformat=empty}
\caption{Figure 3 | Decoder weighting on $\boldsymbol{\epsilon}$-mse, $w_{\boldsymbol{\epsilon}}\left(\lambda_{t}\right)=c_{\mathrm{lf}} \cdot \operatorname{sigmoid}\left(b-\lambda_{t}\right)$, showing which noise levels are penalized (via a loss factor $c_{\mathrm{lf}}=1.6$ in this case) and which noise levels are discounted. In theory, for weightings above 1 the latent model is preferred and for weightings below 1 the decoder is preferred. In practise, the decoder will model information even if the weighting is slightly above 1 .}
\end{figure}

Even with equal weighting, literature has shown that it is difficult to use the latent space in VAEs when the decoder is powerful, a phenomenon referred to as posterior collapse (Razavi et al., 2019). For that reason, we up-weigh the decoder loss with a loss factor (which is equivalent to down-weighting the KL-term). See Figure 3 for a combined view of weighting and the loss factor. We find that we only need a small loss factor (1.3 to 1.7 in most experiments).

Thus, in our experiments we set the decoder ELBO weighting using the loss factor ( $c_{\mathrm{lf}}$ ) and sigmoid bias $b$. These 2 hyper-parameters effectively control the amount of information in the latents. A higher information latent naturally leads to better reconstruction quality. However, by using a more informative latent we defer more of the modelling complexity to the base model.

\subsection*{3.3. Base model: Stage 2 training}

In principle we could use the prior diffusion model as described above to generate $\boldsymbol{z}_{0}$ and subsequently sample $p\left(\boldsymbol{x} \mid \boldsymbol{z}_{0}\right)$ using the diffusion decoder. However we find that a prior trained using an ELBO loss does not produce good samples (see App. B). Because the prior can only be trained on an ELBO weighting in stage 1, it places equal weight on low-frequency and high-frequency content in the latent. Therefore, we find that performance can be improved considerably by retraining the prior model as a base model with a sigmoid weighting. Because only a frozen encoder is required during this stage, the base model size and batch size can be much larger than in stage 1.

The training of the base model largely follows the same procedure as existing Latent Diffusion Models (Rombach et al., 2022). The only difference is that Unified Latents have a fixed amount of noise so there is a fixed logsnr max $\lambda(0)$ for the base model which is the same as the one used in the prior. This deviates from the standard approach where the final logsnr is a hyper-parameter and we use the final prediction $\hat{\boldsymbol{z}}$ instead of $\boldsymbol{z}_{0}$ as the sampled latent.

There are alternative design choices that allow for single stage training. In this case the prior model will already achieve better performance. This requires different weightings of decoder and prior losses, and are discussed in Appendix B.

\section*{4. Related Work}

Our work combines diffusion-based decoding with diffusion-based priors to learn latent representations optimized for generation. We review the most relevant prior work below.

Diffusion Decoders Several works have explored using diffusion models as decoders in VAE-like frameworks. DiffuseVAE (Pandey et al., 2022) trains a conventional MSE autoencoder first, then finetunes a diffusion decoder using the original decoder's output as conditioning. SWYCC (Birodkar et al., 2024) and $\epsilon$-VAE (Zhao et al., 2025) train latents with a diffusion decoder, but still rely on a channel bottleneck for regularization rather than a learned prior. DiVAE (Shi et al., 2022) combines a diffusion decoder with discrete VQ-VAE tokens. In contrast, our approach uses continuous latents regularized by a diffusion prior, providing interpretable control over the bitrate.

Diffusion Priors LSGM (Vahdat et al., 2021) jointly trains a diffusion prior in a VAE framework, but requires a separate encoder entropy term $\mathbb{E}_{q\left(\boldsymbol{z}_{0} \mid \boldsymbol{x}\right)} \log q\left(\boldsymbol{z}_{0} \mid \boldsymbol{x}\right)$ that introduces training instability. Our approach sidesteps this by using a deterministic encoder with fixed noise, absorbing the encoder distribution into the diffusion forward process. This yields a simpler two-term objective (decoder loss + prior loss) while maintaining a tight bound on latent information.

Diffusion Decoder and Prior DiffAE (Preechakul et al., 2022) uses diffusion for both encoding and decoding, but its latent comes from a pre-trained "semantically meaningful" encoder rather than being optimized for generation quality. Our work differs by jointly training the encoder, prior, and decoder, with the explicit goal of maximizing generation efficiency.

Latent Diffusion and Efficient Autoencoders The original Latent Diffusion Model (Rombach et al., 2022) uses a GAN-trained autoencoder with channel-bottlenecked latents and a small KL penalty, but provides no principled way to control latent information. Recent work on efficient autoencoders (Chen et al., 2024) achieves high compression ratios but does not address the interplay between autoencoder design and downstream diffusion modeling. Token-based approaches like TiTok (Yu et al., 2024) compress images to discrete tokens, trading reconstruction quality for faster sampling. Lastly, pretrained semi-supervised encoders like DINO (Caron et al., 2021) can be used to focus on semantically meaningful representations (Shi et al., 2025; Zheng et al., 2025) and obtain impressive generation quality metrics. A downside of these approaches is that PSNR scores are low $(\leq 20)$ causing reconstructions to appear different from their original in particular on high-frequency details.

Latents from Self-Supervised Representation A number of recent works have replaced the AutoEncoder all-together and model a semi-supervised representation like SigLip or Dino instead (Shi et al., 2025; Zheng et al., 2025).

\section*{5. Experiments}

We evaluate Unified Latents on their ability to improve pre-training efficiency-the relationship between training compute and generation quality. We conduct experiments on ImageNet-512 and Kinetics-600 for direct comparison with prior work, and include scaling studies on large-scale text-to-image and text-to-video datasets. We focus on pre-training efficiency and avoid fine-tuning stages (such as
aesthetics fine-tuning) or MS-COCO evaluations, as these introduce confounding factors unrelated to the quality of the learned latents.

\subsection*{5.1. Model Architecture}

Our encoder and decoder models use $2 \times 2$ patching to save compute. The encoder is a Resnet model with [128, 256, 512, 512] channels and 2 residual blocks for downsampling stage and 3 blocks in the final stage.

The prior model is a single level ViT with 8 blocks and 1024 channels. In the base model we use a 2 stage ViT with [512, 1024] channels [6, 16] blocks. The base model is regularized with a dropout rate of 0.1 in both stages.

The decoder is a UVit model (Hoogeboom et al., 2024) with channel counts [128, 256,512] in the convolutional down-sampling and up-sampling stages. The transformer in the middle has 8 blocks and 1024 channels. We use a dropout rate of 0.1 for regularization.

\subsection*{5.2. Evaluation Metrics}

To assess the quality of samples and autoencoder reconstructions we use FID and FVD for images and videos, respectively. When sampling from a base model we denote the FID as gFID. For reconstruction we use the term rFID and use the same samples from the dataset to compute reconstructions and the FID references. This breaks the convention of standard FID where the reference statistics are computed on the entire train (or sometimes eval) dataset. The same approach and the rFID and gFID convention is used by the majority of existing literature.

We also use PSNR (Peak Signal-to-Noise Ratio) to measure how closely reconstructions match their originals. This complements FID, since a reconstruction can be in-distribution (low FID) while still differing substantially from the original image. Additionally, because our models provide an upper bound on latent information, we report the estimated bits per dimension (bpd) in the latent space.

For computational cost, we count FLOPs for all linear projections and attention operations. For training cost, we multiply by 3 to approximate the cost of computing gradients.

\subsection*{5.3. Image Generation}

In this section we test Image Generation performance. The autoencoder operates on a resolution of $512 \times 512$ and downsamples $16 \times 16$ to produce $32 \times 32$ latents. For each experiment, the optimal latent bitrate is chosen so that gFID is highest (for details see Sec. 5.4).

ImageNet First we show scaling performance of Unified Latents in training flops vs generation FID (Figure 4) on ImageNet512 ${ }^{1}$. There are several important things to note. Firstly, UL outperforms other approaches in literature in a training cost vs generation performance trade-off, which means it is the most efficient pre-training approach on this dataset. Secondly, for a fair comparison we train the exact architecture (a 2-level ViT) on Stable Diffusion latents (baselines small SD and medium SD). Here we see that UL is outperforming the baselines to a greater extend. We find that patching is detrimental to the performance of the base model. The UNet (SD) baseline is a small model that uses an additional convolution stack instead of patching the SD latents.

\footnotetext{
${ }^{1}$ The original time of writing was March 2025, in the meantime other (often complimentary) approaches may have discovered.
}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/f31373f8-0b83-4f59-ac12-b02ffd8edde1-08.jpg?height=805&width=1557&top_left_y=294&top_left_x=255}
\captionsetup{labelformat=empty}
\caption{Figure 4 | FID vs. training cost on ImageNet-512. UL outperforms all other approaches on base training compute versus generation equality We assume that one training iteration is three times as expensive as evaluating the model (i.e., forward pass, backprop to inputs, backprop to weights). Note that auto-encoder training cost is not included.}
\end{figure}

AutoEncoder transfer Previous work like Stable Diffusion uses a auto-encoder that is trained on another dataset than ImageNet. To test the effect of using an out-of-distribution autoencoder we also train a base model on Unified Latents trained on an internal text-to-image dataset (tti AE). We did not observe a significant difference in training efficiency. In-distribution autoencoders seem slightly better when training small base models with a low information latent.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/f31373f8-0b83-4f59-ac12-b02ffd8edde1-08.jpg?height=396&width=1559&top_left_y=1708&top_left_x=228}
\captionsetup{labelformat=empty}
\caption{Figure 5 | A selection of samples from a text-to-image trained with Unified Latents}
\end{figure}

Text-To-Image In order to test our methods at scale we trained multiple AutoEncoders on internal Text-To-Image datasets sweeping over loss factor (1.25-1.7). For each AutoEncoder we train base models add various sizes ( 100,300 , and 970 GFlops). To evaluate these models we take 30 k samples without guidance and compute clip and FID scores against the training set.

Figure 5 shows some hand-picked samples from one of the large models. See Figure 10 for additional and non cherry picked samples.

Figure. 6 shows how AutoEncoders with a low latent bitrate lead to better image quality as measure

\begin{table}
\begin{tabular}{lcc}
\hline latents & gFID@30K & clip \\
\hline UL (LF=1.5) & 4.1 & 27.1 \\
Pixel (no latents) & 5.0 & 27.0 \\
StableDiffusion & 6.8 & 27.0 \\
\hline
\end{tabular}
\captionsetup{labelformat=empty}
\caption{Table 1 | Generation quality and text alignment for text-to-image models trained with Unified Latents, pixel diffusion (no latents), and StableDiffusion latents.}
\end{table}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/f31373f8-0b83-4f59-ac12-b02ffd8edde1-09.jpg?height=425&width=1315&top_left_y=699&top_left_x=374}
\captionsetup{labelformat=empty}
\caption{Figure 6 | Image generation quality (left) and text alignment (right) against AutoEncoder Loss Factor for various base model sizes.}
\end{figure}
by $g$ FID. This effect is more pronounced for smaller models. The text alignment (clip) on the other hand suffers slightly from very low loss factors even for smaller models. Indicating that perhaps the decoder would also benefit from text conditioning. However, we also note that text-alignment can be easily improved by applying guidance.

In Table 1 we compare the text-to-image models trained with Unified Latents to models trained with pixel diffusion (Hoogeboom et al., 2024), and Stable Diffusion latents (Rombach et al., 2022). We add additional convolution blocks to deal with the higher resolution but do not compensate the UL model for the additional flops used by the other models. The UL el significantly outperform these baselines on perceptual quality and has a slightly better text-alignment.

\subsection*{5.4. Latent bitrate tuning}

Recall that there is a reconstruction FID vs generation FID trade-off. The goal of the combined auto-encoder and base model stack is to achieve the highest gFID possible. On the other hand, it is trivial to obtain a very good rFID by allowing more and more bits to flow through the latents. This is a problem, because high bitrate latents will be more difficult to model. One way to control the amount of bits in the latent is by changing the loss factor (see Table 2 and Fig. 7). Note that for smaller models, typically lower bitrates are optimal: even though rFID (and thus decoding) is somewhat worse, the smaller capacity models can only fit low bitrate latents properly. On the contrary, larger models are less sensitive to latent bitrates, and can achieve even better performance on higher bitrates.

An alternative method to tune latent bitrates is via the bias in the sigmoid loss, which is entangled with the loss factor. For lower biases, one typically requires higher loss factors. In Figure 8 we show a sweep over decoder bias and loss factors, which demonstrates that several settings give roughly equal performance / latent bitrate curves.

Table 2 | Increasing the loss factor leads to improved reconstruction metrics (rFID, PSNR) at the cost of increased bitrate in the latent encoding. For small models, the loss factor (and bits in the latent) matter a lot. For larger base models the loss factor is less sensitive.

\begin{tabular}{|l|l|l|l|l|l|}
\hline LF & bits/pixel & rFID@50k & PSNR & gFID (small) & gFID (medium) \\
\hline 1.3 & 0.035 & 0.79 & 25.7 & 1.42 & 1.37 \\
\hline 1.5 & 0.059 & 0.47 & 27.6 & 1.54 & 1.31 \\
\hline 1.7 & 0.083 & 0.36 & 28.9 & 1.77 & 1.38 \\
\hline 1.9 & 0.101 & 0.31 & 29.6 & 2.02 & 1.45 \\
\hline 2.1 & 0.116 & 0.27 & 30.1 & 2.38 & 1.58 \\
\hline
\end{tabular}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/f31373f8-0b83-4f59-ac12-b02ffd8edde1-10.jpg?height=303&width=1579&top_left_y=815&top_left_x=242}
\captionsetup{labelformat=empty}
\caption{Figure 7 | Reconstruction quality vs loss factor. Fine details like small text are lost for low bitrate latents.}
\end{figure}

\subsection*{5.5. Latent shape}

For Latent Diffusion Models the downsampling factor and latent channels are the main factors determining the information bottleneck (Rombach et al., 2022). In this experiment we use a fixed spatial downsampling to $32 \times 32$, and vary the number of latent channels (from 4 to 64). The results are in Table 3. From the results we conclude that Unified Latents are mostly insensitive to the number of latent channels. Only for a very low latent channel count the encoder is unable to pass enough information to enable good reconstructions (4 and 8).

In the next experiment, we vary the spatial downsampling ( $8 x$ to $32 x$ ), while using a fixed number of latent channels (32). The results are in Table 4. First, we observe that 32 channels work well for any of the spatial dimensions of the latents. Second, we see that the rFID results are similar for $16 x$ and $8 x$ spatial downsampling (to $32 \times 32$ and to $64 \times 64$ ), while it seems that the former is easier to model for the decoder, resulting in lower gFID numbers.

\subsection*{5.6. L2 Regularization}

It can be cumbersome to train 2 diffusion models simultaneously. Here we find that for a slight decrease in performance it is also possible to first train the encoder using a diffusion prior while placing 12 regularization on the decoder and different loss weightings (see Table 5).

This experiment may raise another question: How about a simpler version where the latents are regularized by an 12 loss / normal prior, as is typical for VAEs? We find that training with a VAE with a normal prior requires higher bitrate latents to reach good reconstruction quality. This then results in more difficult to learn latents and worse gFID.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/f31373f8-0b83-4f59-ac12-b02ffd8edde1-11.jpg?height=385&width=1623&top_left_y=294&top_left_x=221}
\captionsetup{labelformat=empty}
\caption{Figure 8 | Image quality for various latent bitrates (FID vs bits/pixel) for a small model variant. Left: generation (gFID). Middle: reconstruction (rFID). Right: reconstruction PSNR. We sweep over sigmoid shift and loss factor. For each shift, we sweep over loss factors [1.5, 1.75, 2., 3., 4.].}
\end{figure}

\begin{table}
\begin{tabular}{lcc}
\hline \# chan & rFID & gFID@50K \\
\hline 4 & 7.19 & - \\
8 & 1.53 & - \\
16 & 0.54 & 1.76 \\
32 & 0.42 & 1.60 \\
64 & 0.48 & 1.77 \\
\hline
\end{tabular}
\captionsetup{labelformat=empty}
\caption{Table 3 | FID metrics on ImageNet512 are insensitive to latent channel count. The AE is unable to obtain a good reconstruction quality (rFID) with to few channels ( $\leq 8$ ).}
\end{table}

\subsection*{5.7. Video Generation}

Kinetics600 In this experiment we show that our method outperforms literature on k600 on a training cost vs FVD tradeoff. Here we use $4 \times 8 \times 8$ downsampling for 16 frames of $128 \times 128$ kinetics videos. Following Video Diffusion, we condition on 5 frames and generate 11 frames. For MAGVIT and W.A.L.T. due to tokenization choices the models operate on 17 frames, a temporal latent dimensions of 5, and FVD is measured on 5-12 generations. To make comparison more fair, we discard the extra token of processing in the FLOP computation of these baselines.

Here again, UL outperforms other approaches on training cost vs FVD performance (see Figure 9). Note that the small model already achieves 1.7 FVD, whereas the medium model achieves 1.3 FVD which is currently SOTA.

\subsection*{5.8. Ablations}

In this section we aim to ablate our approach by removing the key innovations. We also consider the classic VAE setup where the encoder is allowed to predict a mean and variance. The results are listed in Table 6.

Firstly (A), we want to make sure the prior improves and regularizes the latents. To test this we added a stop-gradient to the prior input so we still get a bitrate estimate but the encoder no longer receives a gradient with respect to the prior. Instead, we regularize the latent with a strongly discounted KL to $\mathcal{N}(0, I)$ like prior works (Rombach et al., 2022). To get a reasonable bitrate and gFID we must reduce the latent channels. The best result reported here uses 8 latent channels vs. 32 in the baseline.

Secondly (B), we ablate the noisy latents by using $\lambda_{z}(0)=10$ which corresponds to adding a very small amount of noise ( $\sigma \approx 0.007$ ). At this precision the prior fails to accurately model the bitrate

\begin{table}
\begin{tabular}{ccc}
\hline latent shape $(h \times w \times c)$ & rFID@50K, & gFID@50K \\
\hline $64 \times 64 \times 32$ & 0.40 & 2.12 \\
$32 \times 32 \times 32$ & 0.41 & 1.63 \\
$16 \times 16 \times 32$ & 1.41 & 1.74 \\
\hline
\end{tabular}
\captionsetup{labelformat=empty}
\caption{Table 4 | FID metrics on ImageNet512 for AutoEncoders with spatial downsampling factors between 8 x and 32x.}
\end{table}

\begin{table}
\begin{tabular}{|l|l|l|l|l|l|}
\hline Prior & Reconstruction loss & Latent bpd with prior & Latent bpd with base model & rFID@50K & gFID@50K \\
\hline Diffusion & Diffusion & 0.079 & 0.079 & 0.86 & 1.4 \\
\hline Diffusion & MSE & 0.072 & 0.072 & 1.1 & 2.4 \\
\hline Normal & Diffusion & 0.39 & 0.26 & 0.83 & 2.5 \\
\hline
\end{tabular}
\captionsetup{labelformat=empty}
\caption{Table 5 | Ablations on the auto-encoder training.}
\end{table}
of the latent and the loss is reduced by simply modelling most information in the decoder. The reconstructions (rFID) are too low quality to train a useful base model.

Thirdly (C), we test what happens if we train on a text-to-image dataset rather than ImageNet. rFID is strongly affected while generation still works well. Other work that trains autoencoders directly on ImageNet data has also reported very low rFID scores (Chen et al., 2024). We hypothesize that this is mostly caused by minor differences in high-frequency statistics that FID seems overly sensitive to compared to human perception.

Lastly (D), we consider a more traditional VAE setup with an encoder that predicts a mean and variance. Prior work (Vahdat et al., 2021) shows that the KL term can be generalized to arbitrary distribution. The generalization adds two entropy terms subtracting the encoder entropy from the entropy. For an encoder distribution $p\left(\boldsymbol{z}_{\text {clean }} \mid x\right)=\mathcal{N}\left(\mu_{z}, \operatorname{diag}\left(\sigma_{z}^{2}\right)\right)$ the extra terms reduce to
$$
\begin{equation*}
\mathcal{L}_{e}=-\frac{1}{2} \log \left[\sigma_{z}^{2} e^{\lambda_{z}(0)}+1\right] \tag{5}
\end{equation*}
$$

For the noisy latent setting $\lambda_{z}(0)=5$ we find that the learned noise quickly drops to 0 and the model becomes unstable. For high precision latents $\lambda_{z}(0)=10$ the encoder does learn to inject additional noise into the latent. The estimate of the KL term is high variance as reported before (Vahdat et al., 2021). The gFID is worse than the baseline. Thus, we conclude that the fixed encoder variance is a useful simplification that increases both stability and performance.

\section*{6. Discussion}

Larger base models benefit from more informative latents. A natural direction for future work is to establish scaling laws for Unified Latents that predict the optimal bitrate given a training budget. Such scaling laws would depend on implementation details including dataset, evaluation metrics, and model architecture, and would be best studied in the context of production-scale foundation models.

While this work focuses primarily on images with some extension to video, the Unified Latent framework appears broadly applicable. With a discrete (diffusion) decoder discrete data like text could in theory be compressed with latents as well.

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 6 | Ablations study on Unified Latents components.}
\begin{tabular}{lccc}
\hline & bits/pixel & rFID@50k & gFID@50k \\
\hline UL baseline (LF=1.5) & 0.059 & $\mathbf{0 . 4 7}$ & $\mathbf{1 . 5 4}$ \\
A. prior model & 0.121 & 1.81 & 7.80 \\
B. noisy latents & 0.008 & 28.27 & - \\
C. ImageNet data & 0.034 & 1.37 & 1.63 \\
D. learned variance & 0.060 & 0.69 & 1.81 \\
\hline
\end{tabular}
\end{table}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/f31373f8-0b83-4f59-ac12-b02ffd8edde1-13.jpg?height=734&width=1394&top_left_y=749&top_left_x=340}
\captionsetup{labelformat=empty}
\caption{Figure 9 | FVD vs. training cost on Kinetics-600. Plotted until convergence.}
\end{figure}

\subsection*{6.1. Limitations}

Existing literature and this work show a trend towards less-informative latents (measured by bitrate or reconstruction PSNR) being easier to model. To what extend are weaker latents moving part of the modelling problem toward the decoder? This work uses U-Net diffusion models, while most prior work uses a GAN based decoder with a discriminator loss but without a noise input (Rombach et al., 2022). A diffusion decoder is strictly more powerful than such GANs because it predicts a distribution rather than a single image. However, the mode-collapsing nature of GAN training might help this class of models producing better looking images with better rFID scores.

Comparison between latent diffusion models is further complicated by differences in AutoEncoder training data. The original Stable Diffusion autoencoder was trained on a large-scale web dataset (Rombach et al., 2022), whereas most of our experiments use only ImageNet. Semi-supervised approaches (Caron et al., 2021; Shi et al., 2025; Zheng et al., 2025) introduce encoders trained on large external datasets, making direct comparison even more challenging.

Finally, diffusion decoders are an order of magnitude more expensive to sample from than GAN based decoders. Without an additional distillation step for the decoder, the computational cost of using Unified Latents is significantly higher than a standard LDM.

\subsection*{6.2. Conclusion}

In summary, we have demonstrated that latent representations can be effectively learned by jointly training an encoder, diffusion prior, and diffusion decoder. This approach outperforms existing methods in both training efficiency and generation quality. Unified Latents provide stable, interpretable control over latent information through simple hyper-parameters, making the reconstruction-modeling trade-off explicit. We believe this principled approach to latent design will prove valuable as latent diffusion models continue to scale.

\section*{References}

Birodkar, V., Barcik, G., Lyon, J., Ioffe, S., Minnen, D., and Dillon, J. V. Sample what you cant compress. Technical report, arXiv, 2024. URL https://arxiv.org/abs/2409. 02529.

Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., and Joulin, A. Emerging properties in self-supervised vision transformers. In 2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021, pp. 9630-9640. IEEE, 2021.

Chen, J., Cai, H., Chen, J., Xie, E., Yang, S., Tang, H., Li, M., Lu, Y., and Han, S. Deep compression autoencoder for efficient high-resolution diffusion models. CoRR, abs/2410.10733, 2024.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS, 2020.

Ho, J., Saharia, C., Chan, W., Fleet, D. J., Norouzi, M., and Salimans, T. Cascaded diffusion models for high fidelity image generation. J. Mach. Learn. Res., 23:47:1-47:33, 2022.

Hoogeboom, E., Mensink, T., Heek, J., Lamerigts, K., Gao, R., and Salimans, T. Simpler diffusion (sid2): 1.5 FID on imagenet512 with pixel-space diffusion. $C o R R$, abs/2410.19324, 2024.

Kingma, D. P. and Gao, R. Understanding the diffusion objective as a weighted integral of elbos. CoRR, abs/2303.00848, 2023.

Kingma, D. P., Salimans, T., Poole, B., and Ho, J. Variational diffusion models. CoRR, abs/2107.00630, 2021.

Pandey, K., Mukherjee, A., Rai, P., and Kumar, A. Diffusevae: Efficient, controllable and highfidelity generation from low-dimensional latents. Technical report, arXiv, 2022. URL https: //arxiv.org/abs/2201.00308.

Preechakul, K., Chatthee, N., Wizadwongsa, S., and Suwajanakorn, S. Diffusion autoencoders: Toward a meaningful and decodable representation. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

Razavi, A., van den Oord, A., Poole, B., and Vinyals, O. Preventing posterior collapse with delta-vaes. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pp. 10674-10685. IEEE, 2022.

Shi, J., Wu, C., Liang, J., Liu, X., and Duan, N. Divae: Photorealistic images synthesis with denoising diffusion decoder, 2022. URL https://arxiv.org/abs/2206.00386.

Shi, M., Wang, H., Zheng, W., Yuan, Z., Wu, X., Wang, X., Wan, P., Zhou, J., and Lu, J. Latent diffusion model without variational autoencoder, 2025. URL https://arxiv.org/abs/2510. 15301.

Vahdat, A., Kreis, K., and Kautz, J. Score-based generative modeling in latent space. In Ranzato, M., Beygelzimer, A., Dauphin, Y. N., Liang, P., and Vaughan, J. W. (eds.), Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS, pp. 11287-11302, 2021.

Yu, Q., Weber, M., Deng, X., Shen, X., Cremers, D., and Chen, L. An image is worth 32 tokens for reconstruction and generation. CoRR, abs/2406.07550, 2024.

Zhao, L., Woo, S., Wan, Z., Li, Y., Zhang, H., Gong, B., Adam, H., Jia, X., and Liu, T. Epsilon-vae: Denoising as visual decoding, 2025. URL https://arxiv.org/abs/2410.04081.

Zheng, B., Ma, N., Tong, S., and Xie, S. Diffusion transformers with representation autoencoders. CoRR, abs/2510.11690, 2025.

\section*{A. Additional samples}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/f31373f8-0b83-4f59-ac12-b02ffd8edde1-17.jpg?height=1081&width=1486&top_left_y=402&top_left_x=264}
\captionsetup{labelformat=empty}
\caption{Figure 10 | Generations from the text-to-image model. Guidance is set to 2 . Images are not cherrypicked. Used prompts: (1) A couple gets caught in the rain, oil on canvas, (2) A lone traveller walks in a misty forest, (3) A walking figure made out of water, (4) In the swamp, a crocodile stealthily surfaces, revealing only its eyes and the tip of its nose as it moves forward, (5) A fox dressed in suit dancing in park, (6) Pouring chocolate sauce over vanilla ice cream in a cone, studio lighting, (7) An astronaut riding a horse, (8) Aurora Borealis Green Loop Winter Mountain Ridges Northern Lights, (9) Sailboat sailing on a sunny day in a mountain lake, (10) A dog driving a car on a suburban street wearing funny sunglasses.}
\end{figure}

\section*{B. End-to-end latent training}

In addition to the 2 -stage training approach described in this paper, we also tried training the encoder, decoder, and base diffusion model end-to-end in a single stage. This can be done in two ways. In our first attempt we shifted the loss of the decoder diffusion model towards more noisy data, following Hoogeboom et al. (2024), combined with the standard ELBO loss on the base model. Both models can then be trained jointly in a stable way, but we did not get FID below 2 using this approach. In a second attempt we trained the base model with a weighted ELBO loss that is equivalent to training this model with unweighted ELBO loss on data with additional added noise (Kingma \& Gao, 2023). This means it is possible to train the decoder and base model jointly, using differently weighted ELBO losses on the base model and decoder, by randomizing the maximum log signal-to-noise ratio of the base model according to a particular truncated logistic distribution. The decoder diffusion model is then modified to condition on the log-SNR of the latents, similar to the conditioning augmentation of Ho et al. (2022). Using the exact settings used in the 2 -stage approach, but training in a single stage, we achieved an FID of about 4 in 400k training steps.