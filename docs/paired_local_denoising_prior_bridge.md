# Paired Local Denoising, Latent Prior Spacetime, and a Lean CSP Wrapper

## Scope

This note is the canonical paired-regime statement for the MMSFM latent prior
plus downstream CSP bridge setup.

The clean claim is:

$$
\boxed{\text{In the paired regime, each local bridge interval is already an exact denoising problem.}}
$$

The diffusion prior helps in two distinct ways:

1. it equips latent space with its own Gaussian denoising spacetime;
2. it regularizes the decoder chart so bridge-noisy latents are more stable and recoverable.

This note does **not** treat the prior as the source of bridge correctness. In
the paired regime, bridge correctness already comes from the exact local bridge
law and its regression characterization.

## 1. Latent Prior Spacetime

Let the encoder produce a clean latent
$$
z_{\mathrm{cl}} = E_\phi(u) \in \mathbb{R}^{d_z}.
$$

The latent prior corrupts it with a Gaussian channel
$$
Z_s^{\mathrm{pr}} = \alpha_p(s)\, z_{\mathrm{cl}} + \sigma_p(s)\,\varepsilon,
\qquad
\varepsilon \sim \mathcal{N}(0, I),
$$
and trains a state-prediction denoiser
$$
\hat z_\omega(Z_s^{\mathrm{pr}}, s) \approx \mathbb{E}[z_{\mathrm{cl}} \mid Z_s^{\mathrm{pr}}, s].
$$

The prior loss is therefore a latent denoising loss, with log-SNR coordinate
$$
\lambda_p(s) = \log \frac{\alpha_p(s)^2}{\sigma_p(s)^2}.
$$

By the same Gaussian-corruption algebra used in the diffusion-spacetime note,
the denoising posterior forms an exponential family with natural parameter
$$
\eta_p(z_s, s) =
\left(
\frac{\alpha_p(s)}{\sigma_p(s)^2} z_s,
-\frac{\alpha_p(s)^2}{2 \sigma_p(s)^2}
\right),
$$
so the prior equips latent space with a Fisher-Rao denoising spacetime indexed
by ``(noisy latent, noise level)``.

## 2. Exact Paired Local Bridge Law

In the paired manuscript, once the left endpoint and retained context are fixed,
each interval reduces to a Dirac-start conditional bridge problem.

For interval ``[t_{i-1}, t_i]``, with left endpoint ``x_{i-1}`` and retained
context ``c``, the right-endpoint law is
$$
\nu_i^c(dy) = \mathrm{Law}(X_i \mid C_{i-1} = c),
$$
and the local reciprocal completion is
$$
Q_i^{c,\mathrm{loc}}(d\omega)
=
\int \nu_i^c(dy)\,
R(d\omega \mid X_{t_{i-1}} = x_{i-1}, X_{t_i} = y).
$$

Its exact controller is
$$
u_i^\star(x, c, t)
=
\mathbb{E}\!\left[\Gamma_i(t, x, X_i)\mid X_t = x, C_{i-1} = c\right],
$$
where
$$
\Gamma_i(t, x, y) = a(t)\,\nabla_x \log r(t, x; t_i, y).
$$

So even before choosing a Gaussian reference, the paired local bridge is already
an endpoint denoiser: from a noisy interior state ``X_t`` and the retained
context, it estimates the next anchor ``X_i``.

In the exact Gaussian hierarchy, the sequential state collapses to the previous
scale. The practical paired-regime condition is therefore just the previous
latent ``X_{i-1}``, not a longer history.

## 3. VE/Brownian Bridge-Local Exponential Family

Choose the latent reference on each interval to be VE/Brownian with variance
clock ``V``:
$$
dX_t = \sqrt{\dot V(t)}\, dW_t,
\qquad
\dot V(t) > 0.
$$

On interval ``[t_{i-1}, t_i]``, write
$$
\Delta V_i = V_i - V_{i-1},
\qquad
\theta_i(t) = \frac{V(t) - V_{i-1}}{\Delta V_i}.
$$

Then, conditional on the endpoints,
$$
X_t
=
(1-\theta_i(t)) X_{i-1}
+ \theta_i(t) X_i
+ \sqrt{\theta_i(t)(1-\theta_i(t))\Delta V_i}\,\varepsilon.
$$

After removing the known left-endpoint contribution, the noisy bridge state is a
Gaussian corruption of the next anchor ``X_i``. The bridge-local posterior is
therefore an exponential family with bridge natural parameter
$$
\eta_i^{\mathrm{br}}(\tilde x, t)
=
\left(
\frac{\theta_i(t)}{\theta_i(t)(1-\theta_i(t))\Delta V_i}\,\tilde x,
-\frac{\theta_i(t)^2}{2\,\theta_i(t)(1-\theta_i(t))\Delta V_i}
\right).
$$

The associated bridge log-SNR is
$$
\lambda_i^{\mathrm{br}}(t)
=
\log\frac{\theta_i(t)}{(1-\theta_i(t))\Delta V_i}.
$$

This is the bridge-side noise coordinate that should be aligned with the prior.
The right matching rule is
$$
\boxed{\text{match bridge and prior in effective log-SNR, not raw time.}}
$$

## 4. Exact State-Prediction Reparameterization

For the VE/Brownian reference, the endpoint-score target simplifies to
$$
u_i^\star(x, c, t)
=
\frac{\dot V(t)}{V_i - V(t)}
\left(
\mathbb{E}[X_i \mid X_t = x, C_{i-1} = c] - x
\right).
$$

With local normalized time ``theta`` on a piecewise-linear interval, this
becomes the familiar paired-regime target
$$
\Gamma_i = \frac{X_i - X_t}{1-\theta}.
$$

For any learned controller ``u_\theta``, define the reparameterized next-anchor
predictor
$$
\widehat X_i = X_t + (1-\theta)\, u_\theta(X_t, X_{i-1}, \lambda_{\mathrm{br}}).
$$

Then the bridge regression loss is exactly equivalent to weighted next-anchor
state prediction:
$$
\left\|\Gamma_i - u_\theta\right\|^2
=
\frac{\left\|X_i - \widehat X_i\right\|^2}{(1-\theta)^2}.
$$

This is the operational rule for the new CSP wrapper: keep the model output as a
drift, but train it through the exact state-prediction reparameterization.

## 5. What The Prior Does And Does Not Change

At the population level in the paired regime, the prior does **not** change:

- the exact local bridge law;
- the exact controller;
- the paired state-collapse theorem;
- the simulation-free regression principle.

What the prior **does** change is the geometry and conditioning of the latent
chart. The small-noise prior expansion yields a decoder Jacobian penalty of the
form
$$
\frac{\sigma_p(0)^2}{2}\,\mathrm{Tr}\, g(z),
\qquad
g(z) = J(z)^\top J(z),
$$
so the prior discourages large local decoder stretch.

For a bridge perturbation ``\delta_t`` with conditional covariance
``\Sigma_i(t) I``, the decoder expansion gives
$$
\mathbb{E}\!\left[
\|D_X(z + \delta_t) - D_X(z)\|^2
\mid X_{i-1}, X_i
\right]
=
\Sigma_i(t)\,\mathrm{Tr}\, g(z) + O(\Sigma_i(t)^{3/2}).
$$

So the mathematically justified recoverability statement is:

$$
\boxed{
\text{The prior does not alter paired-bridge exactness, but it reduces chart distortion,}
}
$$
$$
\boxed{
\text{which makes bridge-noisy latent states more stable to decode and easier to regress from numerically.}
}
$$

## 6. Implementation-Facing Noise Matching Contract

The maintained paired-prior CSP path specializes the general paired-regime
derivation to a shared-``\Delta V`` Brownian bridge and the repository's
existing prior time parameterization.

### Local Bridge Time And Bridge Log-SNR

The local bridge time used by the paired-prior wrapper is the normalized
interval coordinate ``\theta \in (0, 1)``.

Training samples
$$
\theta \sim \mathrm{Unif}[\theta_{\mathrm{trim}}, 1 - \theta_{\mathrm{trim}}],
\qquad
\theta_{\mathrm{trim}} = 0.05 \text{ by default},
$$
and constructs the interior bridge state as
$$
X_t =
(1-\theta) X_{i-1}
+ \theta X_i
+ \sqrt{\theta(1-\theta)\Delta V}\,\varepsilon.
$$

The implemented bridge feature is therefore exactly
$$
\lambda_{\mathrm{br}}
=
\log \frac{\theta}{(1-\theta)\Delta V}.
$$

In repository code and CLI arguments, the shared interval variance increment is
named ``delta_v``. In the math, it is the shared ``\Delta V``.

### Matched Prior Time In The Repository

The paired-prior trainer does **not** feed a raw prior time into the bridge
model. The model consumes ``\lambda_{\mathrm{br}}`` directly through the
existing sinusoidal scalar embedding.

The prior-time map is computed only for diagnostics and saved metadata.

For the current latent prior implementation,
$$
Z_t^{\mathrm{pr}} = (1-t) z_{\mathrm{cl}} + t\,\varepsilon,
$$
so the prior log-SNR is
$$
\lambda_p(t) = \log \frac{(1-t)^2}{t^2},
\qquad
t = \sigma\!\left(-\tfrac12 \lambda_p\right).
$$

The repository therefore uses the matched prior-time diagnostic
$$
s_{\mathrm{match}}
=
\sigma\!\left(
-\tfrac12\, \mathrm{clip}(\lambda_{\mathrm{br}}, -\lambda_p^{\max}, \lambda_p^{\max})
\right).
$$

This aligns the bridge state with the trained prior noise coordinate, but it is
not an additional model input in the current paired-prior CSP path.

### Inference With One Shared ``\Delta V``

At inference time, the solver advances over an absolute generation-time
interval ``[\tau_{i-1}, \tau_i]``, but the learned bridge controller is still
parameterized by the local interval coordinate
$$
\theta(\tau) = \frac{\tau - \tau_{i-1}}{\tau_i - \tau_{i-1}}.
$$
So the drift wrapper converts the absolute solver time to
``\lambda_{\mathrm{br}}(\theta(\tau); \Delta V)`` on the fly and feeds that
scalar feature to the learned drift network.

So the practical inference contract is:

- the solver evolves over absolute interval time ``\tau``, while the model
  feature is still local ``\theta``;
- the model conditions on the previous latent, the interval identity, and the
  bridge-log-SNR feature;
- one shared ``\Delta V`` controls the noise level on every interval.

Because training is written in local ``\theta`` coordinates, the absolute-time
rollout must apply the chain-rule conversion
$$
\frac{dX}{d\tau}
=
\frac{1}{\tau_i - \tau_{i-1}}\,
\frac{dX}{d\theta}.
$$
Equivalently, if the learned network outputs a local-time controller
``u_\theta(\theta, x, c)``, then the absolute-time solver must use
$$
u_\tau(\tau, x, c)
=
\frac{1}{\tau_i - \tau_{i-1}}\,
u_\theta(\theta(\tau), x, c).
$$
If this factor is omitted, the bridge under-transports along the deterministic
path while the diffusion term remains correctly scaled, which makes the
rollout appear not to reach the fine endpoint even though it can still wander
substantially in latent norm.

If an interval has generation-time endpoints ``[\tau_{i-1}, \tau_i]``, the
diffusion amplitude is scaled so that the total variance increment over that
interval remains the shared ``\Delta V``:
$$
\sigma_i = \sqrt{\frac{\Delta V}{\tau_i - \tau_{i-1}}}.
$$

### What Matching Does And Does Not Choose

Matching prior and bridge noise in log-SNR means matching Gaussian denoising
difficulty. It does **not** by itself determine the bridge noise parameter
``\Delta V``.

With trimmed bridge times
$$
\theta \in [\theta_{\mathrm{trim}}, 1-\theta_{\mathrm{trim}}],
$$
and symmetric prior support
$$
\lambda_p \in [-\lambda_p^{\max}, \lambda_p^{\max}],
$$
support matching only constrains ``\Delta V`` to a feasible interval:
$$
\frac{1-\theta_{\mathrm{trim}}}{\theta_{\mathrm{trim}}}
e^{-\lambda_p^{\max}}
\le \Delta V \le
\frac{\theta_{\mathrm{trim}}}{1-\theta_{\mathrm{trim}}}
e^{\lambda_p^{\max}}.
$$

So the maintained contract is:

- log-SNR matching aligns the prior and bridge in denoising coordinates;
- ``delta_v`` remains an explicit bridge hyperparameter;
- the rollout must rescale the local-``\theta`` drift by
  ``(\tau_i-\tau_{i-1})^{-1}`` when integrating over absolute interval time;
- the current wrapper keeps that choice simple, with default ``delta_v = 1.0``.

The paired-prior trainer then uses the exact state-prediction objective
$$
\frac{\|X_i - (X_t + (1-\theta)u_\theta)\|^2}{(1-\theta)^2},
$$
which is the maintained mathematical contract for
`train_csp_paired_prior_from_fae.py`.
