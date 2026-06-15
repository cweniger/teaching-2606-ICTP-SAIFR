# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cweniger/teaching-2606-ICTP-SAIFR/blob/main/notebooks/s1_pytorch_and_npe.ipynb)
#
# # Session 1A — Train and dissect an MLP, then build your first amortised posterior
#
# **Hands-on session 1A (after Lecture 2). Runs on a laptop CPU. ~60 min.**
#
# In the lectures you *watched* MLPs train. Here you run the training
# loop yourself, then take the trained network apart to see what it
# learned. Two blocks:
#
# 1. **Train and dissect an MLP (~38 min).** The PyTorch training loop
#    from the Lecture 2b slides, run by your own hand on a 1D toy.
#    Then look inside: the learned hidden features (the "learnable
#    basis" from the slides), the effect of the learning rate,
#    overfitting made visible, and spectral bias.
# 2. **Your first amortised posterior (~22 min).** Rebuild the
#    homoscedastic Gaussian band from Lecture 2a,
#    `q_φ(θ|x) = N(μ_θ(x), σ_θ²)`, in PyTorch. Train it on the
#    ball-throw simulator and validate against the *exact* analytic
#    posterior. You end on an open question: the single shared width is
#    visibly wrong near the edge of the prior, which is exactly what
#    Session 1B fixes.
#
# **Session 1B (the GW example)** picks up from there: a
# width that depends on `x` (heteroscedastic), a real
# gravitational-wave simulator, and the normalising flows that finally
# capture the awkward posterior shapes.

# %%
# Install the samma_sbi course package from GitHub.
!pip install -q --upgrade --force-reinstall --no-deps git+https://github.com/cweniger/teaching-2606-ICTP-SAIFR.git  # noqa: E501

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import trange   # progress bars; tqdm.auto picks the right style for Colab

from samma_sbi.simulators import BallThrow

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

# %% [markdown]
# **Notation bridge.** The Lecture 2 slides and the PyTorch code use
# different names for the same things. Keep this table handy:
#
# | On the slides | In the code |
# |---|---|
# | parameters `φ` (all weights/biases) | `model.parameters()` |
# | loss `E(φ)` | `loss` |
# | learning rate `η` | `lr` |
# | mean curve `μ_θ(x)` | the network output `mu` |
# | noise width `σ_θ` | a learned scalar (Block 2) |
# | gradient step `φ ← φ − η ∇E` | `opt.step()` |

# %% [markdown]
# ---
#
# ## Block 1 — Train and dissect an MLP
#
# ### 1.1 — Tensors and autograd
#
# A `torch.Tensor` behaves almost exactly like a numpy array, with one
# extra power: if you set `requires_grad=True`, PyTorch records every
# operation you do to it and can compute derivatives automatically.
# That mechanism (autograd) is the whole reason we use PyTorch instead
# of numpy.

# %%
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.arange(3, dtype=torch.float32)
print("a + b      =", a + b)            # elementwise, like numpy
print("a.shape    =", a.shape, " a.dtype =", a.dtype)

# Autograd in one line: define y(x), call .backward(), read x.grad.
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3 + 2 * x
y.backward()
print("dy/dx at x=2:", x.grad.item(), "  (expected 3*4 + 2 = 14)")

# %% [markdown]
# That `.backward()` call is the *exact* same mechanism that will
# compute the gradient of the loss with respect to every weight in a
# network. Nothing more magic happens at scale; there are just more
# tensors carrying `.grad`.
#
# ### 1.2 — An MLP as an `nn.Module`
#
# On the Lecture 2b slides you saw the math-to-code mirror for a small
# network. Here is that network. An `nn.Module` bundles parameters with
# a `forward` method; everything wrapped in `nn.Linear` is
# auto-registered and shows up in `model.parameters()`.

# %%
class TinyMLP(nn.Module):
    def __init__(self, in_dim=1, hidden=32, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


model = TinyMLP()
print(model)
print("Trainable parameters:", sum(p.numel() for p in model.parameters()))

# %% [markdown]
# ### 1.3 — The training loop
#
# Goal: fit `f(x) = sin(x)` on `x ∈ [-π, π]`. This five-step loop is the
# skeleton of every training run in the school; you saw it on the
# slides, now run it.

# %%
# Data
N = 1024
x_train = torch.linspace(-np.pi, np.pi, N).unsqueeze(1)   # (N, 1)
y_train = torch.sin(x_train)

# Model, loss, optimizer
model = TinyMLP(in_dim=1, hidden=32, out_dim=1)
loss_fn = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
losses = []
for step in range(2000):
    opt.zero_grad()                  # 1. clear gradient buffers
    y_pred = model(x_train)          # 2. forward pass
    loss = loss_fn(y_pred, y_train)  # 3. compute the loss E(φ)
    loss.backward()                  # 4. backward pass: fill every .grad
    opt.step()                       # 5. gradient step: φ ← φ − η ∇E
    losses.append(loss.item())

print(f"Final training loss: {losses[-1]:.2e}")

# %%
fig, (ax_loss, ax_fit) = plt.subplots(1, 2, figsize=(9, 3.2))
ax_loss.semilogy(losses)
ax_loss.set_xlabel("step"); ax_loss.set_ylabel("MSE loss"); ax_loss.set_title("training curve")

with torch.no_grad():
    y_fit = model(x_train).numpy().ravel()
ax_fit.plot(x_train.numpy().ravel(), y_train.numpy().ravel(), "k-", lw=1.5, label="truth")
ax_fit.plot(x_train.numpy().ravel(), y_fit, "C0--", lw=1.5, label="MLP fit")
ax_fit.set_xlabel("x"); ax_fit.set_ylabel("y"); ax_fit.legend(); ax_fit.set_title(r"$y=\sin(x)$ fit")
fig.tight_layout(); plt.show()

# %% [markdown]
# **The five steps, and the one bug.** `zero_grad → forward → loss →
# backward → step`. Each `nn.Parameter` carries a `.grad` buffer;
# `backward()` *adds* into it (it does not overwrite), `zero_grad()`
# clears it, `step()` reads it and updates. Forgetting step 1 silently
# accumulates gradients across iterations and is the most common
# PyTorch bug. You can watch the buffer fill and empty:
#
# (One subtlety: by default `zero_grad()` frees the buffer entirely,
# setting `.grad` back to `None` rather than to a tensor of zeros,
# because that is faster. We pass `set_to_none=False` here just so we
# can read off a literal zero norm.)

# %%
W0 = model.net[0].weight
print("after backward,  .grad norm:", float(W0.grad.norm()))
opt.zero_grad(set_to_none=False)        # zero the buffer in place so we can see it
print("after zero_grad, .grad norm:", float(W0.grad.norm()))

# %% [markdown]
# ### 1.4 — Look inside: the learned hidden features
#
# On the slides, a one-hidden-layer network was described as a
# *learnable basis*: the output is a weighted sum of `H` simple
# functions `g(v_j·x + b_j)`, one per hidden unit, whose shapes the
# network learns. Our 1D toy lets us *see* those basis functions
# directly. Below we plot a handful of the last hidden layer's
# activations as functions of `x`, next to the network output they sum
# to.

# %%
with torch.no_grad():
    xg = torch.linspace(-np.pi, np.pi, 400).unsqueeze(1)
    h = xg
    for layer in model.net[:-1]:     # everything except the final Linear
        h = layer(h)
    feats = h.numpy()                # (400, hidden) last-hidden activations
    y_out = model(xg).numpy().ravel()
xg = xg.numpy().ravel()

fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 3.4))
step = max(1, feats.shape[1] // 8)
for j in range(0, feats.shape[1], step):
    axL.plot(xg, feats[:, j], lw=1)
axL.set_xlabel("x"); axL.set_title(r"a few learned hidden features $g(v_j x + b_j)$")
axR.plot(xg, np.sin(xg), "k-", lw=1.5, label="truth")
axR.plot(xg, y_out, "C0--", lw=1.5, label="MLP output")
axR.set_xlabel("x"); axR.legend(); axR.set_title("output = weighted sum of those features")
fig.tight_layout(); plt.show()

# %% [markdown]
# Each hidden feature is a piecewise-linear ReLU "kink". The network
# learned *where* to put each kink and *how strongly* to weight it, so
# that their sum traces `sin(x)`. That is the entire content of "a
# neural network is a learnable basis": no kink alone looks like a
# sine, but a few dozen of them combine into one.
#
# ### ✏️ EXERCISE 1.A — your turn
#
# Change the target from `sin(x)` to `cos(2x)`, retrain, and plot. You
# should not need to change anything except the target.

# %%
# TODO — your code here.
# Hint: set a new y target, re-instantiate the model and optimizer, run
# the same five-step loop.


# %%
# @title Reference solution { display-mode: "form" }
y_train_2 = torch.cos(2 * x_train)
model_2 = TinyMLP(in_dim=1, hidden=32, out_dim=1)
opt_2 = optim.Adam(model_2.parameters(), lr=1e-3)
for _ in range(2000):
    opt_2.zero_grad()
    loss = nn.functional.mse_loss(model_2(x_train), y_train_2)
    loss.backward()
    opt_2.step()
with torch.no_grad():
    plt.figure(figsize=(5, 3))
    plt.plot(x_train.numpy().ravel(), y_train_2.numpy().ravel(), "k-", label="truth")
    plt.plot(x_train.numpy().ravel(), model_2(x_train).numpy().ravel(),
             "C1--", label="MLP fit")
    plt.legend(); plt.xlabel("x"); plt.ylabel("y"); plt.title(r"$y=\cos(2x)$ fit")
    plt.tight_layout(); plt.show()

# %% [markdown]
# ### A reusable training helper
#
# To experiment quickly we wrap the same five steps into two small
# helpers: `build_mlp` makes a plain `Sequential` MLP (same thing as
# `TinyMLP`, written inline), and `train_regression` runs the loop and
# returns the loss history.

# %%
def build_mlp(hidden=32, n_layers=2, in_dim=1, out_dim=1):
    layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


def train_regression(model, x, y, n_steps=2000, lr=1e-3, progress=False, desc=""):
    opt = optim.Adam(model.parameters(), lr=lr)
    losses = []
    # Show a live progress bar with the running loss only when asked, so the
    # quick cells stay clean and the slow ones give feedback.
    iterator = trange(n_steps, desc=desc, leave=False) if progress else range(n_steps)
    for step in iterator:
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward(); opt.step()
        losses.append(loss.item())
        if progress and step % 50 == 0:
            iterator.set_postfix(loss=f"{losses[-1]:.2e}")
    return losses

# %% [markdown]
# ### ✏️ EXERCISE 1.B — the learning rate is the most important knob
#
# The learning rate `η` (`lr`) sets how big each gradient step is. Too
# small and training crawls; too large and it overshoots and may
# diverge. Train the `sin(x)` fit with `lr ∈ {1e-4, 1e-3, 1e-2, 1e-1}`
# and plot the four loss curves on the same (log-scale) axes. Which one
# converges fastest? Which one blows up or oscillates?
#
# This is the step-size / curvature picture from the Lecture 2b slides,
# now under your control.

# %%
# TODO — your code here.
# Hint: for each lr, build a fresh MLP (seed it first for a fair
# comparison) and call train_regression; collect the loss lists.


# %%
# @title Reference solution { display-mode: "form" }
plt.figure(figsize=(5.5, 3.2))
for lr in [1e-4, 1e-3, 1e-2, 1e-1]:
    torch.manual_seed(0)                 # same init for a fair comparison
    m = build_mlp(hidden=32, n_layers=2)
    curve = train_regression(m, x_train, y_train, n_steps=1500, lr=lr)
    plt.semilogy(curve, label=f"lr = {lr:g}")
plt.xlabel("step"); plt.ylabel("MSE loss"); plt.legend(fontsize=8)
plt.title("effect of the learning rate"); plt.tight_layout(); plt.show()

# %% [markdown]
# Typical reading: `1e-3` is the sweet spot here; `1e-4` is correct but
# slow; `1e-2` is faster but noisier; `1e-1` is too large and the loss
# jumps around or rises. There is no universal best value: it depends
# on the problem and the optimizer. Adam (what we use) is more
# forgiving than plain SGD, which is why it is the default.
#
# ### 1.5 — Overfitting made visible
#
# Lecture 2a drew the U-shaped validation curve; Lecture 2b-extra
# motivated early stopping. Here it is on real code. We fit a *big*
# network to a *small, noisy* training set and watch the validation
# loss (measured on a dense, clean grid) turn upward even as the
# training loss keeps falling.

# %%
torch.manual_seed(0)
n_small = 15
x_small = torch.linspace(-np.pi, np.pi, n_small).unsqueeze(1)
y_small = torch.sin(x_small) + 0.1 * torch.randn_like(x_small)   # noisy labels
x_dense = torch.linspace(-np.pi, np.pi, 400).unsqueeze(1)        # clean validation
y_dense = torch.sin(x_dense)

big = build_mlp(hidden=128, n_layers=3)
opt = optim.Adam(big.parameters(), lr=1e-3)
tr_curve, va_curve, steps = [], [], []
for s in trange(6000, desc="overfitting", leave=False):
    opt.zero_grad()
    l = nn.functional.mse_loss(big(x_small), y_small)
    l.backward(); opt.step()
    if s % 25 == 0:
        with torch.no_grad():
            tr_curve.append(l.item())
            va_curve.append(nn.functional.mse_loss(big(x_dense), y_dense).item())
            steps.append(s)

fig, (axc, axf) = plt.subplots(1, 2, figsize=(10, 3.4))
axc.semilogy(steps, tr_curve, label="train (15 noisy points)")
axc.semilogy(steps, va_curve, label="val (clean grid)")
axc.set_xlabel("step"); axc.set_ylabel("MSE"); axc.legend(fontsize=8)
axc.set_title("training keeps falling; validation turns up")
with torch.no_grad():
    axf.plot(x_dense.numpy().ravel(), y_dense.numpy().ravel(), "k-", lw=1.5, label="truth")
    axf.plot(x_dense.numpy().ravel(), big(x_dense).numpy().ravel(), "C3--", lw=1.5, label="overfit MLP")
axf.plot(x_small.numpy().ravel(), y_small.numpy().ravel(), "ko", ms=4, label="training points")
axf.set_xlabel("x"); axf.legend(fontsize=8); axf.set_title("the fit wiggles through the noise")
fig.tight_layout(); plt.show()

# %% [markdown]
# The network has enough capacity to thread every noisy training point,
# so the training loss goes to near zero, but in doing so it invents
# oscillations that are not in the truth: the validation loss rises.
# *Early stopping* means: keep the model from the step where validation
# was lowest, not the final one. **Try it:** drop `n_small` to 8, or
# raise `hidden` to 256, and watch the gap grow.
#
# ### ✏️ EXERCISE 1.C — break it with frequency (spectral bias)
#
# Now a different failure. Push the target frequency up: fit `cos(k·x)`
# with `k = 4`, then `8`, then `12`. At some point the default
# 32-unit, 2-layer MLP, trained for the same number of steps, visibly
# fails: the fit becomes a smooth under-resolved version of the truth.
#
# This is *spectral bias* (the Lecture 2b-extra slide): networks fit
# low-frequency structure long before high-frequency structure. Once
# you find a `k` that breaks the default network, try to recover it
# with the obvious knobs (more units, more layers, more steps). Does
# scaling fix it, or just postpone the failure to higher `k`?

# %%
# TODO — your code here.
# 1. Crank k up to a value where the default MLP visibly fails.
# 2. Try one or two scaling fixes and re-plot.


# %%
# @title Reference solution { display-mode: "form" }
def fit_cos_kx(k, hidden=32, n_layers=2, n_steps=2000, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    y = torch.cos(k * x_train)
    m = build_mlp(hidden=hidden, n_layers=n_layers)
    train_regression(m, x_train, y, n_steps=n_steps, lr=lr,
                     progress=True, desc=f"cos({k}x), {n_layers}L x {hidden}")
    with torch.no_grad():
        return m(x_train).numpy().ravel()


fig, axes = plt.subplots(1, 3, figsize=(13, 3), sharey=True)
xv = x_train.numpy().ravel()
# (a) default network on a high frequency — breaks
axes[0].plot(xv, np.cos(12 * xv), "k-", lw=1.2, label="truth")
axes[0].plot(xv, fit_cos_kx(k=12), "C3--", lw=1.5, label="default MLP")
axes[0].set_title(r"$\cos(12x)$ — 32 hidden, 2 layers, 2k steps"); axes[0].legend(fontsize=8)
# (b) wider + deeper + longer
axes[1].plot(xv, np.cos(12 * xv), "k-", lw=1.2, label="truth")
axes[1].plot(xv, fit_cos_kx(k=12, hidden=256, n_layers=4, n_steps=1000), "C2--", lw=1.5, label="wider/deeper")
axes[1].set_title("256 hidden, 4 layers, 1k steps"); axes[1].legend(fontsize=8)
# (c) push higher still — spectral bias returns
axes[2].plot(xv, np.cos(20 * xv), "k-", lw=1.2, label="truth")
axes[2].plot(xv, fit_cos_kx(k=20, hidden=256, n_layers=4, n_steps=1000), "C1--", lw=1.5, label=r"$\cos(20x)$")
axes[2].set_title("same big MLP, even higher frequency"); axes[2].legend(fontsize=8)
fig.tight_layout(); plt.show()

# %% [markdown]
# The default MLP under-fits `cos(12x)`; scaling it up recovers that
# target; but push to `cos(20x)` and the bigger network gives up too.
# Scaling does not *remove* spectral bias, it just moves the failure to
# higher frequencies. Genuinely fixing it needs architectural tricks
# (Fourier features) that are beyond today.
#
# **You have now seen an MLP from the inside:** the loop that trains
# it, the learned basis it builds, the learning rate that controls it,
# and the two ways it fails (overfitting, spectral bias). Everything
# from here is putting that same machinery to work.

# %% [markdown]
# ---
#
# ## Block 2 — Your first amortised posterior
#
# We now use an MLP for inference. The setup is exactly Lecture 2a's
# **Gaussian band**: model the conditional density of the parameter
# given the data as a Gaussian whose mean is an MLP and whose width is
# a single shared number,
#
# $$ q_\phi(\theta \mid x) = \mathcal{N}\bigl(\theta;\ \mu_\theta(x),\ \sigma_\theta^2\bigr). $$
#
# We train it on `(θ, x)` pairs from a simulator by minimising the
# Gaussian negative log-likelihood. Because the network maps *any* `x`
# to a posterior in one forward pass, inference is **amortised**: train
# once, evaluate on as many observations as you like. This is the
# simplest possible neural posterior estimator (NPE).
#
# ### 2.1 — Simulator and summary
#
# The ball-throw from Lecture 1b. We restrict the prior to `(0.05, π/4)`
# so the range map `θ → r(θ)` is one-to-one (no bimodality yet). The
# observation is the **mean** of `n_balls = 10` landings, the summary
# from Demo 2 of Lecture 1b.

# %%
N_BALLS = 10
sim = BallThrow(prior_low=0.05, prior_high=np.pi / 4)

rng = np.random.default_rng(SEED)
theta_demo = sim.sample_prior(5, rng=rng)
x_demo = sim.simulate_summary(theta_demo, n_balls=N_BALLS, rng=rng)
for t, xi in zip(theta_demo, x_demo):
    print(f"theta = {t:.3f} rad   ->   x_summary = {xi:.3f} m")

# %% [markdown]
# ### 2.2 — Training set
#
# We draw `N_TRAIN = 4000` pairs (cheap, trains in seconds on a laptop)
# and hold out a 10% validation set so we can monitor honest loss.

# %%
def simulate_dataset(sim, n_pairs, n_balls, rng):
    theta = sim.sample_prior(n_pairs, rng=rng)
    x = sim.simulate_summary(theta, n_balls=n_balls, rng=rng)
    theta_t = torch.tensor(theta, dtype=torch.float32).unsqueeze(1)  # (N, 1)
    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(1)          # (N, 1)
    return theta_t, x_t


N_TRAIN, N_VAL = 4000, 400
rng = np.random.default_rng(SEED)
theta_tr, x_tr = simulate_dataset(sim, N_TRAIN, N_BALLS, rng)
theta_va, x_va = simulate_dataset(sim, N_VAL, N_BALLS, rng)
print("train:", theta_tr.shape, "  val:", theta_va.shape)

# %% [markdown]
# The joint `(θ, x)` cloud traces the noise-free range curve plus
# scatter of width `σ/√n_balls`. Reading this cloud column-by-column at
# a fixed `x` is exactly the conditional density `q(θ|x)` from the
# Lecture 2a slides.

# %%
theta_grid = np.linspace(sim.prior_low, sim.prior_high, 1001)
plt.figure(figsize=(4.5, 3))
plt.scatter(theta_tr.numpy(), x_tr.numpy(), s=2, alpha=0.3, label="training pairs")
plt.plot(theta_grid, sim.range_mean(theta_grid), "k-", lw=1.5, label=r"$r(\theta)$")
plt.xlabel(r"$\theta$ [rad]"); plt.ylabel("x (mean landing)"); plt.legend()
plt.tight_layout(); plt.show()

# %% [markdown]
# ### ✏️ EXERCISE 2.A — build the Gaussian band
#
# The mean `μ_θ(x)` is a small MLP. The width is **homoscedastic**:
# one single learned number `σ_θ`, shared across every `x` (exactly
# Lecture 2a). We store it as a learned `log σ²` so it stays positive,
# via an `nn.Parameter`.
#
# Fill in `forward` so it returns `(mu, log_var)`, both of shape
# `(batch, 1)`. The `log_var` is the *same* scalar for every row in the
# batch, broadcast to the batch shape.

# %%
class GaussianBand(nn.Module):
    def __init__(self, in_dim=1, hidden=64):
        super().__init__()
        self.mu_net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.log_var = nn.Parameter(torch.zeros(1))  # one shared log-variance

    def forward(self, x):
        # TODO — your code here.
        # 1. mu = self.mu_net(x)                 -> shape (batch, 1)
        # 2. log_var broadcast to mu's shape     -> self.log_var.expand_as(mu)
        # 3. return (mu, log_var)
        raise NotImplementedError("implement GaussianBand.forward")


# %%
# @title Reference solution { display-mode: "form" }
class GaussianBand(nn.Module):  # noqa: F811
    def __init__(self, in_dim=1, hidden=64):
        super().__init__()
        self.mu_net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.log_var = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mu = self.mu_net(x)
        log_var = self.log_var.expand_as(mu)
        return mu, log_var


# %% [markdown]
# ### ✏️ EXERCISE 2.B — the Gaussian negative log-likelihood
#
# Implement the per-sample Gaussian NLL,
#
# $$ \ell(\theta;\ \mu, \log\sigma^2) =
# \tfrac{1}{2}\!\left[\tfrac{(\theta-\mu)^2}{\sigma^2} + \log\sigma^2\right], $$
#
# averaged over the batch. Drop the constant `½ log(2π)`; it has no
# gradient. This is the loss `E(φ)` from Lecture 2a, with `σ` now
# learned jointly with the mean.

# %%
def gaussian_nll(theta, mu, log_var):
    # TODO — your code here.
    # Return a scalar: the mean over the batch of the per-sample NLL.
    raise NotImplementedError("implement gaussian_nll")


# %%
# @title Reference solution { display-mode: "form" }
def gaussian_nll(theta, mu, log_var):  # noqa: F811
    var = torch.exp(log_var)
    return 0.5 * (((theta - mu) ** 2) / var + log_var).mean()


# %% [markdown]
# ### 2.3 — Train
#
# The same five-step loop, now with mini-batches and a validation pass
# each epoch.

# %%
def train_band(x_tr, theta_tr, x_va, theta_va,
               n_epochs=400, batch_size=256, lr=2e-3, hidden=64, seed=SEED):
    torch.manual_seed(seed)
    model = GaussianBand(in_dim=x_tr.shape[1], hidden=hidden)
    opt = optim.Adam(model.parameters(), lr=lr)
    n = x_tr.shape[0]
    tr_curve, va_curve = [], []
    bar = trange(n_epochs, desc="train band", leave=False)
    for _ in bar:
        perm = torch.randperm(n)
        ep = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            mu, log_var = model(x_tr[idx])
            loss = gaussian_nll(theta_tr[idx], mu, log_var)
            loss.backward(); opt.step()
            ep += loss.item() * idx.numel()
        tr_curve.append(ep / n)
        with torch.no_grad():
            mu_v, lv_v = model(x_va)
            va_curve.append(gaussian_nll(theta_va, mu_v, lv_v).item())
        bar.set_postfix(train=f"{tr_curve[-1]:.3f}", val=f"{va_curve[-1]:.3f}")
    return model, np.array(tr_curve), np.array(va_curve)


model_band, tr_curve, va_curve = train_band(x_tr, theta_tr, x_va, theta_va)

plt.figure(figsize=(5, 3))
plt.plot(tr_curve, label="train"); plt.plot(va_curve, label="val")
plt.xlabel("epoch"); plt.ylabel("Gaussian NLL"); plt.legend()
plt.title("Gaussian band — training"); plt.tight_layout(); plt.show()

# %% [markdown]
# ### 2.4 — Validate, amortise, and find where one width isn't enough
#
# The decisive test: this is a 1D problem, so we have the *exact*
# analytic posterior on a grid (`sim.true_posterior`). We sweep several
# true `θ` values across the prior, simulate one `x_obs` for each, and
# overlay the trained band's posterior on the analytic truth. The
# **same** network produces all of them with no retraining: that is
# amortisation.

# %%
def band_curve(model, x_obs, grid):
    with torch.no_grad():
        mu, log_var = model(torch.tensor([[x_obs]], dtype=torch.float32))
    mu = mu.item(); sg = float(torch.exp(0.5 * log_var).item())
    dens = (1 / (sg * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((grid - mu) / sg) ** 2)
    return dens, mu, sg


theta_trues = [0.15, 0.45, 0.70]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), sharey=True)
for ax, theta_t in zip(axes, theta_trues):
    x_obs = float(sim.simulate_summary(np.array([theta_t]), n_balls=N_BALLS,
                                       rng=np.random.default_rng(int(1000 * theta_t)))[0])
    q, mu, sg = band_curve(model_band, x_obs, theta_grid)
    _, p_true = sim.true_posterior(x_obs, n_balls=N_BALLS, theta_grid=theta_grid)
    ax.plot(theta_grid, p_true, "k-", lw=1.5, label="exact posterior")
    ax.plot(theta_grid, q, "C0--", lw=1.5, label="Gaussian band")
    ax.axvline(theta_t, color="C3", lw=1)
    ax.set_xlabel(r"$\theta$"); ax.set_title(fr"$\theta_{{\rm true}} = {theta_t}$")
    ax.legend(fontsize=8)
axes[0].set_ylabel("density")
fig.suptitle("one trained network, three observations (amortised)")
fig.tight_layout(); plt.show()

# %% [markdown]
# **Read the three panels.** The *mean* is excellent everywhere: the
# network has learned to predict `θ` from `x` across the whole prior,
# in one forward pass. The *width* is the interesting part. It is a
# single shared number, so it cannot be right everywhere:
#
# - near `θ = 0.15` the true posterior is **narrow** (the range curve
#   is steep, so a small `x` error pins `θ` tightly), but the band is
#   too wide;
# - near `θ = 0.70` the true posterior is **wide** (the range curve
#   flattens toward `π/4`, so the same `x` error allows a broad range
#   of `θ`), but the band is too narrow, i.e. overconfident exactly
#   where you should be least sure.
#
# The true posterior width genuinely depends on `x`. A homoscedastic
# model is mathematically incapable of expressing that. This is the
# limit of Lecture 2a's Gaussian band, and it is the limitation
# Session 1B opens with.
#
# ---
#
# ## Where this lands you
#
# - You ran the MLP training loop yourself and dissected the network:
#   its learned basis, its learning rate, and its failure modes.
# - You rebuilt Lecture 2a's Gaussian band in PyTorch and used it as an
#   amortised neural posterior estimator on the ball-throw, validated
#   against the exact posterior.
# - You found the band's wall: a single shared width cannot track an
#   `x`-dependent posterior width.
#
# **Session 1B (the gravitational-wave example)** starts right here. The
# first upgrade is a width that depends on `x` (a *heteroscedastic*
# head, `σ_θ(x)`). Then we point the same machinery at a real GW
# simulator, and finally swap the Gaussian for a normalising flow when
# even an `x`-dependent Gaussian is not flexible enough for the true
# posterior shape.
