# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cweniger/teaching-2606-ICTP-SAIFR/blob/main/notebooks/s1_pytorch_and_npe.ipynb)
#
# # Session 1 — PyTorch crash + Gaussian-head NPE
#
# **Hands-on session 1 (after Lecture 2). Runs on a laptop CPU.**
#
# Two blocks:
#
# 1. **PyTorch crash (~30 min).** Tensors, `autograd`, `nn.Module`,
#    `torch.optim`, the canonical training loop. We end with an MLP that
#    fits `sin(x)` so that you have run a real training loop *yourself*
#    before any SBI machinery appears.
# 2. **Gaussian-head NPE on the ball-throw (~45 min).** Build the
#    simplest neural posterior estimator from scratch using the
#    primitives from Block 1: a small MLP whose output is the mean and
#    log-variance of a Gaussian `q_φ(θ | x)`, trained on
#    `(θ_i, x_i)` pairs from the simulator. Validate against the
#    analytic reference posterior; then change the prior and watch the
#    posterior follow it.
#
# Block 3 (your APP-flavoured example of choice) lives in a separate
# notebook (`s1_app_<choice>.ipynb`).
#
# **Why this order.** Until you have written `loss.backward()` and
# `optimizer.step()` with your own hands, the lecture description of NPE
# is a mystery. After Block 1, NPE in Block 2 is just *fit a parametrised
# distribution by minimising NLL*, which you have now done.

# %%
# Always reinstall to pick up the latest from main — pip would otherwise
# short-circuit on an already-installed version, even when git HEAD has
# changed. --no-deps skips numpy/scipy/matplotlib (Colab ships them).
# !pip install -q --upgrade --force-reinstall --no-deps git+https://github.com/cweniger/teaching-2606-ICTP-SAIFR.git  # noqa: E501

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from samma_sbi.simulators import BallThrow

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

# %% [markdown]
# ---
#
# ## Block 1 — PyTorch crash
#
# Skip this block only if you have already trained a network in PyTorch
# yourself. Reading tutorials does not count.
#
# ### 1.1 — Tensors
#
# A `torch.Tensor` is a multi-dim array with two extra properties: it can
# live on a GPU, and it can carry gradient information. Otherwise it
# behaves almost exactly like a numpy array.

# %%
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.arange(3, dtype=torch.float32)
print("a =", a)
print("b =", b)
print("a + b =", a + b)
print("a.shape =", a.shape, "  a.dtype =", a.dtype)

# %% [markdown]
# **Broadcasting.** Same rules as numpy: dimensions of size 1 stretch to
# match. This is how a batch of inputs flows through a single weight
# matrix without writing a `for` loop.

# %%
W = torch.randn(4, 3)        # 4 outputs, 3 inputs
x_batch = torch.randn(8, 3)  # batch of 8 inputs
# Matrix-multiply each row of x_batch by W^T
y_batch = x_batch @ W.T      # shape (8, 4)
print("y_batch.shape =", y_batch.shape)

# %% [markdown]
# **GPU vs CPU.** We do not need a GPU for any of today's exercises, but
# the pattern is worth seeing once.

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", device)
a_on_device = a.to(device)
# Computations between tensors must happen on the same device.

# %% [markdown]
# ### 1.2 — Autograd in one line
#
# Set `requires_grad=True` on the tensors you want to optimise. PyTorch
# records every operation, and `loss.backward()` populates `.grad` for
# all of them.

# %%
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3 + 2 * x
y.backward()
print("dy/dx at x=2:", x.grad.item(), "  (expected: 3*4 + 2 = 14)")

# %% [markdown]
# ### 1.3 — `nn.Module`
#
# A `Module` bundles parameters with a `forward` method. Parameters
# defined as `nn.Parameter` (or wrapped inside `nn.Linear`, `nn.Conv2d`,
# etc.) are auto-registered and show up in `model.parameters()`.

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
# ### 1.4 — The canonical training loop
#
# Goal: fit `f(x) = sin(x)` on `x ∈ [-π, π]` with the MLP above. This is
# the pattern you will reuse for every neural network in this school.

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
    opt.zero_grad()              # 1. clear old gradients
    y_pred = model(x_train)      # 2. forward pass
    loss = loss_fn(y_pred, y_train)  # 3. compute loss
    loss.backward()              # 4. backward pass: fill .grad
    opt.step()                   # 5. optimizer step: update params
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
fig.tight_layout()
plt.show()

# %% [markdown]
# **Five steps to memorise.** Every training loop in this school has the
# same skeleton:
#
# 1. `opt.zero_grad()` — clear gradient buffers.
# 2. forward pass — compute predictions.
# 3. compute the loss.
# 4. `loss.backward()` — populate `.grad`.
# 5. `opt.step()` — take a gradient step.
#
# Forgetting step 1 silently *adds* gradients across iterations and is
# the most common PyTorch bug.
#
# **What `.grad` actually is.** Each `nn.Parameter` carries a `.grad`
# attribute — a tensor of the same shape, initially `None`. Three
# rules to internalise:
#
# - `loss.backward()` *adds* the freshly computed gradient into
#   `.grad` for every parameter on the computation graph. It does not
#   overwrite.
# - `opt.zero_grad()` walks every parameter the optimizer owns and
#   resets its `.grad` to zero.
# - `opt.step()` reads `.grad` and applies the update rule (Adam,
#   SGD, etc.).
#
# Why does `backward()` accumulate rather than overwrite? Because it
# lets you sum gradients from multiple losses or simulate a larger
# effective batch by calling `backward()` a few times before
# stepping. In a normal training loop you want exactly one fresh
# gradient per step, so the explicit `zero_grad()` is mandatory.
#
# You can inspect `.grad` directly. After the loop above, the first
# linear layer's weight has a populated gradient buffer; if we call
# `zero_grad()` it returns to zero.

# %%
W0 = model.net[0].weight
print("after training, .grad norm:", float(W0.grad.norm()))
opt.zero_grad()
print("after zero_grad,  .grad norm:", float(W0.grad.norm()))

# %% [markdown]
# ### ✏️ EXERCISE 1.A — your turn
#
# Replace `sin(x)` with `cos(2x)`, retrain, and plot. You should not
# need to change anything other than the target.

# %%
# TODO — your code here.
# Hint: change y_train, re-instantiate the model and optimizer, run the
# same five-step loop.


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
# ### ✏️ EXERCISE 1.B — break it, then fix it
#
# Now push the target frequency up: try fitting `cos(k * x)` with
# `k = 4`, then `k = 8`, then `k = 12`. At some point the same
# 32-hidden-unit MLP, trained for the same 2000 steps with the same
# learning rate, will visibly fail — the fit will look like a smooth
# under-resolved approximation of the truth.
#
# That failure is real and is called *spectral bias*: vanilla MLPs
# fit low-frequency targets much more easily than high-frequency ones.
#
# Once you have found a `k` where the default network clearly breaks,
# try to recover the fit by turning the obvious knobs:
#
# - more hidden units (`hidden = 128`, `256`),
# - more layers (add a third or fourth `Linear → ReLU` block),
# - more training steps (5000, 10000),
# - different learning rate (`lr = 3e-3`, `1e-4`).
#
# Which knob helps most? Which one is fighting the spectral bias and
# which one just smooths over poor optimisation? Keep the answer in
# mind; we will see the same trade-offs reappear when we choose
# architectures for `q_φ(θ | x)`.

# %%
# TODO — your code here.
# 1. Crank k up to a value where the default MLP visibly fails.
# 2. Try one or two architectural fixes and re-plot.


# %%
# @title Reference solution { display-mode: "form" }
def fit_cos_kx(k, hidden=32, n_layers=2, n_steps=2000, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    y = torch.cos(k * x_train)
    layers = [nn.Linear(1, hidden), nn.ReLU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    layers += [nn.Linear(hidden, 1)]
    model = nn.Sequential(*layers)
    opt = optim.Adam(model.parameters(), lr=lr)
    for _ in range(n_steps):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(x_train), y)
        loss.backward(); opt.step()
    with torch.no_grad():
        return model(x_train).numpy().ravel(), float(loss)


fig, axes = plt.subplots(1, 3, figsize=(13, 3), sharey=True)
xv = x_train.numpy().ravel()
# (a) default network on a high frequency — breaks
y_pred, loss = fit_cos_kx(k=12)
axes[0].plot(xv, np.cos(12 * xv), "k-", lw=1.2, label="truth")
axes[0].plot(xv, y_pred, "C3--", lw=1.5, label=f"default MLP (loss {loss:.2g})")
axes[0].set_title(r"$\cos(12x)$ — 32 hidden, 2 layers, 2k steps")
axes[0].legend(fontsize=8)
# (b) wider + deeper + longer
y_pred, loss = fit_cos_kx(k=12, hidden=256, n_layers=4, n_steps=8000)
axes[1].plot(xv, np.cos(12 * xv), "k-", lw=1.2, label="truth")
axes[1].plot(xv, y_pred, "C2--", lw=1.5, label=f"wider/deeper (loss {loss:.2g})")
axes[1].set_title("256 hidden, 4 layers, 8k steps")
axes[1].legend(fontsize=8)
# (c) push even further — spectral bias remains visible
y_pred, loss = fit_cos_kx(k=20, hidden=256, n_layers=4, n_steps=8000)
axes[2].plot(xv, np.cos(20 * xv), "k-", lw=1.2, label="truth")
axes[2].plot(xv, y_pred, "C1--", lw=1.5, label=f"$\\cos(20x)$ (loss {loss:.2g})")
axes[2].set_title("same big MLP, even higher frequency")
axes[2].legend(fontsize=8)
fig.tight_layout(); plt.show()

# %% [markdown]
# What the three panels show:
#
# - **(a) default MLP, `cos(12x)`** — clear under-fit. The network
#   essentially averages out the high-frequency oscillations.
# - **(b) wider + deeper + longer** — the same target now sits cleanly
#   inside the fit. Scaling and longer training go a long way against
#   spectral bias.
# - **(c) push higher still** — at `cos(20x)`, even the bigger MLP
#   gives up. Scaling does not *eliminate* spectral bias; it just
#   pushes the failure to higher frequencies. Properly fixing this in
#   general needs architectural tricks (Fourier features, sinusoidal
#   positional encodings) that are beyond Session 1.

# %% [markdown]
# ---
#
# ## Block 2 — Gaussian-head NPE on the ball-throw
#
# We now build the simplest neural posterior estimator. Given a
# simulator that produces pairs `(θ, x)`, we train a network that takes
# an observation `x` and returns the parameters of a Gaussian over
# `θ`:
#
# $$ q_\phi(\theta \mid x) = \mathcal{N}\bigl(\theta;\ \mu_\phi(x),\ \sigma_\phi^2(x)\bigr). $$
#
# The training loss is the *negative log-likelihood of the Gaussian
# evaluated at the true `θ_i`*, averaged over the training set:
#
# $$ \mathcal{L}(\phi) = \tfrac{1}{N} \sum_i
# \tfrac{1}{2} \left[ \tfrac{(\theta_i - \mu_\phi(x_i))^2}{\sigma_\phi^2(x_i)} + \log \sigma_\phi^2(x_i) \right] + \text{const}. $$
#
# That is *all* there is to NPE conceptually. The rest of the school is
# either better choices for the distribution family `q_φ` (flows,
# diffusion, FM), better inputs (summary networks), or diagnostics.
#
# ### 2.1 — Simulator and summary statistic
#
# The ball-throw simulator from Lecture 1b. We restrict the prior to
# `(0.05, π/4)` so that the mapping `θ → r(θ)` is one-to-one — the
# Gaussian head can plausibly capture this posterior. (We will come back
# to the wider, bimodal prior in Session 2 when we have flows.)
#
# The observation is the **mean** of `n_balls = 10` landings, which is
# the same summary used in Demo 2 of Lecture 1b.

# %%
N_BALLS = 10
sim = BallThrow(prior_low=0.05, prior_high=np.pi / 4)

rng = np.random.default_rng(SEED)
theta_demo = sim.sample_prior(5, rng=rng)
x_demo = sim.simulate_summary(theta_demo, n_balls=N_BALLS, rng=rng)
for t, xi in zip(theta_demo, x_demo):
    print(f"theta = {t:.3f} rad   ->   x_summary = {xi:.3f} m")

# %% [markdown]
# ### 2.2 — Training set: simulate `(θ, x)` pairs
#
# Two natural design choices:
#
# - **Number of training pairs `N_TRAIN`.** Each pair costs `n_balls`
#   simulator calls. We use `N_TRAIN = 4000` so the whole training set
#   fits on a laptop and trains in seconds.
# - **Train/val split.** A held-out validation set lets us monitor
#   over-fitting honestly. We use a 90/10 split.

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
print("train:", theta_tr.shape, x_tr.shape)
print("val:  ", theta_va.shape, x_va.shape)

# %% [markdown]
# Quick sanity plot: the (θ, x) joint cloud should trace the noise-free
# range function plus Gaussian scatter of width `σ / √n_balls`.

# %%
theta_grid = np.linspace(sim.prior_low, sim.prior_high, 200)
plt.figure(figsize=(4.5, 3))
plt.scatter(theta_tr.numpy(), x_tr.numpy(), s=2, alpha=0.3, label="training pairs")
plt.plot(theta_grid, sim.range_mean(theta_grid), "k-", lw=1.5, label=r"$r(\theta)$")
plt.xlabel(r"$\theta$ [rad]"); plt.ylabel("x (mean landing)"); plt.legend(); plt.tight_layout(); plt.show()

# %% [markdown]
# ### 2.3 — The Gaussian-head model
#
# A small MLP that takes `x` and outputs two numbers: the mean `μ(x)`
# and the **log-variance** `log σ²(x)`. We output log-variance (not
# variance) so the network is free to predict any real number while
# `σ²` stays positive.
#
# ### ✏️ EXERCISE 2.A — implement `GaussianHead.forward`
#
# Fill in the `forward` method so that it returns `(mu, log_var)` —
# both of shape `(batch, 1)`. The body of the network is provided.

# %%
class GaussianHead(nn.Module):
    def __init__(self, in_dim=1, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head_mu = nn.Linear(hidden, 1)
        self.head_logvar = nn.Linear(hidden, 1)

    def forward(self, x):
        # TODO — your code here.
        # 1. run x through self.trunk
        # 2. apply self.head_mu and self.head_logvar to the trunk output
        # 3. return (mu, log_var)
        raise NotImplementedError("implement GaussianHead.forward")


# %%
# @title Reference solution { display-mode: "form" }
class GaussianHead(nn.Module):  # noqa: F811
    def __init__(self, in_dim=1, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head_mu = nn.Linear(hidden, 1)
        self.head_logvar = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        mu = self.head_mu(h)
        log_var = self.head_logvar(h)
        return mu, log_var


# %% [markdown]
# ### 2.4 — Gaussian negative log-likelihood
#
# ### ✏️ EXERCISE 2.B — implement `gaussian_nll`
#
# Implement the per-sample Gaussian NLL,
#
# $$ \ell(\theta;\ \mu, \log\sigma^2) =
# \tfrac{1}{2}\!\left[\tfrac{(\theta-\mu)^2}{\sigma^2} + \log\sigma^2\right] $$
#
# averaged over the batch. Drop the additive constant `½ log(2π)`; it
# does not affect the gradient.

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
# ### 2.5 — Train the model
#
# Same five-step loop you wrote in Block 1, now with the Gaussian NLL,
# mini-batches, and a validation pass.

# %%
def train_gaussian_npe(
    x_tr, theta_tr, x_va, theta_va,
    n_epochs=80, batch_size=256, lr=1e-3, hidden=64, seed=SEED,
):
    torch.manual_seed(seed)
    model = GaussianHead(in_dim=x_tr.shape[1], hidden=hidden)
    opt = optim.Adam(model.parameters(), lr=lr)

    n = x_tr.shape[0]
    train_curve, val_curve = [], []
    for epoch in range(n_epochs):
        # shuffle indices for mini-batching
        perm = torch.randperm(n)
        ep_train = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            mu, log_var = model(x_tr[idx])
            loss = gaussian_nll(theta_tr[idx], mu, log_var)
            loss.backward()
            opt.step()
            ep_train += loss.item() * idx.numel()
        ep_train /= n
        with torch.no_grad():
            mu_v, log_var_v = model(x_va)
            ep_val = gaussian_nll(theta_va, mu_v, log_var_v).item()
        train_curve.append(ep_train)
        val_curve.append(ep_val)
    return model, np.array(train_curve), np.array(val_curve)


model, tr_curve, va_curve = train_gaussian_npe(
    x_tr, theta_tr, x_va, theta_va
)

plt.figure(figsize=(5, 3))
plt.plot(tr_curve, label="train")
plt.plot(va_curve, label="val")
plt.xlabel("epoch"); plt.ylabel("Gaussian NLL"); plt.legend()
plt.title("Gaussian-head NPE — training"); plt.tight_layout(); plt.show()

# %% [markdown]
# Train and val should both decrease and stay close to each other. If
# `val` plateaus while `train` keeps falling, you are over-fitting —
# usually a sign that the network is too large for the dataset.
#
# ### 2.6 — Posterior at a chosen observation
#
# Generate a single `x_obs` from a known `θ_true`, evaluate the
# Gaussian head, and compare with the analytic reference posterior on a
# fine grid.

# %%
theta_true = float(np.array([0.55]))  # somewhere inside the prior
x_obs = float(sim.simulate_summary(np.array([theta_true]), n_balls=N_BALLS,
                                   rng=np.random.default_rng(123))[0])

with torch.no_grad():
    mu, log_var = model(torch.tensor([[x_obs]], dtype=torch.float32))
    mu, sigma = mu.item(), float(torch.exp(0.5 * log_var).item())

theta_grid = np.linspace(sim.prior_low, sim.prior_high, 1001)
q_phi = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
    -0.5 * ((theta_grid - mu) / sigma) ** 2
)
_, p_true = sim.true_posterior(x_obs, n_balls=N_BALLS, theta_grid=theta_grid)

plt.figure(figsize=(5.5, 3))
plt.plot(theta_grid, p_true, "k-", lw=1.5, label="reference posterior")
plt.plot(theta_grid, q_phi, "C0--", lw=1.5,
         label=fr"$q_\phi(\theta|x)$: $\mathcal{{N}}({mu:.3f},\,{sigma:.3f}^2)$")
plt.axvline(theta_true, color="C3", lw=1, label=r"$\theta_{\rm true}$")
plt.xlabel(r"$\theta$"); plt.ylabel("density"); plt.legend()
plt.title(f"Gaussian-head NPE at x_obs = {x_obs:.3f}")
plt.tight_layout(); plt.show()

# %% [markdown]
# In this regime (one-to-one `r(θ)`, well-inside the prior) the
# Gaussian head sits almost on top of the analytic posterior. That is
# the cleanest validation we can give NPE without resorting to flows.

# %% [markdown]
# ### ✏️ EXERCISE 2.C — evaluate outside the training domain
#
# The network was only ever shown `x_obs` values that the prior could
# plausibly produce: the maximum noise-free range is
# `r(π/4) = v₀²/g ≈ 10.2 m`, and with the summary noise of
# `σ/√n_balls ≈ 0.1 m` essentially every training `x` lies in
# `[0.5, 10.3]`. What does the network do when you feed it an
# `x_obs` that lives well outside this range?
#
# Try `x_obs = 14.0` (impossible — nothing in the prior could
# produce this) and also `x_obs = -1.0` (negative range). Evaluate
# `q_φ(θ | x_obs)` and compare to the reference posterior on the
# grid. The reference is allowed to look weird at the boundary; the
# point is that the network *will not warn you* — it returns a
# perfectly confident Gaussian for any input you give it.
#
# *Take-away.* Amortised inference is only trustworthy on `x_obs`
# values that look like the simulator's outputs. Out-of-distribution
# detection is a real concern in practice and not something the
# Gaussian head provides for free.

# %%
# TODO — your code here.
# Pick two OOD x_obs values, evaluate q_phi, plot alongside the
# reference posterior.


# %%
# @title Reference solution { display-mode: "form" }
fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
for ax, x_ood in zip(axes, [14.0, -1.0]):
    with torch.no_grad():
        mu_o, lv_o = model(torch.tensor([[x_ood]], dtype=torch.float32))
        mu_o = mu_o.item(); sg_o = float(torch.exp(0.5 * lv_o).item())
    q = (1 / (sg_o * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((theta_grid - mu_o) / sg_o) ** 2
    )
    _, p_ref = sim.true_posterior(x_ood, n_balls=N_BALLS, theta_grid=theta_grid)
    ax.plot(theta_grid, p_ref, "k-", lw=1.2, label="reference")
    ax.plot(theta_grid, q, "C0--", lw=1.5,
            label=fr"$q_\phi$: $\mathcal{{N}}({mu_o:.2f},\,{sg_o:.2f}^2)$")
    ax.set_xlabel(r"$\theta$")
    ax.set_title(f"x_obs = {x_ood} (out of distribution)")
    ax.legend(fontsize=8)
axes[0].set_ylabel("density")
fig.tight_layout(); plt.show()

# %% [markdown]
# Notice that the network confidently outputs *some* Gaussian for
# both inputs, even though no realistic `θ` could have produced
# either observation. The width is roughly the typical training-time
# width; the mean is wherever the network's extrapolation lands. The
# reference posterior, by contrast, piles up at the closest possible
# boundary and is essentially zero everywhere.
#
# ### ✏️ EXERCISE 2.D — make the width `x`-independent
#
# Here is the more interesting failure mode: what if we hobble the
# *density family* and ask the network to use a single, learned
# variance that does **not** depend on `x`?
#
# For the ball-throw with prior `(0.05, π/4)`, the noise-free range
# `r(θ) = (v₀²/g) sin(2θ)` flattens near `θ → π/4` — so a fixed
# observation noise translates into a *wide* posterior near the
# prior edge and a *tight* posterior near `θ = 0.05`. The width
# genuinely depends on `x_obs`. A homoscedastic model is forced to
# pick the average; it will be visibly overconfident on some
# observations and visibly underconfident on others.
#
# Implement `GaussianHeadHomo` below by making `log_var` a *single
# learned scalar* (use `nn.Parameter`) instead of a function of `x`.
# Re-use the same training loop and compare against the
# heteroscedastic baseline at two well-separated `x_obs`.

# %%
class GaussianHeadHomo(nn.Module):
    """Same trunk as GaussianHead but with a single, x-independent log-variance."""

    def __init__(self, in_dim=1, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head_mu = nn.Linear(hidden, 1)
        # TODO — your code here.
        # Add a single learnable log-variance parameter (hint: nn.Parameter(torch.zeros(1))).
        # In forward(), return mu of shape (batch, 1) and log_var broadcast to (batch, 1).
        raise NotImplementedError("implement GaussianHeadHomo")

    def forward(self, x):
        raise NotImplementedError("implement GaussianHeadHomo.forward")


# %%
# @title Reference solution { display-mode: "form" }
class GaussianHeadHomo(nn.Module):  # noqa: F811
    def __init__(self, in_dim=1, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head_mu = nn.Linear(hidden, 1)
        self.log_var_const = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        h = self.trunk(x)
        mu = self.head_mu(h)
        log_var = self.log_var_const.expand_as(mu)
        return mu, log_var


def train_homo(x_tr, theta_tr, x_va, theta_va,
               n_epochs=80, batch_size=256, lr=1e-3, hidden=64, seed=SEED):
    torch.manual_seed(seed)
    m = GaussianHeadHomo(in_dim=x_tr.shape[1], hidden=hidden)
    opt = optim.Adam(m.parameters(), lr=lr)
    n = x_tr.shape[0]
    for _ in range(n_epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            mu, lv = m(x_tr[idx])
            loss = gaussian_nll(theta_tr[idx], mu, lv)
            loss.backward(); opt.step()
    return m


model_homo = train_homo(x_tr, theta_tr, x_va, theta_va)

# Pick two x_obs from opposite ends of the range: tight (small θ)
# and wide (θ near π/4).
fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
for ax, theta_t in zip(axes, [0.15, 0.70]):
    xo = float(sim.simulate_summary(np.array([theta_t]), n_balls=N_BALLS,
                                    rng=np.random.default_rng(int(1000 * theta_t)))[0])
    with torch.no_grad():
        mh, lh = model(torch.tensor([[xo]], dtype=torch.float32))      # heteroscedastic
        mH, lH = model_homo(torch.tensor([[xo]], dtype=torch.float32))  # homoscedastic
    mh, sh = mh.item(), float(torch.exp(0.5 * lh).item())
    mH, sH = mH.item(), float(torch.exp(0.5 * lH).item())
    g = lambda m_, s_: (1 / (s_ * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((theta_grid - m_) / s_) ** 2)
    _, p_ref = sim.true_posterior(xo, n_balls=N_BALLS, theta_grid=theta_grid)
    ax.plot(theta_grid, p_ref, "k-", lw=1.2, label="reference")
    ax.plot(theta_grid, g(mh, sh), "C0--", lw=1.5, label=fr"heteroscedastic ($\sigma={sh:.3f}$)")
    ax.plot(theta_grid, g(mH, sH), "C3--", lw=1.5, label=fr"homoscedastic ($\sigma={sH:.3f}$)")
    ax.axvline(theta_t, color="0.5", lw=0.6)
    ax.set_xlabel(r"$\theta$"); ax.set_title(fr"$\theta_{{\rm true}} = {theta_t}$")
    ax.legend(fontsize=8)
axes[0].set_ylabel("density")
fig.tight_layout(); plt.show()

# %% [markdown]
# Read the two panels.
#
# - At `θ_true ≈ 0.15`, the reference posterior is *narrow*. The
#   heteroscedastic model matches it; the homoscedastic one is too
#   wide — it has learned the population-average variance.
# - At `θ_true ≈ 0.70` (near the prior edge), the reference posterior
#   is *wide*. The heteroscedastic model matches; the homoscedastic
#   one is now too narrow — overconfident in exactly the regime where
#   uncertainty is highest.
#
# The mean predictions can still be fine in both cases. The model is
# failing on *width*, not location. This is the lesson Block 2 hangs
# on: the choice of density family matters, and even a "small"
# restriction (force `σ²` to be a scalar) destroys calibration where
# it matters most. Lecture 3 generalises this — flows, FM, diffusion
# all relax the family in more interesting ways than just letting
# `σ²(x)` move.

# %% [markdown]
# ### 2.7 — Prior dependence
#
# A subtle point that is easy to miss: **the NPE posterior is
# prior-conditional**, in a way the rejection-ABC posterior never is.
# We trained `q_φ` on `(θ, x)` pairs drawn from one specific prior;
# if you change the prior, the optimum `q_φ` changes too, even at the
# same `x_obs`.
#
# We illustrate this by retraining on a *narrower* prior and comparing
# the two posteriors at one common `x_obs`.

# %%
sim_narrow = BallThrow(prior_low=0.40, prior_high=np.pi / 4)
rng = np.random.default_rng(SEED + 1)
theta_tr_n, x_tr_n = simulate_dataset(sim_narrow, N_TRAIN, N_BALLS, rng)
theta_va_n, x_va_n = simulate_dataset(sim_narrow, N_VAL, N_BALLS, rng)

model_n, _, _ = train_gaussian_npe(
    x_tr_n, theta_tr_n, x_va_n, theta_va_n
)

# Pick an x_obs that is plausible under *both* priors.
x_obs2 = float(sim.simulate_summary(np.array([0.55]), n_balls=N_BALLS,
                                    rng=np.random.default_rng(7))[0])

def gaussian_curve(model, x_obs, grid):
    with torch.no_grad():
        mu, log_var = model(torch.tensor([[x_obs]], dtype=torch.float32))
        mu = mu.item(); sg = float(torch.exp(0.5 * log_var).item())
    return (1 / (sg * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((grid - mu) / sg) ** 2
    ), mu, sg


q_wide, mu_w, sg_w = gaussian_curve(model, x_obs2, theta_grid)
q_narr, mu_n, sg_n = gaussian_curve(model_n, x_obs2, theta_grid)

plt.figure(figsize=(6, 3.2))
plt.axvspan(sim.prior_low, sim.prior_high, color="C0", alpha=0.08,
            label="wide prior")
plt.axvspan(sim_narrow.prior_low, sim_narrow.prior_high, color="C3", alpha=0.12,
            label="narrow prior")
plt.plot(theta_grid, q_wide, "C0-", lw=1.7,
         label=fr"wide-prior NPE: $\mathcal{{N}}({mu_w:.3f},\,{sg_w:.3f}^2)$")
plt.plot(theta_grid, q_narr, "C3-", lw=1.7,
         label=fr"narrow-prior NPE: $\mathcal{{N}}({mu_n:.3f},\,{sg_n:.3f}^2)$")
plt.axvline(0.55, color="k", lw=0.8, label=r"$\theta_{\rm true}$")
plt.xlabel(r"$\theta$"); plt.ylabel("density"); plt.legend(fontsize=8)
plt.title("Same x_obs, two priors, two NPE posteriors")
plt.tight_layout(); plt.show()

# %% [markdown]
# **Read this carefully.** The two networks see *exactly the same*
# `x_obs`. They produce different posteriors because they have been
# trained to approximate `p(θ | x)` under different priors. The
# narrow-prior network has effectively *no* training pairs from the
# tails of the wide prior, so it has no way to express posterior mass
# there. Out-of-distribution `x_obs` (say, an `x` that would only ever
# be produced by `θ < 0.4`) would simply be mis-handled by the
# narrow-prior network with no warning.
#
# This is the practical price of amortisation: **the prior is baked in
# at training time**.
#
# ---
#
# ## Where this lands you
#
# - You have written the entire NPE training loop yourself, from the
#   five-step PyTorch skeleton up to amortised evaluation.
# - You have a working baseline on a one-parameter problem you can
#   solve analytically.
# - You have seen, explicitly, that the NPE posterior depends on the
#   training-time prior.
#
# **Next:** open `s1_app_<your_choice>.ipynb` and apply the *same*
# `GaussianHead` + `gaussian_nll` machinery to a real
# astroparticle-physics-flavoured simulator (gravitational waves,
# cosmic-ray spectrum, or point sources in images). In Session 2 we
# revisit your APP result with normalising flows, learned summary
# networks, and SBC diagnostics — and discover what the Gaussian head
# was hiding.
