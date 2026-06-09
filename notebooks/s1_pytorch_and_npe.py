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

# %% [markdown]
# ### Exercise 1.A — your turn
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
# **Exercise 2.A.** Fill in the `forward` so that it returns
# `(mu, log_var)` — both of shape `(batch, 1)`. The body of the network
# is provided.

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
# **Exercise 2.B.** Implement the per-sample Gaussian NLL,
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
# ### Exercise 2.C — amortisation
#
# The same trained network produces a posterior for *any* `x_obs` —
# this is what is meant by **amortised** inference: training is
# expensive, evaluation is one forward pass. Pick three different true
# `θ` values, simulate a fresh `x_obs` for each, and overlay all three
# Gaussian-head posteriors on the same axes (with the analytic
# references for comparison).

# %%
# TODO — your code here.


# %%
# @title Reference solution { display-mode: "form" }
fig, ax = plt.subplots(figsize=(5.5, 3))
for theta_t, col in zip([0.30, 0.55, 0.72], ["C0", "C2", "C3"]):
    xo = float(sim.simulate_summary(np.array([theta_t]), n_balls=N_BALLS,
                                    rng=np.random.default_rng(int(1000 * theta_t)))[0])
    with torch.no_grad():
        mu_i, log_var_i = model(torch.tensor([[xo]], dtype=torch.float32))
        mu_i = mu_i.item(); sg_i = float(torch.exp(0.5 * log_var_i).item())
    q = (1 / (sg_i * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((theta_grid - mu_i) / sg_i) ** 2
    )
    _, ptrue_i = sim.true_posterior(xo, n_balls=N_BALLS, theta_grid=theta_grid)
    ax.plot(theta_grid, ptrue_i, color=col, lw=1.2, alpha=0.4)
    ax.plot(theta_grid, q, color=col, lw=1.5, ls="--",
            label=fr"$\theta_{{\rm true}}={theta_t}$")
    ax.axvline(theta_t, color=col, lw=0.6, alpha=0.5)
ax.set_xlabel(r"$\theta$"); ax.set_ylabel("density")
ax.set_title("amortised Gaussian-head posteriors")
ax.legend(); fig.tight_layout(); plt.show()

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
