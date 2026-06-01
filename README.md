# Machine Learning for Astroparticle Physics

**A Crash-Course in Simulation-Based Inference**

Lecture notes and hands-on materials for the
[ICTP-SAIFR SAMMA 2026 PhD school](https://www.ictp-saifr.org/samma2026/),
São Paulo, 2026.

Christoph Weniger ([GRAPPA](https://grappa.amsterdam) / University of Amsterdam).

## Lectures

The slide decks live in [`docs/`](docs/) and are served at
<https://cweniger.github.io/teaching-2606-ICTP-SAIFR/> once GitHub Pages is
enabled. See [`docs/index.html`](docs/index.html) for the lecture index.

Planning documents:

- [`lecture_plan.md`](lecture_plan.md) — slide-level plans per lecture
- [`handson_plan.md`](handson_plan.md) — two-session hands-on plan

## Hands-on (Colab)

The hands-on notebooks live in [`notebooks/`](notebooks/) and run on Google
Colab with one click. Each notebook starts with a cell that installs this
package directly from GitHub:

```python
!pip install -q git+https://github.com/cweniger/teaching-2606-ICTP-SAIFR.git
```

Open-in-Colab badges will be added per notebook as they land.

## Local installation (authors / curious students)

```bash
git clone https://github.com/cweniger/teaching-2606-ICTP-SAIFR.git
cd teaching-2606-ICTP-SAIFR
pip install -e ".[dev,sbi]"
```

## Notebook authoring workflow (authors)

Notebooks are paired with jupytext percent-format `.py` sources. The `.py`
is the source of truth (diff-friendly, agent-editable); the `.ipynb` is
regenerated.

```bash
# initial pairing (once per notebook)
jupytext --set-formats ipynb,py:percent notebooks/<name>.ipynb

# after editing either file
jupytext --sync notebooks/<name>.py
```

## License

MIT.
