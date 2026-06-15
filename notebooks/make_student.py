"""Generate student (no-solution) notebooks from the master notebooks.

The master ``*.py`` notebooks carry the reference solutions in cells
marked ``# @title Reference solution``. Each such cell is preceded by an
exercise prompt and a ``# TODO — your code here`` scaffold cell, so we
can simply *drop* the solution cells and the students keep their
foothold. The ``# @title Optional flow preview`` cell is not a solution
and is kept.

Run from the repo root after editing a master notebook::

    python notebooks/make_student.py

It writes ``notebooks/<name>_student.ipynb`` for each master. The master
notebooks themselves double as the *solutions* notebooks.
"""

from __future__ import annotations

import jupytext
import nbformat

MASTERS = ["s1_pytorch_and_npe", "s1_app_gw"]
SOLUTION_MARK = "# @title Reference solution"
BADGE = "colab.research.google.com/github/cweniger/teaching-2606-ICTP-SAIFR"


def banner(name: str) -> str:
    sol = f"https://{BADGE}/blob/main/notebooks/{name}.ipynb"
    return (
        "> ### ✏️ Exercises version\n"
        "> The reference solutions are removed here so *Run all* can't spoil "
        "the exercises. Work through the `# TODO` cells yourself, then check "
        f"your answers against the [**solutions notebook**]({sol})."
    )


for name in MASTERS:
    nb = jupytext.read(f"notebooks/{name}.py")

    # Drop the reference-solution cells; keep everything else.
    kept = [
        c for c in nb.cells
        if not (c.cell_type == "code"
                and c.source.lstrip().startswith(SOLUTION_MARK))
    ]

    # Point the top "Open in Colab" badge at the student notebook itself.
    if kept and kept[0].cell_type == "markdown":
        kept[0].source = kept[0].source.replace(
            f"notebooks/{name}.ipynb", f"notebooks/{name}_student.ipynb"
        )

    # Add a short banner just after the title cell.
    kept.insert(1, nbformat.v4.new_markdown_cell(banner(name)))

    nb.cells = kept
    out = f"notebooks/{name}_student.ipynb"
    jupytext.write(nb, out)
    n_sol = sum(
        c.cell_type == "code" and c.source.lstrip().startswith(SOLUTION_MARK)
        for c in nb.cells
    )
    print(f"wrote {out}  ({len(kept)} cells, {n_sol} solution cells remaining)")
