# Coursework origins and revision history

This study develops material from Andres Aguirre Torres's Penn State MATH 448 and MATH 451 honors projects, completed in fall 2025.

The source references were the final `MATH 448 Honors Project.ipynb` and `MATH 451 Project.ipynb` notebooks. The original academic files remain separate from this repository.

## September 2026 revision

- Consolidated overlapping Black–Scholes and simulation material into one small Python module.
- Corrected antithetic standard errors to use independent pair means.
- Replaced single-run error claims with repeated experiments at equal payoff-evaluation budgets.
- Added an analytic Poisson-mixture Merton benchmark, focused validation checks, and explicit limiting-case behavior.
- Regenerated the figures and numerical results from a single seeded script.
- Omitted historical GPU comparisons because they did not establish a matched-workload implementation speedup.

The refactor, added code, checks, experiments, and documentation were prepared with AI assistance. This repository does not claim unaided authorship of every implementation detail, novelty of the pricing models, or that the updated results were part of the original 2025 submission.

The earlier demonstration notebook, course instructions, reference PDFs, duplicate exports, and private local material are not included. The computations use synthetic model draws rather than redistributed market datasets.
