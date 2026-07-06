# Data-for-AI Readiness (DAR): dataset, index and code

Replication materials for the paper **"Data before models: statistical capacity and open-data governance as the binding AI input in a catch-up knowledge economy"** (Yesmagambetov, 2026; submitted to *Technology in Society*).

The study measures the **data input** of national AI capability for eleven economies, builds a reproducible **Data-for-AI Readiness (DAR)** index, and relates it to national AI research output. This archive contains everything needed to reproduce the dataset, the index, the regressions and the figures.

## Contents

```
data/
  dataset.csv               # raw assembled panel: 11 economies × SPI pillars, AI output, controls
  dataset_with_index.csv    # dataset + computed DAR / DAR_all5 / DAR_pca and derived intensities
code/
  compute_article5.py       # builds the DAR index, correlations and OLS regressions; prints results
  make_figures_article5.py  # regenerates Figures 1–4 (300 dpi)
figures/                    # fig1 DAR ranking, fig2 conversion quadrant, fig3 under-conversion, fig4 KZ profile
requirements.txt
LICENSE                     # MIT (code)
```

## Economies

Kazakhstan, Uzbekistan, Kyrgyzstan, Poland, Estonia, Turkey, United Arab Emirates, United States, China, Germany, Netherlands.

## Data sources (all public)

- **World Bank Statistical Performance Indicators (SPI)** — five data-system pillars, 2024. From the official `worldbank/SPI` repository (`03_output_data/SPI_index.csv`). Provides DAR components (P2 openness, P4 sources, P5 infrastructure) and population.
- **World Bank World Development Indicators (WDI)** — GDP per capita (PPP), R&D as % of GDP, researchers per million. Retrieved via the DBnomics mirror of WDI.
- **OpenAlex** — AI-related research works by country of authorship (concept `C154945302`, "Artificial intelligence"), 2019–2024, via the OpenAlex API.
- **TOP500** — Kazakhstan's November 2025 debut, used as background compute context only (not a regressor).

Every value is traceable to these sources; see the paper's data appendix (`ARTICLE5_facts.md`) for per-figure provenance.

## Reproduce

```bash
pip install -r requirements.txt
python code/compute_article5.py      # prints DAR ranking, correlations, OLS models, residuals; writes dataset_with_index.csv
python code/make_figures_article5.py # writes figures/fig1–4.png
```

## Key results

- Kazakhstan's DAR is 81.8 (upper-middle of the sample), not data-poor.
- In the size–income–DAR model (n=11, R²=0.951), Kazakhstan has the largest negative residual (−1.15): descriptive under-conversion. Robust to dropping the US and China (n=9, residual −0.95) and to leave-one-out re-estimation (residual always negative, −1.27 to −0.96).
- AI output scales almost one-for-one with the researcher stock (elasticity 0.964, R²=0.977), while DAR adds little: the binding input is research capacity, not data.

## License

Code: MIT (see `LICENSE`). Data are derived from public sources listed above and are shared for research reuse under CC BY 4.0.

## Citation

If you use these materials, please cite the paper and this archive (Zenodo DOI assigned on deposit).

## AI-use note

A large language model assisted with language editing, drafting and analysis/figure code, under the author's full control and responsibility. Data were collected and verified by the author.
