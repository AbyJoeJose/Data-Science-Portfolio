# Market Data Quality Analytics for VaR

A framework that scores market data quality, detects outliers, imputes missing values, and measures how much the defects move Value at Risk in dollars.

Built in Python with NumPy, pandas, SciPy and Plotly. Three class hierarchies, ten sections, eight exhibits, all from public Federal Reserve data.


**Aby Joe Jose** | [GitHub](https://github.com/AbyJoeJose) | [Portfolio](https://abyjoejose.github.io/Data-Science-Portfolio) | [LinkedIn](https://linkedin.com/in/aby-joe-jose-88959021b) | abyjoejose00@gmail.com

## The question

Value at Risk is computed from historical market data. If that data is wrong, the risk number is wrong, and the firm holds the wrong amount of capital against its trading book.

So: if nobody cleaned this data, how wrong would the risk number be?

This notebook answers that in dollars rather than in principle. Everything before the last few sections is the machinery required to get there.

## What it found

Run on a $10,000,000 equally weighted portfolio across equities, two FX pairs and crude oil, 2016 to 2026, at 99% confidence.

**Uncleaned data moved historical simulation VaR by $13,608, or 6.0%.** Running the same dirty panel through the detection and imputation pipeline recovered 92% of that error, leaving a residual of $1,105.

**Different defects damage different models, in opposite directions.**

| Defect | Historical simulation | Parametric | Mechanism |
| --- | --- | --- | --- |
| Bad prints | +$13,728 | +$5,917 | A fake return inflates the tail and the variance together |
| Staleness | no change | $4,420 lower | Zeros dilute variance; the extreme days are untouched |
| Gaps, forward filled | +$11,690 | +$1,006 | The merged two day return becomes the tail; variance barely moves |

Quantile based measures are damaged by merging. Variance based measures are damaged by dilution. A quality process watching only one of them misses the other, which is the argument for grading the data directly rather than inferring its health from whether the risk number looks plausible.

**A fixed threshold is not a fixed standard.** Outliers were injected at six times *local* volatility, which is how a real bad print behaves. A full sample z score caught 60% of them during volatile periods and 10% during calm ones, because a large move in a quiet week is unremarkable against a sample containing March 2020. Scaling by trailing EWMA volatility closed that gap almost entirely, at 65% and 60%.

**The scorecard found a real property of the data on its first run.** The 10 year Treasury series graded C at 8.6% stale. That is not a broken feed: yields are published to 1 basis point, so consecutive identical prints are ordinary. Distinguishing an instrument's quoting convention from a vendor failure is the difference between a useful alert and a noisy one.

**Crude oil settled at negative $36.98 on 20 April 2020.** Any pipeline taking log returns without a sign check produces NaN that day and, if it drops them silently, loses the largest move in the sample. The scorecard flags it and the return calculation masks it explicitly.

**Breaches cluster.** In the backtest, the probability of a breach given a breach the day before was 12.9%, against an unconditional 1.4%. That is volatility clustering in a single number, and it is the evidence for the GARCH weighted extension listed under future work rather than an assertion that one would help.

**Backtesting can be fooled by bad data.** Historical simulation passed the Kupiec test on the dirty panel and failed it on the clean one. The defects raised the VaR threshold, which mechanically produced fewer breaches. A model can pass its backtest because its inputs are corrupted.

## Quick start


To run locally:

```bash
git clone https://github.com/AbyJoeJose/Data-Science-Portfolio.git
cd Data-Science-Portfolio/Market-Data-Quality-VaR
pip install -r requirements.txt
jupyter notebook Market_Data_Quality_VaR.ipynb
```

Then Run All. Runtime is roughly one minute.

The `data/` folder holds both the six source CSVs as downloaded from FRED and the merged panel the notebook builds from them, so the repo runs with no network access and reproduces every number in this file exactly. To watch the merge logic rebuild the panel from source, delete `data/market_panel.csv` and re run.

## Data

Six series across five asset classes, all from FRED (Federal Reserve Economic Data, St. Louis Fed). Multiple asset classes are deliberate: each has its own holiday calendar, so the missing data in the panel is genuine rather than manufactured.

| Series | Asset class | Description |
| --- | --- | --- |
| `SP500` | Equity | S&P 500 index level |
| `DEXUSEU` | FX | US dollars per euro |
| `DEXJPUS` | FX | Japanese yen per US dollar |
| `DCOILWTICO` | Commodity | WTI crude, spot |
| `DGS10` | Rates | 10 year Treasury constant maturity yield |
| `VIXCLS` | Volatility | VIX index |

To refresh with newer data, download each series and replace the file in `data/`, then delete `data/market_panel.csv` so the cache rebuilds:

```
https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500&cosd=2016-01-01
https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSEU&cosd=2016-01-01
https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXJPUS&cosd=2016-01-01
https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILWTICO&cosd=2016-01-01
https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10&cosd=2016-01-01
https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS&cosd=2016-01-01
```

## How it is built

Three class hierarchies, each an abstract base with concrete implementations. Adding a new method means writing one subclass and nothing else.

```
OutlierDetector (ABC)
├── ZScoreDetector                 full sample mean and standard deviation
├── ModifiedZScoreDetector         median and MAD, robust to contamination
├── RollingZScoreDetector          trailing window, adapts to volatility
└── EwmaVolDetector                exponentially weighted volatility

Imputer (ABC)
├── ForwardFillImputer             carry the last observation forward
├── LinearInterpolationImputer     straight line across the gap
└── PeerScaledImputer              scale by a correlated peer's return

VaRModel (ABC)
├── HistoricalSimulationVaR        empirical quantile, no distribution assumed
├── ParametricVaR                  fitted normal
└── MonteCarloVaR                  simulated from a fitted distribution
```

Two supporting classes: `DataQualityReport` scores a panel on five checks and assigns a letter grade, and `VaRBacktest` runs a rolling window backtest with the Kupiec test and the Basel traffic light.

`DefectInjector` manufactures known defects so detectors can be scored against a truth that would otherwise be unobservable.

## Contents

| Section | Content |
| --- | --- |
| 1 | Data loading, business day alignment, return construction |
| 2 | Quality scorecard across five checks with letter grading |
| 3 | Four outlier detectors sharing one interface |
| 4 | Detector benchmark: precision, recall, F1, and recall split by volatility regime |
| 5 | Three imputers, scored on accuracy and on volatility distortion |
| 6 | Three VaR models, with a normality test on the return distribution |
| 7 | The impact of dirty data on VaR, combined and then isolated by defect |
| 8 | Rolling window backtesting, Kupiec test, Basel zone, breach clustering |
| 9 | Vectorisation of the rolling quantile |
| 10 | Limitations and future work |

## Methodology notes

**Returns respect the instrument.** Price series use log returns, guarded so that a non positive price produces NaN rather than a silent error. The 10 year yield is already a rate, so its return is the absolute daily change in percentage points. Treating it as a percentage change would make the near zero period look wildly volatile.

**Rolling estimators are lagged.** The rolling and EWMA detectors both shift their volatility estimate by one day. Without that, the observation being tested contributes to the volatility it is judged against, which is look ahead bias and makes large moves look normal.

**Comparisons are like for like.** Clean, dirty and repaired portfolios are built with the same rule and restricted to the same dates, so the measured difference is data quality rather than sample size.

**Peers are selected on comparability, not just correlation.** VIX is the most correlated series in the panel by absolute value at rho of 0.73, and it is excluded from peer imputation because a volatility index moving several percent a day cannot scale an equity price moving one percent.

**Optimisation is verified, not trusted.** The vectorised rolling VaR is asserted equal to the loop implementation before the speedup is reported. It came out at 14.9 times faster with results identical to zero difference.

## Limitations

**The defects are simulated, not observed.** Real vendor errors are correlated across instruments, cluster around holidays and corporate actions, and are sometimes systematically biased rather than random.

**Detection is univariate.** If every bank stock moves 2% and one moves 9%, that single name is suspicious in a way no single series detector can see. A residual based screen against a factor model would catch it.

**Parametric and Monte Carlo both assume normality**, which the Jarque Bera test in section 6 rejects decisively. Excess kurtosis on this portfolio is 24.3, driven by March 2020. A Student t or a fitted extreme value tail would be more honest.

**VaR here is unconditional.** No EWMA or GARCH weighting, which is why the breach clustering in section 8 persists.

**The portfolio is deliberately trivial.** Equal weights, no FX translation, no position level detail. That keeps the focus on data quality, but it means the dollar figures are illustrative.

**Average Daily Trading Volume is absent** because FRED publishes no volume data. The computation is a rolling mean; the hard part is the production plumbing, which the pipeline pattern here demonstrates on prices instead.

## Future work

1. Cross sectional and factor residual outlier detection across instruments.
2. EWMA and GARCH weighted VaR, then re run the backtest to see whether the clustering disappears.
3. Expected shortfall alongside VaR, since Basel FRTB moved to ES at 97.5%.
4. A vendor comparison layer scoring the same instrument from two sources against each other, which is how a real quality process resolves a disputed print.
5. Package the classes as an installable library with a pytest suite and a scheduled run, so the scorecard becomes a daily artefact rather than a notebook output.

## References

Kupiec, P. (1995). Techniques for Verifying the Accuracy of Risk Measurement Models. *Journal of Derivatives*.

Christoffersen, P. (1998). Evaluating Interval Forecasts. *International Economic Review*.

Basel Committee on Banking Supervision (1996, revised 2019). Supervisory Framework for the Use of Backtesting.

Iglewicz, B. and Hoaglin, D. (1993). How to Detect and Handle Outliers. ASQC Quality Press.

## Stack

Python, NumPy, pandas, SciPy, Plotly.

## Disclaimer

Independent analysis on publicly available data, produced for a portfolio. Not investment advice, not a recommendation to transact, and carrying no affiliation with any financial institution. Dollar figures are illustrative of position mechanics, not forecasts.
