# Asset Return Model Comparison: Geometric Brownian Motion vs. Deep Learning
Geometric Brownian Motion (GBM) assumes returns are normally distributed and independent. Neither holds empirically. This project quantifies how large that gap is and tests whether deep learning can close it.
### Summary
- **Data**: S&P-500 daily returns, 2020 - 2026, sourced via yfinance
- **Part 1**: Empirical analysis of return distributions and volatility structure: we reproduce several stylized facts documented in *Empirical properties of asset returns: stylized facts and statistical issues* (Rama Cont, 2001) for our data set
- **Part 2**: GBM calibrated to historical parameters; systematic comparison reveals failure to reproduce heavy tails and volatility clustering
- **Part 3**: Two neural network architectures trained to capture the statistical structure (volatility clustering) GBM misses
## Key Findings
- The log returns of the S&P-500 data shows significant kurtosis. Directly overlaying the distribution of empirical data with the distribution of the log returns of GBM paths calibrated with the same mean and variance reveals the GBM significantly underestimate the likelyhood of extreme movements (i.e tiny amplitude and large amplitude movements) while overestimating mid-sized movements
- The empirical data also shows significant time variance of volatility as well as volatility clustering. The squared returns show significant autocorrelation with a clear linear decay structure. 
- A GBM model with the same mean and variance, while able to replicate the average amplitude of volotility fluctuations is unable to replicate the same time variance structure in the volatility and shows no volatility clustering. The GBM simulation shows no significant autocorrelation either in log returns or squared log returns. The iid Gaussian increment assumption of the GBM is inconsistent with volatility clustering. 
## Consequences

## Example Results

### Heavy-tailed Return Distribution
![Heavy-tailed S&P-500 Distribution](images/Fat-tailed_S&P500Distribution.png)
### Rolling Volatility: Real Data vs GBM
![Rolling Volatility](images/rolling_volatility_comparison.png)
### Autocorrelation of Squared Returns in Historical Data, Absent in GBM
![Autocorrelation of Squared Returns](images/SP500vsGBMSquaredReturnsAutocorrelation.png)

## Repository Structure
'''text
hybrid_project          
├───data/
├───images/
├───notebooks
│       └───analysis.ipynb
├───src
│   ├───__init__.py
│   ├───data_loader.py
│   ├───features.py
│   ├───gbm.py
│   └───models.py
└───README.md



