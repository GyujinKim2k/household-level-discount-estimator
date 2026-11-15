# PSID Measurement-Error Model for Simulated Panels — Methodology and Provenance

This document explains how the measurement-error model in `psid_noise.py` was derived for the four PSID variables used in the simulator: `income`, `consumption`, `liquid_assets`, and `illiquid_assets`.

For each variable it records:

- which source each number came from,
- what calculation turns those numbers into the noise parameters,
- which choices are judgment calls rather than sourced estimates.

Every number carries one of three tags:

- **[S] Sourced**: taken directly from the cited paper.
- **[D] Derived**: computed here from sourced numbers; the arithmetic is shown.
- **[A] Assumption**: a judgment call, with the reasoning given. Treat these as the first things to revisit or put wide priors on.

---

## 0. Summary

All errors are multiplicative on the magnitude:

```
x_obs = sign(x*) * |x*| * exp(u)
```

so zeros stay zero and signs are preserved.

| Variable | Error model | Default parameters | Prior for SBI |
|---|---|---|---|
| income | `u = b·(log y* − ref_t) + v`, v ~ AR(1) across waves | σ_v = 0.11, b = −0.10, ρ = 0.15 | σ_v ∈ U(0.08, 0.14), b ∈ U(−0.20, 0), ρ ∈ U(0, 0.3) |
| consumption | `u ~ N(0, σ_c²)`, iid across waves | σ_c = 0.18 | σ_c ∈ U(0.12, 0.25) |
| liquid_assets | `u ~ σ·t_ν` (unit-variance t, clipped at ±3), iid across waves | σ calibrated so noise = 50% of Var(Δ log A) | share ∈ U(0.3, 0.7), ν ∈ U(3, 8) |
| illiquid_assets | same as liquid, or component-level (house, mortgage, other) then netted | share = 0.5, ν = 4 | same as liquid |

The four variables' errors are drawn independently of each other. For simulation-based inference, draw the error parameters from the priors on every simulation (`sample_noise_params`) and marginalize over them; do not fix point values.

---

## 1. Primary source for labor-income error: the PSID Validation Study (PSIDVS)

### 1.1 What the PSIDVS is [S]

The only direct validation of the PSID instrument against administrative "truth":

- Workers at one large manufacturing firm were interviewed with a PSID-style questionnaire, and their answers were compared to the firm's payroll records.
- Wave 1 (summer 1983) interviewed 418 workers. Wave 2 (1987) re-interviewed 341 of them plus 151 newly drawn hourly workers.
- All hourly workers were unionized. Earnings dispersion inside the firm is much more compressed than in general-population samples.

Sources:
- Bound, Brown, Duncan & Rodgers (1994), *Journal of Labor Economics* 12(3), 345–368.
- Table values below are taken from the working-paper version, NBER WP 2884 (1989; published in Hartog, Ridder & Theeuwes eds., 1990).
- Cross-check: Gottschalk & Huynh (IZA DP 2327) cite JOLE Table 1 noise ratios of .151 (1982) and .302 (1986), matching the WP values .150 and .303.
- Earlier wave-1 analysis: Duncan & Hill (1985), *JOLE* 3(4), 508–532. They found error variance equal to about 30% of payroll-earnings variance in 1981 and 15% in 1982 [S, via Gottschalk & Huynh].

### 1.2 Numbers taken from NBER WP 2884 [S]

**Table 1 — log annual earnings, salaried + hourly men.**

Column definitions:
- `ratio` = σ_u²/(σ_x²+σ_u²)
- `b_vY` = regression of error on the true value (the mean-reversion slope)
- `r_u` = correlation of errors over time
- `r_X` = correlation of true values over time

| Sample | ratio | b_vY | r_u | r_X |
|---|---|---|---|---|
| PSIDVS 1986 (N=422) | .303 | −.172 | – | – |
| PSIDVS 1982 (N=320) | .150 | −.104 | – | – |
| PSIDVS 4-yr Δ (N=206) | .294 | −.214 | **.073** | .452 |
| CPS–SSA 1977 | .221 | −.138 | – | – |
| CPS–SSA 1976 | .210 | −.190 | – | – |
| CPS–SSA 1-yr Δ | .322 | −.238 | **.372** | .635 |

**Table 2 — hourly workers.**

| Variable | ratio 1986 | ratio 1982 | ratio 4-yr Δ | b_vY 1986 | b_vY 1982 |
|---|---|---|---|---|---|
| ln earnings | .276 | .241 | .260 | −.192 | −.208 |
| ln hours | .366 | .284 | .312 | −.244 | −.226 |
| ln earnings/hour | .670 | .693 | .821 | .021 | −.301 |

**Other facts [S]:**
- Correlation between the errors in ln earnings and ln hours: .407 in 1986 and −.169 in 1982. 1982 was a high-unemployment year at the firm.
- The authors state that the single-firm sample's restricted true variance likely makes measurement error look more serious than it would in a broader sample.
- Only about one-third of unemployment spells in company records were reported in the interview.
- Tables 4–6 give, for hourly workers in 1986, the record SD, report SD, and record–report correlation of log annual measures:
  - earnings: SD_record .181, SD_report .194, corr .806
  - hours: .136, .150, .640
  - hourly wage: .089, .154, .558
- Appendix Table 1: SD of ln annual earnings ≈ 0.22–0.28 in the PSIDVS vs ≈ 0.72 in the national PSID male-head samples.

### 1.3 Absolute error SD backed out from Tables 4–6 [D]

For report R = X + e, `Var(e) = Var(R) + Var(X) − 2·Cov(R, X)`.

| Variable | Calculation | Var(e) | SD(e) |
|---|---|---|---|
| ln earnings | .194² + .181² − 2(.806)(.181)(.194) | 0.0138 | **0.117** |
| ln hours | .150² + .136² − 2(.640)(.136)(.150) | 0.0149 | **0.122** |
| ln hourly wage | .154² + .089² − 2(.558)(.089)(.154) | 0.0163 | **0.128** |

Consistency check: wage = earnings / hours, so Var(e_w) = Var(e_y) + Var(e_h) − 2·corr·SD_y·SD_h = 0.0138 + 0.0149 − 2(.407)(.117)(.122) = 0.0171. This matches the directly computed 0.0163.

---

## 2. `income`

### 2.1 Why the noise ratio cannot be transported directly

The PSIDVS noise *ratios* (0.15–0.30) are inflated by the compressed true variance inside one firm [S, §1.2]. What is plausibly portable is:

- (a) the respondent-level **classical reporting noise** σ_v, and
- (b) a **proportional shrinkage** slope b.

These are then applied to the simulator's own, much wider, true income distribution.

### 2.2 Classical part σ_v = 0.11 [D]

Model: `u = b·(x* − mean) + v`, so `Var(u) = b²·Var(x*) + σ_v²`.

PSIDVS hourly workers, 1986:
- Var(u) = 0.0138 (§1.3)
- Var(x*) = 0.181² = 0.0328
- b = −0.192 (Table 2)

```
σ_v² = 0.0138 − (0.192²)(0.0328) = 0.0138 − 0.0012 = 0.0126  →  σ_v ≈ 0.112
```

Using the all-worker slope b = −0.172 gives σ_v ≈ 0.113. Inside the PSIDVS almost all of the error variance is classical, so **σ_v = 0.11**.

### 2.3 Mean-reversion slope b = −0.10 [D + A]

**Sourced slopes [S]:** −0.10 to −0.21 (PSIDVS) and −0.14 to −0.19 (CPS–SSA).

**Representative-sample target [S]:**
- Blundell, Pistaferri & Saporta-Eksten (BPS; NBER WP 18445, 2012; AER 2016) adopt Bound, Brown & Mathiowetz (2001) estimates for the PSID.
- BBM extrapolate the validation evidence to representative samples. BPS use their shares of measurement error of 0.04 for log earnings, 0.072–0.162 for log hourly wages (BPS pick 0.13), and 0.23 for log hours.

**Calculation [D].** Let PSID-like Var(log y*) ≈ 0.55 (PSID SD ≈ 0.72 → Var ≈ 0.52; the test panel has 0.49–0.64). The observed-variance share of error is:

```
share = [b²·0.55 + 0.0121] / [(1+b)²·0.55 + 0.0121]
```

| b | share | Note |
|---|---|---|
| −0.17 | ≈ 7.3% | Twice the BBM/BPS target |
| −0.15 (σ_v = 0.10) | 5.4% | Simulated; corr(error, truth) = −0.75 |
| −0.10 | ≈ 3.8% | Matches the 0.04 target |

**[A]** b = −0.10 is chosen to reconcile the PSIDVS classical noise with the BBM/BPS representative-sample share. It is the least certain income parameter.

- Transporting a slope estimated in a compressed sample to a dispersed one makes the mean-reversion term dominate the error.
- With b = −0.10, corr(error, truth) is still −0.57 in the test panel.
- Prior: U(−0.20, 0). **Check posterior sensitivity to b.**

### 2.4 Serial correlation of the classical part ρ = 0.15 [D]

PSID has been biennial since 1997, so a 2-year error correlation is needed. Sourced values [S]: 1-year 0.372 (CPS–SSA) and 4-year 0.073 (PSIDVS).

- Geometric interpolation to 2 years: √(0.372 × 0.073) = **0.165**.
- AR(1) extrapolation from the 1-year value: 0.372² = **0.138**.
- → ρ = 0.15. Prior U(0, 0.3).

Implementation: z_t = ρ·z_{t−1} + √(1−ρ²)·ε_t, with v = σ_v·z (stationary variance σ_v²).

### 2.5 Implied properties [D, verified in the synthetic test]

Var(Δu) = b²·Var(Δ log y*) + 2σ_v²(1−ρ).

On a synthetic PSID-like panel (N = 20,000, T = 6 biennial waves, Var(log y*) 0.49–0.64):

| Metric | Value |
|---|---|
| error SD | 0.133 |
| error share of level variance | 3.8% |
| error share of change variance | 17.3% |
| corr(error, truth) | −0.57 |

The change share is below the PSIDVS/CPS values (0.29–0.32) because the simulated true changes are larger than in those samples.

### 2.6 Mean-reversion reference point [S + A]

- The default reference is the wave's cross-sectional mean of log y*, as in Bound & Krueger style mean reversion.
- Pischke (1995) used the PSIDVS to assess whether *transitory* earnings variation is under-reported [S, via Gottschalk & Huynh].
- **[A]** If the simulator has an explicit permanent component, pass it as `income_log_ref`. Errors then shrink transitory deviations instead of cross-sectional ones. This may matter for a savings model whose key inputs are transitory shocks.

### 2.7 Caveats that affect inference

1. **Identification.** Blundell, Pistaferri & Preston (2008) show that income measurement-error variance cannot be separated from transitory-shock variance using income moments [S]. If the simulator's transitory-shock variance is a target parameter, σ_v must carry a tight, externally informed prior, as BPS do by following Meghir & Pistaferri (2004).
2. **Transfers (not implemented).** If `income` includes government transfers, symmetric noise is not enough. Meyer, Mok & Sullivan report PSID dollar reporting rates of about 70% for AFDC/TANF, Unemployment Insurance and Food Stamps, typically under half for SSI, and even lower for Workers' Compensation [S].
   - **TODO:** if the simulator separates transfers, apply a downward bias (non-reporting) to that component.
3. **Timing.** PSID income questions refer to the previous calendar year; wealth is measured at the interview date. **TODO:** align periods in the simulator.
4. **Wages and hours.** Not among the four variables. If added, generate earnings and hours errors (hours SD ≈ 0.12, b ≈ −0.23, error corr with earnings ≈ 0.4) and *derive* the wage. This reproduces division bias automatically.

---

## 3. `consumption`

### 3.1 Sources and numbers

**Blundell, Pistaferri & Preston (2008, AER) [S]** — PSID 1978–1992, nondurable consumption imputed from food via a CEX demand system.
- Measurement error in log consumption is identified by `Var(e_c) = −Cov(Δc_t, Δc_{t+1})` under the martingale property of consumption (their fn. 17).
- Estimated imputation/measurement-error variance: **0.05–0.10** (SD 0.22–0.32).
- Errors are modeled as classical multiplicative and independent of income errors.

BPP Table IV (annual data) and the implied noise share of Var(Δc) = 2|autocov| / var:

| Year | var(Δc) | cov(Δc_{t+1}, Δc_t) | noise share [D] |
|---|---|---|---|
| 1980 | .1275 | −.0526 | 0.83 |
| 1981 | .1197 | −.0573 | 0.96 |
| 1982 | .1322 | −.0641 | 0.97 |
| 1983 | .1532 | −.0691 | 0.90 |
| 1985 | .2019 | −.0872 | 0.86 |
| 1990 | .1751 | −.0602 | 0.69 |
| 1991 | .1646 | −.0696 | 0.85 |

**Alan & Browning (2010, *REStud* 77(4))** estimate that roughly 80% of PSID (food) consumption-growth variance is measurement error. *Secondary citation (arXiv 1810.05287, fn. 25); not verified in the original.*

**BPS (2016)** — 1999–2009 PSID, directly measured nondurables + services (biennial).
- SD of Δ log C is **0.321** (Table 1, Panel B) [S] → Var(Δc) = 0.103 [D].
- They use the same autocovariance identification and treat it as an upper bound under nonseparable preferences [S].
- Their numeric estimate of Var(e_c) was not retrieved.

### 3.2 Calculation of σ_c = 0.18 [D + A]

- **[A]** The post-1999 data are directly measured, so the imputation-error component in BPP is absent. Assume the noise share of biennial Δc lies between 50% and 80%, below BPP's 69–97% and around the Alan–Browning 80%.
- With iid errors, Var(Δc_obs) = Var(Δc*) + 2σ_c², so 2σ_c² = share × 0.103:

```
share 0.5 → σ_c² = 0.026 → σ_c = 0.16
share 0.8 → σ_c² = 0.041 → σ_c = 0.20
```

→ **σ_c = 0.18**. Prior U(0.12, 0.25): the upper end approaches BPP's imputed-data level (0.22–0.32).

### 3.3 Structure [S + A]

- Classical, no mean reversion, iid across waves, independent of income errors. This follows BPP's assumptions.
- Weak supporting evidence: Abe (2008), Japanese diary-vs-recall data, finds no negative correlation between consumption error and true consumption.
- **[A]** Imputed rent: if `consumption` includes owners' imputed rent built from self-reported house value (BPS use 6% of house value [S]), compute it from the *noisy* house value so the two errors correlate as they do in PSID.

### 3.4 Verification [D]

- Synthetic test: −Cov(Δc_t, Δc_{t+1}) = 0.0351 vs σ_c² = 0.0324 — the BPP identification moment is reproduced.
- **TODO:** compute −Cov(Δc_t, Δc_{t+2}) (consecutive biennial waves) on the real PSID extract and set σ_c (or center its prior) on that value.

---

## 4. `liquid_assets` and `illiquid_assets`

### 4.1 Evidence (no PSID validation study exists for wealth)

**Pfeffer, Schoeni, Kennickell & Andreski (2016), *J. Econ. Soc. Meas.* 41(2) [S]:**
- PSID vs SCF: median net worth 6% higher in the SCF; mean 31% higher, with the gap concentrated in the top 1–2% of households.
- PSID measures wealth with many fewer questions than the SCF.
- → Level differences are mainly top-tail coverage, not respondent noise. They are not modeled as noise here.

**Hill (2006), *J. Applied Econometrics* 21(6) [S]:**
- HRS 1998–2000. Respondents with large unexplained asset changes were called back and asked to reconcile; 1,255 households were reconciled.
- As a result, the variance of measured change in net worth for all 11,583 households was **cut in half**.
- Substantial noise remained after reconciliation.

**Juster, Smith & Stafford (1999) and Bosworth & Smart (2009) [S]:**
- Saving measured as wealth differences is much more affected by measurement error than wealth levels.
- PSID/HRS saving results are strongly influenced by measurement error.

### 4.2 Calibration of the noise scale [D + A]

**Target [A]:** errors account for **50%** of Var(Δ log A) among holders. Reasoning:
- In the HRS, whose asset module is long and of high quality, removing a subset of errors alone halved change variance, and noise remained.
- PSID's shorter module is unlikely to be cleaner.
- Prior U(0.3, 0.7).

**Independence across waves [A]:** the HRS evidence is about spurious wave-to-wave jumps that respondents correct when shown prior answers. This indicates transient rather than persistent errors.

**Derivation [D].** With independent errors, Var(Δ log A_obs) = Var(Δ log A*) + 2σ², and share s = 2σ² / Var(Δ log A_obs). So:

```
from observed PSID data:   σ² = s · Var(Δ log A_obs) / 2          (observed=True)
from simulated truth:      σ² = s / (1 − s) · Var(Δ log A*) / 2
```

- Variances are computed as `(IQR / 1.349)²` over wave-to-wave pairs where both waves are positive, because Δ log wealth is fat-tailed.
- `calibrate_asset_sigma` raises an error if there are fewer than 50 such pairs.

### 4.3 Distribution and zeros [D + A]

- **[A]** Student-t with ν = 4 (prior U(3, 8)), rescaled to unit variance and clipped at |u| ≤ 3 (≈ 20×).
  - Reason: Hill's halving came from about 11% of households (1,255 / 11,583) with large discrepancies, so the error distribution is heavy-tailed rather than Gaussian.
- **[A]** Multiplicative noise preserves zeros: holding vs not holding is assumed correctly reported. This keeps the share of zero-liquid-wealth (hand-to-mouth) households intact.
  - Pfeffer & Griffin report item nonresponse below 5% per asset component in the PSID [S].
- Net positions that can be negative keep their sign; the error scales the magnitude.

### 4.4 Illiquid assets by component (optional, recommended if housing is tracked) [S + A]

**House value [S]:**
- Kiel & Zabel (1999): the average owner overvalues their house by 5.1%; recent buyers report values 8.4% above sale prices.
- The literature range of overestimation is 3–16%; two studies find about 2% underestimation.
- Robins & West (1977): owners' errors about 7%.
- A PSID-based study (Boston Fed WP 25-13) finds average misestimation close to zero, but large dispersion: about $60k for one SD.

**Parameter choices [A]:**
- `sigma_house = 0.10`, between the 7% owner error and the wider PSID dispersion, which also contains house-specific deviation from ZIP-level indices.
- `bias_house = 0` by default; set 0–0.05 to mimic overvaluation.
- `sigma_mortgage = 0.05`. Mortgage balances are assumed well reported; this is a judgment, not sourced.
- `sigma_other = 0.30` with t-tails for business, other real estate, vehicles and retirement accounts. This is a judgment.

**Leverage amplification [D]:** equity E = H − M, so SD(error_E / E) ≈ σ_H · H / (H − M). At 80% LTV, a 10% house-value error becomes about 50% of equity. An aggregate-level error on `illiquid_assets` cannot reproduce this, which is why component-level noise is preferred when available.

### 4.5 Verification [D]

Synthetic test (20% zero liquid balances, 30% zero illiquid):

| Variable | σ calibrated | error share of Δ variance | zeros preserved |
|---|---|---|---|
| liquid_assets | 0.461 | 0.485 | yes |
| illiquid_assets | 0.245 | 0.494 | yes |

The σ values depend entirely on the true Δ dispersion in the simulator. Recalibrate on the real PSID extract with `observed=True`.

---

## 5. Module API (`psid_noise.py`)

```python
from psid_noise import (NoiseParams, sample_noise_params, add_psid_noise,
                        calibrate_asset_sigma, illiquid_from_components, noise_report)

rng = np.random.default_rng(seed)
p = sample_noise_params(rng)              # or NoiseParams() for defaults
out = add_psid_noise(income, consumption, liquid, illiquid, params=p, rng=rng,
                     income_log_ref=None) # optional (N,T) log permanent income
out["income"], out["consumption"], out["liquid_assets"], out["illiquid_assets"]
out["params"]                             # parameters used, incl. calibrated asset sigmas
noise_report(true, observed, name="income")   # diagnostics
```

- All panels are shaped (N, T), in levels, one column per biennial wave.
- Errors are independent across the four variables.
- To bypass asset calibration, set `NoiseParams(sigma_liq=..., sigma_ill=...)`.

## 6. Open items for integration

1. Compute −Cov(Δc_t, Δc_{t+2}) on the real PSID extract and center the σ_c prior there (§3.4).
2. Calibrate the asset scales from observed PSID Δ log liquid/illiquid with `observed=True` (§4.2).
3. Check posterior sensitivity to the income mean-reversion slope b (§2.3), and consider `income_log_ref` = permanent component (§2.6).
4. If `income` includes transfers, add downward-biased non-reporting on that component (§2.7).
5. Align income timing (previous calendar year) with wealth (interview date) (§2.7).
6. If housing is simulated separately, switch to `illiquid_from_components` (§4.4).

## 7. Reading notes

- BPP (2008) and BPS (2012 WP / 2016 AER) PDFs were read up to roughly their first 100k characters, which cover the data, measurement-error, and main-results sections.
- Bound, Brown & Mathiowetz (2001) was not read directly. Their extrapolated shares are taken as reported by BPS.
- The PSIDVS numbers come from the NBER WP 2884 version. JOLE 1994 values cross-check for the Table 1 noise ratios.

## References

- Alan, S. & Browning, M. (2010). Estimating intertemporal allocation parameters using synthetic residual estimation. *Review of Economic Studies* 77(4), 1231–1261. https://econpapers.repec.org/RePEc:oup:restud:v:77:y:2010:i:4:p:1231-1261
- Blundell, R., Pistaferri, L. & Preston, I. (2008). Consumption inequality and partial insurance. *American Economic Review* 98(5). https://www.homepages.ucl.ac.uk/~uctp39a/aer_final.pdf
- Blundell, R., Pistaferri, L. & Saporta-Eksten, I. (2016). Consumption inequality and family labor supply. *American Economic Review* 106(2), 387–435. https://web.stanford.edu/~pista/BPS_final.pdf ; WP version: https://economics.yale.edu/sites/default/files/blundell-121024.pdf
- Bosworth, B. & Smart, R. (2009). Evaluating micro-survey estimates of wealth and saving. CRR WP 2009-4. https://crr.bc.edu/wp-content/uploads/2009/02/wp_2009-4.pdf
- Bound, J., Brown, C., Duncan, G. & Rodgers, W. (1994). Evidence on the validity of cross-sectional and longitudinal labor market data. *Journal of Labor Economics* 12(3), 345–368. WP version: https://www.nber.org/papers/w2884.pdf
- Bound, J., Brown, C. & Mathiowetz, N. (2001). Measurement error in survey data. *Handbook of Econometrics* Vol. 5, Ch. 59, 3705–3843. https://ideas.repec.org/h/eee/ecochp/5-59.html
- Duncan, G. & Hill, D. (1985). An investigation of the extent and consequences of measurement error in labor-economic survey data. *Journal of Labor Economics* 3(4), 508–532.
- Gottschalk, P. & Huynh, M. (2006). Are earnings inequality and mobility overstated? IZA DP 2327. https://repec.iza.org/dp2327.pdf
- Hill, D. (2006). Wealth dynamics: reducing noise in panel data. *Journal of Applied Econometrics* 21(6), 845–860. https://econpapers.repec.org/RePEc:wly:japmet:v:21:y:2006:i:6:p:845-860
- Juster, F. T., Smith, J. & Stafford, F. (1999). The measurement and structure of household wealth. *Labour Economics* 6(2). https://econpapers.repec.org/RePEc:wpa:wuwpla:0402009
- Kiel, K. & Zabel, J. (1999). The accuracy of owner-provided house values: the 1978–1991 American Housing Survey. *Real Estate Economics* 27(2), 263–298. https://econpapers.repec.org/RePEc:hcx:wpaper:9906
- Meghir, C. & Pistaferri, L. (2004). Income variance dynamics and heterogeneity. *Econometrica* 72(1), 1–32.
- Meyer, B., Mok, W. & Sullivan, J. (2015). Household surveys in crisis. *Journal of Economic Perspectives* 29(4), 199–226; transfer under-reporting summary: https://www.nber.org/sites/default/files/2023-06/orrc08-12.executive_summary.pdf
- Pfeffer, F., Schoeni, R., Kennickell, A. & Andreski, P. (2016). Measuring wealth and wealth inequality: comparing two U.S. surveys. *Journal of Economic and Social Measurement* 41(2), 103–120. https://pmc.ncbi.nlm.nih.gov/articles/PMC5145259
- Pfeffer, F. & Griffin, J. (2017). Determinants of wealth fluctuation. *methods, data, analyses* 11(1). https://core.ac.uk/download/201756404.pdf
- Pischke, J.-S. (1995). Measurement error and earnings dynamics: some estimates from the PSID Validation Study. *Journal of Business & Economic Statistics* 13(3).
- House value misestimation in the PSID: Federal Reserve Bank of Boston WP 25-13. https://www.bostonfed.org/-/media/Documents/Workingpapers/PDF/2025/WP2513.pdf
