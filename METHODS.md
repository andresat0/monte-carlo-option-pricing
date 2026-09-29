# Models and experiment design

## Conventions

`spot` and `strike` use the same currency units. `maturity` is in years; `rate` is a continuously compounded annual rate; `volatility` is annual diffusion volatility. The asset pays no dividends. All simulations are under the specified risk-neutral measure.

The estimators sample the terminal distribution directly. There is no Euler time grid and therefore no time-discretization error in these terminal-payoff experiments. Monte Carlo sampling error and model misspecification remain.

## Black–Scholes and plain Monte Carlo

For a standard normal draw Z,

$$S_T=S_0\exp\left[(r-\tfrac12\sigma^2)T+\sigma\sqrt{T}Z\right].$$

The discounted payoff is $Y=e^{-rT}(S_T-K)^+$. Its sample mean estimates the call value, and $s_Y/\sqrt{N}$ estimates its standard error. The analytic reference is $S_0\Phi(d_1)-Ke^{-rT}\Phi(d_2)$.

## Antithetic sampling

For M independent normal draws, form

$$A_j=\frac{Y(Z_j)+Y(-Z_j)}{2}.$$

The estimate is the mean of the M pair averages; its standard error is $s_A/\sqrt{M}$. A budget of N=2M payoff evaluations is compared with N independent payoffs for plain Monte Carlo. These budgets match payoff evaluations, not random-number counts or wall-clock time.

Using the sample standard deviation of all 2M correlated payoffs divided by $\sqrt{2M}$ is not the correct uncertainty calculation. The current implementation and test operate on complete pairs.

## Merton jump diffusion

Let $N_T\sim\operatorname{Poisson}(\lambda T)$ and log-jump sizes be iid $L_i\sim N(\mu_J,\delta^2)$, independent of the diffusion. Then

$$S_T=S_0\exp\left[(r-\lambda\kappa-\tfrac12\sigma^2)T+\sigma\sqrt{T}Z+\sum_{i=1}^{N_T}L_i\right],\qquad \kappa=e^{\mu_J+\delta^2/2}-1.$$

Conditional on the count n, the jump sum is normal with mean $n\mu_J$ and variance $n\delta^2$. This supplies exact terminal sampling and enforces $E[e^{-rT}S_T]=S_0$ under the model.

For the independent analytic benchmark, condition on n and sum the lognormal call expectations with Poisson weights. Write $v_n=\sigma^2T+n\delta^2$ and $m_n=\log S_0+(r-\lambda\kappa-\sigma^2/2)T+n\mu_J$. When $v_n>0$,

$$C_n=e^{-rT}\left[e^{m_n+v_n/2}\Phi\left(\frac{m_n+v_n-\log K}{\sqrt{v_n}}\right)-K\Phi\left(\frac{m_n-\log K}{\sqrt{v_n}}\right)\right].$$

The full value is $\sum_n P(N_T=n)C_n$. Zero conditional variance is evaluated as a deterministic payoff.

### Truncation

The code evaluates the stock leg using a tilted Poisson mean $\lambda T e^{\mu_J+\delta^2/2}$. Since the call payoff is nonnegative and bounded by the stock, stopping at count k leaves an omitted contribution no larger than

$$S_0 P\left[\operatorname{Poisson}(\lambda T e^{\mu_J+\delta^2/2})>k\right].$$

The default absolute tolerance is $10^{-10}$ in price units. Floating-point roundoff is separate from this bound. The benchmark rejects Poisson means above 10,000; extreme finite inputs can also exceed floating-point limits.

## Repeated experiments

The master seed is `20260928`. Each estimate receives a separate NumPy `SeedSequence` child. At each of four budgets, 200 independent estimates per method give empirical RMSE and variance relative to the analytic price. The plotted inverse-square-root curve is a scaled theoretical reference, not a fitted convergence-rate estimate.

Empirical coverage is the fraction of 200 pointwise 95% intervals containing the true model price. With 200 replications, coverage itself has sampling uncertainty (about 1.5 percentage points standard error near 95% coverage). The jump sweeps use separate random streams at each parameter value and 200,000 observations per estimate.
