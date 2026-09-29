"""Focused numerical checks; run `python -m unittest -v`."""
import math
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from pricing import (black_scholes_call, plain_call, antithetic_call,
                     merton_terminal, merton_call_mc, merton_call_benchmark)

BASE = (100.0, 100.0, 0.05, 0.2, 1.0)


class PricingChecks(unittest.TestCase):
    def test_black_scholes_known_value(self):
        self.assertAlmostEqual(black_scholes_call(*BASE), 10.450583572185565, places=11)

    def test_black_scholes_against_normal_quadrature(self):
        for spot, strike, rate, vol, maturity in [(80,100,-.01,.35,.4), BASE, (140,90,.04,.1,2)]:
            threshold = (math.log(strike/spot) - (rate-.5*vol**2)*maturity)/(vol*math.sqrt(maturity))
            value = quad(lambda z: math.exp(-rate*maturity) *
                         (spot*math.exp((rate-.5*vol**2)*maturity+vol*math.sqrt(maturity)*z)-strike)*norm.pdf(z),
                         threshold, 12, epsabs=1e-9)[0]
            self.assertAlmostEqual(black_scholes_call(spot,strike,rate,vol,maturity), value, places=8)

    def test_limiting_cases(self):
        self.assertEqual(black_scholes_call(110,100,.05,.2,0), 10)
        self.assertAlmostEqual(black_scholes_call(100,100,.05,0,1), 100-100*math.exp(-.05))
        self.assertAlmostEqual(merton_call_benchmark(*BASE,0,-.05,.2), black_scholes_call(*BASE))
        self.assertAlmostEqual(merton_call_benchmark(*BASE,2,0,0), black_scholes_call(*BASE), places=9)
        self.assertAlmostEqual(merton_call_benchmark(100,100,.05,0,1,2,0,0), black_scholes_call(100,100,.05,0,1), places=9)

    def test_pair_mean_uncertainty(self):
        # Independent reconstruction from explicit two-column antithetic pairs.
        z = np.random.default_rng(7).normal(size=500)
        terminal = 100*np.exp(.03 + .2*np.column_stack((z,-z)))
        pairs = (np.exp(-.05)*np.maximum(terminal-100,0)).mean(axis=1)
        result = antithetic_call(*BASE,1000,np.random.default_rng(7))
        self.assertAlmostEqual(result.price, pairs.mean(), places=13)
        self.assertAlmostEqual(result.standard_error, pairs.std(ddof=1)/math.sqrt(500), places=13)
        self.assertEqual(result.independent_observations,500)
        self.assertEqual(result.payoff_evaluations,1000)

    def test_monte_carlo_against_benchmarks(self):
        # Fixed seeds, broad five-SE tolerance; not exact equality of random estimates.
        for method in [plain_call, antithetic_call]:
            result = method(*BASE,200_000,np.random.default_rng(2026))
            self.assertLess(abs(result.price-black_scholes_call(*BASE)),5*result.standard_error)
        for intensity, mean, std in [(.5,-.05,.2),(1,.1,.3),(.5,-.2,0)]:
            result = merton_call_mc(*BASE,300_000,intensity,mean,std,np.random.default_rng(17))
            benchmark = merton_call_benchmark(*BASE,intensity,mean,std)
            self.assertLess(abs(result.price-benchmark),5*result.standard_error)

    def test_discounted_stock_martingale(self):
        discounted = math.exp(-.05)*merton_terminal(100,.05,.2,1,400_000,.7,-.1,.3,np.random.default_rng(91))
        se = discounted.std(ddof=1)/math.sqrt(len(discounted))
        self.assertLess(abs(discounted.mean()-100),5*se)

    def test_invalid_inputs(self):
        for params in [(-1,100,.05,.2,1),(100,0,.05,.2,1),(100,100,.05,-.2,1),(100,100,float('nan'),.2,1)]:
            with self.assertRaises(ValueError): black_scholes_call(*params)
        for n in [0,1,3,5,10.5,True]:
            with self.assertRaises(ValueError): antithetic_call(*BASE,n,np.random.default_rng(0))
        with self.assertRaises(ValueError): merton_call_benchmark(*BASE,-1,0,.2)
        with self.assertRaises(ValueError): merton_call_benchmark(*BASE,1,0,-.2)


if __name__ == '__main__':
    unittest.main()
