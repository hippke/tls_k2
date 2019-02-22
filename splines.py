import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from sklearn.base import TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import HuberRegressor
#from sklearn.metrics import mean_squared_error
import numpy
#import scipy.signal
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip, LombScargle
from astropy.io import fits
from transitleastsquares import (
    transitleastsquares,
    cleaned_array,
    catalog_info,
    transit_mask
    )

"""



zu schützende länge: aus T14 (stellar density)
gaps: neue spline-stücke verwenden
in iterativer planeten suche: nach jeder maskierung NEU DETRENDEN!
It is commonly accepted that the first mathematical reference to splines is the 1946 paper by Schoenberg, which is probably the first place that the word "spline" is used in connection with smooth, piecewise polynomial approximation.
Schoenberg, Isaac J. (1946). "Contributions to the Problem of Approximation of Equidistant Data by Analytic Functions: Part A.—On the Problem of Smoothing or Graduation. A First Class of Analytic Approximation Formulae" (PDF). Quarterly of Applied Mathematics. 4 (2): 45–99.
Schoenberg, Isaac J. (1946). "Contributions to the Problem of Approximation of Equidistant Data by Analytic Functions: Part B.—On the Problem of Osculatory Interpolation. A Second Class of Analytic Approximation Formulae" (PDF). Quarterly of Applied Mathematics. 4 (2): 112–141.
splines were established by
Ferguson, James C. (1964). "Multivariable Curve Interpolation". Journal of the ACM. 11 (2): 221–228. doi:10.1145/321217.321225.
Ahlberg, J. Harold; Nielson, Edwin N.; Walsh, Joseph L. (1967). The Theory of Splines and Their Applications. New York: Academic Press. ISBN 0-12-044750-9.



good explanation of huber function:
http://statweb.stanford.edu/~owen/reports/hhu.pdf
s. 3f
MEDIAN
======
Python/C++ implementation: https://github.com/craffel/median-filter
https://github.com/suomela/median-filter/tree/master/src
https://github.com/suomela/mf2d
https://bitbucket.org/janto/snippets/src/tip/running_median.py?fileviewer=file-view-default
http://code.activestate.com/recipes/576930/
Ein sliding *mean* hat eine Laufzeit von O(1), ist also unabhängig von der Fensterbreite n. Pro Datenpunkt der Lichtkurve fallen zwei Rechenoperationen an. Ein Wert kommt neu ins Fenster, einer fällt raus. Der neue Durchschnitt ist trivial zu errechnen. Mit steigender Anzahl Datenpunkte x~n in der Lichtkurve ist das Verhalten linear: O(n)
Ein sliding median ist, naiv implementiert, linear abhängig von der Fensterbreite, O(n), da mit jedem Rechenschritt der Median der Punkte im Fenster neu berechnet werden. Großes Fenster, mehr Rechenarbeit. Damit wäre die Laufzeit bei mehr Datenpunkten quadratisch: O(n^2).
Ich hatte einmal erwähnt, dass es beim sliding median einen schnelleren Weg gibt, aber nicht mehr genau gewusst, welchen. Heute bin ich wieder darüber gestolpert. Das Verfahren heißt "Turlach implementation" [1] und hat eine Laufzeit von O(log n), bzw. O(n log n) für viele Werte. Beim "Turlach" wird ein Register aufgebaut, die Werte im Fenster also erstmalig sortiert. Optimal ist dabei "Heapsort". Ein neu hinzukommender Wert wird dann im Register eingefügt, der rausfallende entfernt. Die neu-Sortierung des Registers hat log-Laufzeit (Suomela 2014).
Benchmark SciPy versus Python-Turlach:
https://arxiv.org/abs/1406.1717
Eine Implementierung gibt es z.B. hier: https://github.com/suomela/median-filter
[1] Benannt nach der Implementierung von Berwin A. Turlach (1995, unpublished), Idee aus:
Optimal Median Smoothing (1995)
W. Härdle and W. Steiger
Journal of the Royal Statistical Society. Series C (Applied Statistics)
Vol. 44, No. 2 (1995), pp. 258-264
DOI: 10.2307/2986349

Benannt nach der Implementierung von Berwin A. Turlach (1995, unpublished)
"""

class BSplineFeatures(TransformerMixin):
    """Robust B-Spline regression with scikit-learn"""

    def __init__(self, knots, degree=3, periodic=False):
        self.bsplines = self.get_bspline_basis(knots, degree, periodic=periodic)
        self.nsplines = len(self.bsplines)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        nsamples, nfeatures = X.shape
        features = np.zeros((nsamples, nfeatures * self.nsplines))
        for ispline, spline in enumerate(self.bsplines):
            istart = ispline * nfeatures
            iend = (ispline + 1) * nfeatures
            features[:, istart:iend] = scipy.interpolate.splev(X, spline)
        return features

    def get_bspline_basis(self, knots, degree=3, periodic=False):
        """Get spline coefficients for each basis spline"""
        knots, coeffs, degree = scipy.interpolate.splrep(
            knots, np.zeros(len(knots)), k=degree, per=periodic)
        ncoeffs = len(coeffs)
        bsplines = []
        for ispline in range(len(knots)):
            coeffs = [1.0 if ispl == ispline else 0.0 for ispl in range(ncoeffs)]
            bsplines.append((knots, coeffs, degree))
        return bsplines


def spline_detrender(t, y, no_knots):
    model = make_pipeline(
        BSplineFeatures(
            numpy.linspace(numpy.min(t), numpy.max(t), no_knots)
        ),
        HuberRegressor(),
    ).fit(t[:, numpy.newaxis], y)
    trend = model.predict(t[:, None])
    return y / trend, trend



def load_file(filename):
    """Loads a TESS *spoc* FITS file and returns cleaned arrays TIME, PDCSAP_FLUX"""
    print(filename)
    hdu = fits.open(filename)
    t = hdu[1].data['TIME']
    y = hdu[1].data['PDCSAP_FLUX']  # values with non-zero quality are nan or zero'ed 
    #y = hdu[1].data['FLUX']  # values with non-zero quality are nan or zero'ed 
    #t = t[64:]
    #y = y[64:]
    #t = t[:-64]
    #y = y[:-64]

    t, y = cleaned_array(t, y)  # remove invalid values such as nan, inf, non, negative
    print(
        'Time series from', format(min(t), '.1f'), 
        'to', format(max(t), '.1f'), 
        'with a duration of', format(max(t)-min(t), '.1f'), 'days')
    return t, y, hdu[0].header['TICID']  # filename[36:45]  "trappist"

#hlsp_everest_k2_llc_246599598-c13_kepler_v2.0_lc


def main():
    # test higher:  alles oder ohne low outlier?        using "all, 3x":
    #               0.032   235037      mit ?           
    #               0.1     140068425   mit             
    #               0.1     14901       ohne low        slight over
    #               0.1     100100827   ohne low        slight under, ok
    #               0.1     7079        mit?            
    #               0.1     2760710     mit             


    # bad           0.01    1546


    path = 'P:/P/Dok/tess_alarm/'
    #path = 'P:/P/Dok/k2fits/'
    #path = '/home/michael/Downloads/tess-data-alerts/'

    os.chdir(path)
    for file in glob.glob("*.fits"):
        #if "ktwo200164267-c12_llc" in file:
        if 1==1:
            converged = False
            
            t, y, TIC_ID = load_file(path + file)
            y = y / numpy.median(y)
            y = sigma_clip(y, sigma_lower=20, sigma_upper=3)
            t, y = cleaned_array(t, y)
            y_clipped = y.copy()

            #no_knots = 30
            duration = max(t)-min(t)
            no_knots = int(duration / 0.75)
            print('no_knots', no_knots)
            #if no_knots > 50:
            #    
            #    print('Final run')
            #no_knots = no_knots#*3

            while converged == False:
                y_detrended, trend = spline_detrender(t, y, no_knots)
                #y_detrended = sigma_clip(y_detrended, sigma_lower=10, sigma_upper=3)
                #t, y_detrended = cleaned_array(t, y_detrended)

                std = numpy.std(y_detrended)

                plt.figure()
                fig, axes = plt.subplots(5, 1, figsize=(6, 10))
                ax = axes[0]
                #ax.plot(X[:, 0]-t.min(), y_clipped, 'o', ms=1, c='black', alpha=0.5)
                ax.plot(t-t.min(), y_clipped, 'o', ms=1, c='black', alpha=0.5)
                ax.plot(t-t.min(), trend, '--', lw=1, ms=1, color='red')
                ax.plot(t-t.min(), trend+std, '--', lw=1, ms=1, color='orange')
                ax.plot(t-t.min(), trend-std, '--', lw=1, ms=1, color='orange')

                ax.set_ylabel("Raw flux")
                ax = axes[1]
                ax.plot(t-t.min(), y_detrended, 'o', ms=1, c='black', alpha=0.5)
                ax.set_xlabel("Time (days)")
                ax.set_ylabel("Normalized flux")

                ax = axes[2]

                core_t = []
                core_y = []
                upper_cut = 1+numpy.std(y_detrended)
                lower_cut = 1-numpy.std(y_detrended)
                for i in range(len(y_detrended)):
                    if y_detrended[i] < upper_cut and y_detrended[i] > lower_cut:
                        core_t.append(t[i])
                        core_y.append(y_detrended[i])

                ax.plot(core_t-min(core_t), core_y, 'o', ms=1, c='black', alpha=0.5)

                ax = axes[3]
                period_grid = numpy.linspace(0.01, max(t)-min(t), 10000)
                frequency_grid = 1 / period_grid

                ls = LombScargle(core_t-min(core_t), core_y)
                power = ls.power(frequency_grid)
                #print(max(power))

                if max(power) < 0.02:
                    converged = True
                    print('Fine, quitting. max(power)=', max(power))
                else:
                    print('no_knots old', no_knots)
                    print('max(power)=', max(power))
                    no_knots = int(no_knots * 1.5)
                    print('no_knots new', no_knots)

                if no_knots > 100:
                    converged = True
                    print('Quitting because no_knows > 100')

                ax.plot(1/frequency_grid, power, color='black')


                #print(ls.false_alarm_probability(power.max()))
                #if max(power) > 0.05:  #ls.false_alarm_probability(power.max()) < 0.05:
                #    color='red'
                #else:
                #    color='black'
                #text = "{:10.4f}".format(ls.false_alarm_probability(power.max()))
                ax.text(5, 0.04, str(no_knots), color='black')
                ax.plot((0, 20), (0.05, 0.05), color='red', linestyle='dashed')
                ax.set_ylim(0, 0.06)
                ax.set_xlabel("Period (days)")
                ax.set_ylabel("Power")
                #plt.show()
                
                ax = axes[4]
                #from scipy.stats import norm
                #x_d = np.linspace(-4, 8, 1000)
                #density = sum(norm(xi).pdf(x_d) for xi in y_detrended)
                #plt.fill_between(x_d, density, alpha=0.5)
                #plt.plot(y_detrended, np.full_like(y_detrended, -0.1), '|k', markeredgewidth=1)
                #bins = 50
                
                from scipy.stats import norm
                import matplotlib.mlab as mlab
                #(mu, sigma) = norm.fit(y_detrended)
                n, bins, patches = ax.hist(y_detrended, int(len(t)/200))
                plt.plot(
                    (numpy.mean(y_detrended), numpy.mean(y_detrended)),
                    (0, max(n)),
                    color='orange'
                    )
                plt.plot(
                    (numpy.median(y_detrended), numpy.median(y_detrended)),
                    (0, max(n)),
                    linestyle='dashed',
                    color='red'
                    )
                #ax.plot(bins, mlab.normpdf(bins, mu, sigma), 'r--', linewidth=2)
                
                plt.savefig(str(TIC_ID) + ".png", bbox_inches='tight')
                plt.clf()
                plt.close('all')



if __name__ == "__main__":
    main()
