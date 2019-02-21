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

7.  
Robust (outlier-resistant) means. The uncertainty on a sample median is $\sqrt{\pi /2}\approx 1.253$ times larger than the uncertainty on a sample mean, so in principle, by using robust means instead of medians in our input representation, we could boost the S/N the model sees by up to 25%.
https://iopscience.iop.org/article/10.3847/1538-3881/aa9e09
https://en.wikipedia.org/wiki/Median#Sampling_distribution
https://stats.stackexchange.com/questions/353347/representing-the-uncertainty-of-the-median-in-a-clean-way
https://newonlinecourses.science.psu.edu/stat414/node/316/
https://physics.stackexchange.com/questions/292871/uncertainty-propagation-upon-taking-the-median
http://mathworld.wolfram.com/StatisticalMedian.html

Optimal Median Smoothing (1995)
W. Härdle and W. Steiger
Journal of the Royal Statistical Society. Series C (Applied Statistics)
Vol. 44, No. 2 (1995), pp. 258-264
DOI: 10.2307/2986349

Benannt nach der Implementierung von Berwin A. Turlach (1995, unpublished)


Although the moving average function gives the best noise reduction it is unsuitable for smoothing data which has curvature over m points. A quadratic filter function is unsuitable for getting a derivative of a data curve with an inflection point because a quadratic polynomial does not have one. The optimal choice of polynomial order and number of convolution coefficients will be a compromise between noise reduction and distortion.[27]




zu schützende länge: aus T14 (stellar density)

gaps: neue spline-stücke verwenden

in iterativer planeten suche: nach jeder maskierung NEU DETRENDEN!

It is commonly accepted that the first mathematical reference to splines is the 1946 paper by Schoenberg, which is probably the first place that the word "spline" is used in connection with smooth, piecewise polynomial approximation.
Schoenberg, Isaac J. (1946). "Contributions to the Problem of Approximation of Equidistant Data by Analytic Functions: Part A.—On the Problem of Smoothing or Graduation. A First Class of Analytic Approximation Formulae" (PDF). Quarterly of Applied Mathematics. 4 (2): 45–99.
Schoenberg, Isaac J. (1946). "Contributions to the Problem of Approximation of Equidistant Data by Analytic Functions: Part B.—On the Problem of Osculatory Interpolation. A Second Class of Analytic Approximation Formulae" (PDF). Quarterly of Applied Mathematics. 4 (2): 112–141.

splines were established by
Ferguson, James C. (1964). "Multivariable Curve Interpolation". Journal of the ACM. 11 (2): 221–228. doi:10.1145/321217.321225.
Ahlberg, J. Harold; Nielson, Edwin N.; Walsh, Joseph L. (1967). The Theory of Splines and Their Applications. New York: Academic Press. ISBN 0-12-044750-9.

Univariate B-Splines, i.e. B-Splines where the knot positions lie in a single dimension

Cardinal B-spline
A cardinal B-spline has a constant separation, h, between knots. 

- discontinuity threshold. make custom knot locations. better: use segments
- Why not median?
    - not efficient (paper 20.02. gelesen) compared to robust mean
    - robust mean is asymptotically efficient (95% with 1.35 at Huber)
    - It is advised to set the parameter epsilon to 1.35 to achieve 95% statistical efficiency.
    - No problem toward the edges (median: wrap-around/shrink/constant)
    - for periodic signals with a period shorter than the median width, it will not fit well
1. Technically speaking, we do not NEED a normal distribution for regression analysis. 
2. We simply require a BLUE Estimator (Best Linear Unbiased Estimator). 
3. However, outliers can significantly change the shape of our distribution, and hence our overall results
==> Show kernel density/distribution of detrended y with transits, flares and outliers. Not Gaussian!
==> Show Huber loss next to it: https://en.wikipedia.org/wiki/Huber_loss
Write about Huber Regressor
wieviele outlier sind ok?
Peter J. Huber, Elvezio M. Ronchetti, Robust Statistics Concomitant scale estimates, pg 172
@misc{Owen2007,
  doi = {10.1090/conm/443/08555},
  url = {https://doi.org/10.1090/conm/443/08555},
  year  = {2007},
  publisher = {American Mathematical Society},
  pages = {59--71},
  author = {Art B. Owen},
  title = {A robust hybrid of lasso and ridge regression}
}
@article{Huber1964,
  doi = {10.1214/aoms/1177703732},
  url = {https://doi.org/10.1214/aoms/1177703732},
  year  = {1964},
  month = {mar},
  publisher = {Institute of Mathematical Statistics},
  volume = {35},
  number = {1},
  pages = {73--101},
  author = {Peter J. Huber},
  title = {Robust Estimation of a Location Parameter},
  journal = {The Annals of Mathematical Statistics}
}
@book{huber1981robust,
  title={Robust Statistics},
  author={Huber, P.J.},
  isbn={9780471418054},
  lccn={80018627},
  series={Wiley Series in Probability and Statistics},
  url={https://books.google.de/books?id=hVbhlwEACAAJ},
  year={1981},
  publisher={Wiley}
}
@book{huber2011robust,
  title={Robust Statistics},
  author={Huber, P.J. and Ronchetti, E.M.},
  isbn={9781118210338},
  series={Wiley Series in Probability and Statistics},
  url={https://books.google.mw/books?id=j1OhquR\_j88C},
  year={2011},
  publisher={Wiley}
}
(page 172)
@ARTICLE{2014arXiv1406.1717S,
       author = {{Suomela}, Jukka},
        title = "{Median Filtering is Equivalent to Sorting}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Data Structures and Algorithms},
         year = "2014",
        month = "Jun",
          eid = {arXiv:1406.1717},
        pages = {arXiv:1406.1717},
archivePrefix = {arXiv},
       eprint = {1406.1717},
 primaryClass = {cs.DS},
       adsurl = {https://ui.adsabs.harvard.edu/\#abs/2014arXiv1406.1717S},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
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
Mein Unterbewusstsein brodelt gerade im Detrending, deshalb ist die Mail auch ein "reminder" für mich :-)
Michael
[1] Benannt nach der Implementierung von Berwin A. Turlach (1995, unpublished), Idee aus:
Optimal Median Smoothing (1995)
W. Härdle and W. Steiger
Journal of the Royal Statistical Society. Series C (Applied Statistics)
Vol. 44, No. 2 (1995), pp. 258-264
DOI: 10.2307/2986349
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
    return t, y, hdu[0].header['TICID']  # filename[36:45]

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


    #path = 'P:/P/Dok/tess_alarm/'
    #path = 'P:/P/Dok/k2fits/'
    path = '/home/michael/Downloads/tess-data-alerts/'

    os.chdir(path)
    for file in glob.glob("*.fits"):
        #if "00230982885" in file:
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
                fig, axes = plt.subplots(4, 1, sharex=True, figsize=(6, 10))
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

                plt.savefig(str(TIC_ID) + ".png", bbox_inches='tight')
                plt.clf()
                plt.close('all')

if __name__ == "__main__":
    main()
