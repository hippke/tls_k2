import numpy
import scipy.signal
from astropy.stats import sigma_clip

import matplotlib.pyplot as plt

from astropy.io import fits
from astropy import units as u
from transitleastsquares import cleaned_array

if __name__ == "__main__":

    file = 'hlsp_tess-data-alerts_tess_phot_00012862099-s03_tess_v1_lc.fits'
    hdu = fits.open(file)
    t = hdu[1].data['TIME']
    y = hdu[1].data['PDCSAP_FLUX']

    t, y = cleaned_array(t, y)
    average_cadence = numpy.median(numpy.diff(t))
    print('average_cadence (days)', average_cadence)
    y = sigma_clip(y, sigma_upper=10, sigma_lower=100)
    t, y = cleaned_array(t, y)

    window = 1  # day
    kernel = int(window / average_cadence)
    if kernel % 2 == 0:
        kernel = kernel + 1
    print('kernel window (days)', window)
    print('kernel window (cadences)', kernel)


    from transitleastsquares.helpers import running_median

    y = y / numpy.mean(y)
    y = y - 1
    trend = scipy.signal.medfilt(y, kernel)
    y = y + 1
    trend = trend + 1
    #trend = running_median(y, kernel)
    #trend = scipy.signal.savgol_filter(trend, kernel, 2)
    y_filt = y /trend
    y_filt = sigma_clip(y_filt, sigma_upper=4, sigma_lower=1e10)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    ax = axes[0]
    ax.scatter(t, y, s=1, alpha=0.5, color='black')
    ax.plot(t, trend)
    ax.set_ylabel("Flux (electrons per sec)")
    ax = axes[1]
    ax.scatter(t, y_filt, s=1, alpha=0.5, color='black')
    #ax.set_xlim(t.min(), t.max())
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Normalized flux")

    #plt.scatter(t, y, s=1, alpha=0.5, color='black')
    #plt.show()
    plt.savefig('out.pdf')


    from transitleastsquares import transitleastsquares
    model = transitleastsquares(t, y_filt)
    results = model.power()


    plt.figure()
    ax = plt.gca()
    ax.axvline(results.period, alpha=0.4, lw=3)
    for n in range(2, 10):
        ax.axvline(n*results.period, alpha=0.4, lw=1, linestyle="dashed")
        ax.axvline(results.period / n, alpha=0.4, lw=1, linestyle="dashed")
    plt.ylabel(r'SDE')
    plt.xlabel('Period (days)')
    plt.plot(results.periods, results.power, color='blue', lw=0.5)
    #plt.plot(results.periods, results.power_raw, color='red', lw=0.5)
    #plt.plot((0, 11), (0, 0), color='black')
    plt.xlim(0, max(results.periods))
    plt.savefig('1.pdf')

    plt.figure()
    plt.plot(
        results.model_folded_phase,
        results.model_folded_model,
        color='red')
    plt.scatter(
        results.folded_phase,
        results.folded_y,
        color='blue',
        s=10,
        alpha=0.5,
        zorder=2)
    plt.xlim(0.40, 0.60)
    plt.xlabel('Phase')
    plt.ylabel('Relative flux')
    plt.savefig('2.pdf')

    #print('Period', format(results.period, '.5f'), 'd')
    #print(len(results.transit_times), 'transit times in time series:', \
    #        ['{0:0.5f}'.format(i) for i in results.transit_times])
    #print('Transit depth', format(results.depth, '.5f'))
    #print('Transit duration (days)', format(results.duration, '.5f'))

    # the line number (data file, or array index) 
    # of the peak in the TLS SDE power spectrum
    # Note that the best period is more easily available as results.period
    print('BLS_NO', numpy.argmax(results.power))  
    # print(results.power[numpy.argmax(results.power)])  # or: results.SDE
    # print(results.periods[numpy.argmax(results.power)])  # or: results.period

    # results.depth (float) Best-fit transit depth (measured at the transit bottom)
    # Note that a limb-darkened transit has an "overshoot"
    # Better information from TLS would be:
    # - results.rp_rs: (float) Radius ratio of planet and star using the 
    #                          analytic equations from Heller 2019
    print('BLS_Depth_1_0', results.depth)

    print('BLS_Npointsaftertransit_1_0', 'TBD')  # after_transit_count
    print('BLS_Npointsbeforetransit_1_0', 'TBD')  # before_transit_count
    print('BLS_Npointsintransit_1_0', 'TBD')  # in_transit_count

    """
    # TLS offers:
    transit_count:          (int) The number of transits
    distinct_transit_count: (int) The number of transits with intransit data points
    empty_transit_count:    (int) The number of transits with no intransit data points
    """
    print('BLS_Ntransits_1_0', results.distinct_transit_count)

    # I can the TESS magnitude from MAST, but it takes time for a HTTP request
    #from astroquery.mast import Catalogs
    #result = Catalogs.query_criteria(catalog="Tic", ID=TIC_ID).as_array()
    #mag = result[0][60]
    #print('BLS_OOTmag_1_0', mag)

    print('BLS_Period_1_0', results.period)
    print('BLS_Qingress_1_0', 'TBD')
    print('BLS_Qtran_1_0', results.duration / results.period)
    print('BLS_Rednoise_1_0', 'TBD')
    print('BLS_SDE_1_0', results.SDE)  # Should be used, is *cleanest*
    print('BLS_SN_1_0', results.snr)

    # MUST NOT be used; is not *cleanest*, and is normalized to 1
    # For an explanation, see this tutorial:
    # https://github.com/hippke/tls/blob/master/tutorials/04%20Exploring%20the%20TLS%20spectra%20and%20statistics.ipynb
    print('BLS_SR_1_0', numpy.max(results.SR))
    print('BLS_SignaltoPinknoise_1_0', 'TBD')
    print('BLS_Tc_1_0', results.T0)
    print('BLS_Whitenoise_1_0', 'TBD')
    print('BLS_deltaChis2_1_0', 'TBD')




    """
    BLS_NO  - the number of peak
    BLS_Depth_1_0 - the depth of the transit estimated by the BLS
    BLS_Npointsaftertransit_1_0 - number of points in a window equal to the transit duration 
    BLS_Npointsbeforetransit_1_0 - number of points in a window equal to the transit duration 
    BLS_Npointsintransit_1_0 - number of points inside the transit window
    BLS_Ntransits_1_0 - number of transits
    BLS_OOTmag_1_0 - out of transit tess magnitude
    BLS_Period_1_0 - period of the signal
    BLS_Qingress_1_0 -  ingress/transit duration
    BLS_Qtran_1_0 -  transit duration/period
    BLS_Rednoise_1_0 - estimated red noise in the light curve 
    BLS_SDE_1_0 - one estimation of the BLS signal to noise
    BLS_SN_1_0 - another estimation of the BLS signal to noise
    BLS_SR_1_0 - another estimation of the BLS signal to noise
    BLS_SignaltoPinknoise_1_0 -  single to noise of the transit
    BLS_Tc_1_0 - center of transit
    BLS_Whitenoise_1_0 - estimated white noise in the light curve 
    BLS_deltaChis2_1_0 - deltachis2 compare to a flat line (I don't use this one)
    """
