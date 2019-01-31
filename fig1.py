import numpy
import scipy.signal
from astropy.stats import sigma_clip
import matplotlib.pyplot as plt
#from transitleastsquares import transitleastsquares, cleaned_array, transit_mask
from astropy.io import fits
import matplotlib.gridspec as gridspec
import batman
from numpy import arccos, degrees

def impact_to_inclination(b, semimajor_axis):
    """Converts planet impact parameter b = [0..1.x] to inclination [deg]"""
    return degrees(arccos(b / semimajor_axis))

# Planet c
# P=20.65614
# R=3.48  3.01
# a = 0.1399 au

# b
# 8.99218
# 5.38 Earth, 5.13
# a=0.08036 au


if __name__ == '__main__':
    with fits.open("hlsp_everest_k2_llc_205071984-c02_kepler_v2.0_lc.fits") as hdus:
        data = hdus[1].data
    t = data["TIME"]
    y = data["FLUX"]
    q = data["QUALITY"]
    # Remove flagged EVEREST data points
    m = numpy.isfinite(t) & numpy.isfinite(y)
    for b in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17]:
        m &= (q & (2 ** (b - 1))) == 0
    t = numpy.ascontiguousarray(t[m], dtype=numpy.float64)
    y = numpy.ascontiguousarray(y[m], dtype=numpy.float64)
    y = y / numpy.median(y)
    skip = 64
    t = t[skip:]
    y = y[skip:]
    #t, y = cleaned_array(t, y)
    print(min(t), max(t), max(t)-min(t), len(t))
    trend = scipy.signal.medfilt(y, 31)
    trend = scipy.signal.savgol_filter(trend, 25, 2)
    y_filt = y /trend
    y_filt = sigma_clip(y_filt, sigma_upper=2, sigma_lower=float('inf'))
    #for i in range(len(y_filt)):
    #    print(t[i], ',', y_filt[i])
    # Periods
    # Period 8.99198 d  2067.92701
    # 31.72849 d T0 2070.77327
    # 20.65769 d T0 2066.41880
    # 4.34937 d  T0 2065.87612
    #plt.figure(figsize=(4.25, 4.25 / 1.5))
    plt.figure(figsize=(8, 4))
    #G = gridspec.GridSpec(ncols=2, nrows=3)
    G = gridspec.GridSpec(ncols=1, nrows=2)#, height_ratios=[2, 1, 2], width_ratios=[1, 1])
    #G.update(hspace=0)
    ax0 = plt.subplot(G[0, 0])
    ax0.scatter(t, y, s=0.5, color='black')
    ax0.set_xlim(min(t), max(t))
    ax0.plot(t, trend, color='red', linewidth=0.5)
    #plt.text(min(t)+0.2, 1.01, 'K2 light curve from EVEREST\n(corrected for systematics)')
    plt.xticks(())
    plt.ylabel('Raw flux (normalized)')
    ax0.get_yaxis().set_label_coords(-0.1,0.5)
    G.update(hspace=0.05)
    ax0b = plt.subplot(G[1, 0])
    #G.update(hspace=2)
    ax0b.scatter(t, -(1-y_filt)*10**6, s=0.5, color='black')
    ax0b.set_xlim(min(t), max(t))
    ax0b.set_ylim(-4500, 400)
    #plt.text(min(t)+0.2, -4000, 'Detrended light curve\n(systematics removed)')
    ax0b.set_xlabel('Time (days)')
    ax0b.get_yaxis().set_label_coords(-0.1,0.5)
    plt.ylabel('Detrended flux')
    plt.savefig('reduction.pdf', bbox_inches='tight')
    plt.figure(figsize=(4, 4))
    G = gridspec.GridSpec(ncols=1, nrows=2)
    G.update(hspace=0)
    ax1 = plt.subplot(G[0, 0])
    ax1.set_yticks(numpy.arange(-400, 400, 200))
    #fig, ax = plt.subplots(2, sharex=True, figsize=(4.25, 4.25 / 1.5))
    #ax[0] = plt.subplot(211)
    #ax[1] = plt.subplot(212, sharex=ax[0])
    #fig.subplots_adjust(wspace=0, hspace=0)
    #ax[0].label_outer()
    #fig.subplots_adjust(hspace=0)
    #ax1.get_yaxis().set_tick_params(which='both', direction='out')
    #ax1.get_xaxis().set_tick_params(which='both', direction='out')
    ax1.get_yaxis().set_tick_params(which='both', direction='out')
    ax1.get_xaxis().set_tick_params(which='both', direction='out')
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-500, 300)
    #ax1.set_ylabel(r'Flux')
    #plt.xlabel()
    #G.update(hspace=0)
    period_e = 4.34937  # 4.34937
    T0_e = 2065.88169  # 2065.87612
    x = (t - T0_e + 0.5*period_e) % period_e - 0.5*period_e
    m = numpy.abs(x) < 1000
    ax1.scatter(x[m] * 24, -(1-y_filt[m])*10**6, color='red', s=3, alpha=0.5)
    plt.xticks([])
    ax2 = plt.subplot(G[1, 0])
    G = gridspec.GridSpec(ncols=1, nrows=4)
    G.update(hspace=0)
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-9000, 500)
    ax2.set_ylabel(r'Flux (ppm)')
    ax2.set_xlabel(r'Time from mid-transit (hrs)')

    period_b = 8.99309  # 8.99198
    T0_b = 2067.92523  # 2067.92701
    x = (t - T0_b + 0.5*period_b) % period_b - 0.5*period_b
    m = numpy.abs(x) < 1000
    ax2.scatter(x[m] * 24, -(1-y_filt[m])*10**6, color='black', s=3, alpha=0.5)

    offset1 = 4500
    period = 20.66442  # 20.65769
    T0 = 2066.41515  # 2066.41880
    x = (t - T0 + 0.5*period) % period - 0.5*period
    m = numpy.abs(x) < 1000
    ax2.scatter(x[m] * 24, -(1-y_filt[m])*10**6-offset1, color='orange', s=3, alpha=0.5)
    offset2 = 6500
    period = 31.71657  # 31.72849
    T0 = 2070.79095
    x = (t - T0 + 0.5*period) % period - 0.5*period
    m = numpy.abs(x) < 1000
    ax2.scatter(x[m] * 24, -(1-y_filt[m])*10**6-offset2, color='blue', s=3, alpha=0.5)
    ax1.set_ylabel(r'Flux (ppm)')
    ax1.get_yaxis().set_label_coords(-0.2,0.5)
    ax2.get_yaxis().set_label_coords(-0.2,0.5)
    ax1.text(3.7, 220, 'e', color='red')
    ax2.text(3.7, -900, 'b', color='black')
    ax2.text(3.7, -5200, 'c', color='orange')
    ax2.text(3.7, -7600, 'd', color='blue')



    # b
    # 8.99218
    # 5.38 Earth, 5.13
    # a=0.08036 au

    t = numpy.linspace(-1, 1, 1000)
    b = 0.33
    semimajor_axis = 0.08036 * 149597870700 / 696342000 / 0.87
    ma = batman.TransitParams()
    ma.t0 = 0  # time of inferior conjunction; first transit is X days after start
    ma.per = 8.99218  # orbital period
    ma.rp = 5.28 * 6371 / 696342 / 0.87  # 6371 planet radius (in units of stellar radii)
    ma.a = semimajor_axis  # semi-major axis (in units of stellar radii)
    ma.inc = impact_to_inclination(b, semimajor_axis)  # orbital inclination (in degrees)
    ma.ecc = 0  # eccentricity
    ma.w = 90  # longitude of periastron (in degrees)
    ma.u = [0.51, 0.19]  # limb darkening coefficients
    ma.limb_dark = "quadratic"  # limb darkening model
    m = batman.TransitModel(ma, t)  # initializes model
    f = m.light_curve(ma)  # calculates light curve
    plt.plot(t*24, -(1-f)*10**6, color='gray', linewidth=1, zorder=0)


    # Planet c
    # P=20.65614
    # R=3.48  3.01
    # a = 0.1399 au
    t = numpy.linspace(-1, 1, 1000)
    b = 0.5#0.74
    semimajor_axis = 0.1399 * 149597870700 / 696342000 / 0.87
    ma = batman.TransitParams()
    ma.t0 = 0  # time of inferior conjunction; first transit is X days after start
    ma.per = 20.65614  # orbital period
    ma.rp = 3.1 * 6371 / 696342 / 0.87  # 6371 planet radius (in units of stellar radii)
    ma.a = semimajor_axis  # semi-major axis (in units of stellar radii)
    ma.inc = impact_to_inclination(b, semimajor_axis)  # orbital inclination (in degrees)
    ma.ecc = 0  # eccentricity
    ma.w = 90  # longitude of periastron (in degrees)
    ma.u = [0.51, 0.19]  # limb darkening coefficients
    ma.limb_dark = "quadratic"  # limb darkening model
    m = batman.TransitModel(ma, t)  # initializes model
    f = m.light_curve(ma)  # calculates light curve
    plt.plot(t*24, -(1-f)*10**6-offset1, color='gray', linewidth=1, zorder=0)


    # Planet d
    # P=
    # R=
    # a 
    t = numpy.linspace(-1, 1, 1000)
    b = 0.5
    semimajor_axis =    0.1873 * 149597870700 / 696342000 / 0.87
    ma = batman.TransitParams()
    ma.t0 = 0  # time of inferior conjunction; first transit is X days after start
    ma.per = 31.71922  # orbital period
    ma.rp = 3.45 * 6371 / 696342 / 0.87  # 6371 planet radius (in units of stellar radii)
    ma.a = semimajor_axis  # semi-major axis (in units of stellar radii)
    ma.inc = impact_to_inclination(b, semimajor_axis)
    ma.ecc = 0  # eccentricity
    ma.w = 90  # longitude of periastron (in degrees)
    ma.u = [0.51, 0.19]  # limb darkening coefficients
    ma.limb_dark = "quadratic"  # limb darkening model
    m = batman.TransitModel(ma, t)  # initializes model
    f = m.light_curve(ma)  # calculates light curve
    plt.plot(t*24, -(1-f)*10**6-offset2, color='gray', linewidth=1, zorder=0)


    # Planet e
    # P=
    # R=
    # a 
    t = numpy.linspace(-1, 1, 1000)
    b = 0.29
    semimajor_axis =    0.06 * 149597870700 / 696342000 / 0.87
    ma = batman.TransitParams()
    ma.t0 = 0  # time of inferior conjunction; first transit is X days after start
    ma.per = 4.3  # orbital period
    ma.rp = 1.1 * 6371 / 696342 / 0.87  # 6371 planet radius (in units of stellar radii)
    ma.a = semimajor_axis  # semi-major axis (in units of stellar radii)
    ma.inc = impact_to_inclination(b, semimajor_axis)
    ma.ecc = 0  # eccentricity
    ma.w = 90  # longitude of periastron (in degrees)
    ma.u = [0.51, 0.19]  # limb darkening coefficients
    ma.limb_dark = "quadratic"  # limb darkening model
    m = batman.TransitModel(ma, t)  # initializes model
    f = m.light_curve(ma)  # calculates light curve
    #plt.plot(t*24, -(1-f)*10**6-offset2, color='gray', linewidth=1, zorder=0)






    ax1.plot(t*24, -(1-f)*10**6, color='gray', linewidth=1, zorder=0)



    plt.savefig('fold.pdf', bbox_inches='tight')
