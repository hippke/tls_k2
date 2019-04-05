import numpy
#import time as ttime
import os
#import logging
import scipy
#import everest
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.stats import sigma_clip
from astropy.stats import LombScargle
from transitleastsquares import transitleastsquares, cleaned_array, transit_mask, \
    catalog_info, fold, resample
from transitleastsquares.helpers import running_mean_equal_length
import kplr
from wotan_fast import trend
from numpy import pi
import os
import psutil
import sys
# 3611 unique KICs in  "cumulative_2019.03.10_11.04.40.ods"
from multiprocessing import Pool
import glob



def T14(R_s, M_s, P):
    """Input:  Stellar radius and mass; planetary period
                   Units: Solar radius and mass; days
       Output: Maximum planetary transit duration T_14max
               Unit: Fraction of period P"""

    G = 6.673e-11  # gravitational constant [m^3 / kg / s^2]
    R_sun = 695508000  # radius of the Sun [m]
    R_jup = 69911000  # radius of Jupiter [m]
    M_sun = 1.989 * 10 ** 30  # mass of the Sun [kg]
    SECONDS_PER_DAY = 86400

    P = P * SECONDS_PER_DAY
    R_s = R_sun * R_s
    M_s = M_sun * M_s
    T14max = (R_s + 2 * R_jup) * ((4 * P) / (pi * G * M_s)) ** (1 / 3)  # planet 2 R_jup
    return T14max / SECONDS_PER_DAY


def make_figure(KOI, planet_number, results, t, y, y_filt, trend, rawtime, rawflux, catalog, row, valid=False):
    fig = plt.figure(figsize=(10, 14))
    G = gridspec.GridSpec(9, 3)
    G.update(hspace=0)
    radius = catalog["R_star"][row]
    radius_max = catalog["rad_up"][row]
    radius_min = catalog["rad_down"][row]
    mass = catalog["mass"][row]
    mass_max = catalog["mass_up"][row]
    mass_min = catalog["mass_down"][row]

    # Raw flux
    axes_1a = plt.subplot(G[0:1, :2])
    plt.plot(rawtime, rawflux/numpy.mean(rawflux), "k", linewidth=0.5)
    plt.plot(rawtime, trend/numpy.mean(y), color='red', linewidth=0.5)
    plt.xlim(min(t), max(t))
    plt.xticks(())
    plt.ylabel(r'Raw Flux')

    ppm_flux = -(1-y_filt)*10**6

    # Detrended flux
    G = gridspec.GridSpec(9, 3)
    G.update(hspace=0.4)
    axes_1b = plt.subplot(G[1:2, :2])
    plt.plot(t, ppm_flux, "k", linewidth=0.5)

    top_ppm = -(1-max(y_filt))*10**6
    bottom_ppm = -(1-min(y_filt))*10**6
    y_range = abs(bottom_ppm) + abs(top_ppm)
    x_range = max(t) - min(t)

    offset_y = bottom_ppm + y_range/40
    offset_x = min(t) + x_range/80

    std = numpy.std(ppm_flux)
    text = r'$\sigma=$' + format(std, '.0f') + r'$\,$ppm'
    plt.text(offset_x, offset_y, text, color='red')
    

    plt.ylabel(r'Flux (ppm)')
    plt.xlabel('Time (BKJD, days)')
    plt.xlim(min(t), max(t))

    # Phase fold
    axes_2 = plt.subplot(G[2, 0:2])
    plt.plot((results.model_folded_phase-0.5)*results.period, -(1-results.model_folded_model)*10**6, color='red', zorder=99)
    plt.scatter((results.folded_phase-0.5)*results.period, -(1-results.folded_y)*10**6, color='black', s=2, alpha=0.5, zorder=2)
    plt.xlim(-2*results.duration, 2*results.duration)
    plt.xlabel('Time from mid-transit (days)')
    plt.ylabel('Flux')
    plt.ylim((numpy.percentile(-(1-y_filt)*10**6, 0.1), numpy.percentile(-(1-y_filt)*10**6, 99.5)))
    plt.text(-1.95*results.duration, numpy.percentile(-(1-y_filt)*10**6, 0.1), 'primary')
    plt.plot(
        (results.folded_phase-0.5)*results.period,
        -(1-running_mean_equal_length(results.folded_y, 10))*10**6,
        color='blue', linestyle='dashed', linewidth=1, zorder=3)

    # Phase fold to secondary eclipse
    G = gridspec.GridSpec(9, 3)
    G.update(hspace=0.3)
    axes_3 = plt.subplot(G[2, 2])
    plt.yticks(())
    phases = fold(time=t, period=results.period, T0=results.T0)
    sort_index = numpy.argsort(phases, kind="mergesort")
    phases = phases[sort_index]
    flux = y_filt[sort_index]
    plt.scatter((phases-0.5) * results.period, flux, color='black', s=2, alpha=0.5, zorder=2)
    plt.plot((-0.5, 0.5), (1, 1), color='red')
    plt.xlim(-2*results.duration, 2*results.duration)
    plt.ylim(numpy.percentile(y_filt, 0.1), numpy.percentile(y_filt, 99.5))
    plt.plot(
        (phases-0.5) * results.period,
        running_mean_equal_length(flux, 10),
        color='blue', linestyle='dashed', linewidth=1, zorder=3)
    # Calculate secondary eclipse depth
    intransit_secondary = numpy.where(numpy.logical_and(
        (phases-0.5)*results.period > (-0.5*results.duration),
        (phases-0.5)*results.period < (0.5*results.duration)))
    mean = -(1 - numpy.mean(flux[intransit_secondary]))*10**6
    stabw = -(numpy.std(flux[intransit_secondary]) / numpy.sqrt(len(flux[intransit_secondary]))) * 10**6
    significance_eb = mean / stabw

    plt.scatter((phases[intransit_secondary]-0.5)* results.period, flux[intransit_secondary], color='orange', s=20, alpha=0.5, zorder=0)
    if numpy.isnan(mean):
        mean = 0
    if numpy.isnan(stabw):
        stabw = 0
    if numpy.isnan(significance_eb):
        significance_eb = 99
    text = r'secondary ' + str(int(mean)) + r'$\pm$' + str(int(stabw)) + ' ppm (' + format(significance_eb, '.1f') + r'$\,\sigma$)'
    plt.text(-1.95*results.duration, numpy.percentile(y_filt, 0.1), text)
    # Full phase
    G = gridspec.GridSpec(9, 3)
    G.update(hspace=1)
    axes_5 = plt.subplot(G[3, :])
    plt.plot(results.model_folded_phase, -(1-results.model_folded_model)*10**6, color='red')
    plt.scatter(results.folded_phase, -(1-results.folded_y)*10**6, color='black', s=2, alpha=0.5, zorder=2)
    plt.xlim(0, 1)
    plt.ylim(numpy.percentile(ppm_flux, 0.1), numpy.percentile(ppm_flux, 99.9))
    plt.xlabel('Phase')
    plt.ylabel('Flux')

    # Check if phase gaps
    phase_diffs = abs(results.folded_phase - numpy.roll(results.folded_phase, -1))
    phase_diffs = phase_diffs[:-1]
    largest_phase_peak = max(phase_diffs) / numpy.median(phase_diffs)

    # All transits in time series
    G = gridspec.GridSpec(9, 3)
    G.update(hspace=0)
    axes_6b = plt.subplot(G[4, :])
    y_filt = -(1-y_filt)*10**6
    in_transit = transit_mask(t, results.period, results.duration, results.T0)

    t_diff1 = abs(t - numpy.roll(t, -1))
    t_diff2 = abs(t - numpy.roll(t, +1))
    if max(t_diff1[in_transit]) > 0.1 or max(t_diff2[in_transit]) > 0.1:  # in days
        transit_near_gaps = True
        plt.text(min(t), numpy.percentile(ppm_flux, 0.1), 'Warning: Transits near gaps')
    else:
        transit_near_gaps = False

    transit_touches_edge = False
    if max(t) in t[in_transit]:
        plt.text(min(t), numpy.percentile(ppm_flux, 0.1), 'Warning: Last transit touches end')
        transit_touches_edge = True
    if max(t) in t[in_transit]:
        plt.text(min(t), numpy.percentile(ppm_flux, 0.1), 'Warning: First transit touches start')
        transit_touches_edge = True
    plt.scatter(t[in_transit], y_filt[in_transit], color='red', s=2, zorder=0)
    plt.scatter(t[~in_transit], y_filt[~in_transit], color='black', alpha=0.5, s=2, zorder=0)
    plt.plot(results.model_lightcurve_time, -(1-results.model_lightcurve_model)*10**6, alpha=0.5, color='red', zorder=1)
    plt.xlim(min(t), max(t))
    plt.ylim(numpy.percentile(ppm_flux, 0.1), numpy.percentile(ppm_flux, 99.9))
    plt.xlabel('Flux')
    plt.ylabel('Flux')
    plt.xticks(())

    # Transit depths error bars
    avg = -(1-results.depth_mean[0])*10**6
    if numpy.isnan(results.transit_depths_uncertainties).any():
        step = 0
    else:
        step = max(results.transit_depths_uncertainties)*10**6
    down = avg - step
    G = gridspec.GridSpec(9, 3)
    G.update(hspace=1)
    axes_6 = plt.subplot(G[5, :])
    plt.errorbar(
        results.transit_times,
        -(1-results.transit_depths)*10**6,
        yerr=step,
        fmt='o',
        color='red')
    plt.plot((min(t), max(t)), (0, 0), color='black')
    plt.plot((min(t), max(t)), (avg, avg), color='black', linestyle='dashed')
    plt.xlim(min(t), max(t))
    for transit in range(len(results.transit_times)):
        plt.text(
            results.transit_times[transit],
            down,
            str(int(results.per_transit_count[transit])),
            horizontalalignment='center')
    plt.xlabel('Time (BKJD, days)')
    plt.ylabel('Flux')

    # Test statistic
    G = gridspec.GridSpec(9, 3)
    G.update(hspace=0)
    #G.update(hspace=3)
    axes_7 = plt.subplot(G[6, :])
    axes_7.axvline(results.period, alpha=0.4, lw=3)
    for n in range(2, 5):
        axes_7.axvline(n*results.period, alpha=0.4, lw=1, linestyle="dashed")
        axes_7.axvline(results.period / n, alpha=0.4, lw=1, linestyle="dashed")
    plt.ylabel(r'TLS SDE')
    plt.xticks(())
    plt.plot(results.periods, results.power, color='black', lw=0.5)
    plt.xlim(min(results.periods), max(results.periods))
    plt.text(results.period + 0.1, results.SDE * 0.9, 'SDE=' + format(results.SDE, '.1f'))

    # LS periodogram raw
    freqs = numpy.geomspace(1/min(results.periods), 1/max(results.periods), 10000)
    freqs = numpy.unique(freqs)
    lspower = LombScargle(rawtime, rawflux).power(freqs)
    freqs = 1/freqs

    G = gridspec.GridSpec(9, 3)
    G.update(hspace=0)
    axes_8 = plt.subplot(G[7, :])
    axes_8.axvline(results.period, alpha=0.4, lw=15)

    plt.ylabel(r'LS (raw)')
    plt.plot(freqs, lspower, color='black', lw=0.5)
    plt.xlim(min(results.periods), max(results.periods))
    plt.ylim(0, max(lspower)*1.1)
    plt.xlabel('Period (days)')
    # Calculate Lomb-Scargle periodogram detrended
    freqs = numpy.geomspace(1/min(results.periods), 1/max(results.periods), 10000)
    freqs = numpy.unique(freqs)
    lspower = LombScargle(t, y_filt).power(freqs)
    freqs = 1/freqs

    continuum = numpy.std(lspower)
    lspower = numpy.divide(lspower, continuum)  # Normalize to S/N

    G = gridspec.GridSpec(9, 3)
    G.update(hspace=0)
    axes_9 = plt.subplot(G[8, :])
    peak_index = numpy.argmax(lspower)
    axes_9.axvline(freqs[peak_index], alpha=0.4, lw=10)
    plt.ylabel(r'LS (detrended)')
    plt.plot(freqs, lspower, color='black', lw=0.5)
    plt.xlim(min(results.periods), max(results.periods))
    plt.ylim(0, max(lspower)*1.1)
    plt.xlabel('Period (days)')
    
    # Text
    G = gridspec.GridSpec(9, 3)
    G.update(hspace=0, wspace=0.1)
    axes_8 = plt.subplot(G[0:2, 2])
    plt.xticks(())
    plt.yticks(())
    plt.text(0, 0, 'KOI ' + str(KOI) + '.0' + str(planet_number), fontweight='bold')

    plt.text(0, -0.30, 'SDE=' + format(results.SDE, '.1f') + ', SNR=' + format(results.snr, '.1f'))
    plt.text(0, -0.45, 'P=' + format(results.period, '.5f') + ' +-' + format(results.period_uncertainty, '.5f') + r'$\,$d')
    plt.text(0, -0.60, r'$T_{dur}=$' + format(results.duration, '.5f') + r'$\,$d')
    plt.text(0, -0.75, r'$R_*=$' + format(radius, '.2f') + ' (+' + format(radius_max, '.2f') + ', -' + format(radius_min, '.2f') + ') $\,R_{\odot}$')
    plt.text(0, -0.90, r'$M_*=$' + format(mass, '.2f') + ' (+' + format(mass_max, '.2f') + ', -' + format(mass_min, '.2f') + ') $\,M_{\odot}$')
    plt.text(0, -1.05, r'$R_P/R_*=$' + format(results.rp_rs, '.3f'))

    print('rp_rs', results.rp_rs)

    rp = results.rp_rs * radius  # in solar radii
    sun = 695700
    earth = 6371
    jupi = 69911
    rp_earth = (sun / earth) * rp
    rp_jupi = (sun / jupi) * rp
    plt.text(0, -1.20, r'$R_P=$' + format(rp_earth, '.2f') + '$\,R_{\oplus}=$'+ format(rp_jupi, '.2f') + '$\,R_{Jup}$')
    plt.text(
        0, -1.35, 
        r'$\delta=$' + \
        format((1-results.depth_mean[0])*10**6, '.0f') + ' ppm (' + \
        format((1-results.depth_mean_odd[0])*10**6, '.0f') + ', ' + \
        format((1-results.depth_mean_even[0])*10**6, '.0f') + ')')
    plt.text(0, -1.50, r'odd/even mismatch: ' + format(results.odd_even_mismatch, '.2f') + '$\,\sigma$')
    plt.text(0, -1.65, str(results.distinct_transit_count) + '/' + str(results.transit_count) + ' transits with data')
    plt.xlim(-0.1, 2)
    plt.ylim(-2, 0.1)
    axes_8.axis('off')
    plt.subplots_adjust(hspace=0.4)

    if abs(significance_eb) > 3:
        valid = False
        print('Vetting fail! significance_eb > 3')

    if abs(results.odd_even_mismatch) > 3:
        valid = False
        print('Vetting fail! odd_even_mismatch larger 3 sigma')

    if transit_near_gaps and results.distinct_transit_count < 4:
        valid = False
        print('Vetting fail! Transit near gaps and distinct_transit_count < 4')

    if valid:  # 5-50d
        figure_out_path = 'GOOD5-50d_' + str(KOI) + '_0' + str(planet_number)
    else:
        figure_out_path = '_FAIL5-50d_' + str(KOI) + '_0' + str(planet_number)

    plt.savefig(figure_out_path + '.png', bbox_inches='tight')
    print('Figure made:', figure_out_path)
    plt.close

    return valid


def loadkoi(koi):

    print('Getting LC for KOI', koi)
    client = kplr.API()
    koi = client.koi(koi)
    lcs = koi.get_light_curves(short_cadence=False)

    # Loop over the datasets and read in the data
    time, flux, ferr, quality = [], [], [], []
    for lc in lcs:
        with lc.open() as f:
            # The lightcurve data are in the first FITS HDU
            hdu_data = f[1].data
            time.append(hdu_data["time"])
            flux_segment = hdu_data["pdcsap_flux"]
            quality.append(hdu_data["sap_quality"])

            # Normalize each segment by its median
            flux_segment = flux_segment / numpy.nanmedian(flux_segment)
            flux.append(flux_segment.tolist())

    time = numpy.hstack(time)
    flux = numpy.hstack(flux)

    # Reject data points of non-zero quality
    quality = numpy.hstack(quality)
    m = numpy.any(numpy.isfinite(flux)) & (quality == 0)
    time = time[m]
    flux = flux[m]

    time, flux = cleaned_array(time, flux)
    print('Data points:', len(time))
    return time, flux



def get_planets(KIC):
    # Match with KIC, as it is the same for all planets in the system (KOI is .0X)
    #KIC = catalog["KIC"][row]
    #print('KIC', KIC)
    periods = []
    t0s = []
    tdurs = []
    for row in range(len(catalog["KIC"])):
        #print(row, catalog["KIC"][row], KIC)
        if catalog["KIC"][row] == KIC:
            #print(row, str(catalog["KIC"][row]), str(KIC))
            #print('MATCHED')
            periods.append(catalog["Period"][row])
            t0s.append(catalog["T0"][row])
            tdurs.append(catalog["T14"][row])
    return periods, t0s, tdurs


def seach_one_koi(KOI):

    # Check if vetting file exists already
    if ("_" + str(KOI) + '_0') in " ".join(glob.glob("*.png")):
        print('Vetting sheet for this KOI exists already, skipping KOI', KOI)
    else:
        print('Working on file', KOI)

        time, flux = loadkoi(str(KOI))

        #KOI = 1206.01
        #print(catalog["KOI"])
        #catalog_KOIs = 
        #catalog_KOIs = round(catalog_KOIs, 0)
        #print(catalog_KOIs)

        #print(int(float(KOI)))
        #print(catalog["KOI"].astype(int)[:20])

        row = numpy.argmax(catalog["KOI"].astype(int)==int(float(KOI)))
        KIC = catalog["KIC"][row]
        print('row', row)
        print('KIC', KIC)

        max_t14 = T14(
            R_s=catalog["R_star"][row],
            M_s=catalog["mass"][row],
            P=period_max)
        window = 3 * max_t14

        print(
            'R_star, mass, max_t14, window',
            catalog["R_star"][row],
            catalog["mass"][row],
            max_t14,
            window)

        #plt.scatter(time, flux, s=1)
        #plt.show()
        #print(time)
        #print(flux)

        if time is not None:

            # Remove known planets
            periods, t0s, tdurs = get_planets(KIC)
            print('periods', periods)
            for no in range(len(periods)):
                print('Removing planet', no+1, periods[no], t0s[no], tdurs[no])
                
                intransit = transit_mask(
                    time,
                    periods[no],
                    2 * tdurs[no],
                    t0s[no]
                    )
                flux = flux[~intransit]
                time = time[~intransit]
                time, flux = cleaned_array(time, flux)

            print('Next planet is number', no+2)

            # Detrend data and remove high outliers

            print('Detrending with window...', window)
            #print(time)
            #print(flux)
            trend1 = trend(time, flux, window=window, c=5)
            #plt.scatter(time, flux, s=1, color='black')
            #plt.plot(time, trend1, color='red')
            #plt.show()


            #trend = scipy.signal.medfilt(flux, 31)
            #trend = scipy.signal.savgol_filter(trend, 25, 2)
            rawflux = flux.copy()
            rawtime = time.copy()
            y_filt = flux / trend1
            y_filt = sigma_clip(y_filt, sigma_upper=3, sigma_lower=1e10)
            time, y_filt = cleaned_array(time, y_filt)

            #plt.close()
            #plt.scatter(time, y_filt, s=1, color='black')
            #plt.show()

            a = catalog["ld1"][row]
            b = catalog["ld2"][row]
            print('LD ab = ', a, b)
            model = transitleastsquares(time, y_filt)
            results = model.power(
                #n_transits_min=1,
                u=(a, b),
                R_star=catalog["R_star"][row],
                R_star_max=catalog["R_star"][row] + 2 * catalog["rad_up"][row],
                # sign is OK, is negative in catalog:
                R_star_min=catalog["R_star"][row] + 2 * catalog["rad_down"][row],
                M_star=catalog["mass"][row],
                M_star_max=catalog["mass"][row] + 2 * catalog["mass_up"][row],
                M_star_min=catalog["mass"][row] + 2 * catalog["mass_down"][row],
                oversampling_factor=5,
                duration_grid_step=1.05,
                use_threads=1,
                period_min=period_min,
                period_max=period_max,
                show_progress_bar=False,
                T0_fit_margin=0.1
                )
            tls_worked = True
                #except ValueError:
                #    tls_worked = False

            valid = True
            if tls_worked:
                # Check if phase space has gaps
                bins = numpy.linspace(0, 1, 100)
                digitized = numpy.digitize(results.folded_phase, bins)
                bin_means = [results.folded_phase[digitized == i].mean() 
                    for i in range(1, len(bins))]
                #print('bin_means', bin_means)
                if numpy.isnan(bin_means).any():
                    print('Vetting fail! Phase bins contain NaNs')
                    valid = False

                if results.distinct_transit_count==1 and results.transit_count >= 2:
                    valid = False
                    print('Vetting fail! results.distinct_transit_count==1 and results.transit_count == 2')
                if results.distinct_transit_count==2 and results.transit_count >= 3:
                    valid = False
                    print('Vetting fail! results.distinct_transit_count==2 and results.transit_count == 3')
                if results.SDE < 8 :
                    valid = False
                    print('Vetting fail! results.SDE < 8', results.SDE)
                if results.snr < 7:
                    valid = False
                    print('Vetting fail! results.snr < 7', results.snr)

                upper_transit_depths = results.transit_depths + results.transit_depths_uncertainties
                if results.transit_count == 2 and max(upper_transit_depths) > 1:
                    valid = False
                    print('Vetting fail! 2 transits, only 1 significant')

                upper_transit_depths = results.transit_depths + results.transit_depths_uncertainties
                if results.transit_count == 3 and max(upper_transit_depths) > 1:
                    valid = False
                    print('Vetting fail! 3 transits, not all 3 significant')

                upper_transit_depths = results.transit_depths + results.transit_depths_uncertainties
                if results.transit_count == 4 and max(upper_transit_depths) > 1:
                    valid = False
                    print('Vetting fail! 4 transits, not all 4 significant')

                if results.depth < 0.95:
                    valid = False
                    print('Vetting fail! Transit depth < 0.95', results.depth)

                print('Signal detection efficiency (SDE):', format(results.SDE, '.1f'))
                print('SNR:', format(results.snr, '.1f'))
                print('Search completed')
                #if valid:
                print('Attempting figure...')
                make_figure(
                    KOI=str(KOI),
                    planet_number=no+2,
                    results=results,
                    t=time,
                    y=flux,
                    y_filt=y_filt,
                    trend=trend1,
                    rawtime=rawtime,
                    rawflux=rawflux,
                    catalog=catalog,
                    row=row,
                    valid=valid
                    )
                #else:
                #    print('No figure made, vetting failed!')
            else:
                print('TLS failed')


if __name__ == '__main__':

    unique_kois_remaining = numpy.genfromtxt(
        "all_kois.csv",
        #delimiter=';',
        dtype="f8",
        names=["KOI"]
        )

    catalog = numpy.genfromtxt(
        "selection_for_search.csv",
        skip_header=1,
        delimiter=',',
        dtype="int, f8, str, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8",
        names=[
            "KIC",
            "KOI",
            "KepName",
            "Period",
            "T0",
            "T14",
            "ld1",
            "ld2",
            "R_star",
            "rad_up",
            "rad_down",
            "mass",
            "mass_up",
            "mass_down"
            ]
    )
    rows = range(len(unique_kois_remaining["KOI"]))
    period_min = 5
    period_max = 50  # days

    iterator = unique_kois_remaining["KOI"].astype(float)
    print(iterator)


    pool = Pool(processes=28) 
    pool.map(
        seach_one_koi, iterator)
    """
    for row in rows:
        seach_one_koi(str(unique_kois_remaining["KOI"][row]))
    """
