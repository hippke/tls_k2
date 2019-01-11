import numpy
import scipy
import everest
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
from transitleastsquares import transitleastsquares, cleaned_array, transit_mask, \
    catalog_info


def get_planets(EPIC_id):
    periods = []
    t0s = []
    tdurs = []
    for row in rows:
        if catalog["EPIC_id"][row] == EPIC_id:
            periods.append(catalog["period"][row])
            t0s.append(catalog["t0"][row])
            tdurs.append(catalog["tdur"][row])
    return periods, t0s, tdurs


def loadfile(EPIC_id):
    star = everest.Everest(EPIC_id)
    t = numpy.delete(star.time, star.badmask)
    y = numpy.delete(star.fcor, star.badmask)
    t = numpy.array(t[~numpy.isnan(y)], dtype='float32')
    y = numpy.array(y[~numpy.isnan(y)], dtype='float32')
    return cleaned_array(t, y)


if __name__ == '__main__':
    MEDIAN_KERNEL_WINDOW = 25  # 25 cadences = 12.5 hrs
    catalog = numpy.genfromtxt(
        "k2cands.csv",
        skip_header=1,
        delimiter=',',
        dtype="int32, f8, f8, f8",
        names=["EPIC_id", "period", "t0", "tdur"]
    )
    rows = range(len(catalog["EPIC_id"]))

    for row in rows:
        EPIC_id = catalog["EPIC_id"][row]
        print('New planet EPIC_id', EPIC_id)
        time, flux = loadfile(EPIC_id)

        # Remove known planets
        periods, t0s, tdurs = get_planets(EPIC_id)
        for no in range(len(periods)):
            print('Removing planet', periods[no], t0s[no], tdurs[no])
            intransit = transit_mask(
                time,
                periods[no],
                2 * tdurs[no],
                t0s[no]
                )
            flux = flux[~intransit]
            time = time[~intransit]
            time, flux = cleaned_array(time, flux)

        # Detrend data and remove high outliers
        trend = scipy.signal.medfilt(flux, MEDIAN_KERNEL_WINDOW)
        flux = flux /trend
        flux = sigma_clip(flux, sigma_upper=2, sigma_lower=6)

        # Search flushed data for additional planets
        ab, mass, mass_min, mass_max, radius, radius_min, radius_max = catalog_info(EPIC_id)
        model = transitleastsquares(time, flux)
        results = model.power(
            n_transits_min=1,
            u=ab,
            oversampling_factor=5,
            duration_grid_step=1.05)
        print('Signal detection efficiency (SDE):', format(results.SDE, '.1f'))

        # Figure of power spectrum
        plt.figure()
        ax = plt.gca()
        ax.axvline(results.period, alpha=0.4, lw=3)
        plt.xlim(numpy.min(results.periods), numpy.max(results.periods))
        for n in range(2, 10):
            ax.axvline(n*results.period, alpha=0.4, lw=1, linestyle="dashed")
            ax.axvline(results.period / n, alpha=0.4, lw=1, linestyle="dashed")
        plt.ylabel(r'SDE')
        plt.xlabel('Period (days)')
        plt.plot(results.periods, results.power, color='black', lw=0.5)
        plt.xlim(0, max(results.periods))
        plt.savefig(str(EPIC_ID) + '_power.pdf', bbox_inches='tight')

        # Figure of phase-folded transit
        plt.figure()
        plt.plot(results.model_folded_phase, results.model_folded_model, color='red')
        plt.scatter(results.folded_phase, results.folded_y, color='blue', s=10, alpha=0.5, zorder=2)
        plt.xlim(0.48, 0.52)
        plt.xlabel('Phase')
        plt.ylabel('Relative flux')
        plt.savefig(str(EPIC_ID) + '_fold.pdf', bbox_inches='tight')
