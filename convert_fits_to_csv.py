import numpy
import os
from astropy.io import fits


FITS_ROOT_FOLDER = 'P:/P/Dok/'
CSV_TARGET_FOLDER = 'P:/P/Dok/'


def cleaned_array(t, y, dy=None):
    """Takes numpy arrays with masks and non-float values.
    Returns unmasked cleaned arrays."""

    # Start with empty Python lists and convert to numpy arrays later (reason: speed)
    clean_t = []
    clean_y = []
    if dy is not None:
        clean_dy = []

    # Cleaning numpy arrays with both NaN and None values is not trivial, as the usual
    # mask/delete filters do not accept their simultanous ocurrence.
    # Instead, we iterate over the array once; this is not Pythonic but works reliably.
    for i in range(len(y)):
        if (y[i] is not None) and (y[i] is not numpy.nan) and \
        (y[i] >= 0) and (y[i] < numpy.inf):
            clean_y.append(y[i])
            clean_t.append(t[i])
            if dy is not None:
                clean_dy.append(dy[i])
    clean_t = numpy.array(clean_t, dtype=float)
    clean_y = numpy.array(clean_y, dtype=float)
    if dy is None:
        return clean_t, clean_y
    else:
        clean_dy = numpy.array(clean_dy, dtype=float)
        return clean_t, clean_y, clean_dy


def convert_file(root, name):
    """Takes EVEREST FITS file and saves the good (time, flux) values into a CSV"""

    try:
        fits_filename = os.path.join(root, name)
        with fits.open(fits_filename) as hdus:
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
        t, y = cleaned_array(t, y)
        EPIC_id = name[20:29]
        output_filename = 'EPIC' + str(EPIC_id) + '.csv'
        numpy.savetxt(CSV_TARGET_FOLDER + output_filename, numpy.column_stack((t, y)), fmt='%1.8f')
        print('Conversion OK for ' + output_filename + ' from ' + fits_filename)
    except:
        print('Conversion failed for', fits_filename)


if __name__ == '__main__':
    for root, dirs, files in os.walk(FITS_ROOT_FOLDER):
        for name in files:
            if name.endswith('.fits'):
                convert_file(root, name)
