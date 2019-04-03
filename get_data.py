import numpy
import kplr

def loadkoi(koi):
    print('Getting LC for KOI', koi)
    client = kplr.API()
    koi = client.koi(koi)
    lcs = koi.get_light_curves(short_cadence=False)


if __name__ == '__main__':
    data = numpy.genfromtxt(
        "all_kois.csv",
        dtype="f8",
        names=["KOI"]
        )
    for row in range(len(data["KOI"])):
        loadkoi(data["KOI"][row])
