import time as ttime
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as pl
import emcee

from occultquad import occultquad



def model_no_moon_nested_2(time,ratio_P,a_o_R,impact_B,phase_B,period_B, u1,u2,verbosity=0):
    const=a_o_R
    if verbosity>1:
        print(const)
    z_B_x=const*np.sin(2.0*np.pi*(time/period_B-phase_B))
    z_B_y=impact_B*np.cos(2.0*np.pi*(time/period_B-phase_B))
    z_coord=np.cos(2.0*np.pi*(time/period_B-phase_B))
    z_B_y[z_coord<0.0]=10.0*a_o_R#exclude if behind star

    z_B=np.sqrt(z_B_x**2.0+z_B_y**2.0)


    transit_signal_P=np.ones(len(z_B))

    transit_signal_P[z_B<1.0+ratio_P] = occultquad(z_B[z_B<1.0+ratio_P], u1, u2, ratio_P)[0]

    return transit_signal_P



def transit_model(time, ratio, impact, duration, phase, period,q1,q2,oversample_factor=1,verbosity=1):
    if q1>1.0 or q1<0.0:
        return -np.inf
    if q2>1.0 or q2<0.0:
        return -np.inf
    if np.abs(impact)>1.0+ratio:#No transit
        if verbosity>0:
            print("impact too high (no transit)!")
        return -np.inf
    if np.abs(impact)>1.0:#No aoR calculation possible
        if verbosity>0:
            print("impact too high (aoR)!")
        return -np.inf
    u1=2.0*np.sqrt(q1)*q2
    u2=np.sqrt(q1)*(1.0-2.0*q2)
    a_o_R=period*np.sqrt(1.0-impact**2.0)/(np.pi*duration)
    if a_o_R<=1.0:
        return -np.inf
    if ratio<=0 or ratio>0.5:
        return -np.inf

    exposure_time=time[1]-time[0]
    
    time_h=(np.array(time)[:,None]+(np.arange(oversample_factor+2)[None,1:-1]*1.0/(oversample_factor+1.0)-0.5)*exposure_time).reshape(-1)

    fl=np.nanmean(model_no_moon_nested_2(time_h, ratio, a_o_R, impact, phase, period,u1,u2).reshape(-1,oversample_factor),axis=1)
    if not np.all(np.isfinite(fl)):
        if verbosity>0:
            print("Non finite element(s) in model light curve!")
        return -np.inf
    return fl


if __name__ == "__main__":


    R_star=0.85
    M_star=0.84
    q1=0.45
    q2=0.35

    R_P = 11.
    M_P = 317.
    period = 30.
    impact=0.1
    phase=0.0

    ratio_P=(R_P*0.009168)/R_star
    a_o_R=(M_star*(period/365.25)**2.)**(1./3.)/(R_star*0.00465)

    u1=2.0*np.sqrt(q1)*q2
    u2=np.sqrt(q1)*(1.0-2.0*q2)

    duration=period/np.pi/a_o_R*np.sqrt(1.0-impact**2.)


    #staring guess
    p0 = [ratio_P,impact,duration,phase,period,q1,q2]

    #Faking a light curve.
    noise_level=0.0001
    time = np.arange(10000)*0.5/24.-10.# in days
    flux = transit_model(time,*list(p0))+rnd.rand(len(time))*noise_level
    error_flux = np.ones(len(time))*noise_level

    #pl.scatter(time,flux)
    #pl.show()



    model_kwargs={"verbosity":0,"oversample_factor":5}

    def loglike(params):
        model=transit_model(time,*list(params), **model_kwargs)
        if type(model) == float:
            return -np.inf
        return -0.5*((flux-model)**2.0/error_flux**2.0).sum()

    n_walkers=100

    p_start = np.array(p0)+rnd.rand(n_walkers,len(p0))*0.001



    sampler = emcee.EnsembleSampler(n_walkers,len(p0),loglike,threads=4)

    sampler.run_mcmc(p_start,1000)

    chain=sampler.chain[:]

    ttime.sleep(2)

    for c in chain:
        print(c[:,0])
        pl.plot(c[:,0],c="k",lw=1)
    pl.show()

