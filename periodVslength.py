import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy import stats
from expoFit import exponential_fit

def periodVLength(datas: list[tuple], lengths: list, lengthUnits="mm", fps=60):
    """
    :params: list of tuple containing the data frame, the periods, and the anti-periods, comes from data_analysis
    """
    tErr = 1/fps
    if (b:=len(datas)) != (a:=len(lengths)):
        raise OverflowError(f"Expected size of lengths is {b} but an length {b} array is given")
    periodTime = list()
    tauErr = list()
    Qs = list()
    Qerr = []
    if lengthUnits == "mm":
        lengths = [i/1000 for i in lengths]
    elif lengthUnits == "m":
        pass
    else:
        raise NameError
    params, covs = exponential_fit(datas)
    vars = [np.sqrt(np.diag(cov)) for cov in covs]
    for j, var in enumerate(vars):
        tauErr.append(var[1])
    for j, data in enumerate(datas):
        # data: tuple
        period =(np.mean(data[1]) + np.mean(data[2]))/2
        periodTime.append(period)
        tau = params[j][1]
        relaErr = max(tauErr[j]/tau, tErr / period)
        Q = np.pi * tau/period
        Qerr.append(Q * relaErr)
        print(f"for run {j+1}: {tau} +/- {tauErr[j]}")
        print(f"Q for run {j+1}: {Q} +/- {Q*relaErr}")
        Qs.append(Q)
    print(Qerr)
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2.4, 1]})
    ax1.errorbar(lengths, Qs, label="data", xerr=0.0005, yerr=Qerr, fmt='o')
    # linear fit
    fit = lambda l, a, b: a*l + b
    ppt, pcov = curve_fit(fit, lengths, Qs)
    fitErr = np.sqrt(np.diag(pcov))
    print("For Q Fit")
    print(ppt, fitErr)
    # get r2
    residuals = np.array(Qs) - fit(np.array(lengths), *ppt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((Qs - np.mean(Qs))**2)
    r2 = 1 - ss_res/ss_tot
    print("R^2:", r2)

    ax1.plot(lengths, fit(np.array(lengths), *ppt), label="Fit", color="orange", alpha=0.6)
    ax2.errorbar(lengths, np.array(Qs) - fit(np.array(lengths), *ppt), fmt='o',label="Residue")
    ax2.axhline(0, color="grey", linestyle="--")
    sigma = np.std(np.array(Qs) - fit(np.array(lengths), *ppt))
    ax2.axhline(sigma, color="grey", linestyle="--")
    ax2.axhline(-sigma, color="grey", linestyle="--")
    ax1.set_xlabel("Length (m)")
    ax1.set_ylabel("Q")
    ax2.set_xlabel("Length (m)")
    ax2.set_ylabel("Residue")
    ax1.legend()
    # font size 15
    ax1.text(0.95, 0.01, fr"$Q(L)={ppt[0]:.2f}L+{ppt[1]:.2f}$", ha='right', va='bottom', transform=ax1.transAxes)
    ax1.set_title("Q vs Length")
    plt.tight_layout()
    plt.show()
    fit = lambda L, k ,n: k * np.power(L, n)
    ppt, pcov = curve_fit(fit, lengths, periodTime)
    fitErr = np.sqrt(np.diag(pcov))
    print("For Period Fit")
    print(ppt, fitErr)
    # get r2
    residuals = np.array(periodTime) - fit(np.array(lengths), *ppt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((periodTime - np.mean(periodTime))**2)
    r2 = 1 - ss_res/ss_tot
    print("R^2:", r2)
    # plot the main graph, error bars, and the fit, and residue on another subplot
    # do it like plot_expo
    # plot the fit
    yerr= tErr
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2.4, 1]})
    ax1.errorbar(lengths, periodTime, label="data", xerr=0.0005, yerr=yerr, fmt='o')
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.plot(lengths, fit(np.array(lengths), *ppt), label="Fit", color="orange", alpha=0.6)
    ax2.errorbar(lengths, np.array(periodTime) - fit(np.array(lengths), *ppt), fmt='o',label="Residue")
    sigma =  np.std(np.array(periodTime) - fit(np.array(lengths), *ppt))
    ax2.axhline(sigma, color="grey", linestyle="--")
    ax2.axhline(-sigma, color="grey", linestyle="--")
    ax2.axhline(0, color="grey", linestyle="--")
    ax1.set_xlabel("Length (m)")
    ax1.set_ylabel("Period (s)")
    ax2.set_xlabel("Length (m)")
    ax2.set_ylabel("Residue (s)")
    ax1.legend()
    ax1.set_title("Period vs Length, log-log plot")
    plt.tight_layout()
    # write the results on the bottom right of ax1 as a equation T = kL^n
    ax1.text(0.95, 0.01, fr"$T(L)={ppt[0]:.2f}L^{{{ppt[1]:.2f}}}$", ha='right', va='bottom', transform=ax1.transAxes)
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2.4, 1]})
    ax1.errorbar(lengths, periodTime, label="data", xerr=0.0005, yerr=yerr, fmt='o')
    ax1.plot(lengths, fit(np.array(lengths), *ppt), label="Fit", color="orange", alpha=0.6)
    ax2.errorbar(lengths, np.array(periodTime) - fit(np.array(lengths), *ppt), fmt='o', label="Residue")
    sigma = np.std(np.array(periodTime) - fit(np.array(lengths), *ppt))
    ax2.axhline(sigma, color="grey", linestyle="--")
    ax2.axhline(-sigma, color="grey", linestyle="--")
    ax2.axhline(0, color="grey", linestyle="--")
    ax1.set_xlabel("Length (m)")
    ax1.set_ylabel("Period (s)")
    ax2.set_xlabel("Length (m)")
    ax2.set_ylabel("Residue (s)")
    ax1.legend()
    ax1.set_title("Period vs Length, regular plot")
    plt.tight_layout()
    # write the results on the bottom right of ax1 as a equation T = kL^n
    ax1.text(0.95, 0.01, fr"$T(L)={ppt[0]:.2f}L^{{{ppt[1]:.2f}}}$", ha='right', va='bottom', transform=ax1.transAxes)
    plt.show()
    return




