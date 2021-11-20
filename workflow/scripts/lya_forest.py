from typing import Tuple, Iterable

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.modeling.functional_models import Voigt1D
from rail.creation.degradation import Degrader
from scipy.stats import truncnorm, uniform

WAVE_ALPHA = 1215.67  # wavelength of lyman alpha in angstroms


class _TruncatedPowerLaw:
    def __init__(self, alpha: float, xmin: float, xmax: float):

        # save the params
        self.alpha = alpha
        self.xmin = xmin
        self.xmax = xmax

        # validate the params
        self._validate_params()

        # calculate the normalization
        self._opa = 1 + alpha
        self.norm = self._opa / (xmax ** self._opa - xmin ** self._opa)

    def _validate_params(
        self,
    ):

        # check that they are numbers
        for name in ["alpha", "xmin", "xmax"]:
            param = getattr(self, name)
            if not isinstance(param, float) and not isinstance(param, int):
                raise TypeError(f"{name} must be a number.")

        # check that xmax > xmin
        if self.xmin > self.xmax:
            raise ValueError("xmax must be greater than xmin.")

        # check that xmin >= 0
        if self.xmin < 0:
            raise ValueError("xmin must be non-negative.")

        # if alpha < 0, check that xmin > 0
        if self.alpha < 0 and self.xmin == 0:
            raise ValueError("For negative power laws, xmin must be greater than zero.")

    def _in_bounds(self, x: np.ndarray) -> np.ndarray:
        return (x >= self.xmin) & (x <= self.xmax)

    def _protect_zero(self, x: np.ndarray) -> np.ndarray:
        return np.where((self.alpha < 0) & (x == 0), self.xmin / 2, x)

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return self.norm * x ** self.alpha

    def pdf(self, x: np.ndarray) -> np.ndarray:
        x = self._protect_zero(x)
        return np.where(self._in_bounds(x), self._pdf(x), 0)

    def _cdf(self, x: np.ndarray) -> np.ndarray:
        return self.norm / self._opa * (x ** self._opa - self.xmin ** self._opa)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        x = self._protect_zero(x)
        return np.where(self._in_bounds(x), self._cdf(x), x > self.xmax)

    def ppf(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0, 1)
        return (
            x * (self.xmax ** self._opa - self.xmin ** self._opa)
            + self.xmin ** self._opa
        ) ** (1 / self._opa)

    def sample(self, size: int, seed: int = None) -> np.ndarray:
        u = uniform.rvs(size=size, random_state=seed)
        return self.ppf(u)


def _voigt_profile(wavelen: np.ndarray, b: np.ndarray) -> np.ndarray:

    # some constants
    c = 2.998e18 * u.AA / u.s  # speed of light in AA/s
    wave_alpha = WAVE_ALPHA * u.AA  # lyman alpha wavelength in AA
    nu_alpha = c / wave_alpha  # lyman alpha frequency in 1/s
    A_alpha = 6.25e8  # Einstein coefficient

    # convert wavelength to frequency
    nu = c / (wavelen * u.AA)  # 1/s

    # width of the Lorentzian
    fwhm_L = 2 * (A_alpha / u.s) / (4 * np.pi)  # 1/s

    # width of the thermal profile
    b_AA = b * 1e13 * u.AA / u.s  # convert doppler param km/s -> AA/s
    sig_G = b_AA / np.sqrt(2) * nu_alpha / c  # sigma in 1/s
    fwhm_G = 2 * np.sqrt(2 * np.log(2)) * sig_G

    # set the amplitude of the Lorentzian so that the profile is normalized
    amplitude_L = 2 / (np.pi * fwhm_L)

    # calculate Voigt profile using Astropy's most accurate algorithm
    # we need high accuracy as we will be multiplying by huge column densities
    vp = Voigt1D(
        x_0=nu_alpha,
        amplitude_L=amplitude_L,
        fwhm_L=fwhm_L,
        fwhm_G=fwhm_G[:, None],
        method="wofz",
    )(nu)

    # convert density in frequency space to density in wavelength space
    vp = c * vp / (wavelen * u.AA) ** 2

    return vp.value


class LyAForestExtinction(Degrader):
    def __init__(
        self,
        z_min: float = 1.63,
        z_max: float = 2.36,
        gamma: float = 2.7,
        logNHI_min: float = 12.5,
        logNHI_max: float = 17,
        beta: float = -1.46,
        b_mu: float = 30,
        b_sig: float = 12,
        b_min: float = 24,
        b_max: float = np.inf,
        N_bs: int = 100_000,
        N_clouds: float = None,
    ):

        # save the parameters
        self.params = {
            "z_min": z_min,
            "z_max": z_max,
            "gamma": gamma,
            "logNHI_min": logNHI_min,
            "logNHI_max": logNHI_max,
            "beta": beta,
            "b_mu": b_mu,
            "b_sig": b_sig,
            "b_min": b_min,
            "b_max": b_max,
            "N_bs": N_bs,
        }

        # if N_clouds is passed, use that
        if N_clouds is not None:
            self.params["N_clouds"] = N_clouds
        # else use the normalization from Hu et al 1995
        else:
            self.params["N_clouds"] = self._hu_et_al_norm()

        # setup the distributions
        self._opz_pl = _TruncatedPowerLaw(  # powerlaw: (1+z)^gamma
            self.params["gamma"],
            1 + self.params["z_min"],
            1 + self.params["z_max"],
        )
        self._NHI_pl = _TruncatedPowerLaw(  # powerlaw: NHI^beta
            self.params["beta"],
            10 ** self.params["logNHI_min"],
            10 ** self.params["logNHI_max"],
        )
        self._b_tn = truncnorm(  # truncated normal distribution for b
            a=(self.params["b_min"] - self.params["b_mu"]) / self.params["b_sig"],
            b=(self.params["b_max"] - self.params["b_mu"]) / self.params["b_sig"],
            loc=self.params["b_mu"],
            scale=self.params["b_sig"],
        )

        # load the u band throughput
        self._u_wave, self._u_T = np.genfromtxt(
            "data/raw/lsst_u_bandpass.dat", unpack=True
        )
        u_R = self._u_wave * self._u_T
        u_R /= np.trapz(u_R, self._u_wave)
        self._u_R = lambda wave: np.interp(wave, self._u_wave, u_R)

        # below we will go ahead an sample some b's and compute the associated
        # voigt profiles because this is the computational bottle neck

        # pre-sample b's
        self._presampled_bs = self._b_tn.rvs(N_bs, random_state=0)

        # pre-compute voigt profiles
        # this wavelength grid works for 12 < log10(NHI) < 18
        self._wave_grid = np.linspace(WAVE_ALPHA - 2, WAVE_ALPHA + 2, 100)
        self._precomputed_voigts = {
            b: voigt
            for b, voigt in zip(
                self._presampled_bs,
                _voigt_profile(self._wave_grid, self._presampled_bs),
            )
        }

    def _hu_et_al_norm(
        self,
    ) -> float:
        """Set the average number of clouds using the data from Hu et al 1995:
        https://arxiv.org/abs/astro-ph/9507047v1
        """

        # create an NHI powerlaw that covers both the Hu range of NHI
        # and the range of NHI we are modeling
        NHI_ref_pl = _TruncatedPowerLaw(
            self.params["beta"],
            min(10 ** 12.3, 10 ** self.params["logNHI_min"]),
            max(10 ** 14.41, 10 ** self.params["logNHI_max"]),
        )

        # numbers of clouds in logNHI bins (Table 3)
        logNHI_bins = np.array([12.3, 12.6, 12.9, 13.2, 13.51, 13.81, 14.11, 14.41])
        incompleteness = np.array([0.25, 0.52, 0.82, 1.03, 1.11, 1.08, 1.01])
        N_clouds_per_bin = np.array(
            [109, 188, 197, 173, 133, 88, 75]
        )  # summed over all quasars

        # calculate expected clouds per quasar
        integrated_prob = np.diff(NHI_ref_pl.cdf(10 ** logNHI_bins))
        N_clouds = np.mean(N_clouds_per_bin / (incompleteness * integrated_prob) / 4)

        # N_clouds above is expected number in range 12.3 < log10(NHI) < 14.41
        # need to reweight this number for the range of NHI we are modeling
        N_clouds = (
            N_clouds
            * (
                NHI_ref_pl.cdf(10 ** self.params["logNHI_max"])
                - NHI_ref_pl.cdf(10 ** self.params["logNHI_min"])
            )
            / (NHI_ref_pl.cdf(10 ** 14.41) - NHI_ref_pl.cdf(10 ** 12.3))
        )

        return N_clouds

    def _sample_clouds(
        self, seed: int = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        # draw number of clouds along line of sight from Poisson distribution
        rng = np.random.default_rng(seed)
        N_clouds = rng.poisson(self.params["N_clouds"])

        # draw column densities, doppler parameters, and redshifts
        NHIs = self._NHI_pl.sample(N_clouds, seed=rng.integers(int(1e9)))
        bs = rng.choice(self._presampled_bs, size=N_clouds)
        zs = self._opz_pl.sample(N_clouds, seed=rng.integers(int(1e9))) - 1

        return NHIs, bs, zs

    def _equivalent_width(self, NHIs: np.ndarray, bs: np.ndarray) -> np.ndarray:

        # get the voigt profiles for these doppler parameters
        # voigts = _voigt_profile(self._wave_grid, bs)
        voigts = np.array([self._precomputed_voigts[b] for b in bs])

        # calculate optical depths; normalization and formula from
        # Mo, van den Bosch, & White, Section 16.4.4
        taus = 4.54e-18 * NHIs[:, None] * self._wave_grid * voigts

        # integrate to get equivalent widths
        eqWs = np.trapz(1 - np.exp(-taus), self._wave_grid)

        return eqWs

    def _simulate_lines_of_sight(self, index: Iterable, seed: int = None) -> dict:

        rng = np.random.default_rng(seed)

        lines_of_sight = {}
        for idx in index:

            # simulate clouds along line of sight
            NHIs, bs, zs = self._sample_clouds(seed=rng.integers(int(1e18)))

            # calculate the rest-frame equivalent widths for the clouds
            eqWs = self._equivalent_width(NHIs, bs)

            # calculate the bandpass response at the corresponding wavelengths
            Rs = self._u_R(WAVE_ALPHA * (1 + zs))

            # calculate u band decrements due to lyman alpha scattering in clouds
            u_decr = -2.5 * np.log10(1 - (eqWs * (1 + zs) * Rs).sum())

            # calculate effective optical depth in the u band
            tau_eff = u_decr * np.log(10) / 2.5

            # save info about line of sight
            lines_of_sight[idx] = {
                "N_clouds": NHIs.size,
                "NHI": NHIs,
                "b": bs,
                "z": zs,
                "eqW": eqWs,
                "u_decr": u_decr,
                "tau_eff": tau_eff,
            }

        return lines_of_sight

    def __call__(self, data: pd.DataFrame, seed: int = None) -> pd.DataFrame:

        # simulate lines of sight
        lines_of_sight = self._simulate_lines_of_sight(data.index, seed)

        # pull out u band decrements
        u_decrs = [los["u_decr"] for los in lines_of_sight.values()]

        # apply u band decrements
        obsData = data.copy()
        obsData["u"] = obsData["u"] + np.array(u_decrs)

        return obsData
