import numpy as np
import pandas as pd
from rail.creation.degradation import LSSTErrorModel, QuantityCut

# get values injected to global by snakemake
# pylint: disable=undefined-variable
input_file = snakemake.input[1]
output_file = snakemake.output[0]
config = snakemake.config["catalog"]
# pylint: enable=undefined-variable


# load the raw catalog
catalog = pd.read_csv(
    input_file,  # i<27.1, redshift<3.5
    delim_whitespace=True,
    comment="#",
    nrows=None,
    header=0,
    usecols=[i for i in range(1, 11)],
    names=[
        "redshift",  # truth
        "u",
        "g",
        "r",
        "i",
        "z",
        "y",
        "ey",  # Euclid
        "j",  # Euclid
        "h",  # Euclid
    ],
)

# lyman alpha extinction here !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# add photometric errors to the catalog
global_err_config = config["error_model"]
lsst_err_config = global_err_config.pop("lsst")
euclid_err_config = global_err_config.pop("euclid")
err_seed = global_err_config.pop("seed")

lsst_error_model = LSSTErrorModel(**global_err_config, **lsst_err_config)
catalog = lsst_error_model(catalog, seed=err_seed)

euclid_error_model = LSSTErrorModel(**global_err_config, **euclid_err_config)
catalog = euclid_error_model(catalog, seed=err_seed + 1)


# cut on SNR
catalog = QuantityCut(
    {
        band: {
            **lsst_error_model.get_limiting_mags(SNR, coadded=True),
            **euclid_error_model.get_limiting_mags(SNR),
        }[band]
        for band, SNR in config["SNR_cuts"].items()
    }
)(catalog)


catalog.to_pickle(output_file)
