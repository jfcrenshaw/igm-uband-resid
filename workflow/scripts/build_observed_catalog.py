import pickle

import pandas as pd
from rail.creation.degradation import LSSTErrorModel, QuantityCut

import lya_forest

# get values injected to global by snakemake
# pylint: disable=undefined-variable
input_file = snakemake.input[1]
output_files = snakemake.output
config = snakemake.config["catalog"]
# pylint: enable=undefined-variable


# load the truth catalog
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


# Simulate Lyman Alpha Forest
# ---------------------------

# simulate lines of sight
lya = lya_forest.LyAForestExtinction(**config["lya_extinction"])
lines_of_sight = lya._simulate_lines_of_sight(catalog.index, 0)

# save lines of sight
with open(output_files[0], "wb") as file:
    pickle.dump(lines_of_sight, file)

# apply uband decrements to the catalog
catalog = lya._apply_uband_decrements(catalog, lines_of_sight)


# Simulate Photometric Errors
# ---------------------------

# pull out config values
global_err_config = config["error_model"]
lsst_err_config = global_err_config.pop("lsst")
euclid_err_config = global_err_config.pop("euclid")

# add photometric errors to the catalog
lsst_error_model = LSSTErrorModel(**global_err_config, **lsst_err_config)
catalog = lsst_error_model(catalog, seed=1)
euclid_error_model = LSSTErrorModel(**global_err_config, **euclid_err_config)
catalog = euclid_error_model(catalog, seed=2)


# Apply cuts to SNR
# -----------------
catalog = QuantityCut(
    {
        band: {
            **lsst_error_model.get_limiting_mags(SNR, coadded=True),
            **euclid_error_model.get_limiting_mags(SNR),
        }[band]
        for band, SNR in config["SNR_cuts"].items()
    }
)(catalog)


# Save the final catalog
# ----------------------
catalog.to_pickle(output_files[1])
