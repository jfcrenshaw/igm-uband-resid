import pandas as pd
from rail.creation.degradation import LSSTErrorModel, QuantityCut

import lya_forest

# get values injected to global by snakemake
# pylint: disable=undefined-variable
input_file = snakemake.input[0]
output_files = snakemake.output
config = snakemake.config["catalog"]
# pylint: enable=undefined-variable


# load the truth catalog
truth_catalog = pd.read_csv(
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


# split off galaxies for training the ensembles
# these will not have lyman alpha extinction
train_set = truth_catalog[: config["N_train"]]
test_set = truth_catalog[config["N_train"] :]

# the test set will receive lyman alpha extinction in u band
lya_seed = config["lya_extinction"].pop("seed")
test_set = lya_forest.LyAForestExtinction(**config["lya_extinction"])(
    test_set, lya_seed
)


# now I will add photometric errors and apply quality cuts

global_err_config = config["error_model"]
lsst_err_config = global_err_config.pop("lsst")
euclid_err_config = global_err_config.pop("euclid")
err_seed = global_err_config.pop("seed")

catalogs = [train_set, test_set]

for catalog, file in zip(catalogs, output_files):

    print(catalog.shape)

    # add photometric errors to the catalog
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

    print(catalog.shape)

    catalog.to_pickle(file)
