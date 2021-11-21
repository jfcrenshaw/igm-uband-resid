import pickle

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.example_libraries.optimizers import adam
from pzflow import FlowEnsemble
from pzflow.bijectors import Chain, RollingSplineCoupling, ShiftBounds, StandardScaler
from pzflow.distributions import Joint, Normal, Uniform

# get values injected to global by snakemake
# pylint: disable=undefined-variable
input_file = snakemake.input[1]
output_files = snakemake.output
config = snakemake.config["ensembles"]
# pylint: enable=undefined-variable

# load the catalog and split it into a training and test set
catalog = pd.read_pickle(input_file)

N_train = config["N_train"]
N_test = config["N_test"]

train_set = catalog[:N_train]
test_set = catalog[N_train : N_train + N_test]

# create the bijector and latent distribution for the normalizing flows

# I will use ShiftBounds to map redshifts onto the range [-5, 5]
# I set (min, max) = [-5, 5] for u band so the u band is not transformed
mins = jnp.array([0, -5])
maxes = jnp.array([train_set["redshift"].max() + 0.1, 5])

# I will use StandardScaler to standard scale the u band magnitudes
# I set (mean, std) = (0, 1) for redshifts so redshifts are not transformed
means = jnp.array([0, train_set["u"].mean()])
stds = jnp.array([1, train_set["u"].std()])

# this function will return our bijector
def get_bijector(n_conditions):
    return Chain(
        ShiftBounds(mins, maxes),
        StandardScaler(means, stds),
        RollingSplineCoupling(nlayers=2, n_conditions=n_conditions),
    )


# we will model redshift with a uniform distribution and u band with a Gaussian
# this is because redshift has a compact range while u band doesn't really
latent = Joint(Uniform((-5, 5)), Normal(1))

# now we define a function to train ensembles
def train_ensemble(ens_file: str, loss_file: str, conditional_columns: list):

    # print the name
    print("Training", ens_file)

    # create the ensemble
    ensemble = FlowEnsemble(
        data_columns=["redshift", "u"],
        bijector=get_bijector(len(conditional_columns)),
        conditional_columns=conditional_columns,
        latent=latent,
        N=config["N_flows"],
    )

    # train the ensemble on the given learning rate schedule
    step_sizes = [1e-3, 2e-4, 1e-4]
    N_epochs = [40, 40, 20]
    seeds = [123, 312, 231]
    losses = [
        ensemble.train(
            train_set,
            convolve_errs=True,
            optimizer=adam(step_size=step_size),
            epochs=epochs,
            seed=seed,
        )
        for step_size, epochs, seed in zip(step_sizes, N_epochs, seeds)
    ]

    # repackage the losses from each stage of training so that each losses
    # is a dict of flow_name: all_losses
    losses = {
        fname: [  # for each flow trained in the ensemble...
            float(loss)  # save the list of training losses
            for lossDict in losses
            for loss in lossDict[fname]
        ]
        for fname in losses[0]
    }

    # print the train and test loss
    train_loss = -np.mean(ensemble.log_prob(train_set))
    test_loss = -np.mean(ensemble.log_prob(test_set))
    print(f"train = {train_loss:.3f}    test = {test_loss:.3f}")

    # save the ensemble
    ensemble.save(ens_file)
    # and the losses
    with open(loss_file, "wb") as file:
        pickle.dump(
            {"losses": losses, "train loss": train_loss, "test loss": test_loss}, file
        )


train_ensemble(output_files[0], output_files[2], list("grizy"))
train_ensemble(output_files[1], output_files[3], list("grizy") + ["ey", "j", "h"])
