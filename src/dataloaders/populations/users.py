from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd

from ..decorators import save_pickle, save_tsv
from ..serialize import DATA_ROOT
from .base import DataSplit, Population

nunique = dd.Aggregation(
    name="nunique",
    chunk=lambda s: s.apply(lambda x: list(set(x))),
    agg=lambda s0: s0.obj.groupby(
        level=list(range(s0.obj.index.nlevels))).sum(),
    finalize=lambda s1: s1.apply(lambda final: len(set(final))),
)


@dataclass
class UserPopulation(Population):
    """
    A cohort of users. This object stores static information about each user, as well as creates datasplits.

    :param name: Name of the population
    :param input_csv: path to the use base
    :param seed: Seed for splitting training, validation and test dataset
    :param train_val_test: Fraction of the data to be included in the three data splits.
        Must sum to 1.
    """

    name: str = "users"
    input_csv: Path = DATA_ROOT / "rawdata" / "users.csv"
    earliest_birthday: str = "01-01-1950"

    seed: int = 42
    train_val_test: Tuple[float, float, float] = (0.7, 0.15, 0.15)

    def __post_init__(self) -> None:
        """
        Perform operations right after the object was initialized
        """
        assert sum(self.train_val_test) == 1.0
        self._earliest_birthday = pd.to_datetime(
            self.earliest_birthday, format="%d-%m-%Y")

    @save_pickle(
        DATA_ROOT / "processed/populations/{self.name}/population",
        on_validation_error="error",
    )
    def population(self) -> pd.DataFrame:
        """
        Creates (or loads) the population base. 
        Here you can specify various preprocessing steps to filter out the users based on some specifications
        """

        columns = [
            "USER_ID",
            "BIRTHDAY",
            "SEX",
        ]

        result = dd.read_csv(
            self.input_csv,
            low_memory=False,
            usecols=columns,
            on_bad_lines="error",
            assume_missing=True,
            dtype={
                "USER_ID": int,  # Deal with missing values
                "SEX": "string",
            },
            blocksize="256MB",
        )

        # Code for filtering in here

        result = result.set_index("USER_ID").compute()
        assert isinstance(result, pd.DataFrame)
        return result

    @save_pickle(DATA_ROOT / "processed/populations/{self.name}/data_split")
    def data_split(self) -> DataSplit:
        """
        Split data based on :attr:`seed` using :attr:`train_val_test` as ratios
        """

        ids = self.population().index.to_numpy()
        np.random.default_rng(self.seed).shuffle(ids)
        split_idxs = np.round(np.cumsum(self.train_val_test)
                              * len(ids))[:2].astype(int)
        train_ids, val_ids, test_ids = np.split(ids, split_idxs)
        return DataSplit(
            train=train_ids,
            val=val_ids,
            test=test_ids,
        )
