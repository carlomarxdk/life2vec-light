from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import dask.dataframe as dd
import pandas as pd

from ..decorators import save_parquet
from ..ops import sort_partitions
from ..serialize import DATA_ROOT
from .base import FIELD_TYPE, TokenSource, Binned


@dataclass
class SyntheticLaborSource(TokenSource):
    """This generates tokens based on information from the Synthetic Labor dataset.

    :param input_csv: CSV file from which to load the Synthetic Labor Dataset.
    :param earliest_start: The earliest start date of a hospital encounter.
    """

    name: str = "synth_labor"
    fields: List[FIELD_TYPE] = field(
        default_factory=lambda: [
            Binned("INCOME", prefix="INCOME", n_bins=50),
            "CITY",
            "OCCUPATION",
            "INDUSTRY"]
    )

    input_csv: Path = DATA_ROOT / "rawdata" / "synth_labor.csv"

    @save_parquet(
        DATA_ROOT / "processed/sources/{self.name}/tokenized",
        on_validation_error="error",
        # You might want to run the verification (aka that indexes are indeed sorted)
        verify_index=False,
    )
    def tokenized(self) -> dd.DataFrame:
        """
        Loads the indexed data, then tokenizes it.
        Do some preprocessing on the raw data
        """

        result = (
            self.indexed()
            .pipe(sort_partitions, columns=["RECORD_DATE"])[
                ["RECORD_DATE", *self.field_labels()]
            ]
            .assign(
                OCCUPATION=lambda x: "OCC_" + x.OCCUPATION,
                CITY=lambda x: "CITY_" + x.CITY,
                INDUSTRY=lambda x: "IND_" + x.INDUSTRY,
            )
            .assign(INCOME=lambda x: x.INCOME.where(lambda x: x > 0, pd.NA))
            # This is important for performance
            .reset_index().set_index("USER_ID", npartitions="auto", sorted=True)
        )

        assert isinstance(result, dd.DataFrame)
        return result

    # FOR BIG DATA, it is often useful to save intermediate results
    # @save_parquet(
    #    DATA_ROOT / "interim/sources/{self.name}/indexed",
    #    on_validation_error="recompute",
    # )
    def indexed(self) -> dd.DataFrame:
        """Loads the parsed data, sets the index, then saves the indexed data"""
        result = (self.parsed()
                  .pipe(lambda x: x.categorize(x.select_dtypes("string").columns))
                  .set_index("USER_ID")
                  .pipe(lambda x: x.astype(
                      {k: "string" for k in x.select_dtypes(
                          "category").columns}
                  )
        )
        )

        assert isinstance(result, dd.DataFrame)
        return result

    # @save_parquet(
    #    DATA_ROOT / "interim/sources/{self.name}/parsed",
    #    on_validation_error="error",
    #    verify_index=False,
    # )
    def parsed(self) -> dd.DataFrame:
        """
        Parses the CSV file, applies some basic filtering, then saves the result
        as compressed parquet file, as this is easier to parse than the CSV for the
        next steps
        """

        columns = [
            "USER_ID",
            "OCCUPATION",
            "CITY",
            "INDUSTRY",
            "RECORD_DATE",
            "INCOME"
        ]

        ddf = dd.read_csv(
            self.input_csv,
            low_memory=False,
            usecols=columns,
            on_bad_lines="error",
            assume_missing=True,
            dtype={
                "USER_ID": int,  # Deal with missing values
                "OCCUPATION": "string",
                "CITY": "string",
                "INDUSTRY": "string",
                "INCOME": float
            },
            blocksize="256MB",
        )

        # Drop missing values and deal with datatypes
        # YOU might do some filtering here
        ddf = (
            ddf.assign(
                USER_ID=lambda x: x.USER_ID.astype(int),
                RECORD_DATE=lambda x: dd.to_datetime(
                    x.RECORD_DATE,
                    format="%Y-%m-%d",
                    errors="coerce",
                ),
            )
        )

        assert isinstance(ddf, dd.DataFrame)
        return ddf
