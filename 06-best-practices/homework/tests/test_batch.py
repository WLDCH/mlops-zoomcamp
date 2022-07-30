from datetime import datetime

import numpy as np
import pandas as pd
from deepdiff import DeepDiff

import batch


def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)


data = [
    (None, None, dt(1, 2), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
    (1, 1, dt(1, 2, 0), dt(2, 2, 1)),
]

columns = ["PUlocationID", "DOlocationID", "pickup_datetime", "dropOff_datetime"]
df = pd.DataFrame(data, columns=columns)


def test_prepare_data():

    actual_output = batch.prepare_data(df, ["PUlocationID", "DOlocationID"])
    expected_output = pd.DataFrame(
        columns=columns + ["duration"],
        data=[
            ("-1", "-1", dt(1, 2), dt(1, 10), 8.0),
            ("1", "1", dt(1, 2), dt(1, 10), 8.0),
        ],
    )

    diff = DeepDiff(
        actual_output.to_dict("records"),
        expected_output.to_dict("records"),
        significant_digits=1,
    )
    print(f"diff={diff}")

    assert "type_changes" not in diff
    assert "values_changed" not in diff
