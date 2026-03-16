from typing import Mapping, MutableMapping, Optional, Sequence, Union

import pandas as pd

TableLike = pd.DataFrame
Scalar = Union[str, int, float, bool, None]
OptionalMapping = Optional[Mapping[str, Scalar]]
MutableScalarMapping = MutableMapping[str, Scalar]
StringSequence = Sequence[str]
