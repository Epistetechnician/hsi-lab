from hsi.evaluation.base import Spec
from hsi.evaluation.e1 import E1Spec
from hsi.evaluation.e2 import E2Spec
from hsi.evaluation.e3 import E3Spec
from hsi.evaluation.e4 import E4Spec
from hsi.evaluation.e5 import E5Spec
from hsi.evaluation.e6 import E6Spec
from hsi.evaluation.e7 import E7Spec

SPECS: dict[str, Spec] = {"E1": E1Spec(), "E2": E2Spec(), "E3": E3Spec(), "E4": E4Spec(),
                    "E5": E5Spec(), "E6": E6Spec(), "E7": E7Spec()}
