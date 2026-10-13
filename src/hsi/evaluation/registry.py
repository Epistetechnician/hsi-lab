from hsi.evaluation.base import Spec
from hsi.evaluation.e1 import E1Spec
from hsi.evaluation.e2 import E2Spec
from hsi.evaluation.e3 import E3Spec
from hsi.evaluation.e7 import E7Spec

SPECS: dict[str, Spec] = {"E1": E1Spec(), "E2": E2Spec(), "E3": E3Spec(), "E7": E7Spec()}
