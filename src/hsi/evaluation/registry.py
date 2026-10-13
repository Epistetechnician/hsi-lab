from hsi.evaluation.base import Spec
from hsi.evaluation.e1 import E1Spec
from hsi.evaluation.e2 import E2Spec

SPECS: dict[str, Spec] = {"E1": E1Spec(), "E2": E2Spec()}
