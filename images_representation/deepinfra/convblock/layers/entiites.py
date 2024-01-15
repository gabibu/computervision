

from enum import Enum

class TensorType(Enum):
   D2 = 2
   D3  = 3


class PaddingType(Enum):
   SAME = 1
   VALID = 2


class LayersTypes(Enum):
   AVG = 1
   MAX = 2
   BN = 3
   ACTIV = 4
   DROP = 5
   UP = 6
   CONV = 7