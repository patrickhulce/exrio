from typing import Any, Optional, overload

import numpy as np

TRANSFORM_DIR_FORWARD: int
TRANSFORM_DIR_INVERSE: int

class CPUProcessor:
    def applyRGB(self, pixels: np.ndarray[Any, Any]) -> None: ...
    def applyRGBA(self, pixels: np.ndarray[Any, Any]) -> None: ...

class Processor:
    def getDefaultCPUProcessor(self) -> CPUProcessor: ...

class ConfigInstance:
    def CreateFromBuiltinConfig(self, config: str) -> ConfigInstance: ...
    def CreateFromFile(self, config_path: str) -> ConfigInstance: ...
    @overload
    def getProcessor(self, from_colorspace: str, to_colorspace: str) -> Processor: ...
    @overload
    def getProcessor(
        self, from_colorspace: str, to_colorspace: str, view: str, direction: int
    ) -> Processor: ...
    def getProcessor(
        self,
        from_colorspace: str,
        to_colorspace: str,
        view: Optional[str] = None,
        direction: int = 0,
    ) -> Processor: ...

def Config() -> ConfigInstance: ...
