import os
import sys
from typing import Dict, Any, Tuple
ROCM_SMI_PATH = os.environ.get("ROCM_SMI_PATH", None)
if ROCM_SMI_PATH is not None:
    sys.path.append(ROCM_SMI_PATH)
    import rocm_smi
else:
    raise ValueError("ROCM_SMI_PATH not set")

class RocmProfiler:
    def __init__(self):
        rocm_smi.initializeRsmi()

    def getPower(self, device) -> Dict[str, Any]:
        """Gets the power usage of a given GPU.

        Args:
            device (int): The device index to get the power usage for.

        Returns:
            Dict[str, Any]: A dictionary containing the power usage of the given GPU with the following keys:
                "power": str,
                "power_type": str,
                "unit": str,
                "ret": int
        """
        return rocm_smi.getPower(device)

    def listDevices(self):
        return rocm_smi.listDevices()

    def getMemInfo(self, device):
        (memUsed, memTotal) = rocm_smi.getMemInfo(device, "vram")
        return round(float(memUsed)/float(memTotal) * 100, 2)
    
    def getUtilization(self, device) -> float:
        return rocm_smi.getGpuUse(device)
    
    def getTemp(self, device, sensor=None) -> Tuple[str, float]:
        """Gets the temperature of a given GPU.

        Args:
            device (int): The device index to get the temperature for.
            sensor (str, optional): The sensor to get the temperature for. Defaults to None.

        Returns:
            Tuple[str, float]: A tuple containing the sensor name and the temperature in Celsius.
        """
        if sensor is None:
            return rocm_smi.findFirstAvailableTemp(device)
        else:
            return rocm_smi.getTemp(device, sensor)