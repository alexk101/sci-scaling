import os
import sys
ROCM_SMI_PATH = os.environ.get("ROCM_SMI_PATH", None)
if ROCM_SMI_PATH is not None:
    sys.path.append(ROCM_SMI_PATH)
    import rocm_smi
else:
    raise ValueError("ROCM_SMI_PATH not set")

class RocmProfiler:
    def __init__(self):
        rocm_smi.initializeRsmi()

    def getPower(self, device):
        return rocm_smi.getPower(device)

    def listDevices(self):
        return rocm_smi.listDevices()

    def getMemInfo(self, device):
        (memUsed, memTotal) = rocm_smi.getMemInfo(device, "vram")
        return round(float(memUsed)/float(memTotal) * 100, 2)
    
    def getUtilization(self, device):
        return rocm_smi.getGpuUse(device)
    
    def getTemp(self, device, sensor=None):
        if sensor is None:
            return rocm_smi.findFirstAvailableTemp(device)
        else:
            return rocm_smi.getTemp(device, sensor)