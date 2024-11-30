import asyncio, aiohttp, aiofiles, json, random, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from datetime import dateatime, timedelta
from pathlib import Path
from typing import NamedTuple
from itertools import chain
from pycivitai import CivitAI
import psutil,  speedtest
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# GPU Monitoring setup
class GPUMonitor:
    def __init__(self):
        self.is_nvidia = False
        try:
            import nvidia_smi
            self.nvidia_smi = nvidia_smi
            self.nvidia_smi.nvmlInit()
            self.is_nvidia = True
            print("NVIDIA GPU monitoring enabled")
        except ImportError:
            print("Development environment detected (M3). NVIDIA monitoring will be enabled on deployment.")
            
    async def get_gpu_info(self):
        if not self.is_nvidia:
            return {"gpu_free": None, "gpu_total": None}
        
        try:
            handle = self.nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            info = self.nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            return {
                "gpu_free": info.free / (1024**3), # GB
                "gpu_total": info.total / (1024**3), # GB
                "gpu_used": info.used / (1024**3) # GB
            }
        except Exception as e:
            print(f"GPU monitoring error: {e}")
            return {"gpu_free": None, "gpu_total":None}
        
# Modified Core Configuration
CFG = {
    'API_KEY': "9dca6f30e0b52ae663be7ffb0e15a4c2",
    'PATHS': {
        'base': Path("/home/Ubuntu/apps/ComfyUI"),
        'models': lambda p: p/'models',
        'output': lambda p: p/'output'
    },
    'NET': {'chunk_min': 1024**2, 'chunk_max': 16*1024**2, 'check_interval': 300},
    'SYNC': {'retention_mins': 30, 'check_interval': 60},
    'SYSTEM': {
        'min_space_gb': 30,
        'max_workers': min(psutil.cpu_count()-2, 8),
        'gpu_monitor': GPUMonitor()
    }
}

class Model(NamedTuple):
    name: str; id: str; type: str; dest: Path; size: float; priority: float = 1.0

async def check_system_resources():
    """Universal system resource monitoring"""
    resources = {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_free': psutil.disk_usage(str(CFG['PATHS']['base'])).free / (1024**3)
    }
    
    # Get GPU info regardless of development or deployment environment
    gpu_info = await CFG['SYSTEM']['gpu_monitor'].get_gpu_info()
    resources.update(gpu_info)
    
    return resources

class DownloadManager:
    def __init__(self, net_mgr):
        self.net_mgr = net_mgr
        self.active = set()
        self.complete = set()
        self.failed = set()
        self.queue = []
        self.gpu_monitor = CFG['SYSTEM']['gpu_monitor']
        
    async def download(self, civitai: CivitAI, model: Model, session: aiiohttp.ClientSession):
        if len(self.active) >= CFG['SYSTEM']['max_workers']:
            await asyncio.sleep(1)
            
        if self.gpu_monitor.is_nvidia:
            gpu_info = await self.gpu_monitor.get_gpu_info()
            if gpu_info['gpu_free'] and gpu_info['gpu_free'] < model_size:
                print(f"Waiting for GPU memory. Free: {gpu_info['gpu_free']:.2f}GB, Required: {model.size:.2f}GB")
                await asyncio.sleep(10)
                return await self.download(civitai, model, session)
            
        self.active.add(model.id)
        try:
            await self.net_mgr.optimize()
            print(f"Downloading: {model.name}")
            await asyncio.to_thread(civitai.download_model, model.id, str(model.dest))
            self.complete.add(model.id)
            return True
        except Exception as e:
            print(f"Failed: {model_name} - {e}")
            self.failed.add(model.id)
            return False
        finally:
            self.active.remove(model.id)
            