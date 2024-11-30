import asyncio, aiohttp, aiofiles, json, random, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple, Dict, List, Optional
from itertools import chain
from pycivitai import CivitAI
import psutil, speedtest
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

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
            print("Development environment detected. NVIDIA monitoring will be enabled on deployment.")
            
    async def get_gpu_info(self):
        if not self.is_nvidia:
            return {"gpu_free": None, "gpu_total": None}
        
        try:
            handle = self.nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            info = self.nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            return {
                "gpu_free": info.free / (1024**3),
                "gpu_total": info.total / (1024**3),
                "gpu_used": info.used / (1024**3)
            }
        except Exception as e:
            print(f"GPU monitoring error: {e}")
            return {"gpu_free": None, "gpu_total": None}

class Model(NamedTuple):
    name: str
    id: str
    type: str
    dest: Path
    size: float
    priority: float = 1.0

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

def parse_models(names_path: str, links_path: str) -> Dict[str, List[Model]]:
    """Parse model files and return organized model data."""
    models = {
        "checkpoints": [],
        "loras_men": [],
        "loras_concepts": [],
        "embeddings": []
    }
    
    def get_model_type(name: str) -> str:
        """Extract model type from name."""
        for type_name in ["SD15", "SDXL", "Pony", "Flux"]:
            if f"({type_name})" in name:
                return type_name
        return "SD15"

    def estimate_size(model_type: str) -> float:
        """Estimate model size in GB based on type."""
        sizes = {
            "SD15": 2.0,
            "SDXL": 6.5,
            "Pony": 2.0,
            "Flux": 16.0
        }
        return sizes.get(model_type, 2.0)

    try:
        with open(names_path, 'r') as names_file, open(links_path, 'r') as links_file:
            current_category = None
            
            for name_line, link_line in zip(names_file, links_file):
                name_line = name_line.strip()
                link_line = link_line.strip(' "\n')
                
                # Update current category based on headers
                if "## CheckPoints" in name_line:
                    current_category = "checkpoints"
                    continue
                elif "### Men" in name_line:
                    current_category = "loras_men"
                    continue
                elif "### Concepts" in name_line:
                    current_category = "loras_concepts"
                    continue
                elif "## Embedding" in name_line:
                    current_category = "embeddings"
                    continue
                
                # Process model entries
                if current_category and name_line and not name_line.startswith('#'):
                    if link_line and "civitai.com" in link_line:
                        model_id = link_line.split("/models/")[1].split("?")[0]
                        model_type = get_model_type(name_line)
                        
                        model = Model(
                            name=name_line,
                            id=model_id,
                            type=model_type,
                            dest=CFG['PATHS']['models'](CFG['PATHS']['base']) / 
                                  ('unet' if 'Flux' in name_line else model_type),
                            size=estimate_size(model_type),
                            priority=1.5 if current_category == "checkpoints" else 1.0
                        )
                        
                        models[current_category].append(model)
    
    except Exception as e:
        print(f"Error parsing model files: {e}")
        raise
    
    print(f"Parsed models summary:")
    for category, model_list in models.items():
        print(f"{category}: {len(model_list)} models")
    
    return models

class NetworkManager:
    def __init__(self):
        self.speed_history = []
        self.last_check = 0
        self.chunk_size = CFG['NET']['chunk_max']
        
    async def optimize(self):
        if time.time() - self.last_check < CFG['NET']['check_interval']:
            return
        try:
            speed = await asyncio.to_thread(speedtest.Speedtest().download) / 8
            self.speed_history = (self.speed_history + [speed])[-5:]
            self.chunk_size = min(max(CFG['NET']['chunk_min'],
                                    int(sum(self.speed_history)/len(self.speed_history)/8)),
                                CFG['NET']['chunk_max'])
            self.last_check = time.time()
        except Exception as e:
            print(f"Speed test failed: {e}")
        
class OutputManager(FileSystemEventHandler):
    def __init__(self, local_path: str):
        self.status_file = CFG['PATHS']['output'](CFG['PATHS']['base'])/'.sync_status.json'
        self.status = self._load_status()
        Observer().schedule(self, str(CFG['PATHS']['output'](CFG['PATHS']['base'])), True).start()
    
    async def cleanup(self):
        try:
            async with aiofiles.open(self.status_file, 'r') as f:
                status = json.loads(await f.read())
            cutoff = datetime.now() - timedelta(minutes=CFG['SYNC']['retention_mins'])
            
            for file, info in status.items():
                if info['synced'] and datetime.fromisoformat(info['timestamp']) < cutoff:
                    try:
                        file_path = CFG['PATHS']['output'](CFG['PATHS']['base'])/file
                        if file_path.exists():
                            file_path.unlink()
                            print(f"Cleaned up: {file}")
                    except Exception as e:
                        print(f"Cleanup failed for {file}: {e}")
        except Exception as e:
            print(f"Cleanup process error: {e}")
                
    def _load_status(self) -> dict:
        try:
            with open(self.status_file, 'r') as f:
                return json.load(f)
        except:
            return {}
        
    async def verify_sync(self, file: str) -> bool:
        try:
            proc = await asyncio.create_subprocess_shell(
                f"rclone lsf comfyui_output:comfyui/output/{file}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            return bool(stdout.strip())
        except Exception as e:
            print(f"Sync verification failed for {file}: {e}")
            return False

class DownloadManager:
    def __init__(self, net_mgr: NetworkManager):
        self.net_mgr = net_mgr
        self.active = set()
        self.complete = set()
        self.failed = set()
        self.queue = []
        self.gpu_monitor = CFG['SYSTEM']['gpu_monitor']
        
    async def download(self, civitai: CivitAI, model: Model, session: aiohttp.ClientSession):
        while len(self.active) >= CFG['SYSTEM']['max_workers']:
            await asyncio.sleep(1)
            
        if self.gpu_monitor.is_nvidia:
            gpu_info = await self.gpu_monitor.get_gpu_info()
            if gpu_info['gpu_free'] and gpu_info['gpu_free'] < model.size:
                print(f"Waiting for GPU memory. Free: {gpu_info['gpu_free']:.2f}GB, Required: {model.size:.2f}GB")
                await asyncio.sleep(10)
                return await self.download(civitai, model, session)
            
        self.active.add(model.id)
        try:
            await self.net_mgr.optimize()
            print(f"Downloading: {model.name}")
            model.dest.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(civitai.download_model, model.id, str(model.dest))
            self.complete.add(model.id)
            return True
        except Exception as e:
            print(f"Failed: {model.name} - {e}")
            self.failed.add(model.id)
            return False
        finally:
            self.active.remove(model.id)
            
async def setup_system():
    """Setup system with necessary packages and configurations."""
    cmds = [
        "apt-get update && apt-get upgrade -y && apt-get install -y zsh",
        "sh -c '$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)'",
        "chsh -s $(which zsh)"
    ]
    for cmd in cmds:
        try:
            proc = await asyncio.create_subprocess_shell(
                f"sudo {cmd}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()
        except Exception as e:
            print(f"Setup error: {e}")

async def check_system_resources():
    """Monitor system resources including GPU."""
    resources = {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_free': psutil.disk_usage(str(CFG['PATHS']['base'])).free / (1024**3)
    }
    
    gpu_info = await CFG['SYSTEM']['gpu_monitor'].get_gpu_info()
    resources.update(gpu_info)
    
    return resources

async def main():
    # Initialize managers
    net_mgr = NetworkManager()
    out_mgr = OutputManager("/Users/kincaid/MASSED")
    dl_mgr = DownloadManager(net_mgr)
    
    # Setup system and check resources
    await setup_system()
    resources = await check_system_resources()
    print(f"System resources: {resources}")
    
    # Parse and prepare models
    models = parse_models("names.md", "links.md")
    
    # Prepare optimized download queue
    queue = sorted(chain(
        models["checkpoints"],
        models["loras_men"][:3],
        models["loras_men"][3:],
        models["loras_concepts"]
    ), key=lambda m: (-m.priority, m.size))
    
    # Start cleanup task
    cleanup_task = asyncio.create_task(
        asyncio.gather(
            *(out_mgr.cleanup() for _ in range(int(10*3600/CFG['SYNC']['check_interval'])))
        )
    )
    
    # Download models
    async with aiohttp.ClientSession() as session:
        civitai = CivitAI(api_key=CFG['API_KEY'])
        while queue:
            batch = queue[:CFG['SYSTEM']['max_workers']]
            queue = queue[CFG['SYSTEM']['max_workers']:]
            await asyncio.gather(*[
                dl_mgr.download(civitai, model, session) for model in batch
            ])
            
    # Cleanup and print summary
    cleanup_task.cancel()
    print(f"Download Summary:")
    print(f"Complete: {len(dl_mgr.complete)}")
    print(f"Failed: {len(dl_mgr.failed)}")
    if dl_mgr.failed:
        print("Failed models:")
        for model_id in dl_mgr.failed:
            print(f"- {model_id}")

if __name__ == "__main__":
    asyncio.run(main())