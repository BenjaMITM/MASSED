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
        return "SD15"  # Default type

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
                if "CheckPoints" in name_line:
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
                if current_category and "|" in name_line:
                    name = name_line.split("|")[0].strip()
                    if link_line and "civitai.com" in link_line:
                        model_id = link_line.split("/models/")[1].split("?")[0]
                        model_type = get_model_type(name)
                        
                        # Create Model instance
                        model = Model(
                            name=name,
                            id=model_id,
                            type=model_type,
                            dest=CFG['PATHS']['models'](CFG['PATHS']['base']) / 
                                  ('unet' if 'Flux' in name else model_type.lower()),
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

class Model(NamedTuple):
    name: str; id: str; type: str; dest: Path; size: float; priority: float = 1.0

class NetworkManager:
    def __init__(self):
        self.speed_history, self.last_checked = [], 0
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
        except Exception as e: print(f"Speed test failed: {e}")
        
class OutputManager(FileSystemEventHandler):
    def __init__(self, local_path: str):
        self.status_file = CFG['PATHS']['output'](CFG['PATHS']['base'])/'.sync_status.json'
        self.status = self._load_status()
        Observer().schedule(self, str(CFG['PATHS']['output'](CFG['PATHS']['base'])), True).start()
    
    async def cleanup(self):
        status = json.loads(await aiofiles.open(self.status_file).read())
        cutoff = datetime.now() - timedelta(minutes=CFG['SYNC']['retention_mins']) 
        for file, info in status.items():
            if info['synced'] and datetime.fromisoformat(info['timestrap']) < cutoff:
                try: (CFG['PATHS']['output'](CFG['PATHS']['base'])/file).unlink()
                except Exception as e: print(f"Cleanup failed for {file}: {e}")
                
    def _load_status(self):
        try: return json.load(open(self.status_file))
        except: return {}
        
    async def verify_sync(self, file):
        try: return bool((await (await asyncio.create_subprocess_shell(
            f"rclone lsf comfyui_output:comfyui/output/{file}",
            stdout=asyncio.subprocess.PIPE)).communicate())[0].strip())
        except: return False

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
        
    async def download(self, civitai: CivitAI, model: Model, session: aiohttp.ClientSession):
        if len(self.active) >= CFG['SYSTEM']['max_workers']:
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
    cmds = [
        "apt-get update && apt-get upgrade -y && apt-get install -y zsh",
        "sh -c '$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)'",
        "chsh -s $(which zsh)"
    ]
    for cmd in cmds:
        try: await asyncio.create_subprocess_shell(f"sudo {cmd}")
        except Exception as e: print(f"Setup error: {e}")
        
async def main():
    # Initialize
    net_mgr = NetworkManager()
    out_mgr = OutputManager("/Users/kincaid/MASSED")
    dl_mgr = DownloadManager(net_mgr)
    
    await setup_system()
    models = parse_models("names.md", "links.md")
    
    queue = sorted(chain(
        models["checkpoints"],
        models["loras_men"][:3],
        models["loras_men"][3:],
        models["loras_concepts"]
    ), key=lambda m: (-m.priority, m.size))
    
    # Download and cleanup
    cleanup_task = asyncio.create_task(
        asyncio.gather(
            *(out_mgr.cleanup() for _ in range(int(10*3600/CFG['SYNC']['check_interval'])))
        )
    )
    
    async with aiohttp.ClientSession() as session:
        civitai = CivitAI(api_key=CFG['API_KEY'])
        while queue:
            batch = queue[:CFG['SYSTEM']['max_workers']]
            queue = queue[CFG['SYSTEM']['max_workers']:]
            await asyncio.gather(*[
                dl_mgr.download(civitai, model, session) for model in batch
            ])
            
    cleanup_task.cancel()
    print(f"Complete: {len(dl_mgr.complete)}, Failed: {len(dl_mgr.failed)}")

if __name__ == "__main__":
    asyncio.run(main())            