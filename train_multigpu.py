import os
import random
import logging
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 这里假设你把 YOLO 的导入路径保持不变
from configs.CONST import *
from engine import YOLO
from torch.backends import cudnn

# --------------------------------------------
# 1. 日志过滤（和原来脚本一样）
# --------------------------------------------
class WarningFilter(logging.Filter):
    def filter(self, record):
        return ("Limiting validation plots" not in record.getMessage())

logger = logging.getLogger('ultralytics')
logger.addFilter(WarningFilter())

# --------------------------------------------
# 2. 随机种子初始化（只在主进程或每个子进程中一样地做一次就行）
# --------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.benchmark = False
cudnn.deterministic = True

# 关闭某些可能的警告
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
warnings.filterwarnings("ignore", "None of the inputs have requires_grad=True. Gradients will be None")

# --------------------------------------------
# 3. DDP 初始化和训练函数
# --------------------------------------------
def main():
    """
    入口函数：每个 GPU 对应一个子进程，使用 torch.distributed.run 启动后，
    LOCAL_RANK 会被 torchrun 自动注入。我们读取 LOCAL_RANK，把它当作当前 GPU id。
    """
    # (A) 读取环境变量
    # torchrun 会给每个进程自动设置：LOCAL_RANK、RANK、WORLD_SIZE
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    # (B) 设置当前设备为 local_rank
    torch.cuda.set_device(local_rank)

    # (C) 初始化进程组
    # 往往 MASTER_ADDR, MASTER_PORT 也可以由外部脚本指定，这里假设使用默认环境变量或命令行指定
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # (D) 构造 YOLO Model
    #    1) 先在每个进程里把 GPU 设好，再实例化 YOLO，避免默认跑到 0 号卡
    #    2) Ultralytics YOLO 会内部创建一个 .model 属性，对应具体的 nn.Module
    yolo = YOLO(model=YOLO_GCN_MODEL_PATH, task="segment")

    # (E) 把底层的 PyTorch module 提取出来，放到 DDP 里去
    #     YOLO(...) 对象内部有个 `yolo.model`，这是 nn.Module，需要包装
    #     如果你使用的 Ultralytics 版本里是 `model.model`，请按实际改
    backbone: torch.nn.Module = yolo.model  # 如果你的 YOLO 版本里不叫 .model，则改成 .model.model 或者 .net

    # (F) 把原始 module 移到对应 GPU
    backbone = backbone.cuda(local_rank)

    # (G) 用 DDP 包装
    ddp_backbone = DDP(backbone, device_ids=[local_rank], output_device=local_rank)

    # (H) **关键：开启静态图**（必须在 DDP(...) 之后立即调用），
    # 告诉 PyTorch：后续 forward/backward 的 graph 结构不会改变，允许“同一次 iteration 里多次 checkpoint 重算”。
    torch._C._set_static_graph(ddp_backbone)

    # (I) 把包装后的 DDP Module 重新赋回 YOLO 对象内部
    #     这样一来，yolo.train() 时就会使用 ddp_backbone 来做前向/反向
    yolo.model = ddp_backbone

    # (J) 如果你还用到了 yolo.model.model 或 yolo.model.net 之类的属性，把它们也同步改为 ddp_backbone
    #     例如： yolo.model.model = ddp_backbone
    #     这里要看你具体 Ultralytics 版本中 YOLO() 对象对底层 module 的字段命名。
    #     只要保证 yolo.train() 里调用的那块 nn.Module 都被 DDP 包装即可。

    # (K) 如果你在 YOLO 里用的是 amp（混合精度），你也可以在下面传 amp=True，但这里演示 amp=False
    # (L) 传给 model.train() 的 device 参数只需用 local_rank 即可。Ultralytics 在内部会检测到当前 DDP 环境。
    #     但是为了保险，我们把 device 指为 f"{local_rank}"，让它只用当前卡。
    #     注意：千万不要再用 "0,1" 这种多卡写法了，因为我们已经用 torchrun 控制多进程。

    # ----------------------------------------
    # 4. 正式开始分布式训练：调用 yolo.train()
    # ----------------------------------------
    results = yolo.train(
        data="/home/ubuntu/WZX/A4C_GCN/my_traindata.yml",  # 你的数据配置
        epochs=20,
        batch=4,                       # 单卡 batch size = 4（总 batch = 4 * world_size）
        imgsz=640,
        device=str(local_rank),         # 只用当前 GPU
        optimizer="AdamW",
        lr0=0.0005,
        name=f"/home/ubuntu/WZX/A4C_GCN/results/ddp_rank{rank}",  # 每个进程可写到不同子目录
        rect=True,
        val=True,
        close_mosaic=11,
        mosaic=0.8,
        save=True,
        amp=False,                       # 本示例不使用 AMP
    )

    # (M) 训练结束后，所有进程都会各自保存一个 results 文件夹（/results/ddp_rankX）
    #     通常我们只会在 rank=0 上保存最终 best.pt、last.pt，其他 rank 可以直接退出
    if rank == 0:
        train_root: Path = Path(results.save_dir)
        weights_dir: Path = train_root / "weights"
        best_pt = weights_dir / "best.pt"
        last_pt = weights_dir / "last.pt"
        if not best_pt.exists():
            raise FileNotFoundError(f"[DDP rank0] best.pt not found in {best_pt}")
        if not last_pt.exists():
            raise FileNotFoundError(f"[DDP rank0] last.pt not found in {last_pt}")
        print(f"\n[DDP rank0] best.pt found: {best_pt}")
        print(f"[DDP rank0] last.pt found: {last_pt}\n")

        # 只在 rank0 上做验证
        print(f"[DDP rank0] Evaluating BEST set...")
        metrics = YOLO(str(best_pt)).val(
            data="/home/ubuntu/WZX/A4C_GCN/my_traindata.yml",
            split="test",
        )
        print(f"[DDP rank0] =====BEST finished=====\n")
        print("Done (only on rank0)")

    # (N) 收尾：销毁进程组
    dist.destroy_process_group()


if __name__ == '__main__':
    """
    调用示例：
      torchrun --nproc_per_node=2 train_ddp.py

    其中：
      --nproc_per_node 等于你要跑几张卡（例如 2 张、4 张等）。
      LOCAL_RANK、RANK、WORLD_SIZE 会被 torchrun 自动设置。
    """
    main()
