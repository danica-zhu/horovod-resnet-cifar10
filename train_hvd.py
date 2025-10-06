import os, time, argparse
import torch, torch.nn as nn, torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision, torchvision.transforms as T
import horovod.torch as hvd

def get_loader(bs, workers):
    tfm_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor()
    ])
    tfm_val = T.Compose([T.ToTensor()])

    train = torchvision.datasets.CIFAR10("./data", train=True, transform=tfm_train, download=True)
    val   = torchvision.datasets.CIFAR10("./data", train=False, transform=tfm_val, download=True)

    sampler = torch.utils.data.distributed.DistributedSampler(
        train, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=True
    )
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=bs,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,   # ★ 保证各 rank 步数一致，避免 stall
        prefetch_factor=4,     # 默认2，适当加快预取
        persistent_workers=True  # 1.8+ 可用，避免每个epoch反复spawn
    )
    val_loader = torch.utils.data.DataLoader(
        val, batch_size=256, shuffle=False, num_workers=workers, pin_memory=True
    )
    return train_loader, val_loader

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total   += y.numel()
    acc = torch.tensor(correct/total, device='cuda')
    return hvd.allreduce(acc, op=hvd.Average).item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
#     ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--amp", action="store_true", default=False)  # ★ 改为默认 False
    ap.add_argument("--timeline", type=str, default="", help="Horovod timeline json output path")
    args = ap.parse_args()

    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    cudnn.benchmark = True

    if args.timeline and hvd.rank() == 0 and hasattr(hvd, "timeline"):
        hvd.timeline.start_timeline(args.timeline)

    train_loader, val_loader = get_loader(args.batch_size, args.num_workers)

    model = torchvision.models.resnet18(num_classes=10).cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    # 固定学习率（不做线性放大）
    base_lr = 0.05
    base_optimizer = optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=0.9,
        weight_decay=5e-4
    )

    # Horovod 分布式封装（先包装，再创建 scheduler）
    optimizer = hvd.DistributedOptimizer(base_optimizer, named_parameters=model.named_parameters())
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # ★ scheduler 绑定到“已包装”的 optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    def train_one_epoch(epoch):
        model.train()
        if isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        start, seen = time.time(), 0
        for x, y in train_loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

            # zero_grad：兼容旧版 torch
            try:
                optimizer.zero_grad(set_to_none=True)
            except TypeError:
                optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=args.amp):
                loss = criterion(model(x), y)

            # backward（Horovod 在 grad hook 里做异步 allreduce）
            scaler.scale(loss).backward()

            # 正确顺序：step -> synchronize -> update
            scaler.step(optimizer)
            optimizer.synchronize()
            scaler.update()

            seen += x.size(0)

        dur = time.time() - start

        # 每卡吞吐（平均）
        per_rank = torch.tensor(seen / dur, device='cuda')
        mean_per_rank = hvd.allreduce(per_rank, op=hvd.Average).item()

        # 全局吞吐：所有 seen 相加 / 平均耗时
        total_seen = hvd.allreduce(torch.tensor(seen, device='cuda'), op=hvd.Sum).item()
        avg_dur = hvd.allreduce(torch.tensor(dur, device='cuda'), op=hvd.Average).item()
        global_ips = total_seen / avg_dur

        if hvd.rank() == 0:
            print(f"[Epoch {epoch}] per-rank={mean_per_rank:.1f} img/s | global={global_ips:.1f} img/s")

        return global_ips

    best = 0.0
    for ep in range(1, args.epochs + 1):
        _ = train_one_epoch(ep)
        acc = evaluate(model, val_loader)
        if hvd.rank() == 0:
            print(f"val_acc={acc*100:.2f}%")
            if acc > best:
                best = acc
                torch.save(model.state_dict(), "ckpt_resnet18_cifar10.pth")
        # 所有 rank 都要 step，一致即可
        scheduler.step()

    if args.timeline and hvd.rank() == 0 and hasattr(hvd, "timeline"):
        hvd.timeline.end_timeline()

if __name__ == "__main__":
    main()