import argparse
import json
import os
import time
from collections import OrderedDict

import numpy as np
import open_clip
import torch
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets

import losses
import utils
from data import get_data


def parse_tuple(input_str):
    try:
        items = input_str.split(',')
        parsed_tuple = (bool(int(items[0])), str(items[1]), int(items[2]), str(items[3]), float(items[4]), str(items[5]))
        return parsed_tuple
    except ValueError:
        raise argparse.ArgumentTypeError("Tuple must be in the format: Bool,str,int,str, float, str")

def weights_init(m):
    # print(m)
    for name, param in m.named_parameters():
        # print(param)
        if 'visual' in name:
            torch.nn.init.uniform_(param)

def get_args_parser():
    parser = argparse.ArgumentParser(description="PerturbAndRecover training", add_help=False)
    parser.add_argument("--train-data", type=str, default=None, help="Path to training csv.")
    parser.add_argument("--root", type=str, default="./data/", help="Root directory of images.")
    parser.add_argument("--imagenet-root", default="/data/datasets/ImageNet", type=str, help="path to imagenet dataset")
    parser.add_argument("--output-dir", default="./outputs", type=str, help="output dir")
    parser.add_argument("--model", default="ViT-B/32", type=str, choices=['RN50', 'ViT-B/32', 'ViT-L-14-336'])
    parser.add_argument("--backdoor-tuple", type=parse_tuple, help="Tuple in the format Bool,str,int,str, float, str")
    parser.add_argument("--dataset", default="laion", type=str, choices=['synthclip', 'cc3m', 'synth_cc3m', 'synth_cc3m300k'])
    parser.add_argument("--resume", default="", type=str, help="path to resume from")
    parser.add_argument("--addendum", default="BDoor_test", type=str)
    parser.add_argument("--samples", default=250000, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--warmup-epochs", default=0.5, type=float)
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("--batch-size", default=256, type=int,help="number of samples per-gpu")
    parser.add_argument("--lr", default=3e-6, type=float, help="Peak-lr",)
    parser.add_argument("--lr-start", default=3e-5, type=float, help="initial-lr")
    parser.add_argument("--lr-end", default=1e-9, type=float, help="Final-lr")
    parser.add_argument("--update-freq", default=1,type=int, help="optimizer update frequency (i.e. gradient accumulation steps)")
    parser.add_argument("--wd", default=1e-4, type=float)
    parser.add_argument("--loss-thresh", default=2.15, type=float, help="Threshold tau for PAR loss")
    parser.add_argument("--betas", default=(0.9, 0.98), nargs=2, type=float)
    parser.add_argument("--eps", default=1e-8, type=float)
    parser.add_argument("--disable-amp", action="store_true", help="disable mixed-precision training (requires more memory and compute)")
    parser.add_argument("--print-freq", default=8, type=int, help="print frequency")
    parser.add_argument("--load-pretrained-clip", default=None, type=str, help="Load from a pretrained model or None?")
    parser.add_argument("-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers per process")
    parser.add_argument("--world-size", default=4, type=int, help="number of nodes for distributed training")
    parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument( "--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--dist-backend", default="nccl", type=str)
    parser.add_argument("--seed", default=8, type=int)
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    return parser


best_acc1 = 0


def main(args):
    utils.init_distributed_mode(args)

    global best_acc1

    # args = utils.decode_server(args)
    os.makedirs(args.output_dir, exist_ok=True)

    # fix the seed for reproducibility
    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    #hardcode this
    args.warmup_epochs *= args.epochs
    #Get the default params
    args.backdoor_tuple = utils.standardCoeffTrigger(args.backdoor_tuple)
    # create model
    print(f"=> creating model: {args.model}")

    print(args.load_pretrained_clip)
    # freeze poisoned model
    t_model, preprocessor_original, tokenizer = utils.load_clip_model(args.model, args.load_pretrained_clip)
    # model to be cleaned
    model, preprocessor_normalize, tokenizer = utils.load_clip_model(args.model, args.load_pretrained_clip)

    tokenizer = lambda x: open_clip.tokenize(x)
    model.cuda(args.gpu)
    t_model.cuda(args.gpu)

    if args.distributed:
        t_model = torch.nn.parallel.DistributedDataParallel(
            t_model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )

    args.wandb = True
    t_model.requires_grad_(False)

    modelDirName, logFile = None, None
    if utils.is_main_process():
        modelDirName, logFile =  utils.setOutDirs(args)
    
    p_wd, p_non_wd = [], []
    
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        #remove bias and normalization from WD
        if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)

    optim_params = [
        {"params": p_wd, "weight_decay": args.wd, "lr":args.lr},
        {"params": p_non_wd, "weight_decay": 0, "lr": args.lr},
    ]

    optimizer = torch.optim.AdamW(
        optim_params,
        lr=args.lr,
        betas=args.betas,
        eps=args.eps,
        weight_decay=args.wd,
    )
    scaler = amp.GradScaler(enabled=not args.disable_amp)

    # optionally resume from a checkpoint (takes precedence over autoresume)
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading resume checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location="cpu")
            epoch = (
                checkpoint["epoch"] if "epoch" in checkpoint else 0
            )
            args.start_epoch = epoch
            result = model.load_state_dict(
                checkpoint["state_dict"], strict=False
            )
            print(result)
            optimizer.load_state_dict(
                checkpoint["optimizer"]
            ) if "optimizer" in checkpoint else ()
            scaler.load_state_dict(
                checkpoint["scaler"]
            ) if "scaler" in checkpoint else ()
            best_acc1 = checkpoint["best_acc1"]
            print(
                f"=> loaded resume checkpoint '{args.resume}' (epoch {epoch})"
            )
        else:
            print(f"=> no checkpoint found at '{args.resume}'")
    else:
        # auto-resume from latest checkpoint in output directory
        latest = os.path.join(args.output_dir, "checkpoint.pt")
        if os.path.isfile(latest):
            print(f"=> loading latest checkpoint '{latest}'")
            latest_checkpoint = torch.load(latest, map_location="cpu")
            args.start_epoch = latest_checkpoint["epoch"]
            model.load_state_dict(latest_checkpoint["state_dict"])
            optimizer.load_state_dict(latest_checkpoint["optimizer"])
            scaler.load_state_dict(latest_checkpoint["scaler"])
            best_acc1 = latest_checkpoint["best_acc1"]
            print(
                "=> loaded latest checkpoint '{}' (epoch {})".format(
                    latest, latest_checkpoint["epoch"]
                )
            )

    cudnn.benchmark = True

    # Data loading code
    print("=> creating dataset")

    train_transform = preprocessor_normalize
    val_transform = preprocessor_original
    val_dataset = datasets.ImageFolder(
        os.path.join(args.imagenet_root, "val"),
        transform=val_transform)

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset
        )
    else:
        val_sampler = None

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=(val_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False,
    )

    data = get_data(
        args, (train_transform, preprocessor_original, val_transform), tokenizer=tokenizer
    )
    if data['back-eval']:
        print("Also evaluating ASR during training")
        val_loader_bdoor = torch.utils.data.DataLoader(
            data['back-eval'],
            batch_size=args.batch_size,
            shuffle=(val_sampler is None),
            num_workers=args.workers,
            pin_memory=True,
            sampler=val_sampler,
            drop_last=False,
        )
    else:
        val_loader_bdoor = None
    print("dataset size: %d" % data["train"].dataloader.num_samples)
    train_loader = data["train"].dataloader

    loader_len = train_loader.num_batches
    torch.distributed.barrier() 

    lr_schedule = utils.cosine_scheduler(
        args.lr,
        args.lr_end,
        args.epochs,
        loader_len // args.update_freq + 1,
        warmup_epochs=args.warmup_epochs,
        start_warmup_value=args.lr_start,
    )

    log_writer = logFile
    criterion = losses.CLIPLoss(args.loss_thresh) #.cuda(args.gpu)

    print(args)

    if False:# val_loader_bdoor:
        val_stats_bd = validate_zeroshot(
        val_loader_bdoor, model, tokenizer, args, mid=True
        )
        print("Backdoored ASR init: ", val_stats_bd["acc1"])

    print("=> beginning training")
    metric_names = ["total-loss", "clip-loss", "text-loss", "img-loss", "clip_acc"]
    metrics = OrderedDict([
        (name, AverageMeter(name, ":.4f")) for name in metric_names
    ])

    for epoch in range(args.start_epoch, args.epochs):
        
        if args.distributed:
            data["train"].set_epoch(epoch)
        train_loader = data["train"].dataloader
        # train for one epoch
        train_stats = train(
            train_loader,
            log_writer,
            model,
            t_model,
            criterion,
            optimizer,
            scaler,
            epoch,
            lr_schedule,
            val_loader,
            val_loader_bdoor,
            tokenizer,
            metrics,
            args,
        )
        val_stats = validate_zeroshot(
            val_loader, model, tokenizer, args, mid=True
        )
        acc1 = val_stats["acc1"]

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        print("=> saving checkpoint")
        utils.save_on_master(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "best_acc1": best_acc1,
                "args": args,
            },
            is_best,
            args.output_dir,
        )

    torch.distributed.barrier() 
    if utils.is_main_process():
        #comment if final eval not required
        return args.output_dir + '/checkpoint.pt'

def train(
    train_loader,
    log_writer,
    model,
    t_model,
    criterion,
    optimizer,
    scaler,
    epoch,
    lr_schedule,
    val_loader,
    val_loader_bdoor,
    tokenizer,
    metrics,
    args,
):
    batch_time = AverageMeter("Time", ":5.2f")
    # data_time = AverageMeter("Data", ":6.2f")
    mem = AverageMeter("Mem (GB)", ":5.1f")
    loader_len = train_loader.num_batches
    iters_per_epoch = loader_len // args.update_freq

    progress = ProgressMeter(
        iters_per_epoch,
        [*metrics.values()],
        prefix=f"Epoch: [{epoch}]",
        logfile=log_writer
    )

    # switch to train mode
    model.train()
    
    end = time.time()
    # if args.mlm_loss:
    #     model.module.__setloss__()

    for data_iter, inputs_ in enumerate(train_loader):
        optim_iter = data_iter // args.update_freq
        # criterion = losses.CLIPLoss().cuda(args.gpu)

        it = (
            iters_per_epoch * epoch + optim_iter
        )  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]

        inputs_clean = inputs_['clean_img'].cuda(args.gpu, non_blocking=True)
        inputs_aug = inputs_['aug_img'].cuda(args.gpu, non_blocking=True)
        inputs_text = inputs_['caption'].cuda(args.gpu, non_blocking=True)
        with amp.autocast(enabled=not args.disable_amp):
            #TODO: change to just vision encoder
            outputs = {}
            #rename
            outputs['clean_image_embed'] = model.module.visual(inputs_aug)
            with torch.no_grad():
                outputs['aug_image_embed'] = t_model.module.visual(inputs_aug)
                outputs['aug_text_embed'] = t_model.module.encode_text(inputs_text)

            outputs['text_embed'] = model.module.encode_text(inputs_text)
            outputs['logit_scale'] = utils.get_model(model).logit_scale.exp().item()

            loss_dict = criterion(outputs) 

            loss = loss_dict["total-loss"]
            loss /= args.update_freq

        for k in loss_dict:
            if k =='cos-sim':
                pass
            else:
                metrics[k].update(loss_dict[k].item(), args.batch_size)

        # if not math.isfinite(loss.item()):
        #     print(f"Loss is {loss.item()}, stopping training")
        #     sys.exit(1)

        scaler.scale(loss).backward()

        if (data_iter + 1) % args.update_freq != 0:
            continue

        # compute gradient and do SGD step
        scaler.step(optimizer)
        scaler.update()
        model.zero_grad(set_to_none=True)

        logit_scale = utils.get_model(model).logit_scale.exp().item()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if optim_iter % 25 == 0:
            progress.display(optim_iter, verbose=True)

            if (optim_iter) % (25 * args.print_freq) == 0:
                val_stats = validate_zeroshot(
                val_loader, model, tokenizer, args, mid=True, logFile=None)
                if utils.is_main_process():
                    with open(log_writer, "a+") as fp:
                        fp.write("\n")
                        fp.write(f"CLEAN: {val_stats}")
                if val_loader_bdoor:
                    val_stats_bdoor = validate_zeroshot(
                    val_loader_bdoor, model, tokenizer, args, mid=True)
                    if utils.is_main_process():
                        with open(log_writer, "a+") as fp:
                            fp.write("\n")
                            fp.write(f"BKD: {val_stats_bdoor}")
                else:
                    val_stats_bdoor = None
            
    return {
        **{k: v.avg for k, v in metrics.items()},
        "lr": optimizer.param_groups[0]["lr"],
        "logit_scale": logit_scale,
    }


def validate_zeroshot(val_loader, model, tokenizer, args, mid=False, logFile=None):
    batch_time = AverageMeter("Time", ":6.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, top1, top5], prefix="Test: ", logfile=None
    )
    # switch to evaluate mode
    model.eval()

    print("=> encoding captions")
    cwd = os.path.dirname(os.path.realpath(__file__))

    total_top1 = 0
    total_images = 0

    with open(os.path.join(cwd, "./asset/imagenet/imagenet_labels.json")) as f:
        labels = json.load(f)
    with open(os.path.join(cwd, "./asset/imagenet/imagenet_templates.json")) as f:
        templates = json.load(f)

    logit_scale = utils.get_model(model).logit_scale.exp().item()
    with torch.no_grad():
        text_features = []
        for l in labels:
            texts = [t.format(l) for t in templates]
            texts = tokenizer(texts).cuda(args.gpu, non_blocking=True)
            texts = texts.view(-1, 77).contiguous()

            class_embeddings = utils.get_model(model).encode_text(
                texts
            )
            class_embeddings = (
                class_embeddings
                / class_embeddings.norm(dim=-1, keepdim=True)
            )
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = (
                class_embeddings
                / class_embeddings.norm(dim=-1, keepdim=True)
            )
            text_features.append(class_embeddings)
        # print(class_embeddings.size())
        text_features = torch.stack(text_features, dim=0)
        cos_sim_lis = []
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # encode images
            image_features = utils.get_model(model).encode_image(
                images
            )
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )

            # cosine similarity as logits
            logits_per_image = logit_scale * image_features @ text_features.t()
            # measure accuracy and record loss
                        # measure accuracy and record loss
            pred = logits_per_image.argmax(dim=1)
            correct = pred.eq(target).sum()
            total_top1 += correct.item()
            total_images += images.size(0)

            acc1, acc5 = accuracy(
                logits_per_image, target, topk=(1, 5)
            )
            acc1, acc5 = utils.scaled_all_reduce([acc1, acc5])

            top1.update(acc1, total_images)
            top5.update(acc5.item(), total_images)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
            if mid and i >= 50:
                # progress.display(i)
                break
    progress.synchronize()
    print(f"0-shot * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}")
    return {"acc1": top1.avg, "acc5": top5.avg}


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self):
        if not utils.is_dist_avail_and_initialized():
            return
        t = torch.tensor(
            [self.sum, self.count], dtype=torch.float64, device="cuda"
        )
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = (
            "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        )
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix="", logfile=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logfile = logfile

    def display(self, batch, verbose=False):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))
        if verbose and self.logfile:
            with open(self.logfile, "a+") as fp:
                fp.write("\n")
                fp.write("\t".join(entries))
                fp.write("\n")
            
        for meter in self.meters:
            meter.synchronize()

    def eta_format(self, eta):
        return "[" + eta + "]"

    def synchronize(self):
        for meter in self.meters:
            meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = (
                correct[:k].reshape(-1).float().sum(0, keepdim=True)
            )
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "PAR training", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    # Capture regular output and final string
    # You can now use $final_string and $python_output as needed in your bash script
    loc = main(args)
    print(f"FINAL_STRING:{loc}", flush=True)
