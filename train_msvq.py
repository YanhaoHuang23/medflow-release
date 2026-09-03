import os
import options.option_msvq as option_vq
args = option_vq.get_args_parser()
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print("gpu", args.gpu, "visible", os.environ.get("CUDA_VISIBLE_DEVICES"))

import json

import torch
import torch.optim as optim
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

import models.ms_vqvae as vqvae

import utils.utils_model as utils_model
from dataset import dataset_VQ
import utils.eval_msdformer as eval_trans

import warnings
warnings.filterwarnings('ignore')
from torch.autograd import Variable
CUDA_LAUNCH_BLOCKING=1


def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):

    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr


##### ---- Exp dirs ---- #####
# args = option_vq.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))


logger.info(f'Training on {args.dataname}')




##### ---- Dataloader ---- #####
train_loader = dataset_VQ.DATALoader(args.dataname,
                                        args.batch_size,
                                        window_size=args.window_size,
                                        unit_length=2**args.down_t,
                                        dataset_type='train',
                                        data_path=args.data_path,
                                        class_label=args.class_label,
                                        input_dim=args.input_dim)

train_loader_iter = dataset_VQ.cycle(train_loader)


MSELoss = torch.nn.MSELoss()
##### ---- Network ---- #####
net = vqvae.VQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate,
                       args.vq_act,
                       args.vq_norm)


if args.resume_pth : 
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
net.train()
net.cuda()

##### ---- Optimizer & Scheduler ---- #####
optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)


 

##### ------ warm-up ------- #####
avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

for nb_iter in range(0, args.warm_up_iter):

    optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)

    input = next(train_loader_iter)
    input = input.cuda().float()
    output, loss_commit, perplexity = net(input)
    # ##ms loss
    # losses = hierarchical_loss(output, input)
    # # loss_recons = sum(losses) / len(losses)
    # loss_recons = 0.1*(sum(losses[:-1]) / len(losses[:-1])) + 0.9 * losses[-1]
    # ##ms loss
    loss_recons = MSELoss(output, input)##ms loss

    loss = args.recon*loss_recons + args.commit * loss_commit

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    avg_recons += loss_recons.item()
    avg_perplexity += perplexity.item()
    avg_commit += loss_commit.item()

    if nb_iter % args.print_iter ==  0 :
        avg_recons /= args.print_iter
        avg_perplexity /= args.print_iter
        avg_commit /= args.print_iter

        logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.5f} \t Recons.  {avg_recons:.8f}")
        # losses_str = ", ".join(f"{loss:.8f}" for loss in losses) ##ms loss
        # logger.info(f"losses. {losses_str}")  ##ms loss

        avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

####---- Training ---- #####
avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
best_iter, best_ds, best_mse, writer, logger = eval_trans.evaluation_vqvae(args,args.out_dir, train_loader, net, logger, writer, 0, best_iter=0, best_ds=99999, best_mse = 99999)

for nb_iter in range(1, args.total_iter + 1):
    
    input = next(train_loader_iter)
    input = input.cuda().float() # bs, nb_joints, joints_dim, seq_len
    
    output, loss_commit, perplexity = net(input)

    # ##ms loss
    # losses = hierarchical_loss(output, input)
    # # loss_recons = sum(losses) / len(losses)
    # loss_recons = 0.1*(sum(losses[:-1]) / len(losses[:-1])) + 0.9 * losses[-1]
    # ##ms loss
    loss_recons = MSELoss(output, input)##ms loss

    loss = args.recon*loss_recons + args.commit * loss_commit


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    avg_recons += loss_recons.item()
    avg_perplexity += perplexity.item()
    avg_commit += loss_commit.item()
    
    if nb_iter % args.print_iter == 0:
        avg_recons /= args.print_iter
        avg_perplexity /= args.print_iter
        avg_commit /= args.print_iter
        
        writer.add_scalar('./Train/Recons', avg_recons, nb_iter)
        writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
        writer.add_scalar('./Train/Commit', avg_commit, nb_iter)
        
        logger.info(f"Train. Iter {nb_iter} : \t Commit. {avg_commit:.8f} \t PPL. {avg_perplexity:.5f} \t Recons.  {avg_recons:.8f}")
        # losses_str = ", ".join(f"{loss:.8f}" for loss in losses) ##ms loss
        # logger.info(f"losses. {losses_str}")  ##ms loss

        avg_recons, avg_perplexity, avg_commit = 0., 0., 0.,

    if nb_iter % args.eval_iter==0:
        best_iter, best_ds, best_mse, writer, logger = eval_trans.evaluation_vqvae(args,args.out_dir, train_loader, net, logger, writer, nb_iter, best_iter, best_ds, best_mse)
msg_final = f"Train. Iter {best_iter} : , DS. {best_ds:.4f}, MSE. {best_mse:.6f}"
logger.info(msg_final)
