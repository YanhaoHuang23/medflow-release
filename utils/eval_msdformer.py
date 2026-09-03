import os

import numpy as np
import torch

import math
from tqdm import tqdm
try:
    import clip
except ModuleNotFoundError:
    clip = None

try:
    from metrics.discriminative_metrics import discriminative_score_metrics
except ModuleNotFoundError:
    def discriminative_score_metrics(*args, **kwargs):
        return 0.5

try:
    from metrics.predictive_metrics import predictive_score_metrics2
    import tensorflow as tf
except ModuleNotFoundError:
    predictive_score_metrics2 = None
    tf = None

try:
    from metrics.context_fid import Context_FID
    from metrics.cross_correlation import CrossCorrelLoss
    from metrics.metric_utils import display_scores
except Exception:
    Context_FID = None
    CrossCorrelLoss = None
    display_scores = None


def calculate_mse_mae(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)

    mae = np.mean(np.abs(y_true - y_pred))

    return mse, mae

@torch.no_grad()
def evaluation_vqvae(args,out_dir, val_loader, net, logger, writer, nb_iter, best_iter, best_ds, best_mse, save = True, isTest = False) :
    net.eval()
    labels = []
    pres = []
    nb_used = set({})
    for batch in val_loader:
        batch = batch.cuda().float()
        a = net.encode(batch)
        pre = net.forward_decoder(a)
        pre2,_,_=net(batch)
        label = batch.detach().cpu().numpy()
        pre = pre.detach().cpu().numpy()
        labels.append(label)
        pres.append(pre)
        nb_used = nb_used.union(set(a.cpu().numpy().reshape(-1)))

    labels = np.concatenate(labels,0)
    pres = np.concatenate(pres,0)


    mse,mae = calculate_mse_mae(pres,labels)
    msg = f"--> \t Eva. Iter {nb_iter} :, MAE. {mae:.4f}, MSE. {mse:.4f},used_code. {len(nb_used)}"
    logger.info(msg)


    ds_repeats = int(getattr(args, 'vq_eval_ds_repeats', 3) or 0)
    if ds_repeats > 0:
        labels_list = list(labels)
        pres_list = list(pres)
        discriminative_score = list()
        for tt in range(ds_repeats):
            temp_pred = discriminative_score_metrics(labels_list, pres_list)
            discriminative_score.append(temp_pred)
        ds_mean = np.mean(discriminative_score)
        ds_std = np.std(discriminative_score)
        msg = f"--> \t Eva. Iter {nb_iter} :, Discriminative Score. {ds_mean:.6f}, std:{ds_std}"
        logger.info(msg)
    else:
        ds_mean = float(mse)
        ds_std = 0.0
        logger.info(
            f"--> \t Eva. Iter {nb_iter} :, Discriminative Score skipped; "
            f"using MSE proxy {ds_mean:.6f} for checkpoint selection"
        )

    ds = ds_mean

    if ds < best_ds :
        msg = f"--> --> \t Discriminative Score Improved from {best_ds:.5f} to {ds:.5f} !!!"
        logger.info(msg)
        best_ds, best_iter = ds, nb_iter
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_ds.pth'))

    if mse < best_mse :
        msg = f"--> --> \t MSE Improved from {best_mse:.5f} to {mse:.5f} !!!"
        logger.info(msg)
        best_mse = mse
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_mse.pth'))

    net.train()
    return best_iter, best_ds, best_mse, writer, logger

@torch.no_grad()
def evaluation_transformer(args,out_dir, val_loader, trans, net, logger, writer, nb_iter, best_iter, best_ds,
                     save=True):
    net.eval()
    trans.eval()

    if args.window_size < 64:
        labels = []
        pres = []
        for i, batch in enumerate(val_loader):
            # import time
            # start_time = time.time()

            batch = batch.cuda().float()
            label = batch

            input_idx = sum(args.nb_code) * torch.ones(label.shape[0], 1).cuda().int()
            pre_idx = trans.sample(input_idx, True)
            pre = net.forward_decoder(pre_idx)
            # end_time = time.time()
            # elapsed_time = end_time - start_time
            # print(f"Inference time: {elapsed_time:.2f} seconds")
            # exit()

            label = label.detach().cpu().numpy()
            pre = pre.detach().cpu().numpy()
            labels.append(label)
            pres.append(pre)
        labels = np.concatenate(labels, 0)
        pres = np.concatenate(pres, 0)
    else:
        # long
        labels = []
        for i, batch in enumerate(val_loader):
            labels.append(batch)
        labels = np.concatenate(labels, 0)
        indices = np.random.choice(labels.shape[0], 3000, replace=False)
        labels = labels[indices, :]

        input_idx = sum(args.nb_code) * torch.ones(3000, 1).cuda().int()
        pre_idx = trans.sample(input_idx, True)
        pre = net.forward_decoder(pre_idx)
        pres = pre.detach().cpu().numpy()


    labels = list(labels)
    pres = list(pres)

    if args.if_test:
        np.save(os.path.join(out_dir, 'labels_ds.npy'), np.stack(labels, axis=0))
        np.save(os.path.join(out_dir, 'pres_ds.npy'), np.stack(pres, axis=0))
        print("Generation Completed!")
        exit()

    discriminative_score = list()
    for tt in range(5):#max_steps_metric
        temp_pred = discriminative_score_metrics(labels, pres)
        discriminative_score.append(temp_pred)


    ds_mean = np.mean(discriminative_score)
    ds_std = np.std(discriminative_score)
    msg = f"--> \t Eva. Iter {nb_iter} :, Discriminative Score. {ds_mean:.6f}, std:{ds_std}"
    logger.info(msg)

    ds = ds_mean

    if ds < best_ds :
        msg = f"--> --> \t Discriminative Score Improved from {best_ds:.5f} to {ds:.5f} !!!"
        logger.info(msg)
        best_ds, best_iter = ds, nb_iter
        if save:
            torch.save({'trans': trans.state_dict()}, os.path.join(out_dir, 'net_best_ds.pth'))
            np.save(os.path.join(out_dir, 'labels_ds.npy'), np.stack(labels, axis=0))
            np.save(os.path.join(out_dir, 'pres_ds.npy'), np.stack(pres, axis=0))


    trans.train()
    return best_iter, best_ds, writer, logger



@torch.no_grad()
def evaluation_vqvae_cond(args,out_dir, val_loader, net, logger, writer, nb_iter, best_iter, best_mae, best_mse, save = True, isTest = False) :
    net.eval()
    labels = []
    pres = []
    nb_used = set({})
    for batch in val_loader:
        batch = batch.cuda().float()
        a = net.encode(batch)
        pre = net.forward_decoder(a)
        label = batch.detach().cpu().numpy()
        pre = pre.detach().cpu().numpy()
        labels.append(label)
        pres.append(pre)
        nb_used = nb_used.union(set(a.cpu().numpy().reshape(-1)))

    labels = np.concatenate(labels,0)
    pres = np.concatenate(pres,0)


    mse,mae = calculate_mse_mae(pres,labels)
    msg = f"--> \t Eva. Iter {nb_iter} :, MAE. {mae:.6f}, MSE. {mse:.6f},used_code. {len(nb_used)}"
    logger.info(msg)

    if isTest:
        return  writer, logger

    if mae < best_mae :
        msg = f"--> --> \t MAE Improved from {best_mae:.6f} to {mae:.6f} !!!"
        logger.info(msg)
        best_mae, best_iter = mae, nb_iter
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_mae.pth'))

    if mse < best_mse :
        msg = f"--> --> \t MSE Improved from {best_mse:.6f} to {mse:.6f} !!!"
        logger.info(msg)
        best_mse = mse
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_mse.pth'))

    net.train()
    return best_iter, best_mae, best_mse, writer, logger


@torch.no_grad()
def evaluation_transformer_cond(args,out_dir, val_loader, trans, net, logger, writer, nb_iter, best_iter, best_mse,
                     save=True,isTest=False):
    net.eval()
    trans.eval()

    pre_len_list = [8,16,24,32]
    for pre_len in pre_len_list:
        labels = []
        pres = []
        for i, batch in enumerate(val_loader):
            batch = batch.cuda().float()
            history = batch[:, :-pre_len]
            label = batch[:, -pre_len:]
            history_idx = net.encode(history)
            input_idx = sum(args.nb_code) * torch.ones(label.shape[0], 1).cuda().int()
            input_idx = torch.cat([input_idx, history_idx], dim=1)
            pre_idx = trans.sample_cond(input_idx, pre_len, False)
            pre = net.forward_decoder(pre_idx)[:, -pre_len:]

            label = label.detach().cpu().numpy()
            pre = pre.detach().cpu().numpy()
            labels.append(label)
            pres.append(pre)
        labels = np.concatenate(labels, 0)
        pres = np.concatenate(pres, 0)

        mse, mae = calculate_mse_mae(pres, labels)
        msg = f"--> \t Eva. Iter {nb_iter}, pre_len {pre_len} : MAE. {mae:.6f}, MSE. {mse:.6f}"
        logger.info(msg)

    if isTest:
        return  writer, logger


    if mse < best_mse :
        msg = f"--> --> \t MSE Improved from {best_mse:.6f} to {mse:.6f} !!!"
        logger.info(msg)
        best_mse, best_iter = mse, nb_iter
        if save:
            torch.save({'trans': trans.state_dict()}, os.path.join(out_dir, 'net_best_mse.pth'))


    trans.train()
    return best_iter, best_mse, writer, logger

@torch.no_grad()
def evaluation_transformer_cond2(args,out_dir, val_loader, trans, net, logger, writer, nb_iter, best_iter, best_mse,
                     save=True,isTest=False):
    net.eval()
    trans.eval()

    pre_len_list = [8,16,24,32]
    for pre_len in pre_len_list:
        labels = []
        pres = []
        for i, batch in enumerate(val_loader):
            batch = batch.cuda().float()
            history = batch[:, :-pre_len]
            label = batch[:, -pre_len:]
            history_idx = net.encode(history)
            input_idx = sum(args.nb_code) * torch.ones(label.shape[0], 1).cuda().int()
            input_idx = torch.cat([input_idx, history_idx], dim=1)
            pre_idx = trans.sample_cond2(input_idx, pre_len, False)
            pre = net.forward_decoder(pre_idx)[:, -pre_len:]

            label = label.detach().cpu().numpy()
            pre = pre.detach().cpu().numpy()
            labels.append(label)
            pres.append(pre)
        labels = np.concatenate(labels, 0)
        pres = np.concatenate(pres, 0)

        mse, mae = calculate_mse_mae(pres, labels)
        msg = f"--> \t Eva. Iter {nb_iter}, pre_len {pre_len} : MAE. {mae:.6f}, MSE. {mse:.6f}"
        logger.info(msg)

    if isTest:
        return  writer, logger


    if mse < best_mse :
        msg = f"--> --> \t MSE Improved from {best_mse:.6f} to {mse:.6f} !!!"
        logger.info(msg)
        best_mse, best_iter = mse, nb_iter
        if save:
            torch.save({'trans': trans.state_dict()}, os.path.join(out_dir, 'net_best_mse.pth'))


    trans.train()
    return best_iter, best_mse, writer, logger


@torch.no_grad()
def evaluation_flow_matching(args, out_dir, val_loader, flow_model, net, logger, writer, nb_iter, best_iter, best_ds,
                             best_mse=99999, best_mse_iter=0, save=True):
    net.eval()
    flow_model.eval()

    labels = []
    pres = []

    for _, batch in enumerate(val_loader):
        batch_labels = None
        if isinstance(batch, (list, tuple)):
            batch, batch_labels = batch
            batch_labels = batch_labels.cuda().long()
        batch = batch.cuda().float()
        label = batch

        pre_idx = flow_model.sample_token_indices(
            batch_size=label.shape[0],
            quantizers=net.vqvae.quantizer,
            flow_steps=args.flow_steps,
            solver=args.solver,
            sample_temperature=args.sample_temperature,
            noise_scale=args.noise_scale,
            senior_sampler=getattr(args, 'senior_sampler', None),
            kde_bandwidth_factor=getattr(args, 'kde_bandwidth_factor', 1.0),
            kde_max_centers=getattr(args, 'kde_max_centers', 2000),
            device=label.device,
            dtype=torch.float32,
            labels=batch_labels,
            label_guidance_scale=getattr(args, 'label_guidance_scale', 1.0),
            sampling_mode=getattr(args, 'sampling_mode', 'shared_context'),
            label_conditioned_prior=getattr(args, 'label_conditioned_prior', False),
        )
        pre = net.forward_decoder(pre_idx)

        labels.append(label.detach().cpu().numpy())
        pres.append(pre.detach().cpu().numpy())

    labels = np.concatenate(labels, 0)
    pres = np.concatenate(pres, 0)

    mse, mae = calculate_mse_mae(labels, pres)
    msg = f"--> \t Eva. Iter {nb_iter} :, MAE. {mae:.6f}, MSE. {mse:.6f}"
    logger.info(msg)

    eval_max_samples = int(getattr(args, 'flow_eval_max_samples', 0) or 0)
    if eval_max_samples > 0 and labels.shape[0] > eval_max_samples:
        rng = np.random.default_rng(int(getattr(args, 'seed', 42)) + int(nb_iter))
        indices = rng.choice(labels.shape[0], size=eval_max_samples, replace=False)
        labels_for_ds = labels[indices]
        pres_for_ds = pres[indices]
        logger.info(
            f"--> \t Eva. Iter {nb_iter} :, DS subsample. "
            f"{eval_max_samples}/{labels.shape[0]} samples"
        )
    else:
        labels_for_ds = labels
        pres_for_ds = pres

    ds_repeats = int(getattr(args, 'flow_eval_ds_repeats', 5) or 0)
    if ds_repeats > 0:
        labels_list = list(labels_for_ds)
        pres_list = list(pres_for_ds)
        discriminative_score = []
        for _ in range(ds_repeats):
            temp_pred = discriminative_score_metrics(labels_list, pres_list)
            discriminative_score.append(temp_pred)

        ds_mean = np.mean(discriminative_score)
        ds_std = np.std(discriminative_score)
        msg = f"--> \t Eva. Iter {nb_iter} :, Discriminative Score. {ds_mean:.6f}, std:{ds_std}"
        logger.info(msg)
    else:
        ds_mean = float(mse)
        ds_std = 0.0
        logger.info(
            f"--> \t Eva. Iter {nb_iter} :, Discriminative Score skipped; "
            f"using MSE proxy {ds_mean:.6f} for checkpoint selection"
        )

    if writer is not None:
        writer.add_scalar('./EvalFM/mae', mae, nb_iter)
        writer.add_scalar('./EvalFM/mse', mse, nb_iter)
        writer.add_scalar('./EvalFM/ds', ds_mean, nb_iter)

    ds = ds_mean
    if ds < best_ds:
        msg = f"--> --> \t Discriminative Score Improved from {best_ds:.5f} to {ds:.5f} !!!"
        logger.info(msg)
        best_ds, best_iter = ds, nb_iter
        if save:
            torch.save({'flow': flow_model.state_dict()}, os.path.join(out_dir, 'net_best_ds.pth'))
            np.save(os.path.join(out_dir, 'labels_ds.npy'), labels)
            np.save(os.path.join(out_dir, 'pres_ds.npy'), pres)

    if mse < best_mse:
        msg = f"--> --> \t MSE Improved from {best_mse:.6f} to {mse:.6f} !!!"
        logger.info(msg)
        best_mse, best_mse_iter = mse, nb_iter
        if save:
            torch.save({'flow': flow_model.state_dict()}, os.path.join(out_dir, 'net_best_mse.pth'))
            np.save(os.path.join(out_dir, 'labels_mse.npy'), labels)
            np.save(os.path.join(out_dir, 'pres_mse.npy'), pres)

    flow_model.train()
    return best_iter, best_ds, best_mse_iter, best_mse, writer, logger
