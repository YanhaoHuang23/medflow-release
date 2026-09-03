import torch.nn as nn
from models.encdec import Encoder, Decoder
from models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset,LFQ,QuantizeEMAResetSim,QuantizeEMASim,QuantizeResetSim
import math
import torch
import numpy as np
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class TSVQVAE(nn.Module):
    def __init__(self,
                 args,
                 nb_code=1024,
                 code_dim=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        self.quant = args.quantizer
        self.args = args
        output_emb_width = self.code_dim
        if args.quantizer == "lfq":
            output_emb_width = math.ceil(math.log2(self.num_code))
            self.code_dim = output_emb_width
        if getattr(args, "input_dim", None) is not None:
            input_emb_width = int(args.input_dim)
        elif args.dataname == 'stock':
            input_emb_width = 6
        elif args.dataname == 'energy':
            input_emb_width = 28
        elif args.dataname == 'sine':
            input_emb_width = 5
        elif args.dataname == 'etth':
            input_emb_width = 7
        elif args.dataname == 'fmri':
            input_emb_width = 50
        elif args.dataname == 'mujoco':
            input_emb_width = 14
        elif args.dataname == 'mimic_icustay':
            input_emb_width = 7
        else:
            raise ValueError(f'Unsupported dataname for VQVAE input width: {args.dataname}')
        self.input_emb_width = input_emb_width
        self.encoder = nn.ModuleList([Encoder(input_emb_width, output_emb_width, int(math.log2(args.window_size//pn)), stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm, quantizer=args.quantizer) for pn in args.patch_num])
        self.decoder = nn.ModuleList([Decoder(input_emb_width, output_emb_width, int(math.log2(args.window_size//pn)), stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm) for pn in args.patch_num])
        if args.quantizer == "ema_reset":
            self.quantizer = nn.ModuleList([QuantizeEMAReset(nc, code_dim, args) for nc in args.nb_code])
        elif args.quantizer == "orig":
            self.quantizer = nn.ModuleList([Quantizer(nc, code_dim, 1.0) for nc in args.nb_code])
        elif args.quantizer == "ema":
            self.quantizer = QuantizeEMA(nb_code, code_dim, args)
        elif args.quantizer == "reset":
            self.quantizer = QuantizeReset(nb_code, code_dim, args)
        elif args.quantizer == "lfq":
            self.quantizer = LFQ(self.num_code, self.code_dim, args)
        elif args.quantizer == "ema_reset_sim":
            self.quantizer = nn.ModuleList([QuantizeEMAResetSim(nc, code_dim, args) for nc in args.nb_code])
        elif args.quantizer == "ema_sim":
            self.quantizer = nn.ModuleList([QuantizeEMASim(nc, code_dim, args) for nc in args.nb_code])
        elif args.quantizer == "reset_sim":
            self.quantizer = nn.ModuleList([QuantizeResetSim(nc, code_dim, args) for nc in args.nb_code])



    def preprocess(self, x):
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        x = x.permute(0,2,1)
        return x



    def encode(self, x):

        N, T, _ = x.shape
        code_idxs = torch.empty(N, 0, dtype=torch.int64).to(x.device)
        nc_sum = 0
        x_in = self.preprocess(x)
        for k,nc in enumerate(self.args.nb_code):
            x_encoder = self.encoder[k](x_in)
            x_quantized, loss, perp  = self.quantizer[k](x_encoder)
            x_decoder = self.decoder[k](x_quantized)###shared2 x.shape[-1]
            x_in = x_in - x_decoder

            x_encoder = self.postprocess(x_encoder)
            x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
            code_idx = self.quantizer[k].quantize(x_encoder)
            code_idx = code_idx.view(N, -1)
            code_idx = code_idx + nc_sum
            nc_sum = nc_sum +nc
            code_idxs = torch.cat([code_idxs,code_idx],dim=1)
        return code_idxs


    def forward(self, x):

        x_in = self.preprocess(x)
        x_out = torch.zeros_like(x_in, device=x_in.device)
        total_loss = torch.tensor(0.0, requires_grad=True)
        perplexity = 0.0
        for k in range(len(self.args.patch_num)):
            # Encode
            x_encoder = self.encoder[k](x_in)

            ## quantization
            x_quantized, loss, perp  = self.quantizer[k](x_encoder)

            ## decoder
            x_decoder = self.decoder[k](x_quantized)###shared2 x.shape[-1]
            x_in = x_in - x_decoder
            x_out = x_out + x_decoder
            total_loss = total_loss + loss
            perplexity = perplexity + perp
        x_out = self.postprocess(x_out)
        perplexity = perplexity/len(self.args.patch_num)
        return x_out, total_loss, perplexity

        # ##ms loss
        # x_in = self.preprocess(x)
        # x_out = torch.zeros_like(x_in, device=x_in.device)
        # x_out_list = []
        # total_loss = torch.tensor(0.0, requires_grad=True)
        # perplexity = 0.0
        # for k in range(len(self.args.patch_num)):
        #     # Encode
        #     x_encoder = self.encoder[k](x_in)
        #
        #     ## quantization
        #     x_quantized, loss, perp  = self.quantizer[k](x_encoder)
        #
        #     ## decoder
        #     x_decoder = self.decoder[k](x_quantized)###shared2 x.shape[-1]
        #     x_in = x_in - x_decoder
        #     x_out = x_out + x_decoder
        #     x_out_list.append(self.postprocess(x_out))
        #     total_loss = total_loss + loss
        #     perplexity = perplexity + perp
        # x_out = self.postprocess(x_out)
        # perplexity = perplexity/len(self.args.patch_num)
        # return x_out_list, total_loss, perplexity


    def forward_decoder(self, code_idxs):
        start = 0
        nc_sum = 0
        N = code_idxs.shape[0]
        x_out = torch.zeros([N,self.input_emb_width,self.args.window_size]).to(code_idxs.device)
        for k,pn in enumerate(self.args.patch_num):
            code_idx = code_idxs[:,start:start+pn]
            code_idx = code_idx - nc_sum
            nc_sum = nc_sum + self.args.nb_code[k]
            x_d = self.quantizer[k].dequantize(code_idx)
            x_d = x_d.view(N, -1, self.code_dim).permute(0, 2, 1).contiguous()
            # decoder
            x_decoder = self.decoder[k](x_d)
            x_out = x_out + x_decoder
            start = start + pn

        x_out = self.postprocess(x_out)
        return x_out


class VQVAE(nn.Module):
    def __init__(self,
                 args,
                 nb_code=512,
                 code_dim=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        
        self.vqvae = TSVQVAE(args, nb_code, code_dim, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

    def encode(self, x):

        quants = self.vqvae.encode(x) # (N, T)
        return quants


    def forward(self, x):

        x_out, loss, perplexity = self.vqvae(x)
        return x_out, loss, perplexity

    def forward_decoder(self, x):
        x_out = self.vqvae.forward_decoder(x)
        return x_out
        
