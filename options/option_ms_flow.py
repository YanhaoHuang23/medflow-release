import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(
        description='The second stage CE-guided Flow Matching training for MSDFormer',
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # data
    parser.add_argument('--dataname', type=str, default='fmri', help='dataset directory')
    parser.add_argument('--data-path', type=str, default=None, help='optional dataset root/file path, used by mimic_icustay tuple data')
    parser.add_argument('--input-dim', type=int, default=None, help='optional tuple feature dimension; inferred when omitted')
    parser.add_argument('--feature-names', type=str, default=None, help='optional comma-separated feature names for tuple data')
    parser.add_argument('--class-label', type=int, default=None, help='optional class filter for class-specific unconditional generators')
    parser.add_argument('--batch-size', default=256, type=int, help='flow matching batch size')
    parser.add_argument('--encode-batch-size', default=64, type=int, help='batch size when encoding tokens via VQ')
    parser.add_argument('--window-size', type=int, default=24, help='time window length')

    # optimization
    parser.add_argument('--total-iter', default=100000, type=int, help='number of total iterations to run')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--weight-decay', default=1e-6, type=float, help='weight decay')

    # vqvae arch (stage-1 frozen)
    parser.add_argument('--code-dim', type=int, default=512, help='embedding dimension')
    parser.add_argument('--nb-code', type=int, nargs='+', default=[128, 512, 512], help='list of codebook sizes')
    parser.add_argument('--patch-num', type=int, nargs='+', default=[3, 6, 12], help='list of token numbers per scale')
    parser.add_argument('--mu', type=float, default=0.99, help='ema momentum in VQ codebook')
    parser.add_argument('--down-t', type=int, default=2, help='downsampling rate')
    parser.add_argument('--stride-t', type=int, default=2, help='stride size')
    parser.add_argument('--width', type=int, default=512, help='width of VQ encoder/decoder')
    parser.add_argument('--depth', type=int, default=3, help='depth of VQ encoder/decoder')
    parser.add_argument('--dilation-growth-rate', type=int, default=3, help='dilation growth rate')
    parser.add_argument('--vq-act', type=str, default='relu', choices=['relu', 'silu', 'gelu'], help='VQ activation')
    parser.add_argument('--vq-norm', type=str, default=None, help='VQ norm')
    parser.add_argument('--quantizer', type=str, default='ema_reset_sim', choices=['ema', 'orig', 'ema_reset', 'reset', 'lfq', 'ema_reset_sim', 'ema_sim', 'reset_sim'], help='VQ quantizer')

    # CE-FM senior options
    parser.add_argument('--fm-backbone', type=str, default='dit1d', choices=['dit1d', 'transformer1d'], help='backbone for CE-FM token transport')
    parser.add_argument('--flow-path', type=str, default='ot', choices=['ot'], help='flow path type')
    parser.add_argument('--solver', type=str, default='euler', choices=['euler', 'heun'], help='ODE solver in sampling')
    parser.add_argument('--flow-steps', type=int, default=30, help='number of ODE steps for sampling')
    parser.add_argument('--noise-scale', type=float, default=1.0, help='sampling noise scale')
    parser.add_argument('--sample-temperature', type=float, default=0.9, help='softmax temperature during sampling')
    parser.add_argument('--t-scheduler', type=str, default='cosine', choices=['cosine', 'linear'], help='time scheduler')
    parser.add_argument('--latent-rank', type=int, default=128, help='latent rank for CE-FM priors')
    parser.add_argument('--latent-noise-std', type=float, default=0.01, help='noise std for senior latent perturbation')
    parser.add_argument('--senior-sampler', type=str, default=None, choices=['mixup', 'kde', 'gaussian'], help='explicit sampling rule for senior variant')
    parser.add_argument(
        '--source-prior-mode',
        type=str,
        default='learned',
        choices=['learned', 'gaussian'],
        help='source endpoint prior for CE-FM; gaussian disables the learned source bank at train/sample time',
    )
    parser.add_argument('--train-mixup-prob', type=float, default=0.5, help='probability of train-time manifold mixup in senior variant')
    parser.add_argument('--train-mixup-alpha', type=float, default=1.0, help='beta alpha used in senior train-time manifold mixup')
    parser.add_argument(
        '--class-aware-mixup',
        action='store_true',
        help='for conditional flow, restrict train-time mixup pairs to samples with the same label',
    )
    parser.add_argument('--structure-loss-weight', type=float, default=10.0, help='weight of senior structure regularization')
    parser.add_argument('--senior-mean-reg-weight', type=float, default=0.1, help='weight of senior latent mean regularization')
    parser.add_argument('--senior-std-reg-weight', type=float, default=0.1, help='weight of senior latent std regularization')
    parser.add_argument('--senior-sample-noise-std', type=float, default=0.01, help='noise std added in senior mixup sampling')
    parser.add_argument('--kde-bandwidth-factor', type=float, default=1.0, help='bandwidth scaling for senior KDE sampling')
    parser.add_argument('--kde-max-centers', type=int, default=2000, help='max latent centers used to build senior KDE prior')
    parser.add_argument(
        '--sampling-mode',
        type=str,
        default='shared_context',
        choices=['shared_context', 'ctf_nearest'],
        help='multi-scale sampling mode; shared_context keeps the original sampler context, ctf_nearest conditions finer scales on nearest coarse tokens',
    )
    parser.add_argument(
        '--cross-scale-conditioning',
        type=str,
        default='none',
        choices=['none', 'ctf'],
        help='train-time cross-scale conditioning; ctf conditions finer flow scales on previously generated/real coarse scales',
    )
    parser.add_argument(
        '--dynamic-loss-weight',
        type=float,
        default=0.0,
        help='optional differentiable endpoint dynamic-stat matching loss for Stage-2 token flow',
    )
    parser.add_argument(
        '--dynamic-loss-components',
        type=str,
        default='diff_mean,diff_std,feat_std',
        help='comma-separated dynamic loss components: diff_mean,diff_std,feat_std',
    )
    parser.add_argument(
        '--contrastive-flow-weight',
        type=float,
        default=0.0,
        help='optional contrastive Flow Matching loss weight; 0 disables it',
    )
    parser.add_argument(
        '--contrastive-representation',
        type=str,
        default='token',
        choices=['token', 'velocity'],
        help='contrastive representation: token compares predicted codebook embeddings; velocity compares predicted sampling directions',
    )
    parser.add_argument(
        '--contrastive-negative-mode',
        type=str,
        default='random_nonself',
        choices=['random_nonself', 'different_label'],
        help='negative sampling rule for contrastive token loss',
    )
    parser.add_argument(
        '--contrastive-objective',
        type=str,
        default='margin',
        choices=['margin', 'delta'],
        help='contrastive objective: margin uses relu(margin + d_pos - d_neg), delta uses d_pos - temperature*d_neg',
    )
    parser.add_argument(
        '--contrastive-margin',
        type=float,
        default=1.0,
        help='margin used by --contrastive-objective margin',
    )
    parser.add_argument(
        '--contrastive-temperature',
        type=float,
        default=1.0,
        help='negative-distance multiplier used by --contrastive-objective delta',
    )
    parser.add_argument(
        '--balanced-fm-loss',
        action='store_true',
        help='for conditional flow, average per-sample token CE by class before averaging classes',
    )

    # flow network size
    parser.add_argument('--fm-hidden-dim', type=int, default=512, help='hidden size of flow model')
    parser.add_argument('--fm-depth', type=int, default=6, help='number of blocks for flow model')
    parser.add_argument('--fm-heads', type=int, default=8, help='attention heads')
    parser.add_argument('--fm-dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--conditional-flow', action='store_true', help='condition flow matching on class labels')
    parser.add_argument('--num-classes', type=int, default=2, help='number of labels for conditional flow')
    parser.add_argument(
        '--class-specific-output-head',
        action='store_true',
        help='for conditional flow, use class-specific token output heads on top of the shared backbone',
    )
    parser.add_argument(
        '--class-specific-adapter',
        action='store_true',
        help='for conditional flow, add small class-specific residual adapters inside the shared backbone',
    )
    parser.add_argument(
        '--label-conditioning-mode',
        type=str,
        default='add',
        choices=['add', 'film'],
        help='how label embeddings condition the flow backbone',
    )
    parser.add_argument(
        '--token-type-conditioning',
        type=str,
        default='none',
        choices=['none', 'class', 'class_scale'],
        help='optional token-type embedding added to per-token hidden states; class uses y, class_scale uses y and scale id',
    )
    parser.add_argument(
        '--positive-mixup-multiplier',
        type=float,
        default=1.0,
        help='for conditional flow, add y=positive-mixup-label data-level mixup samples before stage-2 token training; 1.0 disables it',
    )
    parser.add_argument(
        '--positive-mixup-alpha',
        type=float,
        default=0.4,
        help='Beta(alpha, alpha) parameter for positive-class data-level mixup',
    )
    parser.add_argument(
        '--positive-mixup-label',
        type=int,
        default=1,
        help='label to oversample with data-level mixup before stage-2 token training',
    )
    parser.add_argument(
        '--positive-mixup-mode',
        type=str,
        default='random',
        choices=['random', 'cluster', 'knn', 'safe_random', 'danger_random', 'danger_knn'],
        help='data-level positive mixup strategy',
    )
    parser.add_argument(
        '--positive-mixup-knn-k',
        type=int,
        default=64,
        help='for knn positive mixup, sample the second y=1 parent from this many nearest neighbors',
    )
    parser.add_argument(
        '--positive-mixup-safe-quantile',
        type=float,
        default=0.01,
        help='for safe_random positive mixup, feature quantile used as the real y=1 valid range',
    )
    parser.add_argument(
        '--positive-mixup-safe-max-violation-frac',
        type=float,
        default=0.02,
        help='for safe_random positive mixup, max fraction of time-feature values outside the valid range',
    )
    parser.add_argument(
        '--positive-mixup-safe-candidate-multiplier',
        type=float,
        default=3.0,
        help='for safe_random positive mixup, candidates generated per requested kept sample before filtering',
    )
    parser.add_argument(
        '--positive-mixup-danger-spec',
        type=str,
        default='SpO2:low,RR:high,MAP:low,SBP:low',
        help='for danger_* positive mixup, comma-separated feature:low|high tail definition',
    )
    parser.add_argument(
        '--positive-mixup-danger-quantile',
        type=float,
        default=0.10,
        help='for danger_* positive mixup, quantile used to score low/high danger tails',
    )
    parser.add_argument(
        '--positive-mixup-danger-power',
        type=float,
        default=1.0,
        help='for danger_* positive mixup, exponent applied to danger scores before anchor sampling',
    )
    parser.add_argument(
        '--positive-mixup-lambda-min',
        type=float,
        default=0.0,
        help='for danger_* positive mixup, optional minimum dominant anchor lambda after Beta sampling',
    )
    parser.add_argument(
        '--positive-mixup-clusters',
        type=int,
        default=8,
        help='number of KMeans clusters used by --positive-mixup-mode cluster',
    )
    parser.add_argument(
        '--positive-mixup-cluster-balance',
        type=str,
        default='none',
        choices=['none', 'inverse_sqrt', 'inverse'],
        help='cluster sampling weight for cluster-aware positive mixup',
    )
    parser.add_argument(
        '--positive-mixup-cluster-min-count',
        type=int,
        default=0,
        help='for cluster mixup, ignore clusters with fewer than this many positive samples; 0 disables filtering',
    )
    parser.add_argument(
        '--positive-mixup-cluster-cap-multiplier',
        type=float,
        default=0.0,
        help='for cluster mixup, cap extra samples per cluster to count * this value; <=0 disables capping',
    )
    parser.add_argument('--class-balanced-sampler', action='store_true', help='sample conditional flow batches with balanced class probabilities')
    parser.add_argument(
        '--class-loss-weight',
        type=str,
        default='none',
        choices=['none', 'auto_sqrt', 'auto_inverse'],
        help='optional class weight applied to conditional token CE loss',
    )
    parser.add_argument('--label-dropout-prob', type=float, default=0.0, help='drop conditional labels during training for CFG-style sampling')
    parser.add_argument('--label-guidance-scale', type=float, default=1.0, help='CFG-style label guidance scale used during conditional sampling/eval')
    parser.add_argument(
        '--label-conditioned-prior',
        action='store_true',
        help='when sampling conditional flow, draw senior latent priors from the requested class instead of the full imbalanced training pool',
    )

    # resume
    parser.add_argument('--resume-pth', type=str, default=None, help='resume stage-1 vq model path')
    parser.add_argument('--resume-flow', type=str, default=None, help='resume stage-2 flow model path')

    # output
    parser.add_argument('--out-dir', type=str, default='./output_flow/', help='output directory')
    parser.add_argument('--exp-name', type=str, default='MS_FlowCE', help='experiment name')

    # logging / eval
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=2500, type=int, help='evaluation frequency')
    parser.add_argument('--flow-eval-ds-repeats', default=5, type=int,
                        help='number of discriminative-score repeats during Stage-2 flow evaluation')
    parser.add_argument('--flow-eval-max-samples', default=0, type=int,
                        help='if > 0, subsample at most this many validation samples for Stage-2 flow evaluation')
    parser.add_argument('--seed', default=123, type=int, help='seed')
    parser.add_argument('--repeat-times', default=100, type=int, help='repeat token dataset to increase optimization steps')

    # runtime
    parser.add_argument('--if-test', action='store_true', help='only run generation/evaluation')
    parser.add_argument('--gpu', type=str, default='0', help='visible gpus')

    return parser.parse_args()
