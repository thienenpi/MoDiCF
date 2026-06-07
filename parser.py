import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--epoch", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--lr", type=int, default=0.0001, help="Learning rate")
    parser.add_argument("--lr_decay", type=float, default=-1, help="Learning rate decay factor, lr decays to lr*lr_decay")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    parser.add_argument("--embed_channel", type=int, default=16, help="Embedding size")
    parser.add_argument("--embed_size", type=int, default=32, help="Embedding size")
    parser.add_argument("--unet_channels", type=str, default="[16, 32, 64, 128]", help="UNet channels")
    parser.add_argument("--conv1d_kernel_size", type=int, default=3, help="Conv1d kernel size")
    parser.add_argument("--num_sample_steps", type=int, default=1000, help="Number of sample steps")
    parser.add_argument("--T_s", type=int, default=10, help="Number of sampling steps")
    parser.add_argument("--alpha_1", type=float, default=1, help="Reconstruction loss weight")

    parser.add_argument("--L", type=int, default=2, help="Number of layers")
    parser.add_argument('--delta', type=float, default=0.4, help='model_cat_rate')
    parser.add_argument('--eta', type=float, default=0.7, help='id_cat_rate')
    parser.add_argument('--weight_size', nargs='?', default='[64, 64]', help='Output sizes of every layer')
    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]', help='for emb_loss.')
    parser.add_argument('--gnn_embed_size', type=int, default=256, help='Embedding size')
    parser.add_argument('--isSparse', type=bool, default=True, help='isSparse')
    parser.add_argument('--H', default=8, type=int, help='head_num_of_multihead_attention. For multi-model relation.')
    parser.add_argument('--lambda_2', default=1e-5, type=float, help='feat_reg_decay')

    parser.add_argument('--m_topk_rate', default=0.0001, type=float, help='for reconstruct')
    parser.add_argument('--lambda_1', type=float, default=0.09, help='Control the effect of the contrastive auxiliary task')

    parser.add_argument("--dataset", type=str, default="baby", help="Dataset name")
    parser.add_argument("--MR", type=float, default=0.4, help="MR")
    parser.add_argument("--complete", type=str, default="zero", help="Complete strategy; Options: mean, zero, mean, random, none, nn")
    parser.add_argument("--normalize", type=bool, default=True, help="Normalize data")
    parser.add_argument("--reduce_dim", type=bool, default=True, help="Reduce dimensionality")
    parser.add_argument("--dim", type=int, default=128, help="Dimensionality")
    parser.add_argument("--load_dir", type=str, default="./checkpoint/baby/", help="Load directory")
    parser.add_argument("--load_model", type=str, default="best_model.pth", help="Load model name")
    parser.add_argument('--T', default=1, type=int, help='it for ui update')
    parser.add_argument('--tau', default=0.5, type=float, help='')

    parser.add_argument("--gamma", type=float, default=0.01, help="Counterfactual inference parameter (constant gamma, eq.17; used when gamma_mode='fixed')")
    parser.add_argument("--gamma_mode", type=str, default="fixed", choices=["fixed", "rank"],
                        help="Counterfactual gamma mode: 'fixed' = constant gamma (eq.17); 'rank' = rank-dependent margin gamma_{u,i}=max(0, y_{u,i}-y_{u,K+1}) (CountER-style)")
    parser.add_argument("--rank_k", type=int, default=20,
                        help="Reference K for the rank-dependent margin threshold y_{u,K+1} (used when gamma_mode='rank')")
    parser.add_argument("--lambda_cf", type=float, default=1.0,
                        help="Intervention strength lambda for rank-dependent counterfactual calibration (used when gamma_mode='rank')")
    parser.add_argument("--lambda_cf_list", type=str, default="",
                        help="Optional list (e.g. '[0.1,0.5,1.0]') to grid-search lambda in rank mode; empty string disables the sweep")
    parser.add_argument("--alpha_2", type=float, default=0.7, help="Item loss weight")

    parser.add_argument('--Ks', nargs='?', default='[5, 10, 20, 50]', help='K value of ndcg/recall @ k')
    parser.add_argument('--test_flag', nargs='?', default='part', help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument('--verbose', type=int, default=1, help='Interval of evaluation.')
    parser.add_argument('--anchor_rate', default=0.25, type=float, help='anchor_rate')
    parser.add_argument('--sample_num_ii', default=8, type=int, help='sample_num')

    # Checkpoint / resume: persist full training state so an interrupted run can
    # continue instead of restarting from epoch 0.
    parser.add_argument('--ckpt_dir', type=str, default='',
                        help="Directory to save/resume the training checkpoint. Empty -> ./checkpoint/<dataset>/train/")
    parser.add_argument('--save_every', type=int, default=1,
                        help="Save a resume checkpoint every N epochs (the final epoch is always saved)")
    parser.add_argument('--from_scratch', action='store_true',
                        help="Ignore any existing checkpoint in ckpt_dir and start training from epoch 0")

    return parser.parse_args()