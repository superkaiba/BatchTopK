import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(description="BatchTopK Sparse Autoencoder Training")

    # Training parameters
    parser.add_argument("--seed", type=int, default=49, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=4096, help="Training batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num_tokens", type=int, default=int(1e9), help="Number of tokens to train on")
    parser.add_argument("--l1_coeff", type=float, default=0, help="L1 regularization coefficient")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1 parameter")
    parser.add_argument("--beta2", type=float, default=0.99, help="Adam beta2 parameter")
    parser.add_argument("--max_grad_norm", type=float, default=100000, help="Maximum gradient norm")

    # Model parameters
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"], help="Data type for training")
    parser.add_argument("--model_name", type=str, default="gpt2-small", help="Name of the model to use")
    parser.add_argument("--site", type=str, default="resid_pre", help="Activation site in the model")
    parser.add_argument("--layer", type=int, default=8, help="Layer number to train on")
    parser.add_argument("--act_size", type=int, default=768, help="Activation size")
    parser.add_argument("--dict_size", type=int, default=12288, help="Dictionary size")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to train on")
    parser.add_argument("--model_batch_size", type=int, default=512, help="Model batch size")
    parser.add_argument("--num_batches_in_buffer", type=int, default=10, help="Number of batches to keep in buffer")

    # Dataset and logging
    parser.add_argument("--dataset_path", type=str, default="Skylion007/openwebtext", help="Path to the dataset")
    parser.add_argument("--wandb_project", type=str, default="sparse_autoencoders", help="Weights & Biases project name")
    parser.add_argument("--input_unit_norm", action="store_true", default=True, help="Normalize input to unit norm")
    parser.add_argument("--perf_log_freq", type=int, default=1000, help="Performance logging frequency")
    parser.add_argument("--sae_type", type=str, default="topk", help="Type of sparse autoencoder")
    parser.add_argument("--checkpoint_freq", type=int, default=10000, help="Checkpoint frequency")
    parser.add_argument("--n_batches_to_dead", type=int, default=5, help="Number of batches to consider dead features")

    # BatchTopK specific parameters
    parser.add_argument("--top_k", type=int, default=32, help="Number of top activations to keep")
    parser.add_argument("--top_k_aux", type=int, default=512, help="Auxiliary top k value")
    parser.add_argument("--aux_penalty", type=float, default=(1 / 32), help="Auxiliary penalty coefficient")
    parser.add_argument("--bandwidth", type=float, default=0.001, help="Bandwidth for jump ReLU")

    args = parser.parse_args()

    # Convert dtype string to torch dtype
    if args.dtype == "float32":
        args.dtype = torch.float32
    elif args.dtype == "float16":
        args.dtype = torch.float16
    elif args.dtype == "bfloat16":
        args.dtype = torch.bfloat16

    return args
