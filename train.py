from training import train_sae
from sae import VanillaSAE, TopKSAE, BatchTopKSAE, JumpReLUSAE
from activation_store import ActivationsStore
from transformer_lens import HookedTransformer
from args import get_args
import wandb
from transformer_lens.loading_from_pretrained import get_pretrained_model_config
from transformers import AutoModelForCausalLM, AutoConfig


def main():
    args = get_args()

    # Initialize wandb if project name is specified
    if args.wandb_project:
        wandb.init(project=args.wandb_project)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=args.dtype)
    # Initialize model and activation store
    model = HookedTransformer.from_pretrained(args.model_name, hf_model=model).to(args.dtype).to(args.device)
    activations_store = ActivationsStore(model, args)

    # Initialize appropriate SAE based on type
    if args.sae_type == "vanilla":
        sae = VanillaSAE(args)
    elif args.sae_type == "topk":
        sae = TopKSAE(args)
    elif args.sae_type == "batchtopk":
        sae = BatchTopKSAE(args)
    elif args.sae_type == "jumprelu":
        sae = JumpReLUSAE(args)
    else:
        raise ValueError(f"Unknown SAE type: {args.sae_type}")

    # Train the SAE
    train_sae(sae, activations_store, model, args)

    # Close wandb if it was initialized
    if args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
