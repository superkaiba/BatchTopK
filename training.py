import torch
import tqdm
from logs import init_wandb, log_wandb, log_model_performance, save_checkpoint


def train_sae(sae, activations_store, model, args):
    num_batches = args.num_tokens // args.batch_size
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    pbar = tqdm.trange(num_batches)

    wandb_run = init_wandb(args)

    for i in pbar:
        batch = activations_store.next_batch()
        sae_output = sae(batch)
        log_wandb(sae_output, i, wandb_run)
        if i % args.perf_log_freq == 0:
            log_model_performance(wandb_run, i, model, activations_store, sae)

        if i % args.checkpoint_freq == 0:
            save_checkpoint(wandb_run, sae, args, i)

        loss = sae_output["loss"]
        pbar.set_postfix(
            {"Loss": f"{loss.item():.4f}", "L0": f"{sae_output['l0_norm']:.4f}", "L2": f"{sae_output['l2_loss']:.4f}", "L1": f"{sae_output['l1_loss']:.4f}", "L1_norm": f"{sae_output['l1_norm']:.4f}"}
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), args.max_grad_norm)
        sae.make_decoder_weights_and_grad_unit_norm()
        optimizer.step()
        optimizer.zero_grad()

    save_checkpoint(wandb_run, sae, args, i)


def train_sae_group(saes, activations_store, model, args_list):
    num_batches = args_list[0].num_tokens // args_list[0].batch_size
    optimizers = [torch.optim.Adam(sae.parameters(), lr=args.lr, betas=(args.beta1, args.beta2)) for sae, args in zip(saes, args_list)]
    pbar = tqdm.trange(num_batches)

    wandb_run = init_wandb(args_list[0])

    batch_tokens = activations_store.get_batch_tokens()

    for i in pbar:
        batch = activations_store.next_batch()
        counter = 0
        for sae, args, optimizer in zip(saes, args_list, optimizers):
            sae_output = sae(batch)
            loss = sae_output["loss"]
            log_wandb(sae_output, i, wandb_run, index=counter)
            if i % args.perf_log_freq == 0:
                log_model_performance(wandb_run, i, model, activations_store, sae, index=counter, batch_tokens=batch_tokens)

            if i % args.checkpoint_freq == 0:
                save_checkpoint(wandb_run, sae, args, i)

            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "L0": f"{sae_output['l0_norm']:.4f}",
                    "L2": f"{sae_output['l2_loss']:.4f}",
                    "L1": f"{sae_output['l1_loss']:.4f}",
                    "L1_norm": f"{sae_output['l1_norm']:.4f}",
                }
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), args.max_grad_norm)
            sae.make_decoder_weights_and_grad_unit_norm()
            optimizer.step()
            optimizer.zero_grad()
            counter += 1

    for sae, args, optimizer in zip(saes, args_list, optimizers):
        save_checkpoint(wandb_run, sae, args, i)
