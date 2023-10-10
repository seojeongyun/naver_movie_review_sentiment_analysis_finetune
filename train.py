from engine import Trainer
from huggingface_hub import login


if __name__ == '__main__':
    from setproctitle import *
    setproctitle('finetune_NSMC_JY')
    trainer = Trainer('train')
    trainer.train()
    trainer.print_loss()