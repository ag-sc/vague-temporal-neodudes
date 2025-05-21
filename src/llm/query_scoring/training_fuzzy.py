import os.path
import sys
import traceback
from argparse import ArgumentParser
from datetime import datetime

import optuna
import torch
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from optuna import Trial
from optuna.storages import JournalStorage, JournalFileStorage

from llm.query_scoring.dataset import FuzzyQueryScoreDataModule
from llm.query_scoring.models.t5 import QueryScoreT5
import lemon

def objective(trial: Trial,
              batch_size=None,
              learning_rate=None,
              ld=None,
              epochs=None,
              model_name="google/flan-t5-small"):
    try:
        if batch_size is None:
            batch_size = 1
            # trial.suggest_int('batch_size', 1, 10)
        if learning_rate is None:
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-4, log=True)
        if ld is None:
            ld = trial.suggest_float('ld', 0.9, 1.0, log=True)
        if epochs is None:
            epochs = trial.suggest_int('epochs', 1, 5)

        # get from https://zenodo.org/records/12610838/files/query_score_models.tar.zst?download=1 using model query_score_llm_clampfp_1.3902932441715008e-05_0.9013707813420198_64_2_2024-06-21_20-33-02-776942.ckpt
        model = QueryScoreT5.load_from_checkpoint(os.path.join(os.path.dirname(sys.modules["lemon"].__file__), "resources", "query_score_llm_clampfp_1.3902932441715008e-05_0.9013707813420198_64_2_2024-06-21_20-33-02-776942_best_val_loss.ckpt"))
        dm = FuzzyQueryScoreDataModule(tokenizer=model.tokenizer, batch_size=batch_size)
        dm.prepare_data()
        dm.setup("train")

        print("Batch size: ", batch_size, flush=True)
        print("Learning rate: ", learning_rate, flush=True)
        print("Lambda: ", ld, flush=True)
        print("Epochs: ", epochs, flush=True)
        print("Model: ", model_name, flush=True)

        logger = TensorBoardLogger("tb_logs",
                                   name="query_score_llm",
                                   version=None if 'SLURM_ARRAY_TASK_ID' not in os.environ else os.environ['SLURM_ARRAY_TASK_ID'])

        trainer = Trainer(enable_checkpointing=True,
                          logger=logger,
                          #accelerator="cpu",
                          accelerator="cuda",
                          devices=-1,
                          reload_dataloaders_every_n_epochs=1,
                          log_every_n_steps=1,
                          max_epochs=epochs,
                          min_epochs=epochs,
                          strategy="ddp")
        trainer.fit(model, datamodule=dm)
        trainer.save_checkpoint(f"query_score_llm_fuzzy_{learning_rate}_{ld}_{batch_size}_{epochs}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}.ckpt")
        print(trainer.callback_metrics, flush=True)
        ret_val = trainer.callback_metrics["val_loss"]
        torch.cuda.empty_cache()
        trainer.test(model, datamodule=dm)
        print(trainer.callback_metrics, flush=True)
        torch.cuda.empty_cache()
        return ret_val
    except Exception as e:
        print(traceback.format_exc(), flush=True)
        print(e, flush=True)
        raise optuna.TrialPruned()

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--model", type=str, default="google/flan-t5-small")
    argparser.add_argument("--batchsize", type=int, default=80)
    argparser.add_argument("--epochs", type=int, default=None)
    argparser.add_argument("--lr", type=float, default=None)
    argparser.add_argument("--ld", type=float, default=None)
    argparser.add_argument("--trials", type=int, default=20)
    argparser.add_argument("--optunafile", type=str, default=f"fuzzy_optuna.log")
    argparser.add_argument("--studyname", type=str, default=f"Fuzzy Query Score")

    arguments = argparser.parse_args()

    batch_size = arguments.batchsize #64 + 16
    learning_rate = arguments.lr
    ld = arguments.ld
    epochs = arguments.epochs #5
    n_trials = arguments.trials  #10
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(arguments.optunafile)
    )
    #JournalStorage(JournalFileStorage(f"{arguments.optunafile}"))
    study = optuna.create_study(direction='minimize',
                                study_name=f"{arguments.studyname}",
                                storage=storage, load_if_exists=True)
    study.optimize(lambda trial: objective(trial=trial,
                                           batch_size=batch_size,
                                           epochs=int(epochs) if epochs is not None else None,
                                           model_name=arguments.model,
                                           learning_rate=float(learning_rate) if learning_rate is not None else None,
                                           ld=float(ld) if ld is not None else None),
                   n_trials=n_trials,
                   n_jobs=1)
    print(study.best_params)
