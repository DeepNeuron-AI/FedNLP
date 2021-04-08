import argparse
import logging
import os
import sys

import numpy as np
import torch
# this is a temporal import, we will refactor FedML as a package installation
import wandb

wandb.init(mode="disabled")

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))


from data_preprocessing.seq_tagging_preprocessor import TLMPreprocessor
from data_manager.seq_tagging_data_manager import SequenceTaggingDataManager

from model.transformer.model_args import ClassificationArgs

from training.st_transformer_trainer import SeqTaggingTrainer

from experiments.utils.general import set_seed, create_model, add_centralized_args
 


if __name__ == "__main__":
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    parser = add_centralized_args(parser) # add general args.
    # TODO: you can add customized args here.
    args = parser.parse_args()

    # customize the log format
    logging.basicConfig(level=logging.INFO,
                        format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S')
    logging.info(args)

    set_seed(args.manual_seed)

    # device
    device = torch.device("cuda:0")

    # attributes
    attributes = SequenceTaggingDataManager.load_attributes(args.data_file_path)

    # model
    model_args = ClassificationArgs()    
    model_args.model_name = args.model_name
    model_args.model_type = args.model_type
    model_args.load(model_args.model_name)
    model_args.num_labels = len(attributes["label_vocab"])
    model_args.update_from_dict({"num_train_epochs": args.num_train_epochs,
                              "learning_rate": args.learning_rate,
                              "gradient_accumulation_steps": args.gradient_accumulation_steps,
                              "do_lower_case": args.do_lower_case,
                              "manual_seed": args.manual_seed,
                              "reprocess_input_data": True, # for ignoring the cache features.
                              "overwrite_output_dir": True,
                              "max_seq_length": args.max_seq_length,
                              "train_batch_size": args.train_batch_size,
                              "eval_batch_size": args.eval_batch_size,
                              "evaluate_during_training_steps": args.evaluate_during_training_steps,
                              "fp16": args.fp16,
                              "data_file_path": args.data_file_path,
                              "partition_file_path": args.partition_file_path,
                              "partition_method": args.partition_method,
                              "dataset": args.dataset,
                              "output_dir": args.output_dir,
                              "is_debug_mode": args.is_debug_mode
                              })

    num_labels = len(attributes["label_vocab"])
    model_config, model, tokenizer = create_model(model_args, formulation="sequence_tagging")

    # preprocessor
    preprocessor = TLMPreprocessor(args=model_args, label_vocab=attributes["label_vocab"], tokenizer=tokenizer)

    # data manager
    process_id = 0
    num_workers = 1
    dm = SequenceTaggingDataManager(args, model_args, preprocessor)
    train_examples, test_examples, train_dl, test_dl = dm.load_centralized_data()

    # Create a SeqTaggingModel and start train
    trainer = SeqTaggingTrainer(model_args, device, model, train_dl, test_dl, test_examples, tokenizer)
    trainer.train_model()


''' Example Usage:

export CUDA_VISIBLE_DEVICES=0
DATA_NAME=w_nut
CUDA_VISIBLE_DEVICES=0 python -m experiments.centralized.transformer_exps.main_st \
    --dataset ${DATA_NAME} \
    --data_file ./data/fednlp_data/data_files/${DATA_NAME}_data.h5 \
    --partition_file ./data/fednlp_data/partition_files/${DATA_NAME}_partition.h5 \
    --partition_method uniform \
    --model_type distilbert \
    --model_name distilbert-base-uncased  \
    --do_lower_case True \
    --train_batch_size 32 \
    --eval_batch_size 8 \
    --max_seq_length 256 \
    --learning_rate 5e-5 \
    --num_train_epochs 5 \
    --evaluate_during_training_steps 10 \
    --output_dir /tmp/${DATA_NAME}_fed/ \
    --n_gpu 1
'''
