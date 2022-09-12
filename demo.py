'''
Main file for creating simulated data or loading real data
and running MetaRegGNN and sample selection methods.

Usage:
    For data processing:
        python demo.py --mode data 

    For inferences:
        python demo.py --mode infer 

    For more information:
        python demo.py -h
'''

import argparse
import pickle

import torch
import numpy as np

import proposed_method.data_utils as data_utils
import evaluators
from config import Config


parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['data', 'infer'],
                    help="Creates data and topological features OR make inferences on data")

opts = parser.parse_args()

if opts.mode == 'data':
    '''
    Connectome and scores are simulated to the folder specified in config.py.
    '''
    data_utils.create_dataset() 
    print(f"Data and topological features are created and saved at {Config.DATA_FOLDER} successfully.")

elif opts.mode == 'infer':
    '''
    Cross validation will be used to train and generate inferences
    on the data saved in the folder specified in config.py.

    Overall MAE and RMSE will be printed and predictions will be saved
    in same data folder.
    '''
    #print(f"{opts.model} will be run on the data.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mae_evaluator = lambda p, s: np.mean(np.abs(p - s))
    rmse_evaluator = lambda p, s: np.sqrt(np.mean((p - s) ** 2))

    preds, scores, _ = evaluators.evaluate_MetaRegGNN(shuffle=Config.SHUFFLE, random_state=Config.MODEL_SEED,
                                                  dropout=Config.MetaRegGNN.DROPOUT,
                                                  lr=Config.MetaRegGNN.LR, wd=Config.MetaRegGNN.WD, device=device,
                                                  num_epoch=Config.MetaRegGNN.NUM_EPOCH)
    if Config.SampleSelection.SAMPLE_SELECTION:
        mae_arr = [mae_evaluator(p, s) for p, s in zip(preds.values(), scores.values())]
        rmse_arr = [rmse_evaluator(p, s) for p, s in zip(preds.values(), scores.values())]
        print(f"For k in {Config.SampleSelection.K_LIST}:")
        print(f"Mean MAE +- std over k: {np.mean(mae_arr):.3f} +- {np.std(mae_arr):.3f}")
        print(f"Min, Max MAE over k: {np.min(mae_arr):.3f}, {np.max(mae_arr):.3f}")
        print(f"Mean RMSE +- std over k: {np.mean(rmse_arr):.3f} +- {np.std(rmse_arr):.3f}")
        print(f"Min, Max RMSE over k: {np.min(rmse_arr):.3f}, {np.max(rmse_arr):.3f}")
    else:
        print(f"MAE: {mae_evaluator(preds, scores):.3f}")
        print(f"RMSE: {rmse_evaluator(preds, scores):.3f}")

    with open(f"{Config.RESULT_FOLDER}preds.pkl", 'wb') as f:
        pickle.dump(preds, f)
    with open(f"{Config.RESULT_FOLDER}scores.pkl", 'wb') as f:
        pickle.dump(scores, f)

    print(f"Predictions are successfully saved at {Config.RESULT_FOLDER}.")

else:
    raise Exception("Unknown argument.")
