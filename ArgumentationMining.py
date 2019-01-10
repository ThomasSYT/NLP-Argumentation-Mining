import os
from model import bilstm


def run(params):
    # set model output path
    params["model_path"] = os.path.join("results", "{}.model".format(params["model"]))
    params["prediction_path"] = os.path.join("results", "{}_predictions.txt".format(params["model"]))

    # create model
    if params["model"] == "bilstm":
        model = bilstm.ArgMiningBiLSTM(params)
    else:
        raise ValueError("Unknown model: {}".format(params["model"]))

    # get predictions
    model.train_and_predict()


if __name__=='__main__':
    params = {"model": "bilstm",    
              "batch_size": 16,
              "dropout": 0.3,
              "hidden_units": 280,
              "epochs": 1}#70
    run(params)