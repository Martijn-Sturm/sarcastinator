from contextlib import redirect_stdout
import os


def proc_results(result, model, filename, logger, param_dict, **kwargs):
    accs_train = result.history['accuracy']
    accs_val = result.history['val_accuracy']

    loss_train = result.history['loss']
    loss_val = result.history['val_loss']

    max_acc_train = max(accs_train)
    max_loss_train = max(loss_train)

    max_acc_val = max(accs_val)
    max_loss_val = max(loss_val)

    n_params = model.count_params()

    # Summary logging
    logger.info(f"\nresults for: {filename}")
    logger.info(f"Model params: {n_params}")
    logger.info(f"Best training accuracy: {max_acc_train:.4}")
    logger.info(f"Best training loss: {max_loss_train:.4}")
    logger.info(f"Best validation accuracy: {max_acc_val:.4}")
    logger.info(f"Best validation loss: {max_loss_val:.4}\n")

    os.makedirs("./results/", exist_ok=True)
    with open(f"./results/{filename}.txt", "w") as f:
        f.write(f"Model: {filename}\n\n")

        # Kwargs:
        for key, value in param_dict.items():
            f.write(f"{key} \t : \t {value}\n")
        f.write("\n\n")

        # Kwargs:
        if kwargs:
            for key, value in kwargs.items():
                f.write(f"{key} \t : \t {value}\n")
            f.write("\n\n")
        # Training results
        f.write(f"for epoch nr: train acc - train loss - vali acc - vali loss \n")
        i = 1
        for accs_train, loss_train, accs_val, loss_val in zip(accs_train, loss_train, accs_val, loss_val):
            f.write(f"nr:{i} \t\t\t{accs_train:.4}\t\t{loss_train:.4}\t\t{accs_val:.4}\t\t{loss_val:.4} \n")
            i += 1

        # Model summary
        f.write(f"\n\nModel summary:\n")
        with redirect_stdout(f):
            model.summary()

    # Make output dictionary:
    output_dict = param_dict
    output_dict.update({
        "acc train": max_acc_train,
        "loss train": max_loss_train,
        "acc val": max_acc_val,
        "loss val": max_loss_val,
        "name": filename,
        "params": n_params
    })

    return(output_dict)
