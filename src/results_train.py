from contextlib import redirect_stdout
import os


def proc_results(result, model, filename, logger, **kwargs):
    accs_train = result.history['accuracy']
    accs_val = result.history['val_accuracy']

    loss_train = result.history['loss']
    loss_val = result.history['val_loss']

    # Summary logging
    logger.info(f"\nresults for: {filename}")
    logger.info(f"Best training accuracy: {max(accs_train)}")
    logger.info(f"Best training loss: {max(loss_train)}")
    logger.info(f"Best validation accuracy: {max(accs_val)}")
    logger.info(f"Best validation loss: {max(loss_val)}\n")

    os.makedirs("./results/", exist_ok=True)
    with open(f"./results/{filename}.txt", "w") as f:
        f.write(f"Model: {filename}\n\n")

        # Kwargs:
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
