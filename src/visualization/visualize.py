import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from src.logger import infologger
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def conf_matrix(y_test: pd.Series, y_pred: pd.Series, path: str) -> str:
    try:
        curr_time = datetime.now().strftime("%d%m%y-%H%M%S")
    except Exception as e:
        infologger.error(
            f"something wrong with directories, check conf_matrix() for issue. exception: {e}"
        )
    else:
        infologger.info("directories are all set!")
        try:
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        except Exception as e:
            infologger.critical(
                f"unable to plot the confusion metrix, check conf_matrix() for issue. exception: {e}"
            )
        else:
            disp.plot(cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted Label")
            plt.ylabel("Actual Label")
            filename = f"{path}/{curr_time}.png"
            plt.savefig(filename)
            plt.close()
            infologger.info(f"confusion metrix saved successfully, path: {path}")
            return filename


if __name__ == "__main__" : 
    pass
    