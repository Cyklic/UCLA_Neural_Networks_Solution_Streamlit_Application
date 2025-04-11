# from setuptools import find_packages, setup


# setup(
#     name='src',
#     packages=find_packages(),
#     version='0.1.0',
#     description='Credit Risk Model code structuring',
#     author='Leonard Umoru',
#     license='',
# )

import logging
import traceback
from src.data.make_dataset import load_data
from src.visualization.visualize import loss_curve
from src.features.build_features import create_dummy_vars
from src.models.train_model import train_MLPmodel
from src.models.predict_model import evaluate_model

# Set up logging
logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="a"
)

if __name__ == "__main__":
    try:
        logging.info("Pipeline started.")
        
        # Load and preprocess the data
        data_path = "data/raw/admission(in).csv"
        try:
            data = load_data(data_path)
            logging.info("Data loaded successfully from %s", data_path)
        except Exception as e:
            logging.error("Error loading data.")
            logging.error(traceback.format_exc())
            raise

        # Create dummy variables and separate features and target
        try:
            x, y = create_dummy_vars(data)
            logging.info("Dummy variables created successfully.")
        except Exception as e:
            logging.error("Error creating dummy variables.")
            logging.error(traceback.format_exc())
            raise

        # Train the Multilayer Perceptron model
        try:
            MLP, xtest_scaled, ytest = train_MLPmodel(x, y)
            logging.info("MLP model trained successfully.")
        except Exception as e:
            logging.error("Error during model training.")
            logging.error(traceback.format_exc())
            raise

        # Visualize the loss curve
        try:
            loss_curve(MLP)
            logging.info("Loss curve plotted.")
        except Exception as e:
            logging.warning("Failed to plot loss curve.")
            logging.warning(traceback.format_exc())

        # Evaluate the model
        try:
            accuracy, confusion_mat = evaluate_model(MLP, xtest_scaled, ytest)
            logging.info(f"Model evaluation complete. Accuracy: {accuracy}")
            logging.info(f"Confusion Matrix:\n{confusion_mat}")
            print(f"Accuracy: {accuracy}")
            print(f"Confusion Matrix:\n{confusion_mat}")
        except Exception as e:
            logging.error("Error evaluating the model.")
            logging.error(traceback.format_exc())
            raise

        logging.info("Pipeline completed successfully.")

    except Exception as e:
        logging.critical("Pipeline terminated due to an unrecoverable error.")
        logging.critical(traceback.format_exc())
