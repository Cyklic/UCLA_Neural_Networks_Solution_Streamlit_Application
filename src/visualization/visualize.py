
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

def loss_curve(MLP):
    # Plotting loss curve
    loss_values = MLP.loss_curve_

    # Plotting the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label='Loss', color='blue')
    plt.title('Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.savefig("loss_curve.png") 
    plt.show()
    
