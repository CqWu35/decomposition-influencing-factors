import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(models, metrics, metrics_names):
    for metric_name in metrics_names:
        plt.figure(figsize=(16, 9))
        x = np.arange(len(models))
        width = 0.35

        out_of_sample_values = [metrics[m]['out_of_sample'][metrics_names.index(metric_name)] for m in models]
        val_values = [metrics[m]['validation'][metrics_names.index(metric_name)] for m in models]
        plt.bar(x - width/2, out_of_sample_values, width, label='Out of Sample')
        plt.bar(x + width/2, val_values, width, label='Validation')
        for i in range(len(models)):
            plt.text(i - width/2, out_of_sample_values[i], f'{out_of_sample_values[i]:.5f}', ha='center', va='bottom')
            plt.text(i + width/2, val_values[i], f'{val_values[i]:.5f}', ha='center', va='bottom')

        plt.xticks(x, models.keys())
        plt.xlabel('Models')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} Comparison')
        plt.legend()
        plt.show()
