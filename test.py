import evaluation
import numpy as np

labels = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
                   [0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
                   [1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1]])
stride = 3
final_labels = evaluation.final_sequence_from_labeled_windows(labels, stride)
