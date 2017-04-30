import math


def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)

    mini_batches = []

    sample_size = len(features)
    for i in range(0, sample_size, batch_size):
        start_index = i
        mini_batches.append([features[start_index:i + batch_size], labels[start_index: i + batch_size]])

    return mini_batches
