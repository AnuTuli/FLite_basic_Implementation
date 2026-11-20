import numpy as np


def fedavg(weight_sets, sample_counts):

    if len(weight_sets) == 0:
        raise ValueError("No model weight sets provided.")

    if len(weight_sets) != len(sample_counts):
        raise ValueError(
            "Number of weight sets and sample counts must match."
 
        )

    total_samples = sum(sample_counts)

    if total_samples <= 0:
        raise ValueError("Total sample count must be positive.")

    reference = weight_sets[0]

    for weights in weight_sets[1:]:

        if len(weights) != len(reference):
            raise ValueError("Models have incompatible structures.")

        for a, b in zip(reference, weights):

            if a.shape != b.shape:
                raise ValueError(
                    "Model weight shapes do not match."
                )

    aggregated_weights = []

    for layer_index in range(len(reference)):

        weighted_layer = np.zeros_like(
            reference[layer_index],
            dtype=np.float64
        )

        for weights, count in zip(weight_sets, sample_counts):

            weighted_layer += (
                weights[layer_index] * count / total_samples
            )

        aggregated_weights.append(
            weighted_layer.astype(reference[layer_index].dtype)
        )

    return aggregated_weights


def aggregate_local_and_neighbors(
        local_weights,
        local_samples,
        neighbor_weights,
        neighbor_samples):

    all_weights = [local_weights] + list(neighbor_weights)

    all_samples = [local_samples] + list(neighbor_samples)

    return fedavg(all_weights, all_samples)
