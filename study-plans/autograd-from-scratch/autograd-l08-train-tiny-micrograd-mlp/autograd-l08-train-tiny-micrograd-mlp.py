import torch


def train_tiny_micrograd_mlp(
    inputs,
    targets,
    weights,
    biases,
    learning_rate,
    steps,
):
    with torch.no_grad():
        trained_weights = [w.clone() for w in weights]
        trained_biases = [b.clone() for b in biases]

        loss_history = []

        def forward():
            activations = [inputs]
            h = inputs

            for w, b in zip(trained_weights, trained_biases):
                h = torch.tanh(h @ w.T + b)
                activations.append(h)

            predictions = h[:, 0]
            loss = ((predictions - targets) ** 2).sum()

            return predictions, loss, activations

        for _ in range(steps):
            predictions, loss, activations = forward()

            loss_history.append(loss.clone())

            grad_h = (2 * (predictions - targets)).unsqueeze(1)

            weight_grads = [None] * len(trained_weights)
            bias_grads = [None] * len(trained_biases)

            for layer in range(len(trained_weights) - 1, -1, -1):
                h_prev = activations[layer]
                h = activations[layer + 1]
                w = trained_weights[layer]

                grad_z = grad_h * (1 - h * h)

                weight_grads[layer] = grad_z.T @ h_prev
                bias_grads[layer] = grad_z.sum(dim=0)

                if layer > 0:
                    grad_h = grad_z @ w

            for layer in range(len(trained_weights)):
                trained_weights[layer] = (
                    trained_weights[layer]
                    - learning_rate * weight_grads[layer]
                )
                trained_biases[layer] = (
                    trained_biases[layer]
                    - learning_rate * bias_grads[layer]
                )

        final_predictions, final_loss, _ = forward()

        return (
            final_predictions,
            final_loss,
            trained_weights,
            trained_biases,
            loss_history,
        )