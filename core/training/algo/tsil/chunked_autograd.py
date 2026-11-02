"""First-order, activation-bounded replay evaluation for feedforward agents.

Forward scoring keeps no network activations. Backward recomputes only rows
with nonzero upstream derivatives, releasing each chunk before the next.
Loss reductions and weight normalization remain outside this operation, over
the original logical batch. Model parameters and normalizers must stay fixed
between forward and backward; stochastic/batch-coupled networks are unsupported.
"""

import torch
from torch.autograd.function import once_differentiable


class _ChunkedEvaluation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, evaluate, chunk_size, inputs, actions, indices, *parameters):
        ctx.evaluate = evaluate
        ctx.chunk_size = chunk_size
        ctx.save_for_backward(inputs, actions, *parameters)
        result = inputs.new_zeros(inputs.shape[0])
        for rows in indices.split(chunk_size):
            if rows.numel():
                result[rows] = evaluate(inputs[rows], actions[rows]).view(-1)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, derivative):
        inputs, actions, *parameters = ctx.saved_tensors
        # Explicit zeros preserve the dense path's optimizer semantics even
        # when the entire replay loss is zero.
        accumulated = [torch.zeros_like(p) for p in parameters]
        active = torch.nonzero(derivative != 0, as_tuple=False).flatten()
        for rows in active.split(ctx.chunk_size):
            if not rows.numel():
                continue
            with torch.enable_grad():
                values = ctx.evaluate(inputs[rows], actions[rows]).view(-1)
                gradients = torch.autograd.grad(
                    values, parameters, grad_outputs=derivative[rows],
                    allow_unused=False, create_graph=False,
                )
            for total, gradient in zip(accumulated, gradients):
                total.add_(gradient)
        return (None, None, None, None, None, *accumulated)


def chunked_value(agent, states, chunk_size):
    parameters = tuple(p for p in agent.critic.parameters() if p.requires_grad)
    return _ChunkedEvaluation.apply(
        lambda x, _: agent.get_value(x, denorm=False, update=False)[0],
        chunk_size, states, states.new_empty((len(states), 0)),
        torch.arange(len(states), device=states.device), *parameters,
    )


def chunked_logprob(agent, observations, actions, mask, chunk_size):
    parameters = tuple(p for _, p in agent.named_policy_parameters())
    return _ChunkedEvaluation.apply(
        lambda x, a: agent.get_logprob(x, a, update=False),
        chunk_size, observations, actions,
        torch.nonzero(mask, as_tuple=False).flatten(), *parameters,
    )
