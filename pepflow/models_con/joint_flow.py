"""Joint PepFlow peptide flow and DynamicBind receptor flow."""

from contextlib import contextmanager

import torch
from torch import nn


class PeptideConditionAdapter(nn.Module):
    """Map a PepFlow peptide representation into DynamicBind scalar channels."""

    def __init__(self, peptide_dim, receptor_scalar_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(peptide_dim),
            nn.Linear(peptide_dim, receptor_scalar_dim),
            nn.SiLU(),
            nn.Linear(receptor_scalar_dim, receptor_scalar_dim, bias=False),
        )
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, peptide_context):
        return torch.sigmoid(self.gate) * self.network(peptide_context)


class JointPeptideReceptorFlow(nn.Module):
    """Retain both native flows while sharing peptide conditioning.

    PepFlow owns peptide frame, sequence, and torsion prediction. Its amino-acid
    embedding is pooled over generated peptide residues and injected into the
    DynamicBind ligand scalar embedding. DynamicBind then runs its unchanged
    ligand-receptor interaction trunk and receptor translation/rotation/chi
    heads. The ligand graph supplies the noisy peptide geometry.
    """

    def __init__(self, peptide_flow, receptor_flow, peptide_dim, receptor_scalar_dim):
        super().__init__()
        self.peptide_flow = peptide_flow
        self.receptor_flow = receptor_flow
        self.condition_adapter = PeptideConditionAdapter(
            peptide_dim, receptor_scalar_dim
        )

    def peptide_context(self, peptide_batch):
        aa = peptide_batch["aa"].clamp(min=0, max=21)
        residue_context = self.peptide_flow.node_embedder.aatype_embed(aa)
        mask = peptide_batch["generate_mask"].to(residue_context.dtype)
        return (residue_context * mask[..., None]).sum(1) / (
            mask.sum(1, keepdim=True) + 1e-8
        )

    @contextmanager
    def _inject_condition(self, receptor_graph, context, zero_condition=False):
        graph_index = receptor_graph["ligand"].batch
        projected = self.condition_adapter(context)

        def hook(_module, _inputs, output):
            if zero_condition:
                return torch.zeros_like(output)
            return output + projected[graph_index]

        handle = self.receptor_flow.lig_node_embedding.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def receptor_forward(
        self, peptide_batch, receptor_graph, condition_batch=None,
        zero_condition=False,
    ):
        source = peptide_batch if condition_batch is None else condition_batch
        context = self.peptide_context(source)
        with self._inject_condition(
            receptor_graph, context, zero_condition=zero_condition
        ):
            return self.receptor_flow(receptor_graph)

    def forward(self, peptide_batch, receptor_graph):
        peptide_losses = self.peptide_flow(peptide_batch)
        receptor_outputs = self.receptor_forward(peptide_batch, receptor_graph)
        return peptide_losses, receptor_outputs
