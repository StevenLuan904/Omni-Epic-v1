"""Joint PepFlow peptide flow and DynamicBind receptor flow."""

from contextlib import contextmanager
import math

import torch
from torch import nn


CLOCK_SLOT_ORDER = ("peptide_seq", "peptide_struct", "pocket_struct")


class FixedClockConditioner(nn.Module):
    """Encode the three semantically fixed flow clocks without attention."""

    def __init__(
        self,
        output_dims,
        scalar_dim=256,
        clock_type_dim=32,
        fusion_dim=512,
    ):
        super().__init__()
        if scalar_dim < 2 or scalar_dim % 2:
            raise ValueError("scalar_dim must be an even integer of at least two")
        if tuple(output_dims) != CLOCK_SLOT_ORDER:
            raise ValueError(f"output_dims must follow fixed order {CLOCK_SLOT_ORDER}")
        self.scalar_dim = scalar_dim
        self.clock_type = nn.Embedding(len(CLOCK_SLOT_ORDER), clock_type_dim)
        input_dim = len(CLOCK_SLOT_ORDER) * (scalar_dim + clock_type_dim)
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(input_dim, fusion_dim),
                nn.SiLU(),
                nn.Linear(fusion_dim, output_dims[name]),
            )
            for name in CLOCK_SLOT_ORDER
        })

    def _fourier(self, time):
        time = time.reshape(-1).float()
        half = self.scalar_dim // 2
        frequencies = torch.exp(
            -math.log(2056.0)
            * torch.arange(half, device=time.device, dtype=time.dtype)
            / max(half - 1, 1)
        )
        phase = time[:, None] * 2056.0 * frequencies[None, :]
        return torch.cat((phase.sin(), phase.cos()), dim=-1)

    def forward(self, clock_times):
        if tuple(clock_times) != CLOCK_SLOT_ORDER:
            raise ValueError(
                f"clock_times must follow fixed order {CLOCK_SLOT_ORDER}; "
                f"received {tuple(clock_times)}"
            )
        encoded = []
        for index, name in enumerate(CLOCK_SLOT_ORDER):
            scalar = self._fourier(clock_times[name])
            clock_id = torch.full(
                (scalar.shape[0],), index, device=scalar.device, dtype=torch.long
            )
            encoded.append(torch.cat((scalar, self.clock_type(clock_id)), dim=-1))
        fused = torch.cat(encoded, dim=-1)
        return {name: self.projections[name](fused) for name in CLOCK_SLOT_ORDER}


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

    def __init__(
        self,
        peptide_flow,
        receptor_flow,
        peptide_dim,
        receptor_scalar_dim,
        clock_embedding=None,
    ):
        super().__init__()
        self.peptide_flow = peptide_flow
        self.receptor_flow = receptor_flow
        self.condition_adapter = PeptideConditionAdapter(
            peptide_dim, receptor_scalar_dim
        )
        self.clock_conditioner = None
        if clock_embedding is not None:
            slot_order = tuple(clock_embedding["slot_order"])
            if slot_order != CLOCK_SLOT_ORDER:
                raise ValueError(f"slot_order must be {CLOCK_SLOT_ORDER}")
            self.clock_conditioner = FixedClockConditioner(
                {
                    "peptide_seq": peptide_dim,
                    "peptide_struct": peptide_dim,
                    "pocket_struct": receptor_scalar_dim,
                },
                scalar_dim=clock_embedding["scalar_dim"],
                clock_type_dim=clock_embedding["clock_type_dim"],
                fusion_dim=clock_embedding["fusion_dim"],
            )

    def peptide_context(self, peptide_batch):
        aa = peptide_batch["aa"].clamp(min=0, max=21)
        residue_context = self.peptide_flow.node_embedder.aatype_embed(aa)
        mask = peptide_batch["generate_mask"].to(residue_context.dtype)
        return (residue_context * mask[..., None]).sum(1) / (
            mask.sum(1, keepdim=True) + 1e-8
        )

    @contextmanager
    def _inject_condition(
        self, receptor_graph, context, pocket_clock_context=None,
        zero_condition=False,
    ):
        graph_index = receptor_graph["ligand"].batch
        projected = self.condition_adapter(context)

        def hook(_module, _inputs, output):
            condition = torch.zeros_like(projected) if zero_condition else projected
            result = output + condition[graph_index]
            if pocket_clock_context is not None:
                result = result + pocket_clock_context[graph_index]
            return result

        handle = self.receptor_flow.lig_node_embedding.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def receptor_forward(
        self, peptide_batch, receptor_graph, condition_batch=None,
        zero_condition=False, clock_context=None,
    ):
        source = peptide_batch if condition_batch is None else condition_batch
        context = self.peptide_context(source)
        with self._inject_condition(
            receptor_graph,
            context,
            pocket_clock_context=(
                None if clock_context is None else clock_context["pocket_struct"]
            ),
            zero_condition=zero_condition,
        ):
            return self.receptor_flow(receptor_graph)

    def forward(self, peptide_batch, receptor_graph, clock_times=None):
        clock_context = None
        peptide_time = None
        if self.clock_conditioner is not None:
            if clock_times is None:
                batch_size = peptide_batch["aa"].shape[0]
                device = peptide_batch["aa"].device
                peptide_time = torch.rand(batch_size, device=device)
                pocket_time = torch.rand(batch_size, device=device)
                clock_times = {
                    "peptide_seq": peptide_time,
                    "peptide_struct": peptide_time,
                    "pocket_struct": pocket_time,
                }
            clock_context = self.clock_conditioner(clock_times)
            peptide_time = clock_times["peptide_struct"].reshape(-1, 1)
        peptide_losses = self.peptide_flow(
            peptide_batch, t=peptide_time, clock_context=clock_context
        )
        receptor_outputs = self.receptor_forward(
            peptide_batch, receptor_graph, clock_context=clock_context
        )
        return peptide_losses, receptor_outputs
