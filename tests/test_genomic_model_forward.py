from __future__ import annotations

import torch

from neurosynth.genomic.model import HierarchicalVariantTransformer


def test_genomic_model_forward_shapes() -> None:
    gene_vocab = {"UNK": 0, "APOE": 1, "APP": 2, "SNCA": 3}
    model = HierarchicalVariantTransformer(gene_vocab=gene_vocab)

    bsz = 2
    v = 40
    variant_features = torch.randn(bsz, v, 16)
    sequence_context = torch.randn(bsz, v, 256)
    gene_ids = torch.randint(0, 4, (bsz, v))
    consequence = torch.randint(0, 10, (bsz, v))
    mask = torch.ones(bsz, v, dtype=torch.bool)
    apoe = torch.tensor([1, 2], dtype=torch.long)

    out = model(
        variant_features=variant_features,
        sequence_context=sequence_context,
        gene_ids=gene_ids,
        consequence_category=consequence,
        variant_mask=mask,
        apoe_e4_count=apoe,
    )

    assert out["embedding"].shape == (bsz, 512)
    assert out["prs_pred"].shape == (bsz, 3)
    assert out["apoe_logits"].shape == (bsz, 3)
    assert out["pathogenicity_logits"].shape == (bsz, 4)
