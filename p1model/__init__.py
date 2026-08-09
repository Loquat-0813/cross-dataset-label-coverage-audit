"""Small, frozen-backbone modules for P1 taxonomy-aware OV segmentation."""

from .taxonomy import TaxonomyLeafIndex, build_taxonomy_leaf_index

__all__ = ["TaxonomyLeafIndex", "build_taxonomy_leaf_index"]
