"""
team_assigner.py — thin passthrough to the color-heuristic implementation.

The real implementation was renamed to team_assigner_color_bak.py as a
rollback point while a SigLIP+UMAP+KMeans clustering replacement is being
evaluated (see siglip_cluster_test.py). This shim keeps main.py /
fast_common.py importing `from team_assigner import TeamAssigner` working
unchanged until the new approach is validated and swapped in.
"""

from .team_assigner_color_bak import TeamAssigner
