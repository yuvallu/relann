"""
Test for view, transpose, and sqrt operations.

Based on HGT-style code patterns:
- transpose(Q_Linear2_Paper(z2))
- sqrt(d) 
- z2.view(h, d/h)
- (z2.view(h, d/h) * transpose(z1)).view(d)
"""

import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[3]))

from relann.engine import Engine
from relann.era_operations import _to_er_dict
from relann.parser import parse_and_transform_str
from relann.relnn import term_graph_to_module
from relann.torch_utils import full_seed


def main() -> None:
    full_seed(0)

    device = torch.device("cpu")
    num_terms = 5
    num_papers = 3
    num_edges = 7
    d = 64  # dimension
    h = 4   # number of heads
    head_dim = d // h  # 16

    # Create synthetic data
    terms_df = pd.DataFrame({"term_id": list(range(num_terms))})
    terms_emb = torch.randn(num_terms, d, device=device)

    papers_df = pd.DataFrame({"paper_id": list(range(num_papers))})
    papers_emb = torch.randn(num_papers, d, device=device)

    # Term-Paper edges
    edges = [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (0, 2), (1, 2)]
    term_paper_df = pd.DataFrame(
        {"term_id": [s for s, _t in edges], "paper_id": [t for _s, t in edges]}
    )
    term_paper_w = torch.ones(len(edges), 1, device=device)

    # Test program with view, transpose, and sqrt operations
    program_str = "\n".join(
        [
            # Base relations
            "terms(term_id; z) :- terms_all(term_id; z) .",
            "papers(paper_id; z) :- papers_all(paper_id; z) .",
            "term_paper(term_id, paper_id; w) :- term_paper_edges(term_id, paper_id; w) .",
            
            # Test 1: Basic transpose operation
            "Q_Terms(term_id; Linear(64, 64, False)(z)) :- terms(term_id; z) .",
            "Q_Papers(paper_id; Linear(64, 64, False)(z)) :- papers(paper_id; z) .",
            "TransposedQ(term_id; transpose(z)) :- Q_Terms(term_id; z) .",
            
            # Test 2: Basic sqrt operation
            "SqrtD(term_id; sqrt(64)) :- terms(term_id; z) .",  # sqrt(d) where d=64
            
            # Test 3: View operation with hyperparameters
            "ViewedTerms(term_id; view(4, 16)(z)) :- Q_Terms(term_id; z) .",  # view(h, d/h) where h=4, d/h=16
            
            # Test 4: Combined operations - attention-like computation
            # This mimics: K_Linear2_Terms(z1) @ W_ATT_Term_Paper @ transpose(Q_Linear2_Paper(z2)) * Mu_Term_Term_Paper_Paper/sqrt(d)
            "K_Terms(term_id; Linear(64, 64, False)(z)) :- terms(term_id; z) .",
            "Q_Papers2(paper_id; Linear(64, 64, False)(z)) :- papers(paper_id; z) .",
            # For W_ATT and Mu, we need proper tensor shapes - use identity for simplicity
            "W_ATT(term_id, paper_id; w) :- term_paper(term_id, paper_id; w) .",
            "Mu(term_id, paper_id; w) :- term_paper(term_id, paper_id; w) .",
            # Note: This creates a scalar attention score, which is fine for testing the operations
            "ATT_Score(term_id, paper_id; (k @ w @ transpose(q)) * mu / sqrt(64)) :- "
            "    K_Terms(term_id; k), W_ATT(term_id, paper_id; w), Q_Papers2(paper_id; q), Mu(term_id, paper_id; mu) .",
            
            # Test 5: Nested view operations - AGG-Msg pattern
            # This mimics: (z2.view(h, d/h) * transpose(z1)).view(d)
            # Note: In DSL, we need to use function call syntax: view(64)((view(4, 16)(z2) * transpose(z1)))
            # Create MSG and ATT_Values with proper tensor embeddings (64-dim vectors per edge)
            "MSG(term_id, paper_id; Linear(64, 64, False)(z)) :- "
            "    terms(term_id; z), term_paper(term_id, paper_id; w) .",
            "ATT_Values(term_id, paper_id; Linear(64, 64, False)(z)) :- "
            "    papers(paper_id; z), term_paper(term_id, paper_id; w) .",
            "AGG_Msg(paper_id; sum(view(64)((view(4, 16)(z2) * transpose(z1))))) :- "
            "    ATT_Values(term_id, paper_id; z1), MSG(term_id, paper_id; z2) .",
            
            # Output
            "Output(paper_id; z) :- AGG_Msg(paper_id; z) .",
        ]
    )

    print("Parsing program...")
    program = parse_and_transform_str(program_str)
    
    print("Creating engine...")
    engine = Engine(
        db={
            "terms_all": (terms_df, terms_emb),
            "papers_all": (papers_df, papers_emb),
            "term_paper_edges": (term_paper_df, term_paper_w),
        },
        debug=True
    )
    engine.add_program(program)

    print("Building term graph...")
    tg = engine.term_graphs["global"]
    tg = engine.eval_tensor_terms_on_tg(tg)

    print("Creating model...")
    model = term_graph_to_module(tg, param_loader=engine).to(device)
    rels = {
        "terms_all": _to_er_dict((terms_df, terms_emb)),
        "papers_all": _to_er_dict((papers_df, papers_emb)),
        "term_paper_edges": _to_er_dict((term_paper_df, term_paper_w)),
    }
    model.instantiate(rels)

    print("Running forward pass...")
    model.eval()
    with torch.no_grad():
        output_er = model()
    
    # Access the embeddings from the output EmbeddedRelation
    if output_er.embeddings:
        output_tensor = output_er.embeddings[0]
        print(f"Output shape: {output_tensor.shape}")
        print(f"Output sample: {output_tensor[:3]}")
    else:
        print("No embeddings in output")
    
    print("Test completed successfully!")


if __name__ == "__main__":
    main()
