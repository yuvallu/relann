"""Dataset loading utilities for RelNN demos"""

import logging
import numpy as np
import pandas as pd
import torch
import io
import zipfile
import urllib.request
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)
try:
    import torch_geometric.transforms as T
    from torch_geometric.datasets import Planetoid
    from torch_geometric.utils import add_self_loops, degree
except ImportError as exc:
    raise ImportError(
        "relann.datasets loads PyTorch Geometric benchmark datasets (Cora, "
        "Planetoid, DBLP), which need torch-geometric + the PyG sparse stack. "
        "These aren't installed by `pip install relann`; add them with:\n"
        "    pip install --no-build-isolation torch-scatter torch-sparse "
        "torch-cluster torch-geometric \\\n"
        "        -f https://data.pyg.org/whl/torch-2.6.0+cu124.html\n"
        "(swap cu124 for your CUDA tag, or +cpu for a CPU-only host). "
        "See https://github.com/yuvallu/relann/blob/main/docs/install-gpu.md"
    ) from exc
from pathlib import Path
from relann.torch_utils import get_project_root
from relann.era_operations import (
    EmbeddedRelation,
    pretty_print_er,
    _format_embedding_cell,
)


# Cora class names for task description
CORA_CLASS_NAMES = (
    "Case_Based, Genetic_Algorithms, Neural_Networks, Probabilistic_Methods, "
    "Reinforcement_Learning, Rule_Learning, Theory"
)


class CoraDataset:
    """
    Result of loading the Cora citation dataset. Holds the relational db
    (Papers, Citation, Labels, TestLabels, MetaRel), task summary, and metadata.

    - Use .db or .to_dict() for Session(db=...).
    - repr() shows database structure and task summary.
    """

    def __init__(self, raw: dict):
        self._raw = raw
        self._db = {
            "Papers": raw["Papers"],
            "Citation": raw["Citation"],
            "Labels": raw["Labels"],
            "TestLabels": raw["TestLabels"],
            "MetaRel": raw["MetaRel"],
        }
        info = raw.get("dataset_info", {})
        self._task = (
            f"Node classification: predict subject category (one of {info.get('num_classes', 7)} classes:\n"
            f"{CORA_CLASS_NAMES}.\n"
            f"Semi-supervised: {info.get('num_train', 0)} train / {info.get('num_val', 0)} val {info.get('num_test', 0)} test nodes."
        )

    @property
    def db(self):
        """Database dict for Session: Papers, Citation, Labels."""
        return self._db

    @property
    def task(self):
        """Short task summary string."""
        return self._task

    @property
    def node_metadata(self):
        return self._raw.get("node_metadata")

    @property
    def dataset_info(self):
        return self._raw.get("dataset_info", {})

    def to_dict(self):
        """Return db dict suitable for Session(db=...). Same as .db."""
        return self._db

    def __getitem__(self, key):
        """Backward compat: data['node_metadata'], data['dataset_info'], data['Papers'], etc."""
        return self._raw[key]

    def __repr__(self):
        lines = []
        lines.append("Cora dataset")
        lines.append("─" * 50)
        lines.append("Tables:")
        for name, (df, tensor) in self._db.items():
            lines.append(f"  • {name:<9} | Rows: {df.shape[0]:,} | Columns: {list(df.columns)} → {tensor.shape}")
        lines.append("")
        lines.append("Task:")
        lines.append(f"  {self._task}")
        return "\n".join(lines)


def load_cora_dataset():
    """
    Load the Cora citation network dataset and convert it to RelNN format.

    Returns:
        CoraDataset: Object with .db (Papers, Citation, Labels, TestLabels),
            .task (summary string), .to_dict() for Session(db=...), and repr
            showing tables + task.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Cora dataset
    project_root = get_project_root()
    path = project_root / 'data' / 'Planetoid'
    dataset = Planetoid(str(path), 'Cora', transform=T.NormalizeFeatures())
    data = dataset[0].to(device)
    
    num_nodes = data.x.size(0)
    num_features = data.x.size(1)
    num_classes = dataset.num_classes
    
    # Create Papers relation: DataFrame with paper_id + paper features tensor
    papers_df = pd.DataFrame({
        'pid': range(num_nodes),
    })
    papers_tensor = data.x  # Shape: [num_nodes, num_features]
    
    # Create Citation relation: DataFrame with citing, cited + edge normalization weights tensor
    # Add self-loops (required for GCN)
    edge_index, _ = add_self_loops(data.edge_index)
    
    citation_df = pd.DataFrame({
        'citing': edge_index[0].cpu().numpy(),
        'target_id': edge_index[1].cpu().numpy()  # Use target_id to match Labels schema
    })
    
    # Compute normalization factors exactly like PyG's gcn_norm
    source, target = edge_index
    deg = degree(source, data.x.size(0), dtype=data.x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    norm_factor = deg_inv_sqrt[source] * deg_inv_sqrt[target]
    
    # Create edge features tensor with the normalized factors
    citation_tensor = norm_factor.unsqueeze(-1)  # Shape: [num_edges, 1]
    
    # Create Labels relation: extract labels from training nodes (class indices for CrossEntropyLoss)
    train_mask = data.train_mask.cpu().numpy()
    train_indices = torch.where(data.train_mask)[0].cpu().numpy()
    
    labels_df = pd.DataFrame({
        'target_id': train_indices
    })
    
    # Class indices (N, 1) long for nn.CrossEntropyLoss; one scalar per row.
    label_indices = torch.as_tensor(data.y[train_indices].cpu().numpy(), dtype=torch.long, device=device)
    labels_tensor = label_indices.unsqueeze(1)  # Shape: [N, 1] - class indices
    
    # Create TestLabels relation: ground-truth class for each test node (for evaluation).
    # Column name 'cited' matches the key in Output so RelNN can JOIN them directly.
    test_indices = torch.where(data.test_mask)[0].cpu().numpy()
    test_labels_df = pd.DataFrame({'cited': test_indices})
    test_labels_tensor = torch.as_tensor(
        data.y[test_indices].cpu().numpy(), dtype=torch.float32, device=device
    ).unsqueeze(1)  # Shape: [N_test, 1]

    # Store node metadata for accuracy calculation
    node_metadata = pd.DataFrame({
        'node_id': range(num_nodes),
        'label': data.y.cpu().numpy(),
        'is_train': data.train_mask.cpu().numpy(),
        'is_val': data.val_mask.cpu().numpy(),
        'is_test': data.test_mask.cpu().numpy()
    })
    
    dataset_info = {
        'num_nodes': num_nodes,
        'num_features': num_features,
        'num_classes': num_classes,
        'num_edges': edge_index.size(1),
        'num_train': int(train_mask.sum()),
        'num_val': int(data.val_mask.sum().item()),
        'num_test': int(data.test_mask.sum().item()),
        'device': device
    }

    # MetaRel: (source_type, edge_type, target_type) for generic HGT bounded sets.
    meta_df = pd.DataFrame([{"source_type": "Papers", "edge_type": "Citation", "target_type": "Papers"}])
    meta_tensor = torch.empty(len(meta_df), 0)

    raw = {
        'Papers': (papers_df, papers_tensor),
        'Citation': (citation_df, citation_tensor),
        'Labels': (labels_df, labels_tensor),
        'TestLabels': (test_labels_df, test_labels_tensor),
        'MetaRel': (meta_df, meta_tensor),
        'node_metadata': node_metadata,
        'dataset_info': dataset_info
    }
    return CoraDataset(raw)


def evaluate_node_classification(data, pred_result, return_value=False):
    """
    Compare predicted classes with ground-truth labels on the test set and print accuracy.

    Takes two inputs: (1) the dataset dict from load_cora_dataset() which contains the
    real labels and train/val/test split; (2) the prediction result from session.run(pred_program)
    which contains one predicted class per node. Merges them on node id, restricts to test
    nodes, and computes accuracy (fraction of test nodes where pred == label).

    Parameters
    ----------
    data : dict
        Output of load_cora_dataset(); must contain 'node_metadata' with columns
        node_id, label, is_test.
    pred_result : EmbeddedRelation
        Result of session.run(pred_program): .content has node identifiers (e.g. target_id),
        .embeddings[0] is a 1-D tensor of predicted class indices (same row order as .content).

    Returns
    -------
    float
        Test accuracy in [0, 1].
    """
    node_metadata = data['node_metadata']
    pred_df = pred_result.content.copy() if pred_result.content is not None else pd.DataFrame()
    # Use actual node-id column for merge (GCN Output has 'cited'; don't assume row index = node id)
    if 'target_id' not in pred_df.columns and len(pred_df) > 0:
        pred_df = pred_df.copy()
        if 'cited' in pred_df.columns:
            pred_df['target_id'] = pred_df['cited'].values
        elif 'pid' in pred_df.columns:
            pred_df['target_id'] = pred_df['pid'].values
        elif len(pred_df.columns) >= 1:
            pred_df['target_id'] = pred_df.iloc[:, 0].values
        else:
            pred_df['target_id'] = list(pred_df.index) if hasattr(pred_df, 'index') else list(range(len(pred_df)))
    pred_class = pred_result.embeddings[0].cpu().numpy().flatten().astype(int)
    pred_df = pred_df.copy()
    pred_df['_pred_class'] = pred_class

    merged = pred_df.merge(node_metadata, left_on='target_id', right_on='node_id', how='left')
    test_mask = merged['is_test'].fillna(False).astype(bool)
    test_labels = merged.loc[test_mask, 'label'].values
    test_preds = merged.loc[test_mask, '_pred_class'].values

    test_acc = float((test_preds == test_labels).mean())
    n_correct = int(np.sum(test_preds == test_labels))
    n_test = len(test_labels)
    print(f"Test Accuracy: {test_acc:.4f} ({n_correct}/{n_test} correct)")
    if return_value:
        return test_acc


class DBLPDataset:
    """Result of loading the DBLP heterogeneous graph dataset.

    Node types: author, paper, term, conference.
    Edge types: author-paper, paper-author, paper-term, paper-conference, term-paper, conference-paper.
    Task: author node classification (4 classes).
    """

    def __init__(self, raw: dict):
        self._raw = raw
        self._db = raw["db"]

    @property
    def db(self):
        """Database dict for Session: node tables + edge tables."""
        return self._db

    @property
    def node_metadata(self):
        return self._raw.get("node_metadata")

    @property
    def dataset_info(self):
        return self._raw.get("dataset_info", {})

    @property
    def pyg_data(self):
        """Original PyG HeteroData object (for PyG model comparisons)."""
        return self._raw.get("pyg_data")

    def to_dict(self):
        return self._db

    def __getitem__(self, key):
        return self._raw[key]

    def __repr__(self):
        lines = ["DBLP dataset"]
        lines.append("-" * 50)
        lines.append("Tables:")
        for name, (df, tensor) in self._db.items():
            lines.append(f"  {name:<25} | Rows: {df.shape[0]:>6,} | Tensor: {tuple(tensor.shape)}")
        info = self.dataset_info
        if info:
            lines.append(f"\nTask: Author classification ({info.get('num_classes', 4)} classes)")
            lines.append(f"  Train: {info.get('num_train', 0)}, Val: {info.get('num_val', 0)}, Test: {info.get('num_test', 0)}")
        return "\n".join(lines)


def load_dblp_dataset():
    """Load the DBLP heterogeneous graph dataset and convert to RelNN format.

    Returns a DBLPDataset with .db containing:
      - Node tables: Author, Paper, Term, Conference (df + feature tensor)
      - Edge tables: AuthorPaper, PaperAuthor, PaperTerm, PaperConference,
                     TermPaper, ConferencePaper (df + ones tensor)
      - Labels: AuthorLabels (train author ids + class labels)
    """
    from torch_geometric.datasets import DBLP

    project_root = get_project_root()
    dblp_path = project_root / "data" / "DBLP"
    dataset = DBLP(str(dblp_path), transform=T.Constant(node_types="conference"))
    data = dataset[0]

    db = {}

    node_type_map = {
        "author": "Author",
        "paper": "Paper",
        "term": "Term",
        "conference": "Conference",
    }

    for pyg_type, relnn_name in node_type_map.items():
        store = data[pyg_type]
        n = store.x.size(0)
        df = pd.DataFrame({f"{relnn_name.lower()}_id": list(range(n))})
        db[relnn_name] = (df, store.x.float())

    edge_type_map = {
        ("author", "to", "paper"): ("AuthorPaper", "author_id", "paper_id"),
        ("paper", "to", "author"): ("PaperAuthor", "paper_id", "author_id"),
        ("paper", "to", "term"): ("PaperTerm", "paper_id", "term_id"),
        ("paper", "to", "conference"): ("PaperConference", "paper_id", "conference_id"),
        ("term", "to", "paper"): ("TermPaper", "term_id", "paper_id"),
        ("conference", "to", "paper"): ("ConferencePaper", "conference_id", "paper_id"),
    }

    for pyg_et, (relnn_name, src_col, dst_col) in edge_type_map.items():
        ei = data[pyg_et].edge_index
        src, dst = ei[0].cpu().numpy(), ei[1].cpu().numpy()
        df = pd.DataFrame({src_col: src, dst_col: dst})
        w = torch.ones(len(df), 1, dtype=torch.float32)
        db[relnn_name] = (df, w)

    # MetaRel: compile-time schema metadata (source_type, edge_type, target_type).
    # Used by bounded sets to iterate over edge types at compile time.
    meta_rows = []
    for pyg_et, (relnn_name, src_col, dst_col) in edge_type_map.items():
        src_type = node_type_map[pyg_et[0]]
        dst_type = node_type_map[pyg_et[2]]
        meta_rows.append({"source_type": src_type, "edge_type": relnn_name, "target_type": dst_type})
    meta_df = pd.DataFrame(meta_rows)
    db["MetaRel"] = (meta_df, torch.empty(len(meta_df), 0))

    author_store = data["author"]
    train_mask = author_store.train_mask
    train_indices = torch.where(train_mask)[0].cpu().numpy()
    labels_df = pd.DataFrame({"author_id": train_indices})
    labels_tensor = author_store.y[train_indices].long().unsqueeze(1)
    db["AuthorLabels"] = (labels_df, labels_tensor)

    node_metadata = pd.DataFrame({
        "node_id": range(author_store.x.size(0)),
        "label": author_store.y.cpu().numpy(),
        "is_train": author_store.train_mask.cpu().numpy(),
        "is_val": author_store.val_mask.cpu().numpy(),
        "is_test": author_store.test_mask.cpu().numpy(),
    })

    dataset_info = {
        "num_authors": int(author_store.x.size(0)),
        "num_papers": int(data["paper"].x.size(0)),
        "num_terms": int(data["term"].x.size(0)),
        "num_conferences": int(data["conference"].x.size(0)),
        "num_classes": int(author_store.y.max().item() + 1),
        "num_train": int(train_mask.sum().item()),
        "num_val": int(author_store.val_mask.sum().item()),
        "num_test": int(author_store.test_mask.sum().item()),
        "author_features": int(author_store.x.size(1)),
        "paper_features": int(data["paper"].x.size(1)),
        "term_features": int(data["term"].x.size(1)),
        "conference_features": int(data["conference"].x.size(1)),
    }

    raw = {
        "db": db,
        "node_metadata": node_metadata,
        "dataset_info": dataset_info,
        "pyg_data": data,
    }
    return DBLPDataset(raw)


def show_db_structure(db):
    """
    Display a visually enhanced and concise overview of the database structure.
    
    Args:
        db: Dictionary of database tables (name -> (DataFrame, Tensor))
        info: Optional dictionary with dataset info (num_nodes, num_edges, num_classes)
    """
    import textwrap

    # Pretty header for Cora dataset info
    print("╭" + "─" * 54 + "╮")
    print("│  📊 Cora dataset summary unavailable                 │")
    print("╰" + "─" * 54 + "╯")

    print("\n📚 \033[1mDatabase Tables:\033[0m\n")
    for table_name, (df, tensor) in db.items():
        row_str = f"{df.shape[0]:,}"
        col_str = str(list(df.columns))
        tensor_str = str(tensor.shape)
        print(f"  • \033[1m{table_name:<9}\033[0m | Rows: \033[96m{row_str:<6}\033[0m | Columns: {col_str:<38}→  \033[93m{tensor_str}\033[0m")

    # Fancy section explaining Labels
    if 'Labels' in db:
        print("\n" + "💡 " + "\033[1mLabels Table Explained\033[0m")
        description = (
            "The \033[1mLabels\033[0m table contains the node classification task for Cora:\n"
            "    Predicting the \033[95msubject category\033[0m of each paper (one of \033[1m7 classes\033[0m)\n"
            "    based on the paper's content features and citation network structure.\n"
            "    This is a \033[1msemi-supervised learning\033[0m task where only a subset of\n"
            "    nodes have labels."
        )
        print(textwrap.indent(description, "  "))

    print()  # Final newline for spacing


def load_era_join_demo_dataset():
    """
    Load the toy Users / Comments dataset for the ERA Join and Projection demo.

    Two tables: **Users** (user_id) and **Comments** (comment_id, user_id).
    Each row has a 2D embedding. Used to demonstrate Join (index_select) and
    Projection (scatter_add) without distracting setup in the notebook.

    Returns
    -------
    dict
        - 'users_er': EmbeddedRelation for Users
        - 'comments_er': EmbeddedRelation for Comments
        - 'users_schema': list of column names for Users
        - 'comments_schema': list of column names for Comments
    """
    users_df = pd.DataFrame({"user_id": ["u0", "u1", "u2"]})
    comments_df = pd.DataFrame({
        "comment_id": ["c0", "c1", "c2", "c3", "c4"],
        "user_id": ["u0", "u0", "u1", "u2", "u0"],
    })
    user_emb = torch.tensor(
        [[0.1, 0.2], [0.4, -0.5], [0.3, 0.8]], dtype=torch.float32
    )
    comment_emb = torch.tensor(
        [[1.0, 1.0], [0.5, -1.5], [-0.3, 0.7], [0.8, -0.2], [1.5, 1.0]],
        dtype=torch.float32,
    )
    users_schema = ["user_id"]
    comments_schema = ["comment_id", "user_id"]
    users_er = EmbeddedRelation(
        content_schema=users_schema,
        embedding_shapes=[(3, 2)],
        content=users_df,
        embeddings=[user_emb],
    )
    comments_er = EmbeddedRelation(
        content_schema=comments_schema,
        embedding_shapes=[(5, 2)],
        content=comments_df,
        embeddings=[comment_emb],
    )
    return {
        "users_er": users_er,
        "comments_er": comments_er,
        "users_schema": users_schema,
        "comments_schema": comments_schema,
    }


def get_era_join_demo_db(data):
    """
    Build a db dict for Session from load_era_join_demo_dataset().

    Use with: Session(db=get_era_join_demo_db(data)) so that rules
    can reference table names "Users" and "Comments".

    Args
    ----
    data : dict
        Output of load_era_join_demo_dataset().

    Returns
    -------
    dict
        {"Users": users_er, "Comments": comments_er}
    """
    return {
        "Users": data["users_er"],
        "Comments": data["comments_er"],
    }


def show_era_join_demo_tables(data, max_rows=6):
    """
    Display the Users and Comments EmbeddedRelations for the ERA Join demo,
    side by side (no header).

    Args
    ----
    data : dict
        Output of load_era_join_demo_dataset().
    max_rows : int
        Maximum rows to show per table.
    """
    _display_ers_side_by_side(
        data["users_er"],
        data["comments_er"],
        "Users",
        "Comments",
        max_rows=max_rows,
    )


def build_and_run_era_demo_module(session, rule_name):
    """
    For the ERA join/project demo: build the RelNN module for a rule (already in the
    term graph), run instantiate + forward with the session's db, and return
    (module, cache, nodes) so the notebook can inspect ERs at each step.

    Returns
    -------
    module : RelNN
    cache : dict
        module._cache_forward (node_id -> EmbeddedRelation)
    nodes : dict
        {"join_node": str or None, "transform_node": str or None, "agg_node": str or None}
    """
    from relann.relnn import term_graph_to_module

    engine = session.engine
    tg = engine.term_graphs.get("global")
    if tg is None:
        raise ValueError("No global term graph. Define the rule first (e.g. session.run(define_program)).")
    sub_tg = tg.induced_subgraph(node_name=rule_name, direction="ancestors", include_root=True)
    ground_sub_tg = engine.replace_all_vars_in_tg_using_symbol_table(sub_tg, in_place=False)
    data_sources = engine._collect_data_sources(ground_sub_tg)
    if not data_sources:
        raise ValueError(
            f"No data loaders found for rule '{rule_name}'. Ensure the rule references tables in the db."
        )
    engine.eval_tensor_terms_on_tg(ground_sub_tg)
    module = term_graph_to_module(ground_sub_tg, param_loader=engine, engine=engine)
    module.eval()
    with torch.no_grad():
        module.instantiate(data_sources)
        module.forward()
    cache = module._cache_forward
    G = module.graph
    join_node = next((n for n in module._eval_order if G.nodes[n].get("type") == "join"), None)
    transform_node = next((n for n in module._eval_order if G.nodes[n].get("type") == "transformation"), None)
    agg_node = next((n for n in module._eval_order if G.nodes[n].get("type") in ("agg", "aggregation")), None)
    nodes = {"join_node": join_node, "transform_node": transform_node, "agg_node": agg_node}
    return module, cache, nodes


def _display_ers_side_by_side(er_left, er_right, label_left, label_right, max_rows=6):
    """Display two EmbeddedRelations side by side (no header)."""
    try:
        from IPython.display import display, HTML
    except ImportError:
        pretty_print_er(er_left, max_rows=max_rows)
        pretty_print_er(er_right, max_rows=max_rows)
        return
    s_left = pretty_print_er(er_left, max_rows=max_rows, display=False)
    s_right = pretty_print_er(er_right, max_rows=max_rows, display=False)
    if s_left is None or s_right is None:
        pretty_print_er(er_left, max_rows=max_rows)
        pretty_print_er(er_right, max_rows=max_rows)
        return
    html = (
        f'<div style="display:flex; gap:2em; align-items:flex-start; flex-wrap:wrap;">'
        f'<div><p style="margin:0 0 0.5em 0;"><b>{label_left}</b></p>{s_left._repr_html_()}</div>'
        f'<div><p style="margin:0 0 0.5em 0;"><b>{label_right}</b></p>{s_right._repr_html_()}</div>'
        f"</div>"
    )
    display(HTML(html))


def show_era_join_step(module, cache, join_node, max_rows=6, max_embedding_display=6):
    """Display input ERs side by side, join indices (with keys and embeddings), then join result."""
    if join_node is None:
        return
    users_er = cache.get("Users")
    comments_er = cache.get("Comments")
    if users_er is None or comments_er is None:
        return
    print("Input: Users | Comments")
    _display_ers_side_by_side(users_er, comments_er, "Users", "Comments", max_rows=max_rows)
    join_op = getattr(module._operators, join_node, None)
    if join_op is not None and getattr(join_op, "_cached_join", None) is not None:
        _, idx = join_op._cached_join
        idx_left = idx[0].tolist()
        idx_right = idx[1].tolist()
        n = len(idx_left)
        # Build meaningful table: result_row, Users row, Comments row, user_id (Users), user_id (Comments), embedding (Users), embedding (Comments)
        u_content = users_er.content
        c_content = comments_er.content
        u_emb = users_er.embeddings[0].cpu() if users_er.embeddings else None
        c_emb = comments_er.embeddings[0].cpu() if comments_er.embeddings else None
        user_id_col = "user_id" if "user_id" in u_content.columns else u_content.columns[0]
        user_id_right_col = "user_id" if "user_id" in c_content.columns else c_content.columns[1]
        rows = []
        for i in range(n):
            r = {"result row": i, "Users (row)": idx_left[i], "Comments (row)": idx_right[i]}
            r["user_id (Users)"] = u_content.iloc[idx_left[i]][user_id_col]
            r["user_id (Comments)"] = c_content.iloc[idx_right[i]][user_id_right_col]
            if u_emb is not None:
                r["embedding (Users)"] = _format_embedding_cell(u_emb, idx_left[i], max_embedding_display)
            if c_emb is not None:
                r["embedding (Comments)"] = _format_embedding_cell(c_emb, idx_right[i], max_embedding_display)
            rows.append(r)
        join_idx_df = pd.DataFrame(rows)
        print("\nWe compute join indices (which input row → each result row):")
        try:
            from IPython.display import display
            display(join_idx_df.style.hide(axis="index"))
        except ImportError:
            print(join_idx_df.to_string(index=False))
    print("\n  ── join on user_id ──►\n")
    print("Join result:")
    pretty_print_er(cache.get(join_node), max_rows=max_rows)


def show_era_transform_step(cache, join_node, transform_node, max_rows=6):
    """Display join output and transform output side by side."""
    if join_node is None or transform_node is None:
        return
    print("Input (joined ER) | Output (transformed ER)")
    _display_ers_side_by_side(
        cache.get(join_node),
        cache.get(transform_node),
        "Joined ER",
        "Transform output",
        max_rows=max_rows,
    )


def show_era_project_step(module, cache, transform_node, agg_node, max_rows=6):
    """Display transform output and project output side by side, then group index table."""
    if transform_node is None or agg_node is None:
        return
    print("Input (transformed ER) | Output (aggregated by user_id)")
    _display_ers_side_by_side(
        cache.get(transform_node),
        cache.get(agg_node),
        "Transformed ER",
        "Project output",
        max_rows=max_rows,
    )
    agg_op = getattr(module._operators, agg_node, None)
    if agg_op is not None and getattr(agg_op, "_cached_group_idx", None) is not None:
        group_idx = agg_op._cached_group_idx.tolist()
        print("\nGroup index (each input row maps to an output row):")
        try:
            from IPython.display import display
            df = pd.DataFrame({"input row": range(len(group_idx)), "output row": group_idx})
            display(df.style.hide(axis="index"))
        except ImportError:
            print(pd.DataFrame({"input row": range(len(group_idx)), "output row": group_idx}).to_string(index=False))


# ── MovieLens 100K ───────────────────────────────────────────────────────────

_ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

# u.user occupation names (21 categories; order matches the dataset README)
_ML100K_OCCUPATIONS = [
    "administrator", "artist", "doctor", "educator", "engineer", "entertainment",
    "executive", "healthcare", "homemaker", "lawyer", "librarian", "marketing",
    "none", "other", "programmer", "retired", "salesman", "scientist", "student",
    "technician", "writer",
]


class MovieLensDataset:
    """Result of loading the MovieLens 100K dataset.

    Contains Users, Movies, TrainRatings, TestRatings tables suitable for
    Session(db=...).  Also exposes train/test rating DataFrames for evaluation,
    and an ``inductive`` flag indicating whether the split holds out entire users.
    """

    def __init__(self, raw: dict):
        self._raw = raw
        self._db = raw["db"]

    @property
    def db(self):
        return self._db

    @property
    def task(self):
        return self._raw.get("task", "")

    @property
    def dataset_info(self):
        return self._raw.get("dataset_info", {})

    @property
    def train_ratings_df(self):
        return self._raw.get("train_ratings_df")

    @property
    def test_ratings_df(self):
        return self._raw.get("test_ratings_df")

    def to_dict(self):
        return self._db

    def __getitem__(self, key):
        return self._raw[key]

    def __repr__(self):
        info = self.dataset_info
        lines = [f"MovieLens 100K ({'inductive' if info.get('inductive') else 'transductive'})"]
        lines.append("─" * 55)
        lines.append("Tables:")
        for name, (df, tensor) in self._db.items():
            lines.append(f"  • {name:<14} | Rows: {df.shape[0]:>6,} | Tensor: {tuple(tensor.shape)}")
        lines.append("")
        lines.append(f"Task: {self.task}")
        lines.append(f"  Train ratings: {info.get('n_train', 0):,}")
        lines.append(f"  Test ratings:  {info.get('n_test', 0):,}")
        if info.get("inductive"):
            lines.append(f"  Train users:   {info.get('n_train_users', 0):,}")
            lines.append(f"  Test (new) users: {info.get('n_test_users', 0):,}")
        return "\n".join(lines)


def _download_ml100k(dest_dir: Path) -> Path:
    """Download and extract ml-100k.zip to *dest_dir*; returns path to ml-100k/ folder."""
    ml_dir = dest_dir / "ml-100k"
    if (ml_dir / "u.data").exists():
        return ml_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading MovieLens 100K from %s ...", _ML100K_URL)
    resp = urllib.request.urlopen(_ML100K_URL, timeout=30)
    buf = io.BytesIO(resp.read())
    with zipfile.ZipFile(buf) as zf:
        zf.extractall(dest_dir)
    assert (ml_dir / "u.data").exists(), f"Expected {ml_dir / 'u.data'} after extraction"
    return ml_dir


def _load_ml100k_raw(ml_dir: Path):
    """Parse raw MovieLens 100K files into DataFrames + feature tensors."""

    # Users: user_id | age | gender | occupation | zip_code
    users = pd.read_csv(
        ml_dir / "u.user", sep="|", header=None,
        names=["user_id", "age", "gender", "occupation", "zip_code"],
        encoding="latin-1",
    )

    # User features: age (normalised), gender (0/1), occupation one-hot  → 23d
    age_norm = (users["age"].values - users["age"].mean()) / (users["age"].std() + 1e-8)
    gender_bin = (users["gender"] == "M").astype(float).values
    occ_idx = users["occupation"].map({o: i for i, o in enumerate(_ML100K_OCCUPATIONS)}).fillna(0).astype(int).values
    occ_onehot = np.eye(len(_ML100K_OCCUPATIONS))[occ_idx]  # (n_users, 21)
    user_features = np.column_stack([age_norm[:, None], gender_bin[:, None], occ_onehot]).astype(np.float32)
    user_tensor = torch.from_numpy(user_features)
    users_df = pd.DataFrame({"user_id": users["user_id"].values})

    # Movies: movie_id | title | release_date | ... | 18 genre columns
    genre_cols = [
        "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
    ]
    item_cols = ["movie_id", "title", "release_date", "video_release_date", "imdb_url"] + genre_cols
    items = pd.read_csv(
        ml_dir / "u.item", sep="|", header=None, names=item_cols, encoding="latin-1",
    )
    movie_features = items[genre_cols].values.astype(np.float32)  # (n_movies, 19)
    movie_tensor = torch.from_numpy(movie_features)
    movies_df = pd.DataFrame({"movie_id": items["movie_id"].values})

    # Ratings: user_id | item_id | rating | timestamp
    ratings = pd.read_csv(
        ml_dir / "u.data", sep="\t", header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
    )

    return users_df, user_tensor, movies_df, movie_tensor, ratings


def _split_transductive(ml_dir: Path):
    """Standard 80/20 split: u1.base / u1.test (same users in both)."""
    train = pd.read_csv(
        ml_dir / "u1.base", sep="\t", header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    test = pd.read_csv(
        ml_dir / "u1.test", sep="\t", header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    return train, test


def _split_inductive(ratings: pd.DataFrame, test_user_frac: float = 0.2, seed: int = 42):
    """Hold out *test_user_frac* of users entirely for testing (fully inductive)."""
    rng = np.random.default_rng(seed)
    all_users = ratings["user_id"].unique()
    n_test = max(1, int(len(all_users) * test_user_frac))
    test_users = set(rng.choice(all_users, size=n_test, replace=False))
    train_mask = ~ratings["user_id"].isin(test_users)
    return ratings[train_mask].copy(), ratings[~train_mask].copy()


def _ratings_to_relation(ratings_df: pd.DataFrame):
    """Convert a ratings DataFrame to (content_df, embeddings_tensor)."""
    content = ratings_df[["user_id", "movie_id"]].copy().reset_index(drop=True)
    emb = torch.tensor(ratings_df["rating"].values, dtype=torch.float32).unsqueeze(1)
    return content, emb


def load_movielens100k_dataset(inductive: bool = True, test_user_frac: float = 0.2, seed: int = 42):
    """Load MovieLens 100K and return a ``MovieLensDataset``.

    Parameters
    ----------
    inductive : bool
        If True, hold out *test_user_frac* of users entirely (fully inductive).
        If False, use the standard u1.base / u1.test transductive split.
    test_user_frac : float
        Fraction of users to hold out (only used when *inductive=True*).
    seed : int
        Random seed for the inductive split.

    Returns
    -------
    MovieLensDataset
        .db has keys Users, Movies, TrainRatings, TestRatings.
    """
    project_root = get_project_root()
    dest_dir = project_root / "data" / "MovieLens100K"
    ml_dir = _download_ml100k(dest_dir)

    users_df, user_tensor, movies_df, movie_tensor, ratings = _load_ml100k_raw(ml_dir)

    if inductive:
        train_df, test_df = _split_inductive(ratings, test_user_frac=test_user_frac, seed=seed)
    else:
        train_df, test_df = _split_transductive(ml_dir)

    train_content, train_emb = _ratings_to_relation(train_df)
    test_content, test_emb = _ratings_to_relation(test_df)

    db = {
        "Users": (users_df, user_tensor),
        "Movies": (movies_df, movie_tensor),
        "TrainRatings": (train_content, train_emb),
        "TestRatings": (test_content, test_emb),
    }

    n_train_users = train_df["user_id"].nunique()
    n_test_users = test_df["user_id"].nunique()
    info = {
        "n_users": len(users_df),
        "n_movies": len(movies_df),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_train_users": n_train_users,
        "n_test_users": n_test_users,
        "user_feature_dim": user_tensor.shape[1],
        "movie_feature_dim": movie_tensor.shape[1],
        "inductive": inductive,
    }

    task = (
        "Rating prediction (regression): predict the rating a user gives to a movie.\n"
        + (f"Inductive: {n_test_users} users held out entirely for test."
           if inductive else "Transductive: same users in train and test.")
    )

    raw = {
        "db": db,
        "task": task,
        "dataset_info": info,
        "train_ratings_df": train_df,
        "test_ratings_df": test_df,
    }
    return MovieLensDataset(raw)


# ---------------------------------------------------------------------------
# RelBench F1 dataset
# ---------------------------------------------------------------------------

class F1Dataset:
    """Result of loading the RelBench rel-f1 dataset for RelNN.

    Holds the relational db (Drivers, Constructors, Results, TrainLabels),
    the RelBench ``task`` object for evaluation, and metadata.
    """

    def __init__(self, raw: dict):
        self._raw = raw
        self._db = raw["db"]

    @property
    def db(self):
        return self._db

    @property
    def task(self):
        return self._raw["task"]

    @property
    def dataset(self):
        return self._raw["dataset"]

    @property
    def dataset_info(self):
        return self._raw["dataset_info"]

    @property
    def train_table(self):
        return self._raw["train_table"]

    @property
    def val_table(self):
        return self._raw["val_table"]

    @property
    def test_table(self):
        return self._raw["test_table"]

    def to_dict(self):
        return self._db

    def __repr__(self):
        info = self.dataset_info
        lines = [f"RelBench rel-f1 ({info['task_name']})"]
        lines.append("─" * 55)
        lines.append("Tables:")
        for name, (df, tensor) in self._db.items():
            lines.append(f"  * {name:<16} | Rows: {df.shape[0]:>6,} | Tensor: {tuple(tensor.shape)}")
        lines.append("")
        lines.append(f"Task: {info['task_name']} ({info['task_description']})")
        lines.append(f"  Train samples: {info['n_train']:,}")
        lines.append(f"  Val samples:   {info['n_val']:,}")
        lines.append(f"  Test samples:  {info['n_test']:,}")
        lines.append(f"  New test drivers (inductive): {info['n_new_test_drivers']}/{info['n_test_drivers']}")
        return "\n".join(lines)


def _encode_f1_drivers(drivers_df):
    """Encode driver features: nationality one-hot + normalised birth year.

    Returns (content_df, feature_tensor).
    """
    df = drivers_df.copy()
    nationalities = sorted(df["nationality"].dropna().unique())
    nat_map = {n: i for i, n in enumerate(nationalities)}

    nat_idx = df["nationality"].map(nat_map).fillna(0).astype(int).values
    nat_onehot = np.eye(len(nationalities), dtype=np.float32)[nat_idx]

    birth_year = pd.to_datetime(df["dob"], errors="coerce").dt.year.fillna(1970).values.astype(np.float32)
    birth_year_norm = (birth_year - birth_year.mean()) / (birth_year.std() + 1e-8)

    features = np.column_stack([birth_year_norm[:, None], nat_onehot])
    content = pd.DataFrame({"driverId": df["driverId"].values})
    return content, torch.from_numpy(features)


def _encode_f1_constructors(constructors_df):
    """Encode constructor features: nationality one-hot.

    Returns (content_df, feature_tensor).
    """
    df = constructors_df.copy()
    nationalities = sorted(df["nationality"].dropna().unique())
    nat_map = {n: i for i, n in enumerate(nationalities)}

    nat_idx = df["nationality"].map(nat_map).fillna(0).astype(int).values
    nat_onehot = np.eye(len(nationalities), dtype=np.float32)[nat_idx]

    content = pd.DataFrame({"constructorId": df["constructorId"].values})
    return content, torch.from_numpy(nat_onehot)


def _encode_f1_results(results_df):
    """Encode result features: grid, positionOrder, points, laps (normalised).

    Returns (content_df, feature_tensor).
    """
    df = results_df.copy()
    numeric_cols = ["grid", "positionOrder", "points", "laps"]
    features = []
    for col in numeric_cols:
        vals = df[col].fillna(0).values.astype(np.float32)
        std = vals.std() + 1e-8
        features.append(((vals - vals.mean()) / std)[:, None])

    finished = (df["statusId"] == 1).astype(np.float32).values[:, None]
    features.append(finished)

    feature_mat = np.concatenate(features, axis=1).astype(np.float32)

    content = pd.DataFrame({
        "resultId": df["resultId"].values,
        "driverId": df["driverId"].values,
        "raceId": df["raceId"].values,
        "constructorId": df["constructorId"].values,
    })
    return content, torch.from_numpy(feature_mat)


def _encode_f1_races(races_df):
    """Encode race features from calendar/context columns.

    Features:
    - year (normalized)
    - round (normalized)
    - circuit one-hot

    Returns (content_df, feature_tensor).
    """
    df = races_df.copy()
    years = df["year"].fillna(df["year"].median()).values.astype(np.float32)
    rounds = df["round"].fillna(df["round"].median()).values.astype(np.float32)
    year_norm = ((years - years.mean()) / (years.std() + 1e-8))[:, None]
    round_norm = ((rounds - rounds.mean()) / (rounds.std() + 1e-8))[:, None]

    circuits = sorted(df["circuitId"].dropna().unique())
    circuit_map = {c: i for i, c in enumerate(circuits)}
    circuit_idx = df["circuitId"].map(circuit_map).fillna(0).astype(int).values
    circuit_onehot = np.eye(len(circuits), dtype=np.float32)[circuit_idx]

    features = np.concatenate([year_norm, round_norm, circuit_onehot], axis=1).astype(np.float32)
    content = pd.DataFrame({"raceId": df["raceId"].values})
    return content, torch.from_numpy(features)


def _make_f1_labels(task_table_df):
    """Average the target position per driver from a RelBench task table.

    Returns (content_df, label_tensor).
    """
    avg = task_table_df.groupby("driverId")["position"].mean().reset_index()
    content = pd.DataFrame({"driverId": avg["driverId"].values})
    labels = torch.tensor(avg["position"].values, dtype=torch.float32).unsqueeze(1)
    return content, labels


def _make_f1_dnf_labels(task_table_df):
    """Average the binary DNF label per driver from a RelBench task table.

    Returns (content_df, label_tensor) with values in [0, 1].
    """
    avg = task_table_df.groupby("driverId")["did_not_finish"].mean().reset_index()
    content = pd.DataFrame({"driverId": avg["driverId"].values})
    labels = torch.tensor(avg["did_not_finish"].values, dtype=torch.float32).unsqueeze(1)
    return content, labels


def _make_f1_top3_labels(task_table_df):
    """Average the binary top-3 qualifying label per driver from a RelBench task table.

    Returns (content_df, label_tensor) with values in [0, 1].
    """
    avg = task_table_df.groupby("driverId")["qualifying"].mean().reset_index()
    content = pd.DataFrame({"driverId": avg["driverId"].values})
    labels = torch.tensor(avg["qualifying"].values, dtype=torch.float32).unsqueeze(1)
    return content, labels


def _merge_f1_qualifying_standings_into_results_tensor(
    results_content: pd.DataFrame,
    results_tensor: torch.Tensor,
    qualifying_df: pd.DataFrame,
    standings_df: pd.DataFrame,
) -> torch.Tensor:
    """Left-join qualifying + driver-standings numeric features onto each result row.

    RelBench rel-f1 does not ship ``pit_stops`` / ``lap_times``; this uses tables
    that exist in ``relbench.datasets.f1`` (``qualifying``, ``standings``).
    """
    q = qualifying_df[["raceId", "driverId", "number", "position"]].copy()
    q = q.rename(columns={"number": "qual_number", "position": "qual_position"})
    for c in ("qual_number", "qual_position"):
        q[c] = pd.to_numeric(q[c], errors="coerce").fillna(0.0)

    st = standings_df[["raceId", "driverId", "points", "position", "wins"]].copy()
    st = st.rename(
        columns={
            "points": "st_points",
            "position": "st_position",
            "wins": "st_wins",
        }
    )
    for c in ("st_points", "st_position", "st_wins"):
        st[c] = pd.to_numeric(st[c], errors="coerce").fillna(0.0)

    keys = results_content[["raceId", "driverId"]].reset_index(drop=True)
    merged = keys.merge(q, on=["raceId", "driverId"], how="left")
    merged = merged.merge(st, on=["raceId", "driverId"], how="left")
    feat_cols = ["qual_number", "qual_position", "st_points", "st_position", "st_wins"]
    merged[feat_cols] = merged[feat_cols].fillna(0.0)
    extra = merged[feat_cols].values.astype(np.float32)
    for i in range(extra.shape[1]):
        col = extra[:, i]
        mu = float(col.mean())
        sig = float(col.std()) + 1e-8
        extra[:, i] = (col - mu) / sig
    extra_t = torch.from_numpy(extra)
    if extra_t.shape[0] != results_tensor.shape[0]:
        raise ValueError("results_content rows must align with results_tensor")
    return torch.cat([results_tensor, extra_t.to(dtype=results_tensor.dtype)], dim=1)


def load_relbench_f1_dataset(task_name="driver-position", extra_tables: bool = False):
    """Load the RelBench rel-f1 dataset and convert to RelNN db format.

    Requires the ``relbench`` package (``pip install relbench``).

    Parameters
    ----------
    task_name : str
        ``driver-position``, ``driver-dnf``, or ``driver-top3``.
    extra_tables : bool
        If True, append qualifying + driver-standings features (per result row)
        to the ``Results`` embedding tail. Default False keeps the smoke-test
        schema stable.

    Returns
    -------
    F1Dataset
        .db has keys Drivers, Constructors, Races, Results, TrainLabels.
    """
    try:
        from relbench.datasets import get_dataset
        from relbench.tasks import get_task
    except ImportError:
        raise ImportError(
            "relbench is required for the F1 dataset. Install with: pip install relbench"
        )

    dataset = get_dataset("rel-f1", download=True)
    task = get_task("rel-f1", task_name, download=True)
    db_rb = dataset.get_db()

    drivers_content, drivers_tensor = _encode_f1_drivers(db_rb.table_dict["drivers"].df)
    cons_content, cons_tensor = _encode_f1_constructors(db_rb.table_dict["constructors"].df)
    results_content, results_tensor = _encode_f1_results(db_rb.table_dict["results"].df)
    if extra_tables:
        results_tensor = _merge_f1_qualifying_standings_into_results_tensor(
            results_content,
            results_tensor,
            db_rb.table_dict["qualifying"].df,
            db_rb.table_dict["standings"].df,
        )
    races_content, races_tensor = _encode_f1_races(db_rb.table_dict["races"].df)

    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test", mask_input_cols=False)

    _TASK_META = {
        "driver-position": {
            "label_fn": _make_f1_labels,
            "description": "Predict avg finishing position (MAE)",
        },
        "driver-dnf": {
            "label_fn": _make_f1_dnf_labels,
            "description": "Predict if driver will DNF (AUROC)",
        },
        "driver-top3": {
            "label_fn": _make_f1_top3_labels,
            "description": "Predict if driver qualifies top-3 (AUROC)",
        },
    }
    meta = _TASK_META.get(task_name)
    if meta is None:
        raise ValueError(
            f"Unsupported task '{task_name}'. Choose from: {list(_TASK_META)}"
        )

    train_labels_content, train_labels_tensor = meta["label_fn"](train_table.df)

    db = {
        "Drivers": (drivers_content, drivers_tensor),
        "Constructors": (cons_content, cons_tensor),
        "Races": (races_content, races_tensor),
        "Results": (results_content, results_tensor),
        "TrainLabels": (train_labels_content, train_labels_tensor),
    }

    train_drivers = set(train_table.df["driverId"].unique())
    test_drivers = set(test_table.df["driverId"].unique())
    new_test_drivers = test_drivers - train_drivers

    info = {
        "task_name": task_name,
        "task_description": meta["description"],
        "n_train": len(train_table.df),
        "n_val": len(val_table.df),
        "n_test": len(test_table.df),
        "n_train_drivers": len(train_drivers),
        "n_test_drivers": len(test_drivers),
        "n_new_test_drivers": len(new_test_drivers),
        "driver_feature_dim": drivers_tensor.shape[1],
        "constructor_feature_dim": cons_tensor.shape[1],
        "race_feature_dim": races_tensor.shape[1],
        "result_feature_dim": results_tensor.shape[1],
        "extra_relbench_tables": bool(extra_tables),
    }

    raw = {
        "db": db,
        "dataset": dataset,
        "task": task,
        "train_table": train_table,
        "val_table": val_table,
        "test_table": test_table,
        "dataset_info": info,
    }
    return F1Dataset(raw)


# ---------------------------------------------------------------------------
# CTU Hepatitis dataset
# ---------------------------------------------------------------------------

class HepatitisDataset:
    """Result of loading the CTU Hepatitis_std dataset for RelNN.

    Schema (CTU MariaDB `Hepatitis_std`):
      dispat (500 patients): m_id, sex, age, Type (target: '0'=HBV, '1'=HCV)
      Bio (32 biopsies): b_id, fibros, activity
      indis (5691 lab tests): in_id, got, gpt, alb, tbil, dbil, che, ttt, ztt, tcho, tp
      rel11 (621): b_id, m_id  -- patient–biopsy links (all 500 patients covered)
      rel12 (5691): in_id, m_id -- patient–lab links (496/500; 4 missing filled with zeros)

    RelNN relations:
      Patients(m_id; z)           -- 2D: [sex_binary, age_normalized]
      Biopsies(biopsy_id, m_id; z) -- 2D: [fibros_norm, activity_norm]
      Labs(lab_id, m_id; z)        -- 10D: [got,gpt,alb,tbil,dbil,che,ttt,ztt,tcho,tp] normalized
      TrainLabels(m_id; label)     -- 1D binary label (0=HBV, 1=HCV)

    Baseline (ReDeLEx Table 1, arXiv:2506.22199):
      LightGBM test AUC: 0.626
      GraphSAGE/DBFormer test AUC: ~1.0

    Splits: random 70/15/15 on m_id (matching ReDeLEx 'orig.' protocol, seed=42).
    """

    def __init__(self, raw: dict):
        self._raw = raw
        self._db = {
            "Patients": raw["Patients"],
            "Biopsies": raw["Biopsies"],
            "Labs": raw["Labs"],
            "TrainLabels": raw["TrainLabels"],
        }

    @property
    def db(self):
        return self._db

    @property
    def train_ids(self):
        return self._raw["train_ids"]

    @property
    def val_ids(self):
        return self._raw["val_ids"]

    @property
    def test_ids(self):
        return self._raw["test_ids"]

    @property
    def val_labels(self):
        return self._raw["val_labels"]

    @property
    def test_labels(self):
        return self._raw["test_labels"]

    @property
    def dataset_info(self):
        return self._raw["dataset_info"]


def _normalize_col(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mu, sig = arr.mean(), arr.std() + 1e-8
    return ((arr - mu) / sig)


def _parse_numeric_col(series: pd.Series) -> np.ndarray:
    """Parse a varchar column of numeric strings to float32, filling NaN with 0."""
    return pd.to_numeric(series, errors="coerce").fillna(0.0).values.astype(np.float32)


def load_ctu_hepatitis_dataset(cache_dir=None, seed: int = 42):
    """Load CTU Hepatitis_std from MariaDB and convert to RelNN db format.

    Requires ``pymysql``: ``pip install pymysql``

    Downloads all 7 tables from the public CTU MariaDB server and caches them
    locally as parquet files in ``cache_dir`` (defaults to
    ``~/.cache/ctu_hepatitis``). Subsequent calls load from cache.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory for caching table parquet files.
    seed : int
        Random seed for the 70/15/15 train/val/test split on m_id.

    Returns
    -------
    HepatitisDataset
    """
    try:
        import pymysql
    except ImportError:
        raise ImportError(
            "pymysql is required for the Hepatitis dataset. Install with: pip install pymysql"
        )

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "ctu_hepatitis"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    TABLE_NAMES = ["dispat", "Bio", "indis", "rel11", "rel12"]
    tables: dict = {}
    cached = all((cache_dir / f"{t}.parquet").exists() for t in TABLE_NAMES)

    if cached:
        for t in TABLE_NAMES:
            tables[t] = pd.read_parquet(cache_dir / f"{t}.parquet")
    else:
        conn = pymysql.connect(
            host="relational.fel.cvut.cz",
            port=3306,
            user="guest",
            password="ctu-relational",
            database="Hepatitis_std",
            connect_timeout=30,
        )
        try:
            for t in TABLE_NAMES:
                cursor = conn.cursor()
                cursor.execute(f"SELECT * FROM `{t}`")
                rows = cursor.fetchall()
                col_names = [d[0] for d in cursor.description]
                tables[t] = pd.DataFrame(rows, columns=col_names)
                tables[t].to_parquet(cache_dir / f"{t}.parquet", index=False)
        finally:
            conn.close()

    dispat = tables["dispat"]
    bio = tables["Bio"]
    indis = tables["indis"]
    rel11 = tables["rel11"]
    rel12 = tables["rel12"]

    # ---- Patients (dispat features) ----
    sex_bin = (dispat["sex"].str.strip().str.upper() == "M").astype(np.float32).values
    age_num = _normalize_col(_parse_numeric_col(dispat["age"]))
    patient_feat = np.stack([sex_bin, age_num], axis=1)  # (500, 2)
    patients_content = pd.DataFrame({"m_id": dispat["m_id"].values})
    patients_tensor = torch.from_numpy(patient_feat)

    # ---- Biopsies (Bio features joined with rel11 to attach m_id) ----
    fibros_num = _normalize_col(_parse_numeric_col(bio["fibros"]))
    activity_num = _normalize_col(_parse_numeric_col(bio["activity"]))
    bio_feat = np.stack([fibros_num, activity_num], axis=1)  # (32, 2)
    bio_with_feats = bio[["b_id"]].copy()
    bio_with_feats["fibros_n"] = fibros_num
    bio_with_feats["activity_n"] = activity_num
    biopsy_joined = rel11.merge(bio_with_feats, on="b_id", how="inner")
    biopsy_content = pd.DataFrame({
        "biopsy_id": biopsy_joined["b_id"].values,
        "m_id": biopsy_joined["m_id"].values,
    })
    biopsy_feats = biopsy_joined[["fibros_n", "activity_n"]].values.astype(np.float32)
    biopsy_tensor = torch.from_numpy(biopsy_feats)

    # ---- Labs (indis features joined with rel12 to attach m_id) ----
    lab_cols = ["got", "gpt", "alb", "tbil", "dbil", "che", "ttt", "ztt", "tcho", "tp"]
    for col in lab_cols:
        indis[col] = _parse_numeric_col(indis[col])
    lab_normed = np.stack([_normalize_col(indis[c].values) for c in lab_cols], axis=1)
    indis_with_feats = indis[["in_id"]].copy()
    for i, col in enumerate(lab_cols):
        indis_with_feats[f"lab_{i}"] = lab_normed[:, i]
    lab_feat_cols = [f"lab_{i}" for i in range(len(lab_cols))]
    lab_joined = rel12.merge(indis_with_feats, on="in_id", how="inner")

    # Fill the 4 patients without any lab row with zero vectors so all 500 patients
    # are reachable from Labs during join evaluation.
    covered_m_ids = set(lab_joined["m_id"].unique())
    all_m_ids = set(dispat["m_id"].values)
    missing_m_ids = all_m_ids - covered_m_ids
    if missing_m_ids:
        fill_rows = []
        for mid in sorted(missing_m_ids):
            row = {"in_id": -1, "m_id": mid}
            row.update({c: 0.0 for c in lab_feat_cols})
            fill_rows.append(row)
        fill_df = pd.DataFrame(fill_rows)
        lab_joined = pd.concat([lab_joined, fill_df], ignore_index=True)

    # Renumber in_id to ensure uniqueness (fill rows have -1)
    lab_joined = lab_joined.reset_index(drop=True)
    lab_joined["lab_id"] = lab_joined.index
    lab_content = pd.DataFrame({
        "lab_id": lab_joined["lab_id"].values,
        "m_id": lab_joined["m_id"].values.astype(int),
    })
    lab_feats = lab_joined[lab_feat_cols].values.astype(np.float32)
    lab_tensor = torch.from_numpy(lab_feats)

    # ---- Train / Val / Test split (70/15/15 on m_id) ----
    rng = np.random.default_rng(seed)
    m_ids = dispat["m_id"].values.copy()
    rng.shuffle(m_ids)
    n = len(m_ids)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train_ids = set(m_ids[:n_train])
    val_ids = set(m_ids[n_train: n_train + n_val])
    test_ids = set(m_ids[n_train + n_val:])

    # ---- TrainLabels ----
    train_mask = dispat["m_id"].isin(train_ids)
    train_dispat = dispat[train_mask].reset_index(drop=True)
    train_labels_content = pd.DataFrame({"m_id": train_dispat["m_id"].values})
    train_label_vals = pd.to_numeric(train_dispat["Type"], errors="coerce").fillna(0).values.astype(np.float32)
    train_labels_tensor = torch.from_numpy(train_label_vals).unsqueeze(1)

    # ---- Val / Test labels for evaluation ----
    val_mask = dispat["m_id"].isin(val_ids)
    test_mask = dispat["m_id"].isin(test_ids)
    val_dispat = dispat[val_mask].reset_index(drop=True)
    test_dispat = dispat[test_mask].reset_index(drop=True)
    val_labels = pd.to_numeric(val_dispat["Type"], errors="coerce").fillna(0).values.astype(np.float32)
    test_labels = pd.to_numeric(test_dispat["Type"], errors="coerce").fillna(0).values.astype(np.float32)

    db = {
        "Patients": (patients_content, patients_tensor),
        "Biopsies": (biopsy_content, biopsy_tensor),
        "Labs": (lab_content, lab_tensor),
        "TrainLabels": (train_labels_content, train_labels_tensor),
    }

    info = {
        "n_patients": len(dispat),
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "patient_feature_dim": int(patients_tensor.shape[1]),
        "biopsy_feature_dim": int(biopsy_tensor.shape[1]),
        "lab_feature_dim": int(lab_tensor.shape[1]),
        "class_balance": float(pd.to_numeric(dispat["Type"], errors="coerce").mean()),
        "seed": seed,
    }

    raw = {
        "Patients": (patients_content, patients_tensor),
        "Biopsies": (biopsy_content, biopsy_tensor),
        "Labs": (lab_content, lab_tensor),
        "TrainLabels": (train_labels_content, train_labels_tensor),
        "train_ids": sorted(train_ids),
        "val_ids": sorted(val_ids),
        "test_ids": sorted(test_ids),
        "val_labels": val_labels,
        "test_labels": test_labels,
        "val_m_ids": val_dispat["m_id"].values,
        "test_m_ids": test_dispat["m_id"].values,
        "dataset_info": info,
    }
    return HepatitisDataset(raw)


# ---------------------------------------------------------------------------
# DDI (Drug-Drug Interaction) Hypergraph dataset
# ---------------------------------------------------------------------------

def _extract_kmer(smiles: str, k: int = 9) -> List[str]:
    """Extract k-mer substructures from a SMILES string."""
    if len(smiles) < k:
        return [smiles]
    return [smiles[i:i + k] for i in range(len(smiles) - k + 1)]


def _build_hypergraph(
    all_smiles: List[str],
    k: int = 9,
) -> Tuple[Dict[str, int], List[Tuple[int, int]]]:
    """Build substructure vocabulary and incidence list from SMILES strings.

    Returns
    -------
    sub_vocab : dict
        Mapping from substructure string to integer sub_id.
    incidence_rows : list of (sub_id, drug_id)
        Each entry means substructure sub_id belongs to drug drug_id.
    """
    sub_vocab: Dict[str, int] = {}
    incidence_set: set = set()
    for drug_id, smiles in enumerate(all_smiles):
        for sub in _extract_kmer(smiles, k):
            if sub not in sub_vocab:
                sub_vocab[sub] = len(sub_vocab)
            incidence_set.add((sub_vocab[sub], drug_id))
    return sub_vocab, sorted(incidence_set)


class DDIDataset:
    """Result of loading a Drug-Drug Interaction dataset for HyGNN in RelNN.

    Tables:
      - Drug: one-hot identity features (n_drugs, n_drugs)
      - Substructure: all-ones features (n_subs, d)
      - Incidence: hypergraph membership (sub_id, drug_id)
      - TrainPairs: positive + negative DDI pairs for training
      - TestPairs: positive + negative DDI pairs for evaluation
    """

    def __init__(self, raw: dict):
        self._raw = raw
        self._db = raw["db"]

    @property
    def db(self):
        return self._db

    @property
    def dataset_info(self):
        return self._raw.get("dataset_info", {})

    @property
    def test_labels(self):
        """Ground-truth labels for TestPairs (for metric computation)."""
        return self._raw.get("test_labels")

    def to_dict(self):
        return self._db

    def __repr__(self):
        info = self.dataset_info
        lines = [f"DDI dataset ({info.get('source', 'unknown')})"]
        lines.append("─" * 55)
        lines.append("Tables:")
        for name, (df, tensor) in self._db.items():
            lines.append(f"  • {name:<14} | Rows: {df.shape[0]:>6,} | Tensor: {tuple(tensor.shape)}")
        lines.append("")
        lines.append(f"Drugs: {info.get('n_drugs', 0):,}  |  Substructures: {info.get('n_subs', 0):,}")
        sub_m = info.get("substructure_method", "kmer")
        if sub_m == "espf":
            lines.append(
                f"Incidence edges: {info.get('n_incidence', 0):,}  |  ESPF min_support: {info.get('espf_support', '—')}"
            )
        else:
            lines.append(f"Incidence edges: {info.get('n_incidence', 0):,}  |  k-mer k: {info.get('k', 0)}")
        lines.append(f"Train pairs: {info.get('n_train', 0):,}  |  Test pairs: {info.get('n_test', 0):,}")
        return "\n".join(lines)


def load_ddi_dataset(
    source: str = "DrugBank",
    k: int = 9,
    d: int = 128,
    train_frac: float = 0.8,
    seed: int = 42,
):
    """Load a DDI dataset and build a hypergraph for HyGNN in RelNN.

    Requires the ``PyTDC`` package (``pip install PyTDC``).

    Parameters
    ----------
    source : str
        TDC dataset name: ``"DrugBank"`` or ``"TWOSIDES"``.
    k : int
        k-mer size for substructure extraction.
    d : int
        Vertex/edge embedding dimension (used for Substructure initial features).
    train_frac : float
        Fraction of DDI pairs used for training (rest is test).
    seed : int
        Random seed for train/test split and negative sampling.

    Returns
    -------
    DDIDataset
        .db has keys Drug, Substructure, Incidence, TrainPairs, TestPairs.
    """
    try:
        from tdc.multi_pred import DDI
    except ImportError:
        raise ImportError(
            "PyTDC is required for DDI datasets. Install with: pip install PyTDC"
        )

    rng = np.random.default_rng(seed)

    logger.info("Loading DDI dataset '%s' from TDC ...", source)
    ddi_data = DDI(name=source)
    split = ddi_data.get_split()

    all_pairs = pd.concat([split["train"], split["valid"], split["test"]], ignore_index=True)

    # Build drug ID mapping from all unique drugs
    drug_ids_col1 = all_pairs["Drug1_ID"].unique()
    drug_ids_col2 = all_pairs["Drug2_ID"].unique()
    all_drug_ids = sorted(set(drug_ids_col1) | set(drug_ids_col2))
    drug_id_map = {did: i for i, did in enumerate(all_drug_ids)}
    n_drugs = len(all_drug_ids)

    # Map SMILES: drug_int_id -> SMILES
    smiles_map: Dict[int, str] = {}
    for _, row in all_pairs.iterrows():
        d1 = drug_id_map[row["Drug1_ID"]]
        d2 = drug_id_map[row["Drug2_ID"]]
        if d1 not in smiles_map:
            smiles_map[d1] = row["Drug1"]
        if d2 not in smiles_map:
            smiles_map[d2] = row["Drug2"]
    all_smiles = [smiles_map[i] for i in range(n_drugs)]

    # Build hypergraph
    sub_vocab, incidence_rows = _build_hypergraph(all_smiles, k=k)
    n_subs = len(sub_vocab)

    # Build positive pairs (deduplicated, mapped to integer IDs)
    positive_pairs = set()
    for _, row in all_pairs.iterrows():
        d1 = drug_id_map[row["Drug1_ID"]]
        d2 = drug_id_map[row["Drug2_ID"]]
        if d1 != d2:
            a, b = min(d1, d2), max(d1, d2)
            positive_pairs.add((a, b))
    positive_pairs = sorted(positive_pairs)

    # Sample equal number of negative pairs
    positive_set = set(positive_pairs)
    negatives = []
    max_attempts = len(positive_pairs) * 20
    attempts = 0
    while len(negatives) < len(positive_pairs) and attempts < max_attempts:
        a = rng.integers(0, n_drugs)
        b = rng.integers(0, n_drugs)
        if a == b:
            attempts += 1
            continue
        a, b = min(a, b), max(a, b)
        if (a, b) not in positive_set and (a, b) not in set(negatives):
            negatives.append((a, b))
        attempts += 1

    # Combine and shuffle
    all_labeled = [(a, b, 1.0) for a, b in positive_pairs] + \
                  [(a, b, 0.0) for a, b in negatives]
    rng.shuffle(all_labeled)

    n_total = len(all_labeled)
    n_train = int(n_total * train_frac)

    train_data = all_labeled[:n_train]
    test_data = all_labeled[n_train:]

    # Build RelNN DB
    db = {}

    # Drug: one-hot identity features
    db["Drug"] = (
        pd.DataFrame({"drug_id": range(n_drugs)}),
        torch.eye(n_drugs, dtype=torch.float32),
    )

    # Substructure: all-ones features
    db["Substructure"] = (
        pd.DataFrame({"sub_id": range(n_subs)}),
        torch.ones(n_subs, d, dtype=torch.float32),
    )

    # Incidence: structural (sub_id, drug_id)
    db["Incidence"] = (
        pd.DataFrame({
            "sub_id": [r[0] for r in incidence_rows],
            "drug_id": [r[1] for r in incidence_rows],
        }),
        torch.ones(len(incidence_rows), 1, dtype=torch.float32),
    )

    # TrainPairs
    train_d1 = [t[0] for t in train_data]
    train_d2 = [t[1] for t in train_data]
    train_labels = torch.tensor([t[2] for t in train_data], dtype=torch.float32).unsqueeze(1)
    db["TrainPairs"] = (
        pd.DataFrame({"drug1": train_d1, "drug2": train_d2}),
        train_labels,
    )

    # TestPairs
    test_d1 = [t[0] for t in test_data]
    test_d2 = [t[1] for t in test_data]
    test_labels_tensor = torch.tensor([t[2] for t in test_data], dtype=torch.float32).unsqueeze(1)
    db["TestPairs"] = (
        pd.DataFrame({"drug1": test_d1, "drug2": test_d2}),
        test_labels_tensor,
    )

    info = {
        "source": source,
        "n_drugs": n_drugs,
        "n_subs": n_subs,
        "n_incidence": len(incidence_rows),
        "k": k,
        "d": d,
        "n_positive": len(positive_pairs),
        "n_negative": len(negatives),
        "n_train": len(train_data),
        "n_test": len(test_data),
    }

    raw = {
        "db": db,
        "dataset_info": info,
        "test_labels": test_labels_tensor,
    }
    return DDIDataset(raw)


# ── HyGNN reference data (from GitHub) ──────────────────────────────────────

_HYGNN_BASE_URL = (
    "https://raw.githubusercontent.com/"
    "shouhengtuo/HyGNN-Drug-Drug-Interaction-Prediction-via-Hypergraph-Neural-Network/"
    "main/data"
)

_HYGNN_SOURCE_MAP = {"TWOSIDES": 645, "DrugBank": 1706}

# Official HyGNN repo (shouhengtuo/…) ships these hyperparameter grids.
_HYGNN_KMER_K = frozenset({3, 6, 9, 12, 15})
_HYGNN_ESPF_SUPPORT = frozenset({5, 10, 15, 20, 25})
# Allowed ``espf_support`` for ``load_hygnn_dataset`` (matches HyGNN reference files).
HYGNN_ESPF_SUPPORT_VALUES = _HYGNN_ESPF_SUPPORT


def _download_hygnn_file(filename: str, dest_dir: Path) -> Path:
    """Download a single file from the HyGNN GitHub repo, caching locally."""
    local = dest_dir / filename
    if local.exists():
        return local
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = f"{_HYGNN_BASE_URL}/{filename}"
    logger.info("Downloading %s ...", url)
    urllib.request.urlretrieve(url, local)
    return local


def load_hygnn_dataset(
    source: str = "TWOSIDES",
    k: int = 3,
    d: int = 128,
    train_frac: float = 0.9,
    seed: int = 42,
    *,
    substructure_method: str = "kmer",
    espf_support: int = 10,
) -> DDIDataset:
    """Load precomputed HyGNN data from the reference GitHub repository.

    Downloads the hypergraph incidence tensor, DDI edge list, and
    substructure-count file directly from
    `github.com/shouhengtuo/HyGNN-Drug-Drug-Interaction-Prediction-…`
    and converts them into the RelNN DB format expected by the HyGNN demo.

    Substructures can be taken from the authors' **k-mer** decomposition of
    SMILES (default) or from **ESPF** (Extended Substructure Pattern mining
    Framework), matching the files published alongside the HyGNN paper.

    Parameters
    ----------
    source : str
        ``"TWOSIDES"`` (645 drugs) or ``"DrugBank"`` (1706 drugs).
    k : int
        When ``substructure_method=="kmer"``: k-mer length (3, 6, 9, 12, or 15).
        Ignored when using ESPF.
    d : int
        Substructure initial feature dimension (default 128).
    train_frac : float
        Fraction of DDI pairs used for training (default 0.9 = 90 % train,
        10 % test, matching the reference 80/10/10 with train+val merged).
    seed : int
        Random seed for shuffling and negative sampling.
    substructure_method : str
        ``"kmer"`` or ``"espf"``. Selects which precomputed hypergraph files
        to download from the reference repository.
    espf_support : int
        When ``substructure_method=="espf"``: minimum-support parameter for
        which the official repo provides tensors (5, 10, 15, 20, or 25).

    Returns
    -------
    DDIDataset
        ``.db`` has keys Drug, Substructure, Incidence, TrainPairs, TestPairs.
    """
    if source not in _HYGNN_SOURCE_MAP:
        raise ValueError(f"source must be one of {list(_HYGNN_SOURCE_MAP)}, got {source!r}")
    n_drugs = _HYGNN_SOURCE_MAP[source]

    sm = substructure_method.lower().strip()
    if sm not in ("kmer", "espf"):
        raise ValueError(
            f"substructure_method must be 'kmer' or 'espf', got {substructure_method!r}"
        )
    if sm == "kmer" and k not in _HYGNN_KMER_K:
        raise ValueError(
            f"For k-mer, k must be one of {sorted(_HYGNN_KMER_K)}, got {k}"
        )
    if sm == "espf" and espf_support not in _HYGNN_ESPF_SUPPORT:
        raise ValueError(
            f"For ESPF, espf_support must be one of {sorted(_HYGNN_ESPF_SUPPORT)}, "
            f"got {espf_support}"
        )

    project_root = get_project_root()
    dest_dir = project_root / "data" / "HyGNN"

    # --- download files ---------------------------------------------------
    if sm == "kmer":
        suffix = f"kmer_{k}"
    else:
        suffix = f"ESPF_{espf_support}"
    incidence_file = _download_hygnn_file(f"hyG_drug_{n_drugs}_{suffix}.pt", dest_dir)
    edgelist_file = _download_hygnn_file(f"edge_list_regular_graph_{n_drugs}.txt", dest_dir)
    rows_file = _download_hygnn_file(f"rows_drug_{n_drugs}_{suffix}.txt", dest_dir)

    # --- parse incidence tensor -------------------------------------------
    incidence_tensor = torch.load(incidence_file, map_location="cpu", weights_only=True)
    sub_ids = incidence_tensor[:, 0].long().tolist()
    drug_ids = incidence_tensor[:, 1].long().tolist()
    incidence_rows = list(zip(sub_ids, drug_ids))

    with open(rows_file, "r") as f:
        n_subs = int(f.read().strip().split()[-1])

    # --- parse DDI edge list (positive pairs) -----------------------------
    positive_set = set()
    with open(edgelist_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                a, b = int(parts[0]), int(parts[1])
                positive_set.add((min(a, b), max(a, b)))
    positive_pairs = sorted(positive_set)

    # --- sample negatives -------------------------------------------------
    rng = np.random.default_rng(seed)
    neg_set: set = set()
    max_attempts = len(positive_pairs) * 20
    attempts = 0
    while len(neg_set) < len(positive_pairs) and attempts < max_attempts:
        a = int(rng.integers(0, n_drugs))
        b = int(rng.integers(0, n_drugs))
        if a == b:
            attempts += 1
            continue
        pair = (min(a, b), max(a, b))
        if pair not in positive_set and pair not in neg_set:
            neg_set.add(pair)
        attempts += 1
    negatives = sorted(neg_set)

    # --- combine, shuffle, split ------------------------------------------
    all_labeled = [(a, b, 1.0) for a, b in positive_pairs] + \
                  [(a, b, 0.0) for a, b in negatives]
    rng.shuffle(all_labeled)

    n_total = len(all_labeled)
    n_train = int(n_total * train_frac)
    train_data = all_labeled[:n_train]
    test_data = all_labeled[n_train:]

    # --- build RelNN DB ---------------------------------------------------
    db: Dict[str, tuple] = {}

    db["Drug"] = (
        pd.DataFrame({"drug_id": range(n_drugs)}),
        torch.eye(n_drugs, dtype=torch.float32),
    )

    db["Substructure"] = (
        pd.DataFrame({"sub_id": range(n_subs)}),
        torch.ones(n_subs, d, dtype=torch.float32),
    )

    db["Incidence"] = (
        pd.DataFrame({"sub_id": sub_ids, "drug_id": drug_ids}),
        torch.ones(len(incidence_rows), 1, dtype=torch.float32),
    )

    train_labels = torch.tensor([t[2] for t in train_data], dtype=torch.float32).unsqueeze(1)
    db["TrainPairs"] = (
        pd.DataFrame({"drug1": [t[0] for t in train_data], "drug2": [t[1] for t in train_data]}),
        train_labels,
    )

    test_labels_tensor = torch.tensor([t[2] for t in test_data], dtype=torch.float32).unsqueeze(1)
    db["TestPairs"] = (
        pd.DataFrame({"drug1": [t[0] for t in test_data], "drug2": [t[1] for t in test_data]}),
        test_labels_tensor,
    )

    info = {
        "source": f"HyGNN-github/{source}",
        "n_drugs": n_drugs,
        "n_subs": n_subs,
        "n_incidence": len(incidence_rows),
        "substructure_method": sm,
        "k": k if sm == "kmer" else None,
        "espf_support": espf_support if sm == "espf" else None,
        "d": d,
        "n_positive": len(positive_pairs),
        "n_negative": len(negatives),
        "n_train": len(train_data),
        "n_test": len(test_data),
    }

    return DDIDataset({"db": db, "dataset_info": info, "test_labels": test_labels_tensor})
