# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Session
#
# > User-facing interface for managing RelNN state, models, and programs

# %%
import logging
import networkx as nx
from typing import Optional
from pydantic import BaseModel
from relann.pydantic_classes import *
from relann.term_graph import TermGraph, preety_draw_tg
from relann.engine import Engine, pretty_print_params
from relann.torch_utils import full_seed
from relann.era_operations import pretty_print_er
from fastcore.basics import patch

logger = logging.getLogger(__name__)

# %%
import inspect
from typing import Any, Dict, Optional
from relann.parser import parse_and_transform_str, RelnnTransformer
from relann.engine import Engine

# %% [markdown]
# ## Session Class

# %%
class Session:
    """User-facing interface for managing RelNN state, models, and programs."""
    
    def __init__(self, db=None, seed: Optional[int] = None, debug: bool = False, device=None):
        """Initialize a Session with a database.
        
        Args:
            db: Relation database dict.
            seed: Optional random seed.
            debug: Enable debug mode.
            device: Optional torch.device or str (e.g. 'cuda', 'cpu'). When set,
                all embedding tensors in db are moved to this device on init so
                that per-epoch CPU->GPU transfers are avoided during training.
        """
        self.engine = Engine(db=db, seed=seed, debug=debug, device=device)
    
    def define(self, code: str):
        """Parse and execute RelNN code. Returns predict result if present, otherwise None."""
        # Op resolution (Linear, custom modules, etc.) must see the *caller's* globals.
        # When using session.run(code), the immediate caller frame is Session.run, whose
        # f_globals is this module — skip that wrapper so we pick up the real caller.
        fr = inspect.currentframe().f_back
        # Only skip the internal Session.run wrapper frame from this module.
        # User code can legitimately define its own `run()` function.
        if (
            fr is not None
            and fr.f_code.co_name == "run"
            and fr.f_globals.get("__name__") == __name__
            and fr.f_back is not None
        ):
            fr = fr.f_back
        self.engine.set_run_globals(fr.f_globals if fr is not None else {})
        # Parse using the session's engine (so transformer uses correct symbol table)
        transformer = RelnnTransformer(self.engine)
        program = parse_and_transform_str(code, start="program", transformer=transformer)
        # Execute the program (fit/predict statements run automatically)
        result = self.engine.add_program(program)
        return result
    
    def run(self, code: str):
        """Alias for define()."""
        return self.define(code)

# %%
@patch
def show_params(self: Session, show_stats: bool = True, max_name_width: int = 50):
    """Display model parameters in a formatted table."""
    pretty_print_params(self.engine, show_stats=show_stats, max_name_width=max_name_width)

# %%
@patch
def show_term_graph(self: Session, rule_name: Optional[str] = None, namespace: str = "global", **kwargs):
    """Visualize the term graph. Use rule_name for subgraph visualization (if provided)."""
    tg = self.engine.term_graphs.get(namespace)
    if tg is None:
        raise ValueError(f"Namespace '{namespace}' not found")
    
    # The recommended way—see parent/term_graph.py induced_subgraph method—
    # is to use tg.induced_subgraph(rule_name, direction="ancestors", include_root=True)
    # to get all upstream nodes for a given rule or symbol.
    if rule_name is not None:
        subgraph = tg.induced_subgraph(rule_name, direction="ancestors", include_root=True)
        preety_draw_tg(subgraph, **kwargs)
    else:
        preety_draw_tg(tg, **kwargs)

# %%
@patch
def relation(self: Session, name: str):
    """Return the EmbeddedRelation for a named rule from the last fit/predict run."""
    return self.engine.relation(name)

# %%
@patch
def params(self: Session):
    """Return an OrderedDict of {clean_name: nn.Parameter} from the parameter store."""
    return self.engine.params()

# %% [markdown]
# ## Example Usage
