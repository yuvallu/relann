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

# %%
# %% [markdown]
# # Parser
#
# > A module for parsing RelNN DSL code and generating models

# %%
if __name__ == "__main__":
    import sys
    import os

# %%
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath('..'))

# %%
from lark import Lark, Transformer, v_args, Token, Tree
from relann.pydantic_classes import *
import torch.nn as nn
from relann.engine import Engine
from relann.tensor_term_compiler import resolve_op as _resolve_op
from fastcore.basics import patch
import logging
import re
from pydantic import BaseModel
from pathlib import Path

# %%
def _package_parent_dir() -> Path:
    """Directory containing this module (`relann/`), including `relann_grammar.lark`."""
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    # Notebook / interactive: cwd may be project root, the `relann/` dir itself,
    # or some parent. Look for the grammar in the conventional spots.
    cwd = Path.cwd()
    candidates = (
        cwd / "relann",          # cwd == repo root
        cwd,                     # cwd == relann/
        cwd.parent / "relann",   # cwd == repo_root/<sibling> (e.g. examples/)
    )
    for base in candidates:
        grammar = base / "relann_grammar.lark"
        if grammar.is_file():
            return grammar.parent
    # Last-resort fallback so the constant exists even if the file isn't found.
    return cwd / "relann"

_pkg_dir = _package_parent_dir()
GRAMMAR_FILE = str(_pkg_dir / "relann_grammar.lark")

logger = logging.getLogger(__name__)

# %% [markdown]
# # Test lark syntax

# %%
class RelNNSyntaxError(SyntaxError):
    pass

# %%
if __name__ == "__main__":
    # Load grammar from external .lark file for better syntax highlighting
    from lark import Lark

# %%
if __name__ == "__main__":
    relnn_grammer_parser = Lark.open(
        GRAMMAR_FILE,
        start="program",  # Note: if you want to parse only part of the grammar and not an entire program - change this.
        parser="earley",
        maybe_placeholders=False,
        propagate_positions=True
    )

    # TODO
    # 1. color lang_server
    # 2. Write with high order
        # V 2.1. template syntax
        # V 2.2. gounding "|" syntax
        # V 2.3. use "*" syntax for "|" syntax (for example, "hconcat(*x)")
        # 2.4. use "..." syntax for unknown number of attibutes in input to func_def (for example, "RightSoftmax(R:(*...,int)<m>)")
    # 3. add explisit agreggation abstract syntax
    # 4. add fit and predict

    # Quick test:
    # test0_01 = """
    # K_Linear<l,S,i> = Tensor(d,d/h).
    # K_Linear<l,S,i> = Tensor(d+i,d/h).
    # Q_Linear<l,S,i> = Tensor(d,d/h).
    # """
    test2 = """
    d = 10 .
    h = 2 .
    K_Linear<l,S,i> = Tensor(d,d/h).
    Q_Linear<l,S,i> = Tensor(d,d/h).

    def Attention<l,T>(N:(str, str, str;)):
        Q<i>(t; Q_Linear<l,T,i> @ x) :- H<l-1,T>(t; x).
        K<S,i>(s; K_Linear<l,S,i> @ x) :- H<l-1,S>(s; x).
        Mew<S,E,T> = Tensor(1,1).
        Watt<E> = Tensor(d/h,d/h).

        AttHead<S,E,i>(s, e, t; (k @ Watt<E> @ Transpose(q)) @ Mew<S,E,T> / sqrt(d)) :- K<S,i>(s; k), Q<i>(t; q).

        StackedHeads<S,E>(s, e, t; hconcat(*x)) :- Join(Set(AttHead<S,E,i>(s, t; x) | INT(i), i <= 1)).
        StackedHeads(s, e, t; x) :- Union(Set(StackedHeads<S,E>(s, e, t; x) | N(S,E,T))).

        Output(s, e, t; x) :- RightSoftmax(StackedHeads)(s, e, t; x).
    enddef
    """

    tree = relnn_grammer_parser.parse(test2)
    print(tree.pretty())

# %%
if __name__ == "__main__":
    test0_01 = """
    def RightSoftmax(R:(int; m)) -> (int; m):
         Denom(target; sum(exp(x))) :- R(target; x).
         Softmax(target; exp(x)/d) :- R(target; x), Denom(target; d).
    enddef
    """
    tree = relnn_grammer_parser.parse(test0_01)
    print(tree.pretty())

# %%
if __name__ == "__main__":
    # Tests for the RELNN grammar in 010_relann_grammar.lark

    # Test 0.1: Constant assignment and float ambiguity
    test0_01 = """
    m = 10 .
    def RightSoftmax(R:(int; m)) -> (int; m):
         TransformDef = 3 * Tensor(1,1).
         Denom(target; sum(exp(x))) :- R(target; x).
         Softmax(target; exp(x)/d) :- R(target; x), Denom(target; d).
    """

    try:
        tree = relnn_grammer_parser.parse(test0_01)
        print("test0_01 parsed successfully")
        print(tree.pretty())
    except Exception as e:
        print("test0_01 failed:", e)

# %%
if __name__ == "__main__":

    # Test 0.3: Simple relation rules
    test_0_03 = """
    A(a_id; v) :- Author(a_id; v).
    P(p_id; t) :- Paper(p_id; t).
    W(a_id, p_id; w) :- Writes(a_id, p_id; w).
    """

    try:
        tree = relnn_grammer_parser.parse(test_0_03)
        print("test_0_03 parsed successfully")
        print(tree.pretty())
    except Exception as e:
        print("test_0_03 failed:", e)

# %%
if __name__ == "__main__":

    # Test 1: RightSoftmax with ellipsis and transformation
    test1 = """
    def RightSoftmax(R:(int; m)) -> (int; m):
        Denom(target; sum(exp(x))) :- R(target; x).
        Softmax(target; exp(x)/d) :- R(target; x), Denom(target; d).
    """

    try:
        tree = relnn_grammer_parser.parse(test1)
        print("test1 parsed successfully")
        print(tree.pretty())
    except Exception as e:
        print("test1 failed:", e)

# %%
if __name__ == "__main__":

    # Test 2: More complex attention mechanism
    test2 = """
    K_Linear = Tensor(1,1).
    Q_Linear = Tensor(1,1).

    def Attention(N:(str, str, str;)):
        Q(t; Q_Linear(x)) :- H(t; x).
        K(s; K_Linear(x)) :- H(s; x).
        Mew = Tensor(1,1).
        Watt = Tensor(1,1).
        AttHead(s, e, t; (k * Watt * Transpose(q)) * Mew / sqrt(1)) :- K(s; k), Q(t; q).
            # StackedHeads(s, e, t; hconcat(x)) :- Join(AttHead(s, t; x) | INT(i), i <= 1).
        StackedHeads(s, e, t; hconcat(x1,x2)) :- AttHead_1(s, e, t; x1), AttHead_2(s, e, t; x2).
            # StackedHeads2(s, e, t; x) :- Union(StackedHeads(s, e, t; x) | N). 
        StackedHeads2(s, e, t; x) :- StackedHeads__S1__E1(s, e, t; x) | StackedHeads__S2__E2(s, e, t; x).
        Output(s, e, t; x) :- RightSoftmax(StackedHeads)(s, e, t; x).
    """

    try:
        tree = relnn_grammer_parser.parse(test2)
        print("test2 parsed successfully")
        print(tree.pretty())
    except Exception as e:
        print("test2 failed:", e)

# %% [markdown]
# # Test High order

# %%
if __name__ == "__main__":
    full_hgt_test = """
    def RightSoftmax(R:(int; m)) -> (int; m):
        Denom(target; sum(exp(x))) :- R(target; x).
        Softmax(target; exp(x)/d) :- R(target; x), Denom(target; d).

    K_Linear<l,S,i> = Tensor(d, d/h).
    Q_Linear<l,S,i> = Tensor(d, d/h).

    def Attention<l,T>(N:(str, str, str;)) -> (int, int, int):
        Q<i>(t; Q_Linear<l,T,i> @ x) :- H<l-1,T>(t; x).
        K<S,i>(s; K_Linear<l,S,i> @ x) :- H<l-1,S>(s; x).
        Mew<S,E,T> = Tensor(1,1).
        Watt<E> = Tensor(d/h,d/h).
        
        AttHead<S,E,i>(s, e, t; (k @ Watt<E> @ Transpose(q)) @ Mew<S,E,T> / sqrt(d)) :- K<S,i>(s; k), Q<i>(t; q).
        
        StackedHeads<S,E>(s, e, t; hconcat(*x)) :- Join(Set(AttHead<S,E,i>(s, t; x) | INT(i), i <= 1)).
        StackedHeads(s, e, t; x) :- Union(Set(StackedHeads<S,E>(s, e, t; x) | N(S,E,T))). 
        
        Output(s, e, t; x) :- RightSoftmax(StackedHeads)(s, e, t; x).

    M_Linear<l,S,i> = Tensor(d, d/h).

    def MSG<l,S,E,T>() -> (int):
        Wmsg = Tensor(d/h, d/h).
        MSG_HEAD<i>(s; M_Linear<S> @ x @ Wmsg) :- H<l-1,S>(s; x).
        MSG(s; hconcat(*m)) :- Join(Set(MSG_HEAD<i>(s; m) | INT(i), i <= h)).

    def H<0,T>() -> (int):
        H(t; Linear(-1, d)(x)) :- T(t; x).

    def H<l,T>(N:(str,str,str;)) -> (int):
        Htilda<S,E>(t; m * attn) :- Attention<l,T>(N)(s, t; attn), MSG<l,S,E,T>(s, e, t; m).
        Htilda(t; sum(x)) :- Union(Set(Htilda<S,E>(t; x) | N(S,E,T))).
        Out(t; sigmoid(Linear(d, d)(h_tag)) + h_old) :- Htilda(t; h_tag), H<l-1,T>(t; h_old).

    def HGTNet<T, num_layers, num_logits>(N:(str,str,str;)) -> (int):
        Out(idx; Linear(d, num_logits)(x)) :- H<num_layers,T>(N)(idx; x).

    # Assume unique db-wide ids for each node/edge

    A(a_id; v) :- Author(a_id; v), Split(a_id, "train").


    # Assuming that N exists with this schema
    # N('A', 'wrote', 'P').
    # N('P', 'contains_keyword', 'T').
    # N('C', 'publishes', 'P').
    # N('P', 'written_by', 'A').
    # N('T', 'is_contained_in', 'P').
    # N('P', 'published_at', 'C').

    d = 64 .
    h = 2 .
    L = 2 .
    num_logits = 4 .
    net = HGTNet<Author, L, num_logits>.

    ?fit <loss="CrossEntropy", epochs=100, optimizer="Adam", batch_size=32> fit()(; CrossEntropy(y, y_tag)) :- net(N)(a_id; y_tag), AuthorLabels(a_id; y).

    # A(a_id; v) :- Author(a_id; v), Split(a_id, "test").
    # ?pred predict(a; logits) :- net(N)(a; logits).
    """
    relnn_grammer_parser = Lark.open(
        GRAMMAR_FILE,
        start="program",
        parser="earley",
        maybe_placeholders=False,
        propagate_positions=True,
        # define the start no terminals (multiple)
    )

    try:
        tree = relnn_grammer_parser.parse(full_hgt_test)
        print("full_hgt_test parsed successfully")
        print(tree.pretty())
    except Exception as e:
        print("full_hgt_test failed:", e)

# %%
if __name__ == "__main__":
    # Encode/decode bracket syntax tests (formerly called "shuttling")
    content_decode_test_1 = """
    A(a_id,[a_id]; v*[a_id]) :- Author(a_id; v), Split(a_id, "train").
    """

    # TODO - this test should fail but currently passes
    fail_content_decode_test_1 = """
    A(a_id,[a_id]; v*[a_id]) :- Author([a_id]; v), Split(a_id, "train").
    """

    relnn_grammer_parser = Lark.open(
        GRAMMAR_FILE,
        start="program",
        parser="earley",
        maybe_placeholders=False,
        propagate_positions=True,
        # define the start no terminals (multiple)
    )

    try:
        tree = relnn_grammer_parser.parse(content_decode_test_1)
        print("full_hgt_test parsed successfully")
        print(tree.pretty())
    except Exception as e:
        print("full_hgt_test failed:", e)

# %% [markdown]
# # Transfomer

# %%
def get_relnn_grammar_parser(start="program"):
    return Lark.open(
        GRAMMAR_FILE,
        start=start,
        parser="earley",
        maybe_placeholders=False,
        propagate_positions=True,
    )

relnn_grammar_parser = get_relnn_grammar_parser()


_statement_parser = None

def get_statement_parser():
    """Return a cached Lark parser for start='statement'."""
    global _statement_parser
    if _statement_parser is None:
        _statement_parser = get_relnn_grammar_parser(start="statement")
    return _statement_parser

# %%
dtypes = ["int","bool","bytes","complex128","complex256","complex64","datetime64","float128","float16","float32","float64","int16","int32","int64","int8","object","str","timedelta64","uint16","uint32","uint64","uint8","void","Sparse[float64, nan]","category"]
default_agg = 'avg'

class RelnnTransformer(Transformer):
    def __init__(self, engine=None, visit_tokens=True, save_diagnostics=False, line_offset=0,
                 collect_tokens=False, source_text: str = None):
        super().__init__(visit_tokens)
        self.engine = engine
        self.line_offset = line_offset
        self.collect_tokens = collect_tokens
        self.source_text = source_text
        if self.collect_tokens:
            self.save_diagnostics = True
        else:
            self.save_diagnostics = save_diagnostics
        if self.save_diagnostics:
            self.diagnostics = []
        if self.collect_tokens:
            self.token_info = []

    def raise_or_save_diagnostic(self, message, meta):
        if self.save_diagnostics:
            self.diagnostics.append((meta, message, self.line_offset))
        else:
            raise RelNNSyntaxError(message)

    def _record_token(self, meta, category: str, text: str = None, skip_pattern: str = None):
        """Record a token for semantic tokenization (no-op when collect_tokens is False)."""
        if not self.collect_tokens:
            return

        line = meta.line - 1
        start_char = meta.column - 1
        skip_chars = 0

        if skip_pattern and self.source_text:
            lines = self.source_text.split('\n')
            if line >= len(lines):
                return
            line_text = lines[line]
            if start_char >= len(line_text):
                return
            match = re.search(skip_pattern, line_text[start_char:])
            if match:
                start_char += match.end()
                skip_chars = match.end()
                if text:
                    text_pos = line_text.find(text, start_char)
                    if text_pos != -1:
                        start_char = text_pos

        if text is not None:
            length = len(text)
        elif hasattr(meta, 'end_line') and hasattr(meta, 'end_column'):
            length = meta.end_column - meta.column - skip_chars
        else:
            length = 1

        self.token_info.append((line, start_char, length, category))

    def _record_template_brackets(self, meta):
        """Record < and > angle brackets for template coloring."""
        if not self.collect_tokens:
            return
        open_line = meta.line - 1
        open_char = meta.column - 1
        self.token_info.append((open_line, open_char, 1, "template_bracket"))
        close_line = meta.end_line - 1
        close_char = meta.end_column - 2
        self.token_info.append((close_line, close_char, 1, "template_bracket"))

    def _collect_var_names_from_arithterm(self, at):
        """Recursively yield Var.name strings from an ArithTerm tree."""
        if at is None:
            return
        if isinstance(at, Var):
            yield at.name
            return
        if hasattr(at, 'value') and isinstance(at.value, Var):
            yield at.value.name
        if hasattr(at, 'sons') and at.sons:
            for child in at.sons:
                yield from self._collect_var_names_from_arithterm(child)

    def _collect_vars_from_tensor_term(self, tt):
        """Recursively yield Var.name strings from a TensorTerm tree."""
        if tt is None:
            return
        if hasattr(tt, 'value') and isinstance(tt.value, Var):
            yield tt.value.name
        if hasattr(tt, 'sons') and tt.sons:
            for child in tt.sons:
                yield from self._collect_vars_from_tensor_term(child)

    def _record_var_in_source(self, meta, var_name, category):
        """Find var_name in source_text within the meta span and record it as a token."""
        if not self.collect_tokens or not self.source_text:
            return
        lines = self.source_text.split('\n')
        line_idx = meta.line - 1
        if line_idx < 0 or line_idx >= len(lines):
            return
        line_text = lines[line_idx]
        start = meta.column - 1
        end = (meta.end_column - 1) if hasattr(meta, 'end_column') and meta.end_line == meta.line else len(line_text)
        for m in re.finditer(r'\b' + re.escape(var_name) + r'\b', line_text[start:end]):
            col = start + m.start()
            fake_meta = type('M', (), {'line': meta.line, 'column': col + 1,
                                       'end_column': col + 1 + len(var_name)})()
            self._record_token(fake_meta, category, var_name)
            break

# %%
def parse_and_transform_str(input_text: str, start: str = "program", transformer=None,
                            collect_tokens: bool = False) -> BaseModel:
    if transformer is None:
        engine = Engine() if not collect_tokens else None
        transformer = RelnnTransformer(engine, collect_tokens=collect_tokens, source_text=input_text)
    else:
        if collect_tokens and (not hasattr(transformer, 'source_text') or transformer.source_text is None):
            transformer.source_text = input_text
    if start == "statement":
        parser = get_statement_parser()
    else:
        parser = get_relnn_grammar_parser(start=start)
    tree = parser.parse(input_text)
    result = transformer.transform(tree)
    return result

# %%
if __name__ == "__main__":
    # example for tests
    relnn_grammer_parser = get_relnn_grammar_parser(start="program")
    engine = Engine()
    transformer = RelnnTransformer(engine)

# %%
@patch
@v_args(meta=True)
def name(self: RelnnTransformer, meta, items):
    return str(items[0])

# %%
if __name__ == "__main__":
    # Test for name transformation
    relnn_grammar_parser = get_relnn_grammar_parser(start="name")
    input_text = "foobar"
    tree = relnn_grammar_parser.parse(input_text)
    logger.info("Test parse tree for name:\n%s", tree.pretty())
    result = transformer.transform(tree)
    print("Test Result (name):", result)
    assert result == "foobar", f"Expected 'foobar', got {result!r}"

# %%
@patch
@v_args(meta=True)
def var(self: RelnnTransformer, meta, items):
    logger.debug("var received items: %s", items)
    name = items[0]
    logger.debug("extracted name: %s", name)
    var = Var(name=name)
    return var

# %%
if __name__ == "__main__":
    # Test var transformation inline (not in a function)
    relnn_grammer_parser = get_relnn_grammar_parser(start="var")
    input_text = "foobar"
    tree = relnn_grammer_parser.parse(input_text)
    ptree_str = tree.pretty()
    logger.info("Test parse tree for var:\n%s", ptree_str)
    result = transformer.transform(tree)
    print("Test Result (var):", result)
    assert result.name == "foobar", f"Expected Var with name 'foobar', got: {result}"

# %%
@patch
@v_args(meta=True)
def primitive_type(self: RelnnTransformer, meta, items):
    basic_type = items[0].value
    logger.debug("primitive_type received: %s", basic_type)
    return basic_type

# %%
if __name__ == "__main__":
    relnn_grammer_parser = get_relnn_grammar_parser(start="primitive_type")
    input_text = "int"
    tree = relnn_grammer_parser.parse(input_text)

    ptree_str = tree.pretty()
    logger.info("Parse tree structure:\n%s", ptree_str)
    p = transformer.transform(tree)
    print("Result:", p, type(p))

# %%
@patch
@v_args(meta=True)
def content_attr_types(self: RelnnTransformer, meta, items):
    logger.debug("items %s", items)
    return items

# %%
if __name__ == "__main__":
    # Test content_attr_types transformation inline (not in a function)
    relnn_grammer_parser = get_relnn_grammar_parser(start="content_attr_types")
    input_text = "int, float, bool"
    tree = relnn_grammer_parser.parse(input_text)
    ptree_str = tree.pretty()
    logger.info("Test parse tree for content_attr_types:\n%s", ptree_str)
    result = transformer.transform(tree)
    print("Test Result:", result)
    assert result == ["int", "float", "bool"], f"Got: {result}"

# %%
@patch
@v_args(meta=True)
def signed_int(self: RelnnTransformer, meta, items):
    return items[0].value

# %%
if __name__ == "__main__":
    # Test signed_int transformation inline (not in a function)
    relnn_grammer_parser = get_relnn_grammar_parser(start="signed_int")
    for input_text in ["128", "-42", "+7"]:
        tree = relnn_grammer_parser.parse(input_text)
        result = transformer.transform(tree)
        print(f"Test Result (signed_int, input='{input_text}'): {result}")
        assert result == input_text, f"Expected '{input_text}', got: {result}"

# %%
@patch
@v_args(meta=True)
def embedding_dims(self: RelnnTransformer, meta, items):
    logger.debug("embedding_dims received items: %s", items)
    result = []
    for item in items:
        if isinstance(item, str) and item.isdigit():
            value = int(item)
        else:
            value = item
        result.append(value)
    logger.debug("extracted values: %s", result)
    return result

# %%
if __name__ == "__main__":
    # Test embedding_dims transformation inline (not in a function)
    relnn_grammer_parser = get_relnn_grammar_parser(start="embedding_dims")
    input_text = "128, d, 256"
    tree = relnn_grammer_parser.parse(input_text)
    ptree_str = tree.pretty()
    logger.info("Test parse tree for embedding_dims:\n%s", ptree_str)
    result = transformer.transform(tree)
    print("Test Result:", result)

    # Verify the result
    assert len(result) == 3, f"Expected 3 items, got {len(result)}"
    # First number should be an integer
    assert isinstance(result[0], int) and result[0] == 128, f"Expected int 128, got {type(result[0])} {result[0]}"
    # Middle item should be a Var
    assert isinstance(result[1], Var) and result[1].name == "d", f"Expected Var with name 'd', got {result[1]}"
    # Last number should be an integer
    assert isinstance(result[2], int) and result[2] == 256, f"Expected int 256, got {type(result[2])} {result[2]}"

# %%
@patch
@v_args(meta=True)
def er_schema(self: RelnnTransformer, meta, items):
    content_attr_types = items[0]
    embedding_dims = items[1] if len(items) > 1 else []
    return ERSchema(content_attr_types=content_attr_types, embedding_dims=embedding_dims)

# %%
if __name__ == "__main__":
    # Test er_schema transformation inline (not in a function)
    relnn_grammer_parser = get_relnn_grammar_parser(start="er_schema")

    # Test case 1: er_schema with content_attr_types and embedding_dims
    input_text = "(int, float; 128, 64)"
    tree = relnn_grammer_parser.parse(input_text)
    ptree_str = tree.pretty()
    logger.info("Test parse tree for er_schema:\n%s", ptree_str)
    result = transformer.transform(tree)
    print("Test Result (with dims):", result)

    # Verify the result
    assert isinstance(result, ERSchema), f"Expected ERSchema, got {type(result)}"
    # Check basic types
    assert result.content_attr_types == ["int", "float"], f"Unexpected content_attr_types: {result.content_attr_types}"
    # Check that dimensions are integers
    assert all(isinstance(d, int) for d in result.embedding_dims), f"Expected all integer dims, got: {result.embedding_dims}"
    assert result.embedding_dims == [128, 64], f"Unexpected embedding_dims: {result.embedding_dims}"

    # Test case 2: er_schema with only content_attr_types (no embedding_dims)
    input_text = "(int, bool, float)"
    tree = relnn_grammer_parser.parse(input_text)
    ptree_str = tree.pretty()
    logger.info("Test parse tree for er_schema (no dims):\n%s", ptree_str)
    result = transformer.transform(tree)
    print("Test Result (no dims):", result)

    # Verify the result
    assert isinstance(result, ERSchema), f"Expected ERSchema, got {type(result)}"
    # Check basic types
    assert result.content_attr_types == ["int", "bool", "float"], f"Unexpected content_attr_types: {result.content_attr_types}"
    assert result.embedding_dims == [], f"Expected empty embedding_dims list, got: {result.embedding_dims}"

# %%
@patch
@v_args(meta=True)
def number(self: RelnnTransformer, meta, items):
    val = items[0]
    if isinstance(val, Token):
        s = str(val)
        self._record_token(meta, "number", s)
        if "." in s or "e" in s.lower():
            return float(s)
        try:
            return int(s)
        except ValueError:
            return float(s)
    return val

@patch
@v_args(meta=True)
def string(self: RelnnTransformer, meta, items):
    val = items[0]
    if isinstance(val, Token) and val.type == "ESCAPED_STRING":
        return str(val)[1:-1]
    return str(val)

@patch
@v_args(meta=True)
def true(self: RelnnTransformer, meta, items):
    if self.collect_tokens:
        self._record_token(meta, "number", "True")
    return True

@patch
@v_args(meta=True)
def false(self: RelnnTransformer, meta, items):
    if self.collect_tokens:
        self._record_token(meta, "number", "False")
    return False

@patch
@v_args(meta=True)
def primitive(self: RelnnTransformer, meta, items):
    """Transform a primitive value to its Python representation."""
    val = items[0]
    # Guarantee bool is mapped to built-in True/False, not e.g. numpy.bool_
    if isinstance(val, bool):
        return bool(val)
    if isinstance(val, (int, float, str)):
        return val
    if isinstance(val, Token):
        return str(val)
    return val

# %%
if __name__ == "__main__":
    # Inline tests for primitive transformer
    relnn_grammar_parser = get_relnn_grammar_parser(start="primitive")

    # Int
    tree = relnn_grammar_parser.parse("42")
    assert transformer.transform(tree) == 42

    # Float
    tree = relnn_grammar_parser.parse("3.14")
    assert transformer.transform(tree) == 3.14

    # Bool True
    tree = relnn_grammar_parser.parse("True")
    assert transformer.transform(tree) is True

    # Bool False
    tree = relnn_grammar_parser.parse("False")
    assert transformer.transform(tree) is False

    # String
    tree = relnn_grammar_parser.parse('"hello"')
    assert transformer.transform(tree) == "hello"

# %%
@patch
@v_args(meta=True)
def template_param(self: RelnnTransformer, meta, items):
    logger.debug("template_param received items: %s", items)
    param = items[0]
    if self.collect_tokens:
        if hasattr(param, "name"):
            text = param.name
        elif isinstance(param, (int, float, str, bool)):
            text = str(param)
        else:
            text = None
        if text:
            self._record_token(meta, "template_param", text)
    logger.debug("template_param returning: %s", param)
    return param

@patch
@v_args(meta=True)
def template_params(self: RelnnTransformer, meta, items):
    logger.debug("template_params received items: %s", items)
    if self.collect_tokens:
        self._record_template_brackets(meta)
    params = [item for item in items if not isinstance(item, Token)]
    logger.debug("template_params returning: %s", params)
    return params

# %%
if __name__ == "__main__":
    # Test template_params and template_param transformation
    relnn_grammer_parser = get_relnn_grammar_parser(start="template_params")

    # Test case 1: Multiple variable parameters
    input_text = "<l, S, i>"
    tree = relnn_grammer_parser.parse(input_text)
    ptree_str = tree.pretty()
    logger.info("Test parse tree for template_params (vars):\n%s", ptree_str)
    result = transformer.transform(tree)
    print("Test Result (vars):", result)

    # Verify the result
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert len(result) == 3, f"Expected 3 parameters, got {len(result)}"
    assert all(isinstance(p, Var) for p in result), "All parameters should be Var objects"
    assert [p.name for p in result] == ["l", "S", "i"], f"Unexpected parameter names: {[p.name for p in result]}"

    # Test case 2: Mixed variables and primitives
    input_text = "<d, 64, T>"
    tree = relnn_grammer_parser.parse(input_text)
    ptree_str = tree.pretty()
    logger.info("Test parse tree for template_params (mixed):\n%s", ptree_str)
    result = transformer.transform(tree)
    print("Test Result (mixed):", result)

    # Verify the result
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert len(result) == 3, f"Expected 3 parameters, got {len(result)}"
    assert isinstance(result[0], Var) and result[0].name == "d", f"First param should be Var('d'), got {result[0]}"
    assert result[1] == 64, f"Second param should be '64', got {result[1]}"
    assert isinstance(result[2], Var) and result[2].name == "T", f"Third param should be Var('T'), got {result[2]}"

# %%
@patch
@v_args(meta=True)
def arith_term(self: RelnnTransformer, meta, items):
    """
    Build an ArithTerm from an arith_term tree or leaf.
    Always returns an ArithTerm.
    """
    _opmap = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "div": "/",
        "pow": "**",
        "neg": "-",  # unary minus
    }

    def _to_term(x):
        if isinstance(x, ArithTerm):
            return x
        if isinstance(x, Var):
            return ArithTerm(value=x)
        if isinstance(x, (int, float, bool, str)):
            return ArithTerm(value=x)
        from lark import Token
        if isinstance(x, Token):
            if x.type == "SIGNED_NUMBER" or x.type == "NUMBER":
                sval = str(x)
                if "." in sval or "e" in sval.lower():
                    return ArithTerm(value=float(sval))
                else:
                    return ArithTerm(value=int(sval))
            return ArithTerm(value=Var(name=str(x)))
        from lark.tree import Tree
        if isinstance(x, Tree):
            op = x.data
            if op == "number":
                tok = x.children[0]
                sval = str(tok)
                if "." in sval or "e" in sval.lower():
                    return ArithTerm(value=float(sval))
                else:
                    return ArithTerm(value=int(sval))
            if op == "string":
                tok = x.children[0]
                s = str(tok)
                return ArithTerm(value=s[1:-1] if len(s) > 1 and ((s[0] == "'" and s[-1] == "'") or (s[0] == '"' and s[-1] == '"')) else s)
            if op == "true":
                return ArithTerm(value=True)
            if op == "false":
                return ArithTerm(value=False)
            if op == "bool":
                return ArithTerm(value=bool(x.children[0]))
            if op == "var":
                v = x.children[0]
                if isinstance(v, Var):
                    return ArithTerm(value=v)
                else:
                    return ArithTerm(value=Var(name=str(v)))
            if op in _opmap:
                mapped_op = _opmap[op]
                children = [_to_term(child) for child in x.children]
                return ArithTerm(op=mapped_op, sons=children)
            if op == "arith_term":
                return _to_term(x.children[0])
        return ArithTerm(value=x)

    if len(items) == 1:
        return _to_term(items[0])
    op = getattr(meta, "rule_name", None)
    from lark.tree import Tree
    if not op and items and isinstance(items[0], Tree) and hasattr(items[0], "data"):
        if items[0].data in _opmap:
            op = items[0].data
    sons = [_to_term(it) for it in items]
    if op == "neg":
        return ArithTerm(op="-", sons=sons)
    if op in _opmap:
        return ArithTerm(op=_opmap[op], sons=sons)
    if len(sons) == 1:
        return sons[0]
    return ArithTerm(sons=sons)

# %%
if __name__ == "__main__":
    # Test arith_term transformation
    relnn_grammar_parser = get_relnn_grammar_parser(start="arith_term")

    # Test case 1: Single variable
    input_text = "x"
    tree = relnn_grammar_parser.parse(input_text)
    result = transformer.transform(tree)
    print("arith_term test (var):", result)
    assert isinstance(result, ArithTerm)
    assert isinstance(result.value, Var)
    assert result.value.name == "x"

    # Test case 2: Single number
    input_text = "42"
    tree = relnn_grammar_parser.parse(input_text)
    result = transformer.transform(tree)
    print("arith_term test (number):", result)
    assert isinstance(result, ArithTerm)
    assert result.value == 42.0 or result.value == 42  # Accept float or int

    # Test case 3: Addition of two variables
    input_text = "a + b"
    tree = relnn_grammar_parser.parse(input_text)
    result = transformer.transform(tree)
    print("arith_term test (add):", result)
    assert isinstance(result, ArithTerm)
    assert result.op == "+"
    assert all(isinstance(child, ArithTerm) for child in result.sons)

    # Test case 4: Negation
    input_text = "-y"
    tree = relnn_grammar_parser.parse(input_text)
    result = transformer.transform(tree)
    print("arith_term test (neg):", result)
    assert isinstance(result, ArithTerm)
    assert result.op == "-"

    # Test case 5: Parentheses and mixed operations
    input_text = "(a + b) * 3"
    tree = relnn_grammar_parser.parse(input_text)
    result = transformer.transform(tree)
    print("arith_term test (paren+mul):", result)
    assert isinstance(result, ArithTerm)
    assert result.op == "*"

# %%
@patch
@v_args(meta=True)
def template_args(self: RelnnTransformer, meta, items):
    logger.debug("template_args received items: %s", items)
    if self.collect_tokens:
        self._record_template_brackets(meta)
        if self.source_text:
            lines = self.source_text.split('\n')
            line_idx = meta.line - 1
            if 0 <= line_idx < len(lines):
                line_text = lines[line_idx]
                start = meta.column - 1
                end = (meta.end_column - 1) if hasattr(meta, 'end_column') and meta.end_line == meta.line else len(line_text)
                span = line_text[start:end]
                inner = span[1:-1] if len(span) >= 2 else span
                inner_offset = start + 1
                for part in re.split(r',', inner):
                    val = part.strip()
                    if val:
                        idx = inner.find(part, 0)
                        col = inner_offset + idx + (len(part) - len(part.lstrip()))
                        fm = type('M', (), {'line': meta.line, 'column': col + 1,
                                            'end_column': col + 1 + len(val)})()
                        self._record_token(fm, "template_param", val)
                        inner_offset_shift = idx + len(part)
                        inner = inner[inner_offset_shift:]
                        inner_offset += inner_offset_shift
    args = [item for item in items if not isinstance(item, Token)]
    logger.debug("template_args returning: %s", args)
    return args

# %%
if __name__ == "__main__":
    # Test template_args transformation
    relnn_grammer_parser = get_relnn_grammar_parser(start="template_args")

    # Test case 1: Simple variable arguments (but arithmetic expressions allowed!)
    input_text = "<l-1, S, i>"
    tree = relnn_grammer_parser.parse(input_text)
    ptree_str = tree.pretty()
    logger.info("Test parse tree for template_args (vars):\n%s", ptree_str)
    result = transformer.transform(tree)
    print("Test Result (vars):", result)
    # Verify the result
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert len(result) == 3, f"Expected 3 arguments, got {len(result)}"
    assert all(isinstance(arg, ArithTerm) for arg in result), "All arguments should be ArithTerm objects"
    # The second and third arguments should be Var-valued ArithTerms, the first is a subtraction ArithTerm
    assert result[0].op == "-", f"Expected '-' operation for first arg, got {result[0].op}"
    assert getattr(result[1].value, "name", None) == "S", f"Expected S var as second arg, got {result[1]}"
    assert getattr(result[2].value, "name", None) == "i", f"Expected i var as third arg, got {result[2]}"

    # Test case 2: Mixed arguments with arithmetic expressions
    input_text = "<l-1, T, 64>"
    tree = relnn_grammer_parser.parse(input_text)
    ptree_str = tree.pretty()
    logger.info("Test parse tree for template_args (mixed):\n%s", ptree_str)
    result = transformer.transform(tree)
    print("Test Result (mixed):", result)

    # Verify the result
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert len(result) == 3, f"Expected 3 arguments, got {len(result)}"
    # First arg should be a subtraction expression as ArithTerm
    assert isinstance(result[0], ArithTerm) and result[0].op == "-", f"First arg should be ArithTerm subtraction, got {result[0]}"
    # Second arg should be an ArithTerm wrapping Var('T')
    assert isinstance(result[1], ArithTerm) and getattr(result[1].value, "name", None) == "T", f"Second arg should be ArithTerm holding Var('T'), got {result[1]}"
    # Third arg should be an ArithTerm wrapping 64
    assert isinstance(result[2], ArithTerm) and (result[2].value == 64 or result[2].value == 64.0), f"Third arg should be ArithTerm holding 64, got {result[2]}"

# %%
@patch
@v_args(meta=True)
def er_param(self: RelnnTransformer, meta, items):
    name = items[0]
    if self.collect_tokens:
        self._record_token(meta, "er_name", str(name))
    er_schema = items[1] if len(items) > 1 else None
    return ErParam(name=name, er_schema=er_schema)

#| export
@patch
@v_args(meta=True)
def er_params(self: RelnnTransformer, meta, items):
    return items

# %%
if __name__ == "__main__":
    # Test er_param transformation inline (not in a function)
    relnn_grammar_parser = get_relnn_grammar_parser(start="er_param")

    # Test case 1: er_param with name and er_schema
    input_text = "a: (int, float; 32, 16)"
    tree = relnn_grammar_parser.parse(input_text)
    ptree_str = tree.pretty()
    logger.info("Test parse tree for er_param (with er_schema):\n%s", ptree_str)
    result = transformer.transform(tree)
    print("Test Result (with er_schema):", result)

    assert isinstance(result, ErParam), f"Expected ErParam, got {type(result)}"
    assert result.name == "a", f"Unexpected name: {result.name}"
    assert isinstance(result.er_schema, ERSchema), "er_schema should be ERSchema instance"
    assert result.er_schema.content_attr_types == ["int", "float"]
    assert result.er_schema.embedding_dims == [32, 16]

    # Test case 2: er_param with name only (no er_schema)
    input_text = "user_id"
    tree = relnn_grammar_parser.parse(input_text)
    ptree_str = tree.pretty()
    logger.info("Test parse tree for er_param (name only):\n%s", ptree_str)
    result = transformer.transform(tree)
    print("Test Result (no er_schema):", result)

    assert isinstance(result, ErParam), f"Expected ErParam, got {type(result)}"
    assert result.name == "user_id", f"Unexpected name: {result.name}"
    assert result.er_schema is None, "er_schema should be None when not provided"

# %%
@patch
@v_args(tree=True)
def hyper_param_list(self: RelnnTransformer, tree):
    """
    Transform rule: comma-separated list of constants (and possibly TensorOps) as hyperparameters.
    Uses tree=True so we can access tree.meta for the full span.
    NOTE: With tree=True, Lark pre-transforms children, so tree.children are
    already ArithTerm/Var/primitives (not raw Tree nodes).
    """
    values = [child for child in tree.children if not isinstance(child, Token)]
    if self.collect_tokens and self.source_text and hasattr(tree, "meta") and tree.meta is not None:
        span_meta = tree.meta
        lines = self.source_text.split('\n')
        line_idx = span_meta.line - 1
        if 0 <= line_idx < len(lines):
            line_text = lines[line_idx]
            start = span_meta.column - 1
            end = (span_meta.end_column - 1) if hasattr(span_meta, 'end_column') and span_meta.end_line == span_meta.line else len(line_text)
            all_var_names = set()
            for val in values:
                if isinstance(val, Var):
                    all_var_names.add(val.name)
                elif isinstance(val, ArithTerm):
                    for vname in self._collect_var_names_from_arithterm(val):
                        all_var_names.add(vname)
            for vname in all_var_names:
                for m in re.finditer(r'\b' + re.escape(vname) + r'\b', line_text[start:end]):
                    col = start + m.start()
                    fm = type('M', (), {'line': span_meta.line, 'column': col + 1,
                                        'end_column': col + 1 + len(vname)})()
                    self._record_token(fm, "hyper_param", vname)
            for op_str in ['**', '/', '+', '-', '*']:
                search_start = start
                while search_start < end:
                    idx = line_text.find(op_str, search_start, end)
                    if idx < 0:
                        break
                    if op_str == '*' and idx + 1 < len(line_text) and line_text[idx + 1] == '*':
                        search_start = idx + 2
                        continue
                    fm = type('M', (), {'line': span_meta.line, 'column': idx + 1,
                                        'end_column': idx + 1 + len(op_str)})()
                    self._record_token(fm, "torch_op", op_str)
                    search_start = idx + len(op_str)
    logger.debug("hyper_param_list received values: %s", values)
    return values

# %%
if __name__ == "__main__":
    # Inline tests for hyper_param_list
    relnn_grammar_parser = get_relnn_grammar_parser(start="hyper_param_list")

    # Test case 1: List of integer constants
    input_text = "1, 2, 3, 4"
    tree = relnn_grammar_parser.parse(input_text)
    ptree_str = tree.pretty()
    logger.info("Test parse tree for hyper_param_list (simple constants):\n%s", ptree_str)
    result = transformer.transform(tree)
    print("Test Result (simple constants):", result)

    # Continue test on: list of arithmetic expressions as hyperparameters
    input_text = "1 + 2, 3 * 4, x, y - 1"
    tree = relnn_grammar_parser.parse(input_text)
    ptree_str = tree.pretty()
    logger.info("Test parse tree for hyper_param_list (mixed arith and vars):\n%s", ptree_str)
    result = transformer.transform(tree)
    print("Test Result (mixed arith and vars):", result)

    # Continue test on: single value (should still be a list of length 1)
    input_text = "42"
    tree = relnn_grammar_parser.parse(input_text)
    ptree_str = tree.pretty()
    logger.info("Test parse tree for hyper_param_list (single value):\n%s", ptree_str)
    result = transformer.transform(tree)
    print("Test Result (single value):", result)

# %%
@patch
@v_args(meta=True)
def module_class(self: RelnnTransformer, meta, items):
    """Returns: String representing the module class name"""
    class_name = str(items[0])
    if self.collect_tokens:
        self._record_token(meta, "constructor", class_name)
    run_globals = self.engine.get_run_globals() if self.engine is not None else {}
    resolved = _resolve_op(class_name, run_globals)
    logger.debug("module_class: resolved '%s' => %r", class_name, resolved)
    if resolved is None:
        self.raise_or_save_diagnostic(
            f"Module class '{class_name}' is not found in run scope or torch.nn/torch. "
            "Import it (e.g. from torch.nn import Linear, ReLU) in the scope that calls session.run().",
            meta=meta)
    return class_name

# %%
if __name__ == "__main__":
    # Test parse tree and transformer for module_class rule
    # Setup the necessary objects if not already
    from lark.exceptions import VisitError

# %%
if __name__ == "__main__":
    # Test parse tree and transformer for module_class rule
    # Setup the necessary objects if not already

    engine = Engine()
    transformer = RelnnTransformer(engine=engine)
    relnn_grammar_parser = get_relnn_grammar_parser(start="module_class")

    # Test 1: Valid module class (registered/builtin)
    input_text = "Linear"
    tree = relnn_grammar_parser.parse(input_text)
    ptree_str = tree.pretty()
    logger.info("Test parse tree for module_class:\n%s", ptree_str)
    result = transformer.transform(tree)
    print("Test Result (module_class, valid):", result)
    assert result == "Linear", f"Expected 'Linear', got {result}"

    # Test 2: Invalid module class (not registered)
    input_text = "NonExistentModule"
    tree = relnn_grammar_parser.parse(input_text)
    try:
        result = transformer.transform(tree)
        # NOTE: The transformer raises RelNNSyntaxError inside the transformer rule,
        # but Lark wraps all exceptions in VisitError unless you catch VisitError.
        # So RelNNSyntaxError will be wrapped in a VisitError.
        # To properly handle this, we need to catch VisitError and unwrap.
        assert False, "Expected exception for non-existent module class"
    except Exception as e:
        # Lark wraps all exceptions thrown from the rule in VisitError
        if isinstance(e, VisitError):
            # The original exception can be accessed via e.orig_exc
            orig = e.orig_exc
            print("Caught VisitError for non-existent module_class.", str(orig))
            assert isinstance(orig, RelNNSyntaxError)
            assert "not found" in str(orig)
        elif isinstance(e, RelNNSyntaxError):
            print("Caught expected error for non-existent module_class:", str(e))
            assert "not found" in str(e)
        else:
            raise

# %%
@patch
@v_args(meta=True)
def module_instance(self: RelnnTransformer, meta, items):
    instance_name = str(items[0])
    if self.collect_tokens:
        self._record_token(meta, "instance", instance_name)
    if len(items) > 1 and items[1] is not None:
        return (instance_name, items[1])
    return instance_name

# %%
if __name__ == "__main__":
    # Real short test for module_instance rule
    relnn_grammar_parser = get_relnn_grammar_parser(start="module_instance")
    tree = relnn_grammar_parser.parse("foo")
    result = transformer.transform(tree)
    print("Short test:", result)
    assert result == "foo"

# %%
@patch
@v_args(meta=True)
def var_templated(self: RelnnTransformer, meta, items):
    name = items[0]
    template_args = items[1] if len(items) > 1 else []
    if template_args is None:
        template_args = []
    if self.collect_tokens:
        self._record_token(meta, "transform_ref", name if isinstance(name, str) else str(name))
    template_params = [arg.value for arg in template_args]
    var_templated_obj = VarTemplated(name=name, template_params=template_params)
    return TensorTerm(value=var_templated_obj)

@patch
@v_args(meta=True)
def tensor_term(self: RelnnTransformer, meta, items):
    logger.debug("tensor_term received items: %s", items)

    if len(items) == 0:
        return None

    # Grammar uses ?-inlining, so only 1 child expected
    if len(items) != 1:
        raise ValueError(
            f"tensor_term expected exactly 1 child (grammar uses ?-inlining), got {len(items)}: {items}"
        )
    child = items[0]
    if self.collect_tokens and isinstance(child, Var):
        self._record_token(meta, "transform_ref", child.name)
    if isinstance(child, TensorTerm):
        return child
    return TensorTerm(value=child)

# %%
if __name__ == "__main__":
    # Test tensor_term
    from typing import List, Union

# %%
def _make_tensor_binary_op(op_str, items):
    """Helper: build a TensorTerm for a binary tensor operation."""
    left, right = items[0], items[1]
    assert isinstance(left, TensorTerm) and isinstance(right, TensorTerm), \
        f"Binary op '{op_str}' expects TensorTerm operands, got {type(left).__name__}, {type(right).__name__}"
    return TensorTerm(op=TensorOp(op=op_str), sons=[left, right])

@patch
@v_args(meta=True)
def tensor_add(self: RelnnTransformer, meta, items):
    return _make_tensor_binary_op("+", items)

@patch
@v_args(meta=True)
def tensor_sub(self: RelnnTransformer, meta, items):
    return _make_tensor_binary_op("-", items)

@patch
@v_args(meta=True)
def tensor_mul(self: RelnnTransformer, meta, items):
    return _make_tensor_binary_op("*", items)

@patch
@v_args(meta=True)
def tensor_div(self: RelnnTransformer, meta, items):
    return _make_tensor_binary_op("/", items)

@patch
@v_args(meta=True)
def tensor_matmul(self: RelnnTransformer, meta, items):
    return _make_tensor_binary_op("@", items)

@patch
@v_args(meta=True)
def tensor_pow(self: RelnnTransformer, meta, items):
    return _make_tensor_binary_op("**", items)

@patch
@v_args(meta=True)
def tensor_eq_op(self: RelnnTransformer, meta, items):
    return _make_tensor_binary_op("==", items)

@patch
@v_args(meta=True)
def tensor_atom(self: RelnnTransformer, meta, items):
    """Leaf-level tensor_term: wraps raw values (var, number) in TensorTerm."""
    if len(items) == 0:
        return None
    if len(items) == 1:
        child = items[0]
        if isinstance(child, TensorTerm):
            return child
        return TensorTerm(value=child)
    return TensorTerm(sons=[
        TensorTerm(value=c) if not isinstance(c, TensorTerm) else c
        for c in items
    ])

# %%
@patch
@v_args(meta=True)
def encode_item(self: RelnnTransformer, meta, items):
    """Single item inside RHS ``[...]``: ``col``, ``Enc(col)``, or ``Enc(hps)(col)``."""
    from lark import Token
    flat = [x for x in items if not isinstance(x, Token)]
    if len(flat) == 1 and isinstance(flat[0], Var):
        return EncodeItem(column=flat[0])
    if len(flat) == 2:
        enc_name = str(flat[0])
        if self.collect_tokens:
            self._record_token(meta, "constructor", enc_name)
        return EncodeItem(column=flat[1], encoder_name=enc_name)
    if len(flat) == 3:
        enc_name = str(flat[0])
        if self.collect_tokens:
            self._record_token(meta, "constructor", enc_name)
        return EncodeItem(column=flat[2], encoder_name=enc_name, encoder_params=flat[1])
    raise RelNNSyntaxError(f"Invalid encode_item: {flat!r}")


@patch
@v_args(meta=True)
def content_encode(self: RelnnTransformer, meta, items):
    from lark import Token
    parts = [x for x in items if not isinstance(x, Token)]
    return ContentEncode(items=parts)


@patch
@v_args(meta=True)
def content_decode(self: RelnnTransformer, meta, items):
    """LHS bracket in predict rules: ``[var]``, ``[Dec(var)]``, ``[Dec(hps)(var)]``."""
    from lark import Token
    flat = [x for x in items if not isinstance(x, Token)]
    if len(flat) == 1 and isinstance(flat[0], Var):
        return ContentDecode(column=flat[0])
    if len(flat) == 2:
        dname = str(flat[0])
        if self.collect_tokens:
            self._record_token(meta, "constructor", dname)
        return ContentDecode(column=flat[1], decoder_name=dname)
    if len(flat) == 3:
        dname = str(flat[0])
        if self.collect_tokens:
            self._record_token(meta, "constructor", dname)
        return ContentDecode(column=flat[2], decoder_name=dname, decoder_params=flat[1])
    raise RelNNSyntaxError(f"Invalid content_decode: {flat!r}")

# %%
if __name__ == "__main__":
    # Test cases for tensor arithmetic operator precedence

    # Recreate transformer so it picks up tensor_add/sub/mul/etc. handlers
    # that were @patch-ed in the cell above (Lark caches handler lookup at init)
    engine = Engine()
    transformer = RelnnTransformer(engine=engine)

    # Helper for getting parser for an inline rule
    term_rule_parser = get_relnn_grammar_parser(start="tensor_term")

    # Test basic binary op: number + number
    input_text = "1 + 2"
    tree = term_rule_parser.parse(input_text)
    logger.info("Test parse tree for tensor_term (1 + 2):\n%s", tree.pretty())
    result = transformer.transform(tree)
    assert isinstance(result, TensorTerm)
    assert isinstance(result.op, TensorOp)
    assert result.op.op == "+"
    assert len(result.sons) == 2
    assert result.sons[0].value == 1
    assert result.sons[1].value == 2
    print("Test Result (tensor_term 1 + 2):", result)

    # Test nested binary op: (1 + 2) * 3
    input_text = "(1 + 2) * 3"
    tree = term_rule_parser.parse(input_text)
    logger.info("Test parse tree for tensor_term ((1 + 2) * 3):\n%s", tree.pretty())
    result = transformer.transform(tree)
    assert isinstance(result, TensorTerm)
    assert isinstance(result.op, TensorOp)
    assert result.op.op == "*"
    assert isinstance(result.sons[0], TensorTerm)
    assert isinstance(result.sons[0].op, TensorOp)
    assert result.sons[0].op.op == "+"
    assert result.sons[0].sons[0].value == 1
    assert result.sons[0].sons[1].value == 2
    assert result.sons[1].value == 3
    print("Test Result (tensor_term (1 + 2) * 3):", result)

    # Test with variable names: x - y
    input_text = "x - y"
    tree = term_rule_parser.parse(input_text)
    logger.info("Test parse tree for tensor_term (x - y):\n%s", tree.pretty())
    result = transformer.transform(tree)
    assert isinstance(result, TensorTerm)
    assert isinstance(result.op, TensorOp)
    assert result.op.op == "-"
    assert isinstance(result.sons[0].value, Var)
    assert result.sons[0].value.name == "x"
    assert isinstance(result.sons[1].value, Var)
    assert result.sons[1].value.name == "y"
    print("Test Result (tensor_term x - y):", result)

    # Test with @ operator: a @ b
    input_text = "a @ b"
    tree = term_rule_parser.parse(input_text)
    logger.info("Test parse tree for tensor_term (a @ b):\n%s", tree.pretty())
    result = transformer.transform(tree)
    assert isinstance(result, TensorTerm)
    assert isinstance(result.op, TensorOp)
    assert result.op.op == "@"
    assert result.sons[0].value.name == "a"
    assert result.sons[1].value.name == "b"
    print("Test Result (tensor_term a @ b):", result)

    # --- Atom/leaf tests (moved from earlier cell to run after tensor_atom handler) ---

    # Variable as tensor_atom
    input_text = "x"
    tree = term_rule_parser.parse(input_text)
    result = transformer.transform(tree)
    assert isinstance(result, TensorTerm), f"Expected TensorTerm, got {type(result)}"
    assert result.value is not None
    assert result.op is None
    assert result.sons is None
    print("Test Result (tensor_atom var):", result)

    # Number as tensor_atom
    input_text = "42"
    tree = term_rule_parser.parse(input_text)
    result = transformer.transform(tree)
    assert isinstance(result, TensorTerm)
    assert result.value == 42 or result.value == 42.0
    assert result.op is None
    assert result.sons is None
    print("Test Result (tensor_atom number):", result)

    # VarTemplated (e.g., "Foo<64, x>")
    input_text = "Foo<64, x>"
    tree = term_rule_parser.parse(input_text)
    result = transformer.transform(tree)
    assert isinstance(result, TensorTerm), f"Expected TensorTerm, got {type(result)}"
    assert isinstance(result.value, VarTemplated)
    assert result.value.name == "Foo"
    assert result.value.template_params[0] == 64
    assert isinstance(result.value.template_params[1], Var)
    assert result.value.template_params[1].name == "x"
    print("Test Result (tensor_atom var_templated):", result)

# %%
@patch
@v_args(meta=True)
def module_arguments(self: RelnnTransformer, meta, items):
    logger.debug("module_arguments received items: %s", items)
    return items

# %%
@patch
@v_args(meta=True)
def module_instance_call(self: RelnnTransformer, meta, children):
    logger.debug("module_instance_call received children: %s", children)
    module_instance = children[0]
    module_args = children[1] if len(children) > 1 else []
    if self.collect_tokens and module_args:
        for arg in module_args:
            for vname in self._collect_vars_from_tensor_term(arg):
                self._record_var_in_source(meta, vname, "embedding_expr")
    name = module_instance[0] if isinstance(module_instance, tuple) else module_instance
    template_args = module_instance[1] if isinstance(module_instance, tuple) else None
    return TensorTerm(
        op=TensorOp(op=name, template_args=template_args),
        sons=module_args or None,
    )

# %%
if __name__ == "__main__":
    # Get a parser starting from "module_instance_call"
    relnn_grammar_parser = get_relnn_grammar_parser(start="module_call")

    # Case: FooInstance(x, y)
    input_text = "FooInstance(x, y)"
    tree = relnn_grammar_parser.parse(input_text)
    ptree_str = tree.pretty()
    logger.info("Test parse tree for module_instance_call:\n%s", ptree_str)
    result = transformer.transform(tree)

    print("Test Result (module_instance_call):", result)
    assert isinstance(result, TensorTerm)
    assert isinstance(result.op, TensorOp)
    # The module_instance is now the .op.op (the function), the sons are args (list of TensorTerms)
    assert result.op.op == "FooInstance" or (hasattr(result.op.op, "value") and result.op.op.value == "FooInstance")
    sons = result.sons

    # Sons should be [TensorTerm(x), TensorTerm(y)]
    assert len(sons) == 2
    assert isinstance(sons[0], TensorTerm)
    assert isinstance(sons[0].value, Var)
    assert sons[0].value.name == "x"
    assert isinstance(sons[1], TensorTerm)
    assert isinstance(sons[1].value, Var)
    assert sons[1].value.name == "y"

    # Additional case: Bar(z)
    input_text = "Bar(z)"
    tree = relnn_grammar_parser.parse(input_text)
    result = transformer.transform(tree)
    assert isinstance(result, TensorTerm)
    assert result.op.op == "Bar" or (hasattr(result.op.op, "value") and result.op.op.value == "Bar")
    assert len(result.sons) == 1
    assert isinstance(result.sons[0], TensorTerm)
    assert isinstance(result.sons[0].value, Var)
    assert result.sons[0].value.name == "z"
    print("Test Result (module_instance_call, Bar):", result)

# %%
@patch
@v_args(meta=True)
def module_ctor_call(self: RelnnTransformer, meta, children):
    if len(children) != 3:
        raise ValueError(f"module_ctor_call: Expected exactly 3 children, got {len(children)}: {children}")
    logger.debug("module_ctor_call received children: %s", children)
    module_class = children[0]
    hyper_params = children[1]
    module_args = children[2]
    if self.collect_tokens and module_args:
        for arg in module_args:
            for vname in self._collect_vars_from_tensor_term(arg):
                self._record_var_in_source(meta, vname, "embedding_expr")
    return TensorTerm(
        op=TensorOp(op=module_class, hyper_params=hyper_params),
        sons=module_args
    )

# %%
if __name__ == "__main__":
    # ---- Test cases for module_ctor_call ----

    # Get a parser starting from "module_ctor_call"
    relnn_grammar_parser_ctor = get_relnn_grammar_parser(start="module_call")

    # Linear(128, 32)(x)
    input_text = "Linear(128, 32)(x)"
    tree = relnn_grammar_parser_ctor.parse(input_text)
    ptree_str = tree.pretty()
    logger.info("Test parse tree for module_ctor_call:\n%s", ptree_str)
    result = transformer.transform(tree)

    print("Test Result (module_ctor_call):", result)
    assert isinstance(result, TensorTerm)
    assert isinstance(result.op, TensorOp)
    assert (result.op.op == "Linear" or (hasattr(result.op.op, "value") and result.op.op.value == "Linear"))
    assert result.op.hyper_params is not None
    # Should be two hyper params: 128, 32
    assert len(result.op.hyper_params) == 2
    hp0, hp1 = result.op.hyper_params
    if isinstance(hp0, BaseModel):
        assert getattr(hp0, "value", None) == 128
    else:
        assert hp0 == 128
    if isinstance(hp1, BaseModel):
        assert getattr(hp1, "value", None) == 32
    else:
        assert hp1 == 32
    # Should be one argument "x" as a TensorTerm with value Var(name='x')
    assert result.sons is not None and len(result.sons) == 1
    arg0 = result.sons[0]
    assert isinstance(arg0, TensorTerm)
    assert isinstance(arg0.value, Var)
    assert arg0.value.name == "x"

    # Another test: Linear(7)(Concat(x, y)) â€” use explicit Concat for multiple tensors
    input_text = "Linear(7)(Concat(x, y))"
    tree = relnn_grammar_parser_ctor.parse(input_text)
    result = transformer.transform(tree)
    assert isinstance(result, TensorTerm)
    assert result.op.op == "Linear" or (hasattr(result.op.op, "value") and result.op.op.value == "Linear")
    assert result.op.hyper_params is not None
    hp = result.op.hyper_params[0]
    if isinstance(hp, BaseModel):
        assert getattr(hp, "value", None) == 7
    else:
        assert hp == 7
    assert len(result.sons) == 1
    concat_child = result.sons[0]
    assert isinstance(concat_child, TensorTerm) and concat_child.op.op == "Concat"
    assert len(concat_child.sons) == 2
    assert all(isinstance(s.value, Var) for s in concat_child.sons)
    names = [s.value.name for s in concat_child.sons]
    assert names == ["x", "y"]
    print("Test Result (module_ctor_call, Linear with Concat):", result)

# %%
@patch
@v_args(meta=True)
def tensor_sequential(self: RelnnTransformer, meta, children):
    logger.debug("tensor_sequential received children: %s", children)
    if len(children) == 0:
        return None
    if len(children) != 1:
        raise ValueError(f"tensor_sequential: Only single tensor_term supported currently (got {len(children)})")
    # TODO: add support for multiple tensor_terms in a tensor_sequential
    return children[0]

# %%
if __name__ == "__main__":
    # ---- Inline test cases for tensor_sequential ----
    # Only one argument allowed
    relnn_grammar_parser = get_relnn_grammar_parser(start="tensor_sequential")
    transformer = RelnnTransformer(engine=engine)
    input_text = "x"
    tree = relnn_grammar_parser.parse(input_text)
    ptree_str = tree.pretty()
    logger.info("Test parse tree for tensor_sequential:\n%s", ptree_str)
    result = transformer.transform(tree)
    # Should return Var(name='x') or a TensorTerm containing Var
    assert hasattr(result, "name") and result.name == "x" or (hasattr(result, "value") and getattr(result.value, "name", None) == "x")
    print("Test Result (tensor_sequential, one):", result)

# %%
@patch
@v_args(meta=True)
def aggregation_fn(self: RelnnTransformer, meta, children):
    logger.debug("aggregation_fn called with meta: %s, children: %s", meta, children)
    if len(children) != 1:
        raise ValueError(f"aggregation_fn expected 1 child, got {len(children)}")
    agg_name = str(children[0])
    if self.collect_tokens:
        self._record_token(meta, "embedding_agg", agg_name)
    valid_aggs = ["min", "max", "add", "sum", "mean", "avg", "count"]
    logger.debug("aggregation_fn received: %s", agg_name)
    if agg_name not in valid_aggs:
        self.raise_or_save_diagnostic(f"Invalid aggregation function: {agg_name}", meta)
    return agg_name

@patch
@v_args(meta=True)
def embedding_expression_with_agg(self: RelnnTransformer, meta, items):
    logger.debug("embedding_expression_with_agg received items: %s", items)
    if len(items) == 0:
        return EmbeddingExpression(aggregation_fn=None, tensor_term=None)
    self._record_token(meta, "embedding_expr", skip_pattern=r';\s*')
    agg_name, tensor_term = items
    if self.collect_tokens and tensor_term is not None:
        for vname in self._collect_vars_from_tensor_term(tensor_term):
            self._record_var_in_source(meta, vname, "embedding_expr")
    return EmbeddingExpression(
        aggregation_fn=agg_name,
        tensor_term=tensor_term
    )

@patch
@v_args(meta=True)
def embedding_expression_default_agg(self: RelnnTransformer, meta, items):
    logger.debug("embedding_expression_default_agg received items: %s", items)
    if len(items) == 0:
        return EmbeddingExpression(aggregation_fn=None, tensor_term=None)
    self._record_token(meta, "embedding_expr", skip_pattern=r';\s*')
    tensor_term = items[0]
    if self.collect_tokens and tensor_term is not None:
        for vname in self._collect_vars_from_tensor_term(tensor_term):
            self._record_var_in_source(meta, vname, "embedding_expr")
    # Safety net: if Earley picked the wrong parse and the root tensor_term is
    # an aggregation function call (e.g. sum(z*w) parsed as module_instance_call),
    # extract the aggregation and use the inner term.
    if isinstance(tensor_term, TensorTerm) and isinstance(getattr(tensor_term, 'op', None), TensorOp):
        if (tensor_term.op.op in ALLOWED_AGGREGATIONS
                and tensor_term.sons and len(tensor_term.sons) == 1):
            logger.debug("embedding_expression_default_agg: extracting aggregation '%s' from tensor_term", tensor_term.op.op)
            return EmbeddingExpression(
                aggregation_fn=tensor_term.op.op,
                tensor_term=tensor_term.sons[0]
            )
    return EmbeddingExpression(tensor_term=tensor_term)

@patch
@v_args(meta=True)
def embedding_expression_empty(self: RelnnTransformer, meta, items):
    logger.debug("embedding_expression_empty called")
    return EmbeddingExpression(aggregation_fn=None, tensor_term=None)

# %%
if __name__ == "__main__":
    # Test embedding_expression transformations
    relnn_grammar_parser = get_relnn_grammar_parser(start="embedding_expression")

    # Test case 1: Expression with aggregation
    input_text = ";sum(x)"
    tree = relnn_grammar_parser.parse(input_text)
    ptree_str = tree.pretty()
    logger.info("Test parse tree for embedding_expression (with agg):\n%s", ptree_str)
    result = transformer.transform(tree)
    print("Test Result (with agg):", result)

    # Verify the result
    assert isinstance(result, EmbeddingExpression), f"Expected EmbeddingExpression, got {type(result)}"
    assert result.aggregation_fn == "sum", f"Expected 'sum' aggregation, got {result.aggregation_fn}"
    assert isinstance(result.tensor_term, TensorTerm), f"Expected TensorTerm, got {type(result.tensor_term)}"
    assert isinstance(result.tensor_term.value, Var), f"Expected Var value, got {type(result.tensor_term.value)}"
    assert result.tensor_term.value.name == "x", f"Expected variable name 'x', got {result.tensor_term.value.name}"

    # Test case 2: Expression without aggregation
    input_text = ";x"
    tree = relnn_grammar_parser.parse(input_text)
    ptree_str = tree.pretty()
    logger.info("Test parse tree for embedding_expression (no agg):\n%s", ptree_str)
    result = transformer.transform(tree)
    print("Test Result (no agg):", result)

    # Verify the result
    assert isinstance(result, EmbeddingExpression)
    assert result.aggregation_fn is None, f"Expected no aggregation, got {result.aggregation_fn}"
    assert isinstance(result.tensor_term, TensorTerm)
    assert isinstance(result.tensor_term.value, Var) and result.tensor_term.value.name == "x"

# %%
@patch
@v_args(meta=True)
def derived_content_attrs(self: RelnnTransformer, meta, items):
    logger.debug("derived_content_attrs received items: %s", items)
    if self.collect_tokens:
        for item in items:
            if isinstance(item, Var):
                self._record_var_in_source(meta, item.name, "content_attr")
    return list(items)

# %%
@patch
@v_args(meta=True)
def derived_er_def(self: RelnnTransformer, meta, items):
    """Transform a derived ER definition into a DerivedER object."""
    logger.debug("derived_er_def received items: %s", items)
    name = items[0]

    if isinstance(name, str):
        self._record_token(meta, "er_name", name)

    if len(items) == 4:
        template_params = items[1]
        derived_content_attrs = items[2]
        embedding_expression = items[3]
    else:
        template_params = None
        derived_content_attrs = items[1]
        embedding_expression = items[2]

    return DerivedER(
        name=name,
        template_params=template_params,
        derived_content_attrs=derived_content_attrs,
        embedding_expression=embedding_expression
    )

# %%
if __name__ == "__main__":
    # Test derived_er_def transformation
    relnn_grammar_parser = get_relnn_grammar_parser(start="derived_er_def")

    # Test case 1: Simple derived ER without template params (aggregation: sum)
    input_text = "Output(x; sum(y))"
    tree = relnn_grammar_parser.parse(input_text)
    ptree_str = tree.pretty()
    logger.info("Test parse tree for derived_er_def (simple):\n%s", ptree_str)
    result = transformer.transform(tree)
    print("Test Result (simple):", result)

    # Verify the result
    assert isinstance(result, DerivedER), f"Expected DerivedER, got {type(result)}"
    assert result.name == "Output", f"Expected name 'Output', got {result.name}"
    assert result.template_params is None, "Expected no template params"
    assert len(result.derived_content_attrs) == 1, f"Expected 1 content attr, got {len(result.derived_content_attrs)}"
    assert isinstance(result.derived_content_attrs[0], Var)
    assert result.derived_content_attrs[0].name == "x"
    assert isinstance(result.embedding_expression, EmbeddingExpression)
    assert result.embedding_expression.aggregation_fn == "sum"
    assert isinstance(result.embedding_expression.tensor_term, TensorTerm)
    assert result.embedding_expression.tensor_term.value.name == "y"

    # Test case 2: Derived ER with aggregation function avg
    input_text_avg = "Output(x; avg(y))"
    tree_avg = relnn_grammar_parser.parse(input_text_avg)
    ptree_str_avg = tree_avg.pretty()
    logger.info("Test parse tree for derived_er_def (avg):\n%s", ptree_str_avg)
    result_avg = transformer.transform(tree_avg)
    print("Test Result (avg):", result_avg)

    # Verify the avg result
    assert isinstance(result_avg, DerivedER), f"Expected DerivedER, got {type(result_avg)}"
    assert result_avg.name == "Output", f"Expected name 'Output', got {result_avg.name}"
    assert result_avg.template_params is None, "Expected no template params"
    assert len(result_avg.derived_content_attrs) == 1, f"Expected 1 content attr, got {len(result_avg.derived_content_attrs)}"
    assert isinstance(result_avg.derived_content_attrs[0], Var)
    assert result_avg.derived_content_attrs[0].name == "x"
    assert isinstance(result_avg.embedding_expression, EmbeddingExpression)
    assert result_avg.embedding_expression.aggregation_fn == "avg"
    assert isinstance(result_avg.embedding_expression.tensor_term, TensorTerm)
    assert result_avg.embedding_expression.tensor_term.value.name == "y"

# %%
@patch
@v_args(meta=True)
def embedding_var(self: RelnnTransformer, meta, items):
    if not items:
        return None
    var_or_number = items[0]
    if isinstance(var_or_number, Var):
        self._record_token(meta, "embedding_expr", var_or_number.name, skip_pattern=r';\s*')
    elif isinstance(var_or_number, (int, float)):
        self._record_token(meta, "embedding_expr", str(var_or_number), skip_pattern=r';\s*')
    return var_or_number

# %%
@patch
@v_args(meta=True)
def content_attr(self: RelnnTransformer, meta, items):
    if isinstance(items[0], Var):
        var = items[0]
        self._record_token(meta, "content_attr", var.name)
        return var
    val = items[0]
    if isinstance(val, (int, float)):
        return val
    # String primitives (e.g. 'Author') used in bounding relations.
    # Wrap in Var so the engine can distinguish bound vs free positions.
    if isinstance(val, str):
        return Var(name=val)
    return Var(name=str(val))

@patch
@v_args(meta=True)
def content_attrs(self: RelnnTransformer, meta, items):
    return items

# %%
@patch
@v_args(meta=True)
def er_ref(self: RelnnTransformer, meta, items):
    name = items[0]
    if isinstance(name, str):
        self._record_token(meta, "er_name", name)
    template_args = items[1] if len(items) > 1 else None
    if template_args == []:
        template_args = None
    return ERRef(name=name, template_args=template_args)

@patch
@v_args(meta=True)
def arguments(self: RelnnTransformer, meta, items):
    return items

# %%
@patch
@v_args(meta=True)
def er_instance(self: RelnnTransformer, meta, items):
    name = items[0]
    if isinstance(name, str):
        self._record_token(meta, "er_name", name)
    if len(items) == 4:
        template_args = items[1]
        content_attrs = items[2]
        embedding_var = items[3]
    elif len(items) == 3:
        template_args = None
        content_attrs = items[1]
        embedding_var = items[2]
    else:
        raise ValueError(f"Unexpected number of items in er_instance(): {items}")
    return EmbeddedRelation(
        name=name,
        template_args=template_args,
        arguments=None,
        content_attrs=content_attrs,
        embedding_var=embedding_var
    )

@patch
@v_args(meta=True)
def function_call(self: RelnnTransformer, meta, items):
    name = items[0]
    if self.collect_tokens:
        self._record_var_in_source(meta, str(name), "function_name")
    if len(items) == 5:
        template_args = items[1]
        arguments = items[2]
        content_attrs = items[3]
        embedding_var = items[4]
    elif len(items) == 4:
        template_args = None
        arguments = items[1]
        content_attrs = items[2]
        embedding_var = items[3]
    else:
        raise ValueError(f"Unexpected number of items in function_call(): {items}")
    return EmbeddedRelation(
        name=name,
        template_args=template_args,
        arguments=arguments,
        content_attrs=content_attrs,
        embedding_var=embedding_var
    )

# %%
if __name__ == "__main__":
    # Test cases for 'er' rule (er_instance and function_call)

    # Helper for getting parser for the 'er' rule
    er_rule_parser = get_relnn_grammar_parser(start="er")

    # --- Test simple er_instance ---
    input_text = "Foo(x; y)"
    tree = er_rule_parser.parse(input_text)
    logger.info("Test parse tree for er_instance (simple):\n%s", tree.pretty())
    result = transformer.transform(tree)
    assert isinstance(result, EmbeddedRelation)
    assert result.name == "Foo"
    assert result.template_args is None
    assert result.arguments is None
    assert isinstance(result.content_attrs, list)
    assert result.embedding_var is not None
    print("Test Result (er_instance, simple):", result)

    # --- Test er_instance with template_args ---
    input_text = "Bar<T>(a; b)"
    tree = er_rule_parser.parse(input_text)
    logger.info("Test parse tree for er_instance (with template_args):\n%s", tree.pretty())
    result = transformer.transform(tree)
    assert isinstance(result, EmbeddedRelation)
    assert result.name == "Bar"
    assert result.template_args is not None
    assert result.arguments is None
    assert isinstance(result.content_attrs, list)
    assert result.embedding_var is not None
    print("Test Result (er_instance, with template_args):", result)

    # --- Test function_call form ---
    input_text = "Fun(a)(x; y)"
    tree = er_rule_parser.parse(input_text)
    logger.info("Test parse tree for function_call form:\n%s", tree.pretty())
    result = transformer.transform(tree)
    assert isinstance(result, EmbeddedRelation)
    assert result.name == "Fun"
    assert result.arguments is not None
    assert isinstance(result.content_attrs, list)
    assert result.embedding_var is not None
    print("Test Result (function_call):", result)

    # --- Test function_call with template_args ---
    input_text = "Baz<T>(a)(x; y)"
    tree = er_rule_parser.parse(input_text)
    logger.info("Test parse tree for function_call with template_args:\n%s", tree.pretty())
    result = transformer.transform(tree)
    assert isinstance(result, EmbeddedRelation)
    assert result.name == "Baz"
    assert result.template_args is not None
    assert result.arguments is not None
    assert isinstance(result.content_attrs, list)
    assert result.embedding_var is not None
    print("Test Result (function_call with template_args):", result)

# %%
@patch
@v_args(meta=True)
def rel_op(self: RelnnTransformer, meta, items):
    op = items[0]
    logger.debug("rel_op received: %s", op)
    return str(op)

# %%
@patch
@v_args(meta=True)
def comp_op(self: RelnnTransformer, meta, items):
    op = str(items[0])
    logger.debug("comp_op received: %s", op)
    return op

# %%
@patch
@v_args(meta=True)
def comparison_expression(self: RelnnTransformer, meta, items):
    left = items[0]
    comp_op = items[1]
    right = items[2]
    logger.debug("comparison_expression received: left=%s, comp_op=%s, right=%s", left, comp_op, right)
    return ComparisonExpression(lhs=left, comp_op=comp_op, rhs=right)

# %%
@patch
@v_args(meta=True)
def filter_expressions(self: RelnnTransformer, meta, items):
    logger.debug("filter_expressions received items: %s", items)
    return list(items)

# %%
if __name__ == "__main__":
    # Inline tests for filter_expressions
    relnn_grammar_parser = get_relnn_grammar_parser(start="filter_expressions")

    # Test case 1: Single comparison expression
    input_text = ", x > 1"
    tree = relnn_grammar_parser.parse(input_text)
    transformer = RelnnTransformer({}, {})
    result = transformer.transform(tree)
    print("filter_expressions test (single):", result)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].comp_op == ">"

    # Test case 2: Multiple expressions separated by commas
    input_text = ", x > 1, y < 2"
    tree = relnn_grammar_parser.parse(input_text)
    result = transformer.transform(tree)
    print("filter_expressions test (multiple comma):", result)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].comp_op == ">"
    assert result[1].comp_op == "<"

    # Test case 3: Multiple expressions separated by |
    input_text = ", a == b, c != d"
    tree = relnn_grammar_parser.parse(input_text)
    result = transformer.transform(tree)
    print("filter_expressions test (multiple pipe):", result)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].comp_op == "=="
    assert result[1].comp_op == "!="

    # Test case 4: Mixed operator types in sequence
    input_text = ",x - 1 >= 1, y == 5, z < 0"
    tree = relnn_grammar_parser.parse(input_text)
    result = transformer.transform(tree)
    print("filter_expressions test (mixed sep):", result)
    assert isinstance(result, list)
    assert len(result) == 3
    # Now the first expression is 'x - 1 >= 1'
    assert result[0].lhs.value is None
    assert result[0].lhs.op == '-'  # the left is x-1
    assert result[0].lhs.sons[0].value.name == "x"
    assert result[0].lhs.sons[1].value == 1
    assert result[0].comp_op == ">="
    assert result[0].rhs.value == 1
    assert result[1].comp_op == "=="
    assert result[1].lhs.value.name == "y"
    assert result[1].rhs.value == 5
    assert result[2].comp_op == "<"
    assert result[2].lhs.value.name == "z"
    assert result[2].rhs.value == 0

    # Test case 5: Empty input should yield empty list
    input_text = ""
    tree = relnn_grammar_parser.parse(input_text)
    result = transformer.transform(tree)
    print("filter_expressions test (empty):", result)
    assert isinstance(result, list)
    assert len(result) == 0

# %%
@patch
@v_args(meta=True)
def rhs(self: RelnnTransformer, meta, items):
    if len(items) == 0:
        self.raise_or_save_diagnostic("rhs: no items parsed", meta=meta)

    *relations_and_ops, filter_expressions = items
    embedded_relations = [item for i, item in enumerate(relations_and_ops) if i % 2 == 0]
    relation_ops = [item for i, item in enumerate(relations_and_ops) if i % 2 == 1]
    rel_ops_list = [str(op) for op in relation_ops] if relation_ops else None
    filter_exprs = filter_expressions if isinstance(filter_expressions, list) else []
    return RHS(ers=embedded_relations, rel_ops=rel_ops_list, filter_expressions=filter_exprs)

# %%
@patch
@v_args(meta=True)
def regular_rule(self: RelnnTransformer, meta, items):
    lhs, rhs = items
    return Rule(lhs=lhs, rhs=rhs)

@patch
@v_args(meta=True)
def splat(self: RelnnTransformer, meta, items):
    """Transform splat: "*" tensor_atom into a TensorTerm with op='splat'."""
    if self.collect_tokens and self.source_text:
        lines = self.source_text.split('\n')
        line_idx = meta.line - 1
        if 0 <= line_idx < len(lines):
            line_text = lines[line_idx]
            start = meta.column - 1
            idx = line_text.find('*', start)
            if idx >= 0:
                fm = type('M', (), {'line': meta.line, 'column': idx + 1, 'end_column': idx + 2})()
                self._record_token(fm, "torch_op", "*")
    inner = items[0]
    if not isinstance(inner, TensorTerm):
        inner = TensorTerm(value=inner)
    return TensorTerm(op=TensorOp(op="splat"), sons=[inner])

@patch
@v_args(meta=True)
def condition_only_bounding(self: RelnnTransformer, meta, items):
    conditions = [item for item in items if isinstance(item, ComparisonExpression)]
    logger.debug("condition_only_bounding: %d conditions", len(conditions))
    return ([], conditions)

@patch
@v_args(meta=True)
def bounding_condition(self: RelnnTransformer, meta, items):
    logger.debug("bounding_condition received items: %s", items)
    return list(items)

@patch
@v_args(meta=True)
def bounding(self: RelnnTransformer, meta, items):
    *ers_raw, conditions = items
    bounding_ers = [e for e in ers_raw if isinstance(e, EmbeddedRelation)]
    bounding_conditions = conditions if isinstance(conditions, list) else []
    logger.debug("bounding: %d ers, %d conditions", len(bounding_ers), len(bounding_conditions))
    return (bounding_ers, bounding_conditions)

@patch
@v_args(meta=True)
def bounded_rhs(self: RelnnTransformer, meta, items):
    rel_op_name = str(items[0])
    main_er = items[1]
    bounding_ers, bounding_conditions = items[2]
    if self.collect_tokens:
        for kw in [rel_op_name, "Set"]:
            self._record_var_in_source(meta, kw, "keyword")
        if self.source_text:
            lines = self.source_text.split('\n')
            line_idx = meta.line - 1
            if 0 <= line_idx < len(lines):
                line_text = lines[line_idx]
                start = meta.column - 1
                end = (meta.end_column - 1) if hasattr(meta, 'end_column') and meta.end_line == meta.line else len(line_text)
                idx = line_text.find('|', start, end)
                if idx >= 0:
                    fm = type('M', (), {'line': meta.line, 'column': idx + 1, 'end_column': idx + 2})()
                    self._record_token(fm, "keyword", "|")
    logger.debug("bounded_rhs: op=%s, main_er=%s, %d bounding_ers",
                 rel_op_name, main_er.name, len(bounding_ers))
    return BoundedRHS(
        rel_op_name=rel_op_name,
        main_er=main_er,
        bounding_ers=bounding_ers,
        bounding_conditions=bounding_conditions,
    )

@patch
@v_args(meta=True)
def bounded_rule(self: RelnnTransformer, meta, items):
    lhs, brhs = items
    return Rule(lhs=lhs, rhs=brhs)

# %%
if __name__ == "__main__":
    # Test rule transformation inline, with only variable arguments (no terminal/string constants)
    relnn_grammer_parser = get_relnn_grammar_parser(start="rule")
    # input_text = "MyRule(a; x) :- AnotherRel(a; x), Split(a, y)."
    input_text = "MyRule(a; Linear(x) + x) :- AnotherRel(a; x), Split(a, y)."
    tree = relnn_grammer_parser.parse(input_text)
    ptree_str = tree.pretty()
    logger.info("Test parse tree for rule (only vars):\n%s", ptree_str)
    result = transformer.transform(tree)
    print("Test Result (rule, only vars):", result)

    # Verify the result
    assert isinstance(result, Rule), f"Expected Rule, got {type(result)}"
    assert hasattr(result, "lhs")
    assert hasattr(result, "rhs")
    assert result.lhs is not None, "lhs should be non-None"
    assert result.rhs is not None, "rhs should be non-None"

# %%
@patch
@v_args(meta=True)
def transform_def(self: RelnnTransformer, meta, items):
    """Grammar: name template_params? "=" tensor_term "." """
    if len(items) == 3:
        name, template_params, tensor_term = items
    elif len(items) == 2:
        name, tensor_term = items
        template_params = None
    else:
        raise ValueError(f"transform_def: unexpected items {items}")

    if isinstance(name, str):
        self._record_token(meta, "er_name", name)

    return TransformDef(
        name=name,
        template_params=template_params,
        tensor_term=tensor_term,
    )

# %%
if __name__ == "__main__":

    # Test case 3: TransformDef with module constructor call in tensor term
    input_text = "Lin = Linear(16,32) ."
    relnn_grammar_parser = get_relnn_grammar_parser(start="transform_def")
    tree = relnn_grammar_parser.parse(input_text)
    ptree_str = tree.pretty()
    logger.info("Test parse tree for transform_def_stmt (module ctor):\n%s", ptree_str)
    result = transformer.transform(tree)
    print("Test Result (transform_def_stmt, module ctor):", result)

    assert isinstance(result, TransformDef), f"Expected TransformDef, got {type(result)}"
    assert result.name == "Lin"
    assert result.template_params is None or result.template_params == []
    assert hasattr(result.tensor_term, "op"), "tensor_term.op missing"
    assert hasattr(result.tensor_term.op, "op"), "tensor_term.op.op missing"
    assert result.tensor_term.op.op == "Linear", f"Expected tensor_term.op.op == 'Linear', got {result.tensor_term.op.op}"
    assert hasattr(result.tensor_term.op, "hyper_params"), "tensor_term.op.hyper_params missing"
    # `Linear(16, 32)` stores its ctor args in `tensor_term.sons` (each son is a
    # TensorTerm with `value` set), NOT in `tensor_term.op.hyper_params`. Pinned
    # by `tests/repro/test_parser_ctor_args_in_sons.py`.
    assert result.tensor_term.op.hyper_params is None
    assert [getattr(x, "value", x) for x in result.tensor_term.sons] == [16, 32], (
        f"Expected ctor args [16, 32] in tensor_term.sons; got "
        f"{[getattr(x, 'value', x) for x in (result.tensor_term.sons or [])]}"
    )

# %%
if __name__ == "__main__":
    result

# %%
if __name__ == "__main__":
    # Test TransformDef transformation
    relnn_grammar_parser = get_relnn_grammar_parser(start="transform_def")

    # Test case 1: Simple transform def with just a name and a number
    input_text = "foo = 33 ."
    tree = relnn_grammar_parser.parse(input_text)
    ptree_str = tree.pretty()
    logger.info("Test parse tree for transform_def_stmt (simple):\n%s", ptree_str)
    result = transformer.transform(tree)
    print("Test Result (transform_def_stmt, simple):", result)

    assert isinstance(result, TransformDef), f"Expected TransformDef, got {type(result)}"
    assert result.name == "foo"
    assert result.template_params is None
    assert result.tensor_term.value == 33 or result.tensor_term.value == 33.0

    # Test case 2: TransformDef with template parameters
    input_text = "K_Linear<l,S,i> = 12 ."
    tree = relnn_grammar_parser.parse(input_text)
    ptree_str = tree.pretty()
    logger.info("Test parse tree for transform_def_stmt (template params):\n%s", ptree_str)
    result = transformer.transform(tree)
    print("Test Result (transform_def_stmt, template params):", result)

    assert isinstance(result, TransformDef), f"Expected TransformDef, got {type(result)}"
    assert result.name == "K_Linear"
    assert isinstance(result.template_params, list)
    assert [v.name for v in result.template_params] == ["l", "S", "i"]
    assert result.tensor_term.value == 12 or result.tensor_term.value == 12.0

    # Test case 3: TransformDef with module constructor call in tensor term
    # TODO: add this test: input_text = "Lin32 = Linear(16,32) ." , currently it doen't work.

# %%
@patch
@v_args(meta=True)
def program(self: RelnnTransformer, meta, items):
    """Returns: Program object containing all statements"""
    logger.debug("program received items: %s", items)
    statements = items if isinstance(items, list) else [items]
    return Program(statements=statements)

# %%
if __name__ == "__main__":
    # Test program transformation
    relnn_grammer_parser = get_relnn_grammar_parser(start="program")

    # Test case 1: Program with transform_defs and multiple rules using ',' in the RHS
    test_program = """
    d = 10 .
    h = 2 .
    A(a_id; v) :- Author(a_id; v).
    B(x, y; z) :- Book(x; y), Person(y; z).
    R(a, b, c; d) :- Foo(a, b; d), Bar(b, c; d).
    """

    tree = relnn_grammer_parser.parse(test_program)
    ptree_str = tree.pretty()
    logger.info("Test parse tree for program:\n%s", ptree_str)
    result = transformer.transform(tree)
    print("Test Result (program):", result)

    # Verify the result
    assert isinstance(result, Program), f"Expected Program, got {type(result)}"
    assert hasattr(result, "statements"), "Program should have statements attribute"
    assert isinstance(result.statements, list), "statements should be a list"
    assert len(result.statements) == 5, f"Expected 5 statements, got {len(result.statements)}"

    # Verify first statement is a TransformDef
    assert isinstance(result.statements[0], TransformDef), f"First statement should be TransformDef, got {type(result.statements[0])}"
    assert result.statements[0].name == "d", "First transform_def should be named 'd'"

    # Verify second statement is a TransformDef
    assert isinstance(result.statements[1], TransformDef), "Second statement should be TransformDef"
    assert result.statements[1].name == "h", "Second transform_def should be named 'h'"

    # Verify third, fourth, fifth statements are Rules
    assert isinstance(result.statements[2], Rule), f"Third statement should be Rule, got {type(result.statements[2])}"
    assert result.statements[2].lhs.name == "A", "Third rule should have derived ER named 'A'"

    assert isinstance(result.statements[3], Rule), f"Fourth statement should be Rule, got {type(result.statements[3])}"
    assert result.statements[3].lhs.name == "B", "Fourth rule should have derived ER named 'B'"
    # Ensure there are two ERs (Book and Person) in the RHS
    assert hasattr(result.statements[3].rhs, 'ers'), "Fourth rule RHS should have 'ers' attribute (list of ER atoms)"
    assert len(result.statements[3].rhs.ers) == 2, "Fourth rule should have two ER atoms in RHS"

    assert isinstance(result.statements[4], Rule), f"Fifth statement should be Rule, got {type(result.statements[4])}"
    assert result.statements[4].lhs.name == "R", "Fifth rule should have derived ER named 'R'"
    # Ensure there are two ERs (Foo and Bar) in the RHS
    assert hasattr(result.statements[4].rhs, 'ers'), "Fifth rule RHS should have 'ers' attribute (list of ER atoms)"
    assert len(result.statements[4].rhs.ers) == 2, "Fifth rule should have two ER atoms in RHS"

    print("All assertions passed!")

# %%
@patch
@v_args(meta=True)
def function_body(self: RelnnTransformer, meta, items):
    logger.debug("function_body received items: %s", items)
    return items

# %%
@patch
@v_args(meta=True)
def function_def(self: RelnnTransformer, meta, items):
    # Items ordering can be:
    # [name, er_params, function_body]
    # [name, template_params, er_params, function_body]
    # [name, er_params, return_type, function_body]
    # [name, template_params, er_params, return_type, function_body]
    logger.debug("function_def received items: %s", items)
    function_body = items[-1]
    return_type = None
    if len(items) > 3 and not isinstance(items[-2], list):
        return_type = items[-2]
        tail = items[:-2]
    else:
        tail = items[:-1]

    name = tail[0]
    if len(tail) == 3:
        template_params = tail[1]
        er_params = tail[2]
    else:
        template_params = None
        er_params = tail[1]

    if self.collect_tokens and self.source_text:
        lines = self.source_text.split('\n')
        line_idx = meta.line - 1
        if 0 <= line_idx < len(lines):
            line_text = lines[line_idx]
            def_idx = line_text.find('def ')
            if def_idx >= 0:
                fm = type('M', (), {'line': meta.line, 'column': def_idx + 1,
                                    'end_column': def_idx + 4})()
                self._record_token(fm, "control_keyword", "def")
        self._record_var_in_source(meta, str(name), "function_name")
        end_line = meta.end_line - 1
        if 0 <= end_line < len(lines):
            end_text = lines[end_line]
            enddef_idx = end_text.find('enddef')
            if enddef_idx >= 0:
                fm = type('M', (), {'line': meta.end_line, 'column': enddef_idx + 1,
                                    'end_column': enddef_idx + 7})()
                self._record_token(fm, "control_keyword", "enddef")

    return FunctionDef(
        name=name,
        template_params=template_params,
        er_params=er_params,
        return_type=return_type,
        function_body=function_body,
    )

# %%
if __name__ == "__main__":

    # Inline tests for function_def transformation (non-template only, per instructions)

    # Get the parser and transformer
    relnn_grammar_parser = get_relnn_grammar_parser(start="function_def")
    transformer = RelnnTransformer(Engine())

    # Test case 1: function_def with no template params and no return_type
    input_text = """
    def Foo(X):
        MyRule(a; Linear(x) + x) :- AnotherRel(a; x), X(a; y).
        enddef
    """
    tree = relnn_grammar_parser.parse(input_text)
    result = transformer.transform(tree)
    print("Test Result (function_def, simple):", result)
    assert result.name == "Foo"
    assert result.template_params is None
    assert isinstance(result.er_params, list)
    assert result.return_type is None
    assert hasattr(result, "function_body")

# %%
if __name__ == "__main__":
    result

# %%
if __name__ == "__main__":

    # Test case 2: function_def with return_type (ERSchema)
    input_text = """
    def Zed(X) -> (int, int):
        S(a) :- X(a), Input1(a, b) .
    enddef
    """
    tree = relnn_grammar_parser.parse(input_text)
    result = transformer.transform(tree)
    print("Test Result (function_def, with return_type):", result)
    assert result.name == "Zed"
    assert result.template_params is None
    assert isinstance(result.er_params, list)
    assert result.return_type is not None
    assert hasattr(result, "function_body")

# %%
if __name__ == "__main__":

    # Test: function_def with template params, no return_type
    input_text = """
    def Bar<T>(X):
        Rule1(a; x) :- X(a; x).
    enddef
    """
    tree = relnn_grammar_parser.parse(input_text)
    result = transformer.transform(tree)
    print("Test Result (function_def, template params, no return_type):", result)

    # Test: function_def with template params and return_type
    input_text = """
    def Buz<T>(X) -> (int, int):
        Rule2(a; x) :- X(a; x).
    enddef
    """
    tree = relnn_grammar_parser.parse(input_text)
    result = transformer.transform(tree)
    print("Test Result (function_def, template params, with return_type):", result)

    # Test: function_def with multiple er_params, no template, no return_type
    input_text = """
    def Qux(X, Y):
        Rule3(a; x) :- X(a; x), Y(a; y).
    enddef
    """
    tree = relnn_grammar_parser.parse(input_text)
    result = transformer.transform(tree)
    print("Test Result (function_def, multiple er_params, no template, no return_type):", result)

    # Test: function_def with multiple er_params, with template and return_type
    input_text = """
    def Quux<T, S>(A, B) -> (float;):
        Rule4(a; x) :- A(a; x), B(a; y).
    enddef
    """
    tree = relnn_grammar_parser.parse(input_text)
    result = transformer.transform(tree)
    print("Test Result (function_def, multiple er_params, template, with return_type):", result)

# %%
@patch
@v_args(meta=True)
def fit_param(self: RelnnTransformer, meta, items):
    name = str(items[0]) if not isinstance(items[0], str) else items[0]
    value = items[1]
    if self.collect_tokens:
        self._record_token(meta, "param_name", name)
    return (name, value)

@patch
@v_args(meta=True)
def fit_params(self: RelnnTransformer, meta, items):
    params = dict(items)
    return params

@patch
@v_args(meta=True)
def fit(self: RelnnTransformer, meta, items):
    self._record_token(meta, "keyword", "?fit")
    fit_params = items[0]
    rule = items[1]
    return FitStatement(fit_params=fit_params, rule=rule)

# %%
if __name__ == "__main__":

    relnn_grammar_parser = get_relnn_grammar_parser(start="program")
    transformer = RelnnTransformer(Engine())
    # Test: fit statement with multiple parameters
    input_text = "?fit <lr=0.01, epochs=10> Rule5(a; x) :- A(a; x), B(a; y) ."
    tree = relnn_grammar_parser.parse(input_text)
    result = transformer.transform(tree)
    print("Test Result (fit statement with multiple fit_params):", result)

    # Test: fit statement with single parameter
    input_text = "?fit <batch_size=32> Rule6(a; y) :- C(a; y), D(a; z)."
    tree = relnn_grammar_parser.parse(input_text)
    result = transformer.transform(tree)
    print("Test Result (fit statement with single fit_param):", result)

    # Test: fit statement with no parameters
    input_text = "?fit <> Rule7(b; w) :- E(b; w)."
    tree = relnn_grammar_parser.parse(input_text)
    result = transformer.transform(tree)
    print("Test Result (fit statement with no fit_params):", result)

# %%
@patch
@v_args(meta=True)
def predict(self: RelnnTransformer, meta, items):
    self._record_token(meta, "keyword", "?pred")
    rule = items[0]
    return PredictStatement(rule=rule)

# %%
if __name__ == "__main__":

    # Import the parser/transformer setup, use "predict" as start
    relnn_grammar_parser = get_relnn_grammar_parser(start="predict")
    transformer = RelnnTransformer(Engine())

    # Test: predict statement
    input_text = "?pred Rule8(a; z) :- F(a; z), G(a; w)."
    tree = relnn_grammar_parser.parse(input_text)
    result = transformer.transform(tree)
    print("Test Result (predict statement):", result)

    assert isinstance(result, PredictStatement), f"Expected PredictStatement, got {type(result)}"
    assert hasattr(result, "rule")
    assert result.rule is not None, "rule should not be None"

# %% [markdown]
# ## Example of how to program the Transformer
