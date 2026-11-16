"""Symbolic user for ColBench backend-programming tasks.

Parse the private training reference into requirement facts, match the agent's
question by keywords, and render selected facts as text without a model call.
The disclosure and matching knobs are editable. A task-level mixture with an
LLM user can reduce dependence on this keyword-matching interface.
"""

import ast
import re
from typing import Any, Dict, List, Optional, Set

# Rendered instead of a fact when the question matches nothing well enough. The
# upstream human-simulator prompt says outright to say you don't know when the
# answer is not in the hidden information, so this is the faithful behaviour --
# and it is what makes match_threshold a difficulty knob rather than a mute.
NO_MATCH_REPLY = "I am not sure about that. Could you ask about something more specific?"
NO_BUDGET_REPLY = "I have told you what I can. Please go ahead with your best solution."

_OP_PROSE = [
    (">=", " is at least "),
    ("<=", " is at most "),
    ("==", " equals "),
    ("!=", " is not "),
    (" > ", " is greater than "),
    (" < ", " is less than "),
    (" * ", " times "),
    (" / ", " divided by "),
    (" + ", " plus "),
    (" - ", " minus "),
    (" % ", " modulo "),
]

# Words that hint at a fact kind but are too common to identify a fact on their
# own; they add a fraction of a point, never a whole one.
_GENERIC = {
    "return": {"return", "returns", "returned", "output", "result", "give", "gives"},
    "shape": {"shape", "format", "structure", "keys", "dict", "dictionary", "type", "list", "tuple"},
    "branch": {"when", "if", "case", "condition", "happens", "otherwise", "else"},
    "loop": {"each", "every", "all", "loop", "iterate", "over", "list"},
    "edge": {"edge", "empty", "zero", "negative", "none", "null", "invalid", "error"},
}

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+\.?\d*")


def _tokens(text: str) -> Set[str]:
    """Question tokens, plus the pieces of any snake_case identifier in it."""
    out: Set[str] = set()
    for t in _TOKEN_RE.findall(text or ""):
        t = t.lower()
        out.add(t)
        if "_" in t:
            out.update(p for p in t.split("_") if p)
    return out


def _prose(expr: str) -> str:
    out = f" {expr} "
    for op, word in _OP_PROSE:
        out = out.replace(op, word)
    return re.sub(r"\s+", " ", out).strip()


def _names_in(node: ast.AST) -> Set[str]:
    found: Set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            found.add(n.id.lower())
            found.update(p for p in n.id.lower().split("_") if p)
        elif isinstance(n, ast.Attribute):
            found.add(n.attr.lower())
        elif isinstance(n, ast.Constant):
            if isinstance(n.value, str):
                found.update(_tokens(n.value))
            elif isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
                found.add(str(n.value))
    return found


class Fact:
    """One disclosable requirement, with the words that make it findable."""

    __slots__ = ("kind", "text", "core", "generic", "polarity", "order")

    def __init__(self, kind: str, text: str, core: Set[str], generic: Set[str],
                 order: int, polarity: Optional[Set[str]] = None) -> None:
        self.kind = kind
        self.text = text
        self.core = {c for c in core if c}
        self.generic = generic
        self.polarity = polarity or set()
        self.order = order

    def score(self, q: Set[str]) -> float:
        """How well a question matches, in [0, 1].

        Absolute overlap, deliberately NOT normalised by |core|. Dividing by the
        fact's own size makes the *shorter* fact win on a tie, and in an
        if/else pair the negated branch is always the shorter one -- so
        "what happens when the year is 1970 or later" reliably retrieved the
        `not (year >= 1970)` branch, the opposite of what was asked.

        Polarity carries the rest: the two branches of a comparison share every
        identifier, and the only thing separating them in the question is a word
        like "later" versus "before".
        """
        if not self.core:
            return 0.6 if (self.generic & q) else 0.0
        score = min(0.8, 0.4 * len(self.core & q))
        if self.polarity & q:
            score += 0.3
        if self.generic & q:
            # Worth clearing the default threshold on its own. A question can be
            # squarely about the return value while sharing no identifier with
            # it ("what should this give back?"); scoring that below threshold
            # left generic questions unanswered, which is not a hard
            # environment, just a mute one.
            score += 0.35
        return min(1.0, score)

    def __repr__(self) -> str:  # pragma: no cover -- debugging aid
        return f"Fact({self.kind}, {self.text[:50]!r})"


_POLARITY = {
    "up": {"later", "after", "above", "over", "greater", "least", "more", "newer",
           "higher", "exceeds", "beyond", "since"},
    "down": {"before", "earlier", "below", "under", "less", "fewer", "most",
             "older", "lower", "within"},
    "eq": {"exactly", "equal", "equals", "same", "matches"},
}


def _polarity_hints(cond_proses: List[str]) -> Set[str]:
    """Direction words for a branch, so an if/else pair is distinguishable.

    Read off the rendered condition rather than the AST because the negated
    branch is built as the string ``not (<cond>)``; a leading negation simply
    swaps which direction the branch describes.
    """
    hints: Set[str] = set()
    for c in cond_proses:
        negated = c.strip().startswith("not (")
        up = ("is at least" in c) or ("is greater than" in c)
        down = ("is at most" in c) or ("is less than" in c)
        if negated:
            up, down = down, up
        if up:
            hints |= _POLARITY["up"]
        if down:
            hints |= _POLARITY["down"]
        if "equals" in c:
            hints |= _POLARITY["eq"]
    return hints


def _fn_of(tree: ast.AST) -> Optional[ast.FunctionDef]:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return node
    return None


def _return_shape(node: ast.Return) -> Optional[str]:
    v = node.value
    if isinstance(v, ast.Dict):
        keys = [k.value for k in v.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
        if keys:
            return "a dictionary with the keys " + ", ".join(repr(k) for k in keys)
    if isinstance(v, ast.Tuple):
        return f"a tuple of {len(v.elts)} values"
    if isinstance(v, ast.List):
        return "a list"
    if isinstance(v, ast.Constant):
        if isinstance(v.value, bool):
            return "a boolean"
        if isinstance(v.value, str):
            return "a string"
        if isinstance(v.value, (int, float)):
            return "a number"
    return None


def extract_facts(code: str) -> List[Fact]:
    """Parse a reference solution into the requirements a user could state."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    fn = _fn_of(tree)
    if fn is None:
        return []

    facts: List[Fact] = []
    counter = [0]

    def add(kind: str, text: str, core: Set[str], polarity: Optional[Set[str]] = None) -> None:
        counter[0] += 1
        facts.append(Fact(kind, text, core, _GENERIC.get(kind, set()), counter[0], polarity))

    shapes: List[str] = []

    def walk(stmts: List[ast.stmt], conds: List[str]) -> None:
        for node in stmts:
            if isinstance(node, ast.If):
                cond = ast.unparse(node.test)
                walk(node.body, conds + [cond])
                if node.orelse:
                    walk(node.orelse, conds + [f"not ({cond})"])
            elif isinstance(node, (ast.For, ast.While)):
                target = ast.unparse(node.target) if isinstance(node, ast.For) else ""
                it = ast.unparse(node.iter) if isinstance(node, ast.For) else ast.unparse(node.test)
                add("loop",
                    f"The function goes through {_prose(it)}"
                    + (f", one {target} at a time." if target else "."),
                    _names_in(node.iter if isinstance(node, ast.For) else node.test))
                walk(node.body, conds)
            elif isinstance(node, ast.Return) and node.value is not None:
                shape = _return_shape(node)
                if shape and shape not in shapes:
                    shapes.append(shape)
                value = _prose(ast.unparse(node.value))
                core = _names_in(node.value)
                if conds:
                    proses = [_prose(c) for c in conds]
                    when = " and ".join(proses)
                    core |= set().union(*[_names_in(ast.parse(c, mode="eval")) for c in conds])
                    add("branch", f"When {when}, the function should return {value}.",
                        core, _polarity_hints(proses))
                else:
                    add("return", f"The function should return {value}.", core)
            elif isinstance(node, (ast.Assign, ast.AugAssign)):
                tgt = ast.unparse(node.targets[0] if isinstance(node, ast.Assign) else node.target)
                val = _prose(ast.unparse(node.value))
                core = _names_in(node.value) | _tokens(tgt)
                if conds:
                    proses = [_prose(c) for c in conds]
                    when = " and ".join(proses)
                    core |= set().union(*[_names_in(ast.parse(c, mode="eval")) for c in conds])
                    add("branch", f"When {when}, {tgt} becomes {val}.", core,
                        _polarity_hints(proses))
                else:
                    add("branch", f"{tgt} is {val}.", core)

    walk(fn.body, [])

    if shapes:
        facts.insert(0, Fact("shape",
                             "The function should return " + " or ".join(shapes) + ".",
                             set(), _GENERIC["shape"] | _GENERIC["return"], 0))
    return facts


class FlowUser:
    """Episode-scoped symbolic user: holds the facts and what it has said.

    All four knobs are read per reply, not captured at construction, so a
    mid-run write to ``env_policy.json`` takes effect on the next turn of
    episodes that are already in flight.
    """

    def __init__(self, code: str, knobs: Optional[Dict[str, Any]] = None) -> None:
        self.facts = extract_facts(code)
        self.knobs = knobs or {}
        self.revealed: Set[int] = set()
        self.reveal_count = 0
        self.unmatched_streak = 0

    def _k(self, name: str, default: Any) -> Any:
        v = self.knobs.get(name, default) if isinstance(self.knobs, dict) else default
        return default if v is None else v

    def reply(self, question: str) -> str:
        if not self.facts:
            return NO_MATCH_REPLY

        max_reveals = int(self._k("max_reveals", 0) or 0)
        if max_reveals and self.reveal_count >= max_reveals:
            return NO_BUDGET_REPLY

        per_reply = max(1, int(self._k("facts_per_reply", 1)))
        threshold = float(self._k("match_threshold", 0.34))
        volunteer_interval = int(self._k("volunteer_interval", 2) or 0)
        reveal_shape = bool(self._k("reveal_return_shape", True))

        q = _tokens(question)
        visible = [
            (i, f) for i, f in enumerate(self.facts)
            if reveal_shape or f.kind != "shape"
        ]
        if not visible:
            return NO_MATCH_REPLY

        ranked = sorted(
            ((f.score(q), -f.order, i, f) for i, f in visible),
            reverse=True,
        )
        above = [(i, f) for sc, _, i, f in ranked if sc >= threshold]
        # Prefer something the agent has not heard yet. Picking the top-scoring
        # fact outright made a *different* question restate the same fact,
        # because one broad fact can outrank every other for several phrasings:
        # a step-20 trajectory showed two distinct questions answered with the
        # identical "weights is {...}" line, so the second turn bought nothing.
        # That caps extractable information at roughly one fact however many
        # questions are asked -- i.e. interviewing stops paying, which is the
        # whole point of the flow user. Restating is kept as the fallback for
        # when nothing new clears the threshold, so asking the same question
        # twice still gets an answer rather than a refusal.
        fresh_above = [(i, f) for i, f in above if i not in self.revealed]
        picked = (fresh_above or above)[:per_reply]

        if not picked:
            self.unmatched_streak += 1
            # Volunteering mirrors TravelGym's elicitation_interval: after a run
            # of off-target turns the user offers something unprompted, so a
            # policy that never asks a matching question is not simply starved.
            fresh = [(i, f) for _, _, i, f in ranked if i not in self.revealed]
            if volunteer_interval and self.unmatched_streak >= volunteer_interval and fresh:
                self.unmatched_streak = 0
                i, f = fresh[0]
                self.revealed.add(i)
                self.reveal_count += 1
                return f"One thing I should mention: {f.text[0].lower() + f.text[1:]}"
            return NO_MATCH_REPLY

        self.unmatched_streak = 0
        out = []
        for i, f in picked:
            # Restating something already disclosed costs nothing: a person asked
            # the same question twice would repeat themselves, and charging the
            # budget for it would instead surface an unrelated fact.
            if i in self.revealed:
                out.append(f.text)
                continue
            if max_reveals and self.reveal_count >= max_reveals:
                break
            self.revealed.add(i)
            self.reveal_count += 1
            out.append(f.text)
        return " ".join(out) if out else NO_BUDGET_REPLY
