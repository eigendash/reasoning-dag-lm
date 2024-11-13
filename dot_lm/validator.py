"""Online validator for typed reasoning traces.

A trace is a sequence of node blocks. Each block opens with a role token and a
`@node` record, then carries typed records and one free-text line:

    <proposer>@node id=2 role=proposer
    @edge src=1 dst=2 kind=use
    @prop id=2 {"gt":[4,2]}
    x>02: 04>02
    <critic>@node id=3 role=critic
    @edge src=2 dst=3 kind=critique
    @status target=2 mark=validated
    ok

The validator is a finite-control machine over single tokens. At every position it
can report which tokens would keep the trace well formed (`allowed`), so a decoder
can mask everything else, and it can reject a finished trace (`validate`). Both use
the same code path, so a trace the decoder is allowed to produce is a trace the
checker accepts.

Rules enforced, following the paper's serialization:
  * node ids are fresh and strictly increasing; every edge ends at the current node
    and starts at an earlier one, so the provenance graph is acyclic by construction;
  * edge kinds obey the role table: critique goes proposer -> critic, refine goes
    proposer/critic -> proposer, use goes problem/proposer -> proposer/summarizer;
  * a critic emits exactly one critique edge, then one status for that target;
    a status may only be written for a proposition still `active` (first writer wins);
  * a summarizer may only draw `use` edges from validated proposers, or the problem;
  * typed `@prop` blocks must parse and type-check under the solver;
  * free-text lines never start with `@`.

One deliberate narrowing: the paper's grammar lets record kinds appear in any order
inside a block. Here each role has a fixed order, which keeps the machine small and
the next record type predictable.
"""

from dataclasses import dataclass, field

from . import solver
from .tokens import CHARS, EOS, ROLE_OF

TEXT_CHARS = frozenset(c for c in CHARS if c not in "@\n")
PROP_CHARS = frozenset('{}[]":, -0123456789abcdefghijklmnopqrstuvwxyz')


class Violation(Exception):
    def __init__(self, message, index=None):
        super().__init__(message if index is None else f"token {index}: {message}")
        self.index = index


@dataclass
class Node:
    id: int
    role: str
    text: str = ""
    prop: dict | None = None
    state: str = "initial"  # initial | active | validated | invalidated


@dataclass
class Diagram:
    """What the extraction map returns for an accepted trace."""

    nodes: dict = field(default_factory=dict)
    edges: list = field(default_factory=list)  # (src, dst, kind)
    statuses: list = field(default_factory=list)  # (target, mark, critic)

    def selected(self, summarizer_id):
        """Validated propositions a summarizer drew `use` edges from."""
        return [
            s for s, d, k in self.edges
            if d == summarizer_id and k == "use" and self.nodes[s].role == "proposer"
        ]

    def summaries(self):
        return [n for n in self.nodes.values() if n.role == "summarizer"]

    def answer(self):
        done = self.summaries()
        return done[-1].text if done else None

    def to_dot(self):
        shape = {"problem": "box", "proposer": "ellipse", "critic": "diamond", "summarizer": "doubleoctagon"}
        lines = ["digraph trace {"]
        for n in self.nodes.values():
            label = f"{n.id} {n.role}\\n{n.text}".replace('"', "'")
            style = {"validated": "green", "invalidated": "red"}.get(n.state, "black")
            lines.append(f'  n{n.id} [label="{label}", shape={shape[n.role]}, color={style}];')
        for s, d, k in self.edges:
            lines.append(f'  n{s} -> n{d} [label="{k}"];')
        lines.append("}")
        return "\n".join(lines)


class OnlineValidator:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.statuses = []
        self.phase = "start"  # start | role | line | end
        self.role = None
        self.cur = 0
        self.stage = None
        self.buf = ""
        self.used = set()
        self.target = None
        self.last_role = None
        self._cands, self._prop_ok, self._text_ok = [], False, False

    # ------------------------------------------------------------------ queries
    def _with(self, role, *states):
        return [n.id for n in self.nodes.values() if n.role == role and (not states or n.state in states)]

    def _edges_for(self, kinds):
        out = []
        for kind, roles, states in kinds:
            for node in self.nodes.values():
                if node.id < self.cur and node.role in roles and (not states or node.state in states):
                    if (node.id, kind) not in self.used:
                        out.append(f"@edge src={node.id} dst={self.cur} kind={kind}")
        return out

    def _enter(self, stage):
        self.stage, self.buf = stage, ""
        cands, prop_ok, text_ok = [], False, False
        role = self.role
        if stage == "node":
            cands = [f"@node id={self.cur} role={role}"]
        elif role == "problem":
            text_ok = True
        elif role == "proposer":
            if stage in ("edge0", "edges"):
                cands = self._edges_for([("use", ("problem", "proposer"), ()), ("refine", ("proposer", "critic"), ())])
                prop_ok = stage == "edges"
            else:
                text_ok = True
        elif role == "critic":
            if stage == "edge0":
                cands = self._edges_for([("critique", ("proposer",), ("active",))])
            elif stage == "status":
                cands = [f"@status target={self.target} mark={m}" for m in ("validated", "invalidated")]
            else:
                text_ok = True
        elif role == "summarizer":
            cands = self._edges_for([("use", ("problem",), ()), ("use", ("proposer",), ("validated",))])
            text_ok = stage == "edges"
        self._cands, self._prop_ok, self._text_ok = cands, prop_ok, text_ok

    def allowed(self):
        if self.phase == "start":
            return {"<problem>"}
        if self.phase == "end":
            return set()
        if self.phase == "role":
            if self.last_role == "summarizer":
                return {EOS}
            out = {"<proposer>"}
            active = self._with("proposer", "active")
            if active:
                out.add("<critic>")
            elif self._with("proposer", "validated"):
                out.add("<summarizer>")
            return out

        buf, prefix = self.buf, f"@prop id={self.cur} "
        if buf and buf[0] != "@":
            return set(TEXT_CHARS) | {"\n"}
        if self._prop_ok and buf.startswith(prefix):
            out = set(PROP_CHARS)
            if self._json_ok(buf[len(prefix):]):
                out.add("\n")
            return out
        out = set()
        if not buf:
            out |= {c[0] for c in self._cands}
            out |= {"@"} if self._prop_ok else set()
            if self._text_ok:
                out |= TEXT_CHARS
            return out
        for cand in self._cands:
            if cand.startswith(buf):
                out.add(cand[len(buf)] if len(cand) > len(buf) else "\n")
        if self._prop_ok and prefix.startswith(buf) and len(buf) < len(prefix):
            out.add(prefix[len(buf)])
        return out

    @staticmethod
    def _json_ok(body):
        try:
            solver.parse_prop(body)
            return True
        except solver.PropError:
            return False

    # ---------------------------------------------------------------- transitions
    def advance(self, token):
        if token not in self.allowed():
            raise Violation(f"{token!r} is not allowed here (phase={self.phase}, stage={self.stage}, line={self.buf!r})")
        if self.phase in ("start", "role"):
            if token == EOS:
                self.phase = "end"
                return
            self.role = ROLE_OF[token]
            self.cur = max(self.nodes, default=0) + 1
            self.phase = "line"
            self._enter("node")
        elif token != "\n":
            self.buf += token
        else:
            self._finish_line()

    def _finish_line(self):
        line, role = self.buf, self.role
        if self.stage == "node":
            state = "active" if role == "proposer" else "initial"
            self.nodes[self.cur] = Node(self.cur, role, state=state)
            self.used = set()
            self._enter("text" if role == "problem" else "edge0")
        elif line.startswith("@edge"):
            fields = dict(part.split("=") for part in line.split()[1:])
            src, kind = int(fields["src"]), fields["kind"]
            self.edges.append((src, self.cur, kind))
            self.used.add((src, kind))
            if role == "critic":
                self.target = src
            self._enter("status" if role == "critic" else "edges")
        elif line.startswith("@prop"):
            self.nodes[self.cur].prop = solver.parse_prop(line[len(f"@prop id={self.cur} "):])
            self._enter("text")
        elif line.startswith("@status"):
            fields = dict(part.split("=") for part in line.split()[1:])
            self.nodes[self.target].state = fields["mark"]
            self.statuses.append((self.target, fields["mark"], self.cur))
            self._enter("text")
        else:
            self.nodes[self.cur].text = line
            self.last_role, self.phase = role, "role"

    def diagram(self):
        return Diagram(dict(self.nodes), list(self.edges), list(self.statuses))


def feed(tokens, validator=None):
    validator = validator or OnlineValidator()
    for i, token in enumerate(tokens):
        try:
            validator.advance(token)
        except Violation as err:
            raise Violation(str(err), i) from None
    return validator


def validate(tokens):
    """Return the extracted Diagram, or raise Violation if the trace is not accepted."""
    validator = feed(tokens)
    if validator.phase != "end":
        raise Violation("trace is not terminated")
    return validator.diagram()


def is_valid(tokens):
    try:
        validate(tokens)
        return True
    except Violation:
        return False
