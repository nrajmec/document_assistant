# Document Assistant - Writeup

Notes on how I built this and how it actually behaves, not just how it's
supposed to behave. Pulled some of the examples below straight out of a real
session file instead of making them up, so a couple of them are messier than
you'd expect from a demo.

## Architecture & Routing Decisions

The graph (`src/agent.py`) is basically flat:

```
classify_intent -> (qa_agent | summarization_agent | calculation_agent) -> update_memory -> END
```

I went with one router node feeding three worker nodes instead of anything
deeper, mostly because a single turn only ever needs one decision. There's no
loop-back to `classify_intent`, so multi-turn conversation isn't a graph
feature at all - it's just the same graph getting called again with the
previous thread's state attached.

`classify_intent` forces the routing decision through
`llm.with_structured_output(UserIntent)`. One thing I only noticed while
writing this up: `"unknown"` doesn't go anywhere special, it just falls
through to `qa_agent` in the if/elif/else. There's actually an `"end"` branch
wired up in `add_conditional_edges` that looked intentional when I wrote it,
but nothing ever sets `next_step` to `"end"`, so it's dead.

All three worker agents share the exact same four tools - `document_search`,
`document_reader`, `document_statistics`, `calculator`. I didn't bother
scoping tools per agent; what changes between them is just the system prompt
and which response schema gets passed to `create_agent`. That means, in
theory, the QA agent could call the calculator if the model decided to - it's
prompt discipline keeping it in line, not an actual restriction. Good enough
for this scope, but worth knowing if this ever needs to be locked down harder.

Also no vector DB anywhere. `SimulatedRetriever` (`src/retrieval.py`) is just
five hardcoded documents in a dict, searched by keyword overlap and a regex
parser for amount phrases like "over $50,000" or "around $25,000". Fine for a
handful of documents, wouldn't scale past that.

## State & Memory

This ended up being three different things layered on top of each other,
which I didn't fully appreciate until I went looking for a bug:

- **The LangGraph checkpoint.** `InMemorySaver`, keyed by `thread_id =
  session_id`. This is what actually keeps `messages` and
  `conversation_summary` around between turns - but only in memory, only for
  the life of the process.
- **The session JSON file** (`./sessions/<id>.json`). Holds `document_context`
  and a `conversation_history` list. Written after every turn. Fair warning:
  `conversation_history` is the *entire* raw graph state per turn, not a
  trimmed log, so the file balloons fast - the sample one in this repo is
  well over 100KB after a handful of messages.
- **The rolling summary.** `update_memory` runs after every turn and
  re-summarizes the whole conversation from scratch (not incrementally) into
  `conversation_summary` and `active_documents`.

The gap I found: if you resume a `session_id` in a fresh process, only the
JSON file's `document_context` comes back. The checkpoint doesn't, because it
was never on disk to begin with - it lived in that first process's memory.
So the session *looks* fully restored (docs are there, file's intact) but the
agent starts that first turn with no prior messages and no summary. Real
continuity only exists within one running process. I'm noting this rather
than "fixing" it since fixing it means swapping `InMemorySaver` for a
persistent checkpointer, which is a bigger change than this writeup covers.

Tool call logging is a separate, simpler thing - `ToolLogger` just appends
every call to `./logs/tool_usage_*.json` per process run. Not tied to a
session at all.

## Structured Output

Two mechanisms depending on whether the node uses tools:

- `classify_intent` and `update_memory` are single LLM calls with
  `llm.with_structured_output(Schema)` - no tools involved, just force the
  shape.
- The three worker agents go through `create_agent(..., response_format=
  Schema)`, which lets the model call tools in a loop and only forces the
  schema on the final answer.

Two things I ran into while looking at actual output that surprised me:

1. The text printed to the CLI isn't the clean `explanation` field - it's
   `final_state["messages"][-1].content`, and `create_agent`'s final message
   for a structured response is basically a Python repr of the object. So
   what you actually see is something like `"Returning structured response:
   expression='69300' result=69300.0 explanation='...' units='USD'"`. The
   individual fields exist, they're just not pulled out anywhere in
   `assistant.py` - only `intent`, `sources`, `tools_used`, and the summary
   get surfaced separately.
2. `Field(default_factory=datetime.now)` isn't as safe as it sounds once a
   field is inside a `response_format` schema - the model is asked to fill
   in the whole object, so it can (and did, in one real trace) just make up
   a timestamp. I found one response with `timestamp=datetime.datetime(2023,
   10, 4, ...)`, nowhere near when that run actually happened.

Outside the LLM side, the `calculator` tool does its own version of
structured enforcement on the input: it regex-checks the expression against
`^[0-9\s\.\+\-\*/%\(\)]+$` before ever calling `eval()`, so something like
`__import__('os').system(...)` gets rejected before it's evaluated.

## Example Conversations

The first two are copied out of an actual saved session
(`sessions/a38d7c41-...json`), not written for this doc. The last two I made
up to cover the QA and summarization paths, since the saved session happened
to only exercise calculation.

**"What is the total amount due on invoice INV-002?"** - got classified as
`calculation`, not `qa`, which I initially thought was a bug until I reread
the intent prompt: the calculation bucket explicitly covers document
questions "that may require calculations," so this is a defensible call, just
not the one I expected. Agent ran `document_reader("INV-002")`, then
`calculator("69300")` even though there's nothing to compute - the
calculation prompt insists on using the calculator "no matter how simple,"
and it followed that literally. Ended in `CalculationResponse(result=69300.0,
...)`.

**"Calculate the sum of all invoice totals"** - same session, later turn.
`document_statistics()` first, then `document_search(type=invoice)`, then
`document_reader` on INV-001 and INV-003 (INV-002 wasn't re-read - it was
still in the checkpoint from the earlier turn), then
`calculator("20000 + 69300 + 214500")` → `303800.0`. This is the one place
where I could actually confirm the cross-turn memory was doing something
real, rather than just trusting the design.

**"Who is the client on invoice INV-001?"** (illustrative) - routes to `qa`,
`document_reader("INV-001")` finds "Client: Acme Corporation", answer comes
back as `AnswerResponse(answer="Acme Corporation", sources=["INV-001"])`.

**"Summarize claim CLM-001"** (illustrative) - routes to `summarization`,
`document_reader("CLM-001")` pulls the expense breakdown, comes back as
`SummarizationResponse(key_points=[...], document_ids=["CLM-001"])`.
