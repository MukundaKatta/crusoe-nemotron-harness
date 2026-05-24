# 90-second demo script

Three shots. No voice tricks. Plain narration in the speaker's own voice.

## Shot 1 (0:00 - 0:25): The problem

Terminal on screen. Run the bare Nemotron provider across 10 tasks:

```bash
.venv/bin/python -c "
from crusoe_nemotron_harness import FakeNemotronProvider
p = FakeNemotronProvider(seed=3)
tokens = sum(p.complete(f'task {i}').input_tokens for i in range(1, 11))
print(f'tasks: 10  tokens approx: {tokens}  cost: unknown  budget: unknown  egress: unknown')
"
```

Narration:

> Here is a Nemotron agent running ten tasks against Crusoe Managed Inference. It tells me how many tokens it spent. It does not tell me what they cost, whether it tried to reach hosts I never approved, whether it called tools with bad args, or whether it stayed under the budget I funded.

## Shot 2 (0:25 - 1:05): The harness

Same terminal, run the demo:

```bash
.venv/bin/python examples/leaderboard_demo.py
```

Pause on Scene 2's RunReport. Highlight rows in order:

> Wrap the same agent in NemotronHarness. Same ten tasks, same seed. Now I see total cost in fractions of a cent. I see tokens used against my cap. I see p95 latency at one second where p50 was 124 milliseconds. The harness blocked an attempted fetch to a host I never approved. Then in Scene 3, I drop the token cap and the harness aborts the run on the fourth call, before the agent can burn through more spend.

## Shot 3 (1:05 - 1:30): How it ships

Show the table in README.md mapping each concern to a small module, then `DEPLOY.md` in the editor.

Narration:

> Every concern is owned by a small module modeled on a library I already shipped: claude-cost, agentguard, agentvet, agentsnap, agenttrace, token-budget-pool. The harness is the seam that pulls them together for Crusoe Managed Inference. Swapping the fake provider for live Crusoe is one provider line plus a 10-line requests shim. Full code is in DEPLOY.md. Repo is MukundaKatta slash crusoe-nemotron-harness. Thanks.

## Recording checklist

- Terminal font at least 18pt so the RunReport rows read on mobile.
- Run `clear` between Shot 1 and Shot 2 so the RunReport lands on a clean screen.
- Mention the track name "Your Harness, Our Inference" once near the end.
- No captions on the speaker's face; let the terminal carry the story.
