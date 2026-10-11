# BetterOpenfrontAi

A reinforcement-learning AI for [OpenFront](https://openfront.io), trained by
self-play inside the real game engine (vendored in `OpenFrontIO/`).

| Folder | What it is |
| --- | --- |
| `env/` | TypeScript environment around the engine: observation encoder, action codec, reward, multi-seat games, the TCP bridge workers, and the in-page agent for live games |
| `train/` | Python: PPO self-play trainer, evaluation, lobby driver, dashboard server |
| `dashboard/` | The control dashboard (plain HTML/JS, served by `train/control_server.py`) |
| `OpenFrontIO/` | The game (engine, client, server), used for training and local play |

## Setup

```bash
cd OpenFrontIO && npm run inst          # game deps
cd ../env && npm ci                     # env deps
cd ../train && pip install -r requirements.txt
```

## Dashboard

```bash
cd train && python control_server.py    # http://127.0.0.1:8765/
```

Everything below can be started from the dashboard; each panel runs the
same command line shown here and keeps its output.

## The three ways to see the AI play

**AI vs AI**: evaluation puts several AIs in the same games, next to the
engine's bots and nations, and records every game.

```bash
cd train
python evaluate.py --seats checkpoints/selfplay_v3/selfplay_v3_latest.pt scripted --games 8 --replay-dir replays/eval
```

**AI vs you**: start the local game, host a private lobby, give the AI its ID.

```bash
cd OpenFrontIO && npm run dev           # http://localhost:9000
# create a private lobby in the browser, copy the 8-character ID, then:
cd train && python play_lobby.py AbCd1234 --model checkpoints/selfplay_v3/selfplay_v3_latest.pt
# start the game once "OpenFrontAI" is in your player list
```

**AI in a real openfront.io lobby**: pass the lobby link (or `--host openfront` with the ID).

```bash
python play_lobby.py https://openfront.io/w2/game/cP9iiNqCDz --model <checkpoint.pt>
python play_lobby.py --list                                  # public lobbies right now
python play_lobby.py --public ffa --model <checkpoint.pt>    # next public FFA lobby
```

The AI plays in a visible browser window as a normal, named player. It
reads the client's own game state and sends intents through the client's
own handlers. Any human check the site shows is yours to answer. If
Cloudflare refuses the automated browser (the check never appears, or
keeps coming back), openfront.io is not available to the bot; nothing here
works around that.

## Watching replays

Training records every 100th game, evaluation records all of them
(`train/replays/`). The dashboard's Replays panel has a Watch button, or:

```bash
cd train && python watch_replay.py replays/selfplay/<file>.json   # needs npm run dev
```

## Training

The main recipe trains one AI on a mix of scenarios at once (alone among
tribes and nations, 4-AI free-for-all, two AIs spawned close, a crowded
map, and an easy no-nations map), with a curriculum: nations start on
Easy and the mix leans on easy games; it moves to Medium nations and a
nation-heavy mix as survival in the nation scenarios passes 50% / 40%.

```bash
cd train && python train.py --config configs/main.yaml --set run.total_steps=3000000
```

```bash
cd train
python train.py --config configs/selfplay.yaml --set run.name=selfplay_v4 --set run.total_steps=100000
python train.py --config configs/smoke.yaml     # seconds-long end-to-end check
```

Self-play: 4 agent seats per game (learner plus frozen snapshots of past
selves) among 20 bots and the map's nations; 30% of games spawn the seats
close together. Checkpoints go to `train/checkpoints/<run>/`, metrics to
`train/runs/<run>/metrics.jsonl` (and TensorBoard). Create
`checkpoints/<run>/STOP` to stop a run cleanly.

### Reward (v4, `env/src/Reward.ts`)

Potentials (paid as they change, all given back on death):

| Term | Default | Why |
| --- | --- | --- |
| land | `1.0 * log(1 + tiles/100)` | territory |
| structures | `0.2 * log(1 + levels)` | building (defense posts excluded) |
| cities | `0.5 * log(1 + city levels)` | the economy core |
| population cap | `0.3 * log(1 + max troops/100k)` | land and cities raise it |
| income | `1.0 * log(1 + gold per minute/100k)` | trade, trains, conquest (smoothed over a minute) |

Events and terminals:

| Term | Default | Why |
| --- | --- | --- |
| kill | +0.5 | eliminations this seat caused |
| win / death / placement | +5 / -2 / +1 x land rank | terminals |
| lowTroops | up to -0.05 per step below 30% of max troops | do not expand yourself empty |
| betrayal | -1 per broken alliance | no request/break farming |
| diplomacy | -0.005 per request, break, embargo, target | no spam |
| advantage | land from a tribe or a weaker player counts 1.5x | clear easy land, attack from strength |
| defend | +0.3 per repelled real attack | hold the line |
| defensePost | -0.1 per post started | posts must earn their keep |
| crowdedPost | extra -0.5 per post within 20 tiles of another own post | spread posts along the front |
| boatLaunch | +0.05 per boat with at least 10% of max troops, unless it fills the last of the 3 slots | keep boating, but never park all three boats out |
| boatLanding | +0.2 per such boat that lands and takes its tile | boats that achieve something |
| nuke | +0.05 per unit of damage of an own nuke that lands (hostile structure levels + hostile tiles/1000, own and allied losses x2 against it; at most ±3 a step) | nukes that hit something worth the gold; shot-down ones pay nothing |

Action rules: donations, alliance breaking and embargoes are off
(`env.allow_donations`, `env.allow_betrayal`, `env.allow_embargo`), attacks must leave
10% of max troops at home (`env.troop_reserve`), and build sites are typed
by terrain (defense posts ~10 tiles behind a front with another player,
ports on the coast, cities and factories inside the territory). Training
games last 20 game minutes; evaluation games 60.

Nukes: atom bomb, hydrogen bomb and MIRV are three launch actions aimed at
a map cell with hostile land. They open when a silo is ready and the bomb
is affordable; the game launches from the closest ready silo. The AI sees
hostile SAM coverage, nukes in flight (its own and incoming), what it can
afford and its ready silos. Checkpoints from before nukes (schema v3) load
and resume: the new inputs start at zero weight, so the AI plays exactly as
before until it learns them.

Death reports: `OPENFRONT_DEATH_LOG=<file> python evaluate.py ...` appends
one JSON line per AI death (when, peak size, who took its land, troops
against its strongest neighbour, nukes aimed at it).

## Tests

```bash
cd env && npx vitest run
cd train && python -m pytest tests
cd train && python tools/local_lobby_smoke.py --seconds 60   # needs npm run dev
```
