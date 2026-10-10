import pytest

from openfront.lobby import pick_public, resolve_lobby, summarize_lobbies


@pytest.mark.parametrize("raw", [
    "https://openfront.io/w3/game/cP9iiNqCDz",
    "https://openfront.io/w3/game/cP9iiNqCDz?lobby&s=x1",
    "openfront.io/game/cP9iiNqCDz?live",
])
def test_openfront_links_with_ten_character_ids(raw):
    assert resolve_lobby(raw, "local") == ("https://openfront.io", "cP9iiNqCDz")


def test_bare_ids_of_both_lengths():
    assert resolve_lobby("AbCd1234", "local") == ("http://localhost:9000", "AbCd1234")
    assert resolve_lobby("cP9iiNqCDz", "openfront") == ("https://openfront.io", "cP9iiNqCDz")
    with pytest.raises(ValueError):
        resolve_lobby("abc", "local")


def payload():
    def game(gid, players, cap, starts=None):
        g = {"gameID": gid, "numClients": players, "gameConfig": {"gameMap": "World", "gameMode": "Free For All",
                                                                  "maxPlayers": cap}}
        if starts is not None:
            g["startsAt"] = 1_000_000 + starts * 1000
        return g
    return {"serverTime": 1_000_000, "games": {
        "ffa": [game("soonFFA001", 10, 30, starts=5), game("nextFFA002", 4, 30, starts=40),
                game("fullFFA003", 30, 30, starts=60), game("waitFFA004", 2, 30)],
        "team": [game("team000001", 20, 60, starts=20)],
        "hosted": [game("hosted0001", 3, 100, starts=15)],
    }}


def test_summary_rows_sorted_by_start():
    rows = summarize_lobbies(payload())
    assert rows[0]["id"] == "soonFFA001" and rows[0]["starts_in"] == 5
    assert rows[-1]["starts_in"] is None


def test_pick_skips_too_soon_full_and_hosted():
    rows = summarize_lobbies(payload())
    assert pick_public(rows, "ffa")["id"] == "nextFFA002"
    assert pick_public(rows, "any")["id"] == "team000001"
    assert pick_public([r for r in rows if r["id"] != "nextFFA002"], "ffa")["id"] == "waitFFA004"
    assert pick_public(rows, "special") is None
