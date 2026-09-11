/* Phase-8 replay loader. Zero changes to the game client needed.
 *
 * Setup, once:
 *   1. Boot the vendored client dev server (`npm run dev` in OpenFrontIO/)
 *      and open the page it serves.
 *   2. Make a new bookmark, paste the minified line at the bottom of this
 *      file as its URL. Or paste the readable function below into the dev
 *      console on the game page.
 *
 * Use: click the bookmark on the game page, pick a recorded episode JSON
 * (train/replays/...), and the game loads it as a replay. Drive with the
 * built-in replay speed controls. The dev build skips the version check,
 * so training records always load.
 */
(function loadTrainingReplay() {
  const picker = document.createElement("input");
  picker.type = "file";
  picker.accept = ".json,application/json";
  picker.onchange = async () => {
    const file = picker.files[0];
    if (!file) return;
    let record;
    try {
      record = JSON.parse(await file.text());
    } catch {
      alert("Not a JSON file: " + file.name);
      return;
    }
    if (!record || !record.info || !Array.isArray(record.turns)) {
      alert("Not a replay file: missing info/turns: " + file.name);
      return;
    }
    document.dispatchEvent(
      new CustomEvent("join-lobby", {
        detail: {
          gameID: record.info.gameID,
          gameRecord: record,
          source: "private",
        },
      }),
    );
  };
  picker.click();
})();

// Bookmarklet URL (minified). Create a bookmark with this as the address:
// javascript:(function(){const p=document.createElement("input");p.type="file";p.accept=".json,application/json";p.onchange=async()=>{const f=p.files[0];if(!f)return;let r;try{r=JSON.parse(await f.text())}catch{alert("Not a JSON file: "+f.name);return}if(!r||!r.info||!Array.isArray(r.turns)){alert("Not a replay file: "+f.name);return}document.dispatchEvent(new CustomEvent("join-lobby",{detail:{gameID:r.info.gameID,gameRecord:r,source:"private"}}))};p.click()})();
