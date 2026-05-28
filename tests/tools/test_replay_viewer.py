import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest


def test_orbit_wars_replay_viewer_uses_kaggle_renderer_style() -> None:
    viewer = Path(__file__).parents[2] / "tools" / "orbit_wars_replay_viewer.html"
    html = viewer.read_text(encoding="utf-8")

    assert 'const PLAYER_COLORS = ["#0072B2", "#D55E00", "#009E73", "#F0E442"]' in html
    assert 'const NEUTRAL_COLOR = "#888888"' in html
    assert "rgba(255, 200, 50, 0.6)" in html
    assert "drawCometTrails" in html
    assert "drawFleets" in html
    assert "Fleet #" in html
    assert "Production" in html
    assert "Ships / turn" in html
    assert "Production / turn" in html
    assert "timelineFrameFromPointer" in html
    assert 'id="timelineChart"' in html
    assert "Open replay JSON or JSONL" in html
    assert "Open replay JSON or JSONL to plot turns" in html


def test_orbit_wars_replay_viewer_accepts_kaggle_replay_json() -> None:
    if shutil.which("node") is None:
        pytest.skip("node is required to execute replay viewer JavaScript")

    viewer = Path(__file__).parents[2] / "tools" / "orbit_wars_replay_viewer.html"
    script = textwrap.dedent(
        """
        const fs = require("fs");
        const vm = require("vm");
        const html = fs.readFileSync(process.argv[2], "utf8");
        const source = html.match(/<script>([\\s\\S]*)<\\/script>/)[1];
        function element() {
          return {
            children: [],
            className: "",
            innerHTML: "",
            style: {},
            textContent: "",
            value: "0",
            options: [],
            selectedIndex: 0,
            addEventListener() {},
            append(...nodes) { this.children.push(...nodes); },
            appendChild(node) { this.children.push(node); return node; },
            replaceChildren(...nodes) { this.children = [...nodes]; },
            classList: { add() {}, remove() {}, toggle() {} },
            getBoundingClientRect() {
              return { width: 300, height: 180, left: 0 };
            },
            getContext() {
              return {
                scale() {},
                fillRect() {},
                fillText() {},
                beginPath() {},
                moveTo() {},
                lineTo() {},
                stroke() {},
              };
            },
            querySelector() { return null; },
            setPointerCapture() {},
            hasPointerCapture() { return false; },
            releasePointerCapture() {},
          };
        }
        const context = {
          console,
          JSON,
          Math,
          Map,
          Set,
          Array,
          Number,
          String,
          window: {
            devicePixelRatio: 1,
            addEventListener() {},
            clearInterval() {},
            setInterval() {},
          },
          document: {
            getElementById() { return element(); },
            querySelectorAll() { return []; },
            createElement() { return element(); },
            createTextNode(text) { return { textContent: String(text) }; },
          },
        };
        vm.createContext(context);
        vm.runInContext(source, context);
        const replay = {
          id: 76093780,
          info: { EpisodeId: 76093780, TeamNames: ["Alpha", "Beta"], seed: 123 },
          configuration: { episodeSteps: 500 },
          steps: [
            [
              {
                observation: {
                  step: 0,
                  planets: [[0, 0, 10, 10, 1, 12, 1]],
                  fleets: [],
                  comets: [],
                },
                reward: 0,
                status: "ACTIVE",
              },
              { observation: { step: 0 }, reward: 0, status: "ACTIVE" },
            ],
            [
              {
                observation: {
                  step: 1,
                  planets: [[0, 1, 10, 10, 1, 14, 1]],
                  fleets: [],
                  comets: [],
                },
                reward: -1,
                status: "DONE",
              },
              { observation: { step: 1 }, reward: 1, status: "DONE" },
            ],
          ],
        };
        const rows = context.parseReplayText(JSON.stringify(replay, null, 2));
        if (rows.length !== 1) throw new Error(`expected one row, got ${rows.length}`);
        const row = rows[0];
        if (row.source !== "kaggle") throw new Error(`unexpected source ${row.source}`);
        if (row.player_count !== 2) {
          throw new Error(`unexpected player count ${row.player_count}`);
        }
        if (row.frames.length !== 2) {
          throw new Error(`unexpected frame count ${row.frames.length}`);
        }
        if (row.frames[1].terminal !== true) {
          throw new Error("final frame should be terminal");
        }
        if (context.playerName(row, 1) !== "Beta") {
          throw new Error("Kaggle team names should label players");
        }
        """
    )

    subprocess.run(["node", "-", str(viewer)], input=script, text=True, check=True)
