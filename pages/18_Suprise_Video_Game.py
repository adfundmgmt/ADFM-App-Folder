import json
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Mini Classic RPG (3D)", layout="wide")
st.title("Mini Classic RPG (3D)")

with st.sidebar:
    st.subheader("Settings")
    player_class = st.selectbox("Class", ["Warrior", "Mage", "Rogue"], index=0)
    difficulty = st.selectbox("Difficulty", ["Chill", "Normal", "Hard"], index=1)
    quality = st.selectbox("Graphics", ["Ultra", "High", "Medium", "Low"], index=1)
    show_fps = st.toggle("Show FPS", value=False)
    reduce_motion = st.toggle("Reduce motion", value=False)
    seed = st.number_input("Seed", min_value=1, max_value=999999999, value=1337, step=1)
    music_on = st.toggle("Ambient soundtrack", value=True)
    music_vol = st.slider("Soundtrack volume", min_value=0, max_value=100, value=35, step=1)
    reset_save = st.button("Reset save")

cfg = {
    "player_class": player_class,
    "difficulty": difficulty,
    "quality": quality,
    "show_fps": show_fps,
    "reduce_motion": reduce_motion,
    "seed": int(seed),
    "reset_save": bool(reset_save),
    "music_on": bool(music_on),
    "music_vol": int(music_vol),
}

st.caption("WASD move | Click to target | Space interact/loot/sell | 1-4 abilities | I inventory | Q quest log | R respawn | P pause")

html = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    html, body { margin:0; padding:0; background: transparent; }
    .wrap { width:100%; display:flex; justify-content:center; }
    .frame { position: relative; width: min(1400px, 98vw); }
    #webgl {
      width: 100%;
      height: auto;
      display:block;
      border-radius: 20px;
      border: 1px solid rgba(255,255,255,0.10);
      background: radial-gradient(1100px 700px at 30% 20%, rgba(120,170,255,0.14), transparent),
                  radial-gradient(1000px 650px at 78% 22%, rgba(255,120,200,0.10), transparent),
                  linear-gradient(180deg, rgba(8,10,14,1), rgba(4,5,7,1));
      box-shadow: 0 18px 60px rgba(0,0,0,0.55);
      outline:none;
    }

    .hudTop {
      position:absolute;
      left: 14px;
      right: 14px;
      top: 14px;
      display:flex;
      justify-content:space-between;
      gap: 10px;
      pointer-events:none;
      z-index: 10;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      color: rgba(245,245,250,0.92);
      font-size: 13px;
    }
    .pill {
      padding: 10px 12px;
      border-radius: 14px;
      border: 1px solid rgba(255,255,255,0.10);
      background: rgba(255,255,255,0.06);
      backdrop-filter: blur(8px);
      box-shadow: 0 10px 30px rgba(0,0,0,0.30);
      white-space: nowrap;
    }

    .unitFrames {
      position:absolute;
      left: 14px;
      top: 56px;
      display:flex;
      gap: 10px;
      z-index: 11;
      pointer-events:none;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      color: rgba(245,245,250,0.92);
    }
    .frameCard {
      width: 280px;
      border-radius: 16px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.06);
      backdrop-filter: blur(10px);
      padding: 10px 10px 8px 10px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    }
    .frameRow { display:flex; justify-content:space-between; align-items:center; gap: 10px; }
    .name { font-weight: 760; font-size: 13.5px; }
    .lvl { opacity: 0.82; font-size: 12.5px; }
    .barWrap { margin-top: 7px; height: 10px; border-radius: 999px; background: rgba(0,0,0,0.30); border: 1px solid rgba(255,255,255,0.10); overflow:hidden; }
    .bar { height: 100%; width: 50%; border-radius: 999px; }

    .xp {
      position:absolute;
      left: 14px;
      right: 14px;
      bottom: 14px;
      height: 10px;
      border-radius: 999px;
      background: rgba(0,0,0,0.38);
      border: 1px solid rgba(255,255,255,0.12);
      overflow:hidden;
      z-index: 12;
      pointer-events:none;
    }
    .xpFill { height:100%; width: 0%; background: linear-gradient(90deg, rgba(92,255,176,0.92), rgba(120,170,255,0.80)); }

    .hotbar {
      position:absolute;
      left: 50%;
      bottom: 28px;
      transform: translateX(-50%);
      display:flex;
      gap: 10px;
      z-index: 13;
      pointer-events:none;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    }
    .slot {
      width: 54px; height: 54px;
      border-radius: 16px;
      border: 1px solid rgba(255,255,255,0.14);
      background: rgba(255,255,255,0.06);
      box-shadow: 0 10px 30px rgba(0,0,0,0.35);
      position:relative;
      overflow:hidden;
    }
    .slotKey {
      position:absolute; left: 8px; top: 6px;
      font-size: 12px; opacity: 0.88;
    }
    .slotName {
      position:absolute; left: 8px; bottom: 6px;
      font-size: 10.5px; opacity: 0.86;
    }
    .cd {
      position:absolute; inset:0;
      background: rgba(0,0,0,0.55);
      transform: translateY(0%);
    }

    .panel {
      position:absolute;
      right: 14px;
      top: 56px;
      width: 360px;
      border-radius: 18px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(8,10,14,0.70);
      backdrop-filter: blur(10px);
      box-shadow: 0 16px 50px rgba(0,0,0,0.55);
      padding: 12px 12px 10px 12px;
      z-index: 14;
      color: rgba(245,245,250,0.92);
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      display:none;
      pointer-events:auto;
    }
    .panelTitle { font-size: 15px; font-weight: 780; margin-bottom: 8px; }
    .panelSub { opacity:0.82; font-size: 13px; line-height:1.35; }
    .invGrid { display:grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 10px; }
    .invItem {
      height: 58px;
      border-radius: 16px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.06);
      padding: 8px;
      font-size: 12px;
      display:flex; flex-direction:column; justify-content:space-between;
      user-select:none;
    }
    .invItem span { opacity:0.82; font-size: 11.5px; }

    .chat {
      position:absolute;
      left: 14px;
      bottom: 44px;
      width: 420px;
      max-height: 200px;
      overflow:hidden;
      border-radius: 18px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(8,10,14,0.62);
      backdrop-filter: blur(8px);
      box-shadow: 0 16px 50px rgba(0,0,0,0.45);
      padding: 10px 10px 8px 10px;
      z-index: 14;
      color: rgba(245,245,250,0.92);
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 12px;
      pointer-events:none;
    }
    .line { opacity:0.92; margin-bottom: 4px; white-space:nowrap; text-overflow:ellipsis; overflow:hidden; }

    .minimap {
      position:absolute;
      right: 14px;
      bottom: 44px;
      width: 180px;
      height: 180px;
      border-radius: 18px;
      border: 1px solid rgba(255,255,255,0.14);
      background: rgba(255,255,255,0.06);
      backdrop-filter: blur(10px);
      box-shadow: 0 16px 50px rgba(0,0,0,0.55);
      z-index: 14;
      pointer-events:none;
    }

    .overlay {
      position:absolute;
      inset:0;
      display:flex;
      align-items:center;
      justify-content:center;
      pointer-events:none;
      z-index: 20;
      color: rgba(245,245,250,0.92);
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    }
    .card {
      width: min(820px, 92%);
      border-radius: 20px;
      border: 1px solid rgba(255,255,255,0.14);
      background: rgba(255,255,255,0.07);
      padding: 18px 18px 16px 18px;
      box-shadow: 0 18px 70px rgba(0,0,0,0.60);
      text-align:left;
    }
    .title { font-size: 22px; font-weight: 820; margin:0 0 8px 0; }
    .sub { opacity:0.90; margin:0 0 10px 0; line-height:1.45; }
    .kbd {
      display:inline-block;
      padding: 2px 8px;
      border-radius: 9px;
      border: 1px solid rgba(255,255,255,0.16);
      background: rgba(0,0,0,0.22);
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 12px;
      margin-right: 6px;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="frame">

      <div class="hudTop">
        <div class="pill" id="pillL">Zone: Greenhollow | Gold: 0 | Bags: 0/16</div>
        <div class="pill" id="pillM">Quest: ...</div>
        <div class="pill" id="pillR">Ping: 12ms</div>
      </div>

      <div class="unitFrames">
        <div class="frameCard">
          <div class="frameRow">
            <div class="name" id="pName">Adventurer</div>
            <div class="lvl" id="pLvl">Lv 1</div>
          </div>
          <div class="barWrap"><div class="bar" id="pHP" style="background: linear-gradient(90deg, rgba(92,255,176,0.92), rgba(92,255,176,0.55)); width: 100%;"></div></div>
          <div class="barWrap" style="margin-top:6px;"><div class="bar" id="pMP" style="background: linear-gradient(90deg, rgba(120,170,255,0.90), rgba(120,170,255,0.55)); width: 100%;"></div></div>
        </div>

        <div class="frameCard" id="tFrame" style="display:none;">
          <div class="frameRow">
            <div class="name" id="tName">Target</div>
            <div class="lvl" id="tLvl">Lv 1</div>
          </div>
          <div class="barWrap"><div class="bar" id="tHP" style="background: linear-gradient(90deg, rgba(255,110,110,0.92), rgba(255,110,110,0.55)); width: 100%;"></div></div>
          <div class="panelSub" id="tNote" style="margin-top:7px;">Click to target. Get in range to fight.</div>
        </div>
      </div>

      <canvas id="webgl" width="1400" height="780" tabindex="0"></canvas>

      <div class="xp"><div class="xpFill" id="xpFill"></div></div>

      <div class="hotbar">
        <div class="slot"><div class="slotKey">1</div><div class="slotName" id="a1n">Ability</div><div class="cd" id="a1cd"></div></div>
        <div class="slot"><div class="slotKey">2</div><div class="slotName" id="a2n">Ability</div><div class="cd" id="a2cd"></div></div>
        <div class="slot"><div class="slotKey">3</div><div class="slotName" id="a3n">Ability</div><div class="cd" id="a3cd"></div></div>
        <div class="slot"><div class="slotKey">4</div><div class="slotName" id="a4n">Ability</div><div class="cd" id="a4cd"></div></div>
      </div>

      <div class="chat" id="chat"></div>
      <canvas class="minimap" id="minimap" width="180" height="180"></canvas>

      <div class="panel" id="invPanel">
        <div class="panelTitle">Inventory</div>
        <div class="panelSub" id="invSub">Click loot on the ground with Space.</div>
        <div class="invGrid" id="invGrid"></div>
      </div>

      <div class="panel" id="questPanel">
        <div class="panelTitle">Quest Log</div>
        <div class="panelSub" id="questText"></div>
      </div>

      <div class="overlay" id="overlay" style="display:flex;">
        <div class="card">
          <div class="title">Mini Classic RPG (3D)</div>
          <p class="sub">
            A classic forest zone loop built for Streamlit. Low poly 3D, real targeting, real quests, real progression.
          </p>
          <p class="sub">
            <span class="kbd">W</span><span class="kbd">A</span><span class="kbd">S</span><span class="kbd">D</span> move |
            <span class="kbd">Click</span> target |
            <span class="kbd">Space</span> interact/loot/sell |
            <span class="kbd">1-4</span> abilities |
            <span class="kbd">I</span> inventory |
            <span class="kbd">Q</span> quest |
            <span class="kbd">P</span> pause |
            <span class="kbd">R</span> respawn
          </p>
          <p class="sub" style="opacity:0.78; margin:0;">Click the game area, then press 1 to pull.</p>
        </div>
      </div>

    </div>
  </div>

  <script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>

<script>
(() => {
  const CONFIG = __CONFIG_JSON__;

  const canvas = document.getElementById("webgl");
  const overlay = document.getElementById("overlay");

  const pillL = document.getElementById("pillL");
  const pillM = document.getElementById("pillM");
  const pillR = document.getElementById("pillR");

  const pName = document.getElementById("pName");
  const pLvl  = document.getElementById("pLvl");
  const pHP   = document.getElementById("pHP");
  const pMP   = document.getElementById("pMP");

  const tFrame = document.getElementById("tFrame");
  const tName  = document.getElementById("tName");
  const tLvl   = document.getElementById("tLvl");
  const tHP    = document.getElementById("tHP");
  const tNote  = document.getElementById("tNote");

  const xpFill = document.getElementById("xpFill");

  const a1n = document.getElementById("a1n");
  const a2n = document.getElementById("a2n");
  const a3n = document.getElementById("a3n");
  const a4n = document.getElementById("a4n");
  const a1cd = document.getElementById("a1cd");
  const a2cd = document.getElementById("a2cd");
  const a3cd = document.getElementById("a3cd");
  const a4cd = document.getElementById("a4cd");

  const chatEl = document.getElementById("chat");
  const invPanel = document.getElementById("invPanel");
  const questPanel = document.getElementById("questPanel");
  const invGrid = document.getElementById("invGrid");
  const questText = document.getElementById("questText");

  const mm = document.getElementById("minimap");
  const mctx = mm.getContext("2d");

  const W = canvas.width;
  const H = canvas.height;

  if (typeof THREE === "undefined") {
    overlay.style.display = "flex";
    overlay.innerHTML = `
      <div class="card">
        <div class="title">3D library failed to load</div>
        <p class="sub">Your environment blocked external scripts. If you deploy on Streamlit Cloud, CDN access typically works.</p>
        <p class="sub" style="opacity:0.78; margin:0;">If needed, I can refactor to bundle Three.js locally.</p>
      </div>
    `;
    return;
  }

  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  const lerp  = (a, b, t) => a + (b - a) * t;

  function mulberry32(a) {
    return function() {
      let t = a += 0x6D2B79F5;
      t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    }
  }

  function dist2(ax, ay, bx, by) {
    const dx = ax - bx, dy = ay - by;
    return Math.hypot(dx, dy);
  }

  const Q = (() => {
    const rm = !!CONFIG.reduce_motion;
    const q = (CONFIG.quality || "High").toLowerCase();
    if (q === "ultra") return { grass: 1100, trees: 70, fog: 0.00085, shakeMul: rm ? 0.55 : 1.0 };
    if (q === "high")  return { grass: 850,  trees: 58, fog: 0.0010,  shakeMul: rm ? 0.55 : 1.0 };
    if (q === "medium")return { grass: 650,  trees: 46, fog: 0.0012,  shakeMul: rm ? 0.45 : 0.85 };
    return               { grass: 450,  trees: 34, fog: 0.0015,  shakeMul: rm ? 0.40 : 0.75 };
  })();

  const D = (() => {
    const d = (CONFIG.difficulty || "Normal").toLowerCase();
    if (d === "chill")  return { mobHp: 0.85, mobDmg: 0.85, mobSp: 0.92, xp: 1.10, gold: 1.05 };
    if (d === "hard")   return { mobHp: 1.18, mobDmg: 1.18, mobSp: 1.08, xp: 0.95, gold: 0.95 };
    return               { mobHp: 1.00, mobDmg: 1.00, mobSp: 1.00, xp: 1.00, gold: 1.00 };
  })();

  const SAVE_KEY = "mini_classic_rpg_3d_save_v2";

  function loadSave() {
    try {
      const raw = localStorage.getItem(SAVE_KEY);
      if (!raw) return null;
      return JSON.parse(raw);
    } catch {
      return null;
    }
  }
  function saveGame() {
    try {
      const payload = {
        gold: state.gold,
        lvl: player.lvl,
        xp: player.xp,
        xpNeed: player.xpNeed,
        inv: inventory.slice(0, 16),
        questIndex: questIndex,
        questState: {
          have: activeQuest().have,
          done: activeQuest().done
        },
        seed: state.seed
      };
      localStorage.setItem(SAVE_KEY, JSON.stringify(payload));
    } catch {}
  }
  function clearSave() {
    try { localStorage.removeItem(SAVE_KEY); } catch {}
  }

  function chat(line) {
    chatLines.unshift(line);
    if (chatLines.length > 9) chatLines.pop();
    chatEl.innerHTML = chatLines.map(s => `<div class="line">${s}</div>`).join("");
  }

  const keys = new Set();
  let mouseN = { x: 0, y: 0 };

  function canvasToLocal(e) {
    const r = canvas.getBoundingClientRect();
    const x = (e.clientX - r.left) * (W / r.width);
    const y = (e.clientY - r.top) * (H / r.height);
    return { x, y };
  }

  const world = { w: 4200, h: 3600 };

  const landmarks = {
    camp:  { x: world.w * 0.5,  y: world.h * 0.5,  name: "Campfire" },
    pond:  { x: world.w * 0.68, y: world.h * 0.56, name: "Pond" },
    oak:   { x: world.w * 0.35, y: world.h * 0.72, name: "Old Oak" },
    hill:  { x: world.w * 0.56, y: world.h * 0.26, name: "Windy Hill" },
    shrine:{ x: world.w * 0.22, y: world.h * 0.33, name: "Stone Shrine" },
  };

  const state = {
    seed: CONFIG.seed || 1337,
    rng: mulberry32(CONFIG.seed || 1337),
    running: false,
    paused: false,
    t: 0,
    shake: 0,
    gold: 0,
    zone: "Greenhollow Forest",
    ping: 12
  };

  const player = {
    cls: (CONFIG.player_class || "Warrior"),
    name: "Adventurer",
    x: landmarks.camp.x,
    y: landmarks.camp.y,
    vx: 0,
    vy: 0,
    r: 24,

    lvl: 1,
    xp: 0,
    xpNeed: 120,

    hp: 100,
    hpMax: 100,

    mp: 60,
    mpMax: 60,

    armor: 0.10,

    gcd: 0,
    aaCd: 0,
    aaSpeed: 1.8,
    range: 110,

    targetId: null,

    a1: 0, a2: 0, a3: 0, a4: 0
  };

  const abilitiesByClass = {
    Warrior: [
      { key:"1", name:"Strike", cd: 3.5, cost: 0,  dmg: 22, type:"melee", color:0xff72dd },
      { key:"2", name:"Sunder", cd: 8.0, cost: 0,  dmg: 14, type:"melee", debuff:"armor", color:0x78aaff },
      { key:"3", name:"Shout",  cd: 20,  cost: 0,  buff:"armor", color:0x5cffb0 },
      { key:"4", name:"Charge", cd: 14,  cost: 0,  dash:true, dmg: 10, type:"melee", color:0xffd45c }
    ],
    Mage: [
      { key:"1", name:"Frostbolt", cd: 2.2, cost: 12, dmg: 20, type:"ranged", slow:true, color:0x78d2ff },
      { key:"2", name:"Fireball",  cd: 3.8, cost: 16, dmg: 28, type:"ranged", dot:true,  color:0xffaa6e },
      { key:"3", name:"Nova",      cd: 16,  cost: 18, dmg: 14, aoe:true, root:true, color:0xff72dd },
      { key:"4", name:"Blink",     cd: 12,  cost: 10, blink:true, color:0x5cffb0 }
    ],
    Rogue: [
      { key:"1", name:"Stab",    cd: 2.6, cost: 0,  dmg: 20, type:"melee", color:0xff72dd },
      { key:"2", name:"Poison",  cd: 10,  cost: 0,  dmg: 10, type:"melee", dot:true,  color:0x5cffb0 },
      { key:"3", name:"Vanish",  cd: 22,  cost: 0,  stealth:true, color:0x78aaff },
      { key:"4", name:"Dash",    cd: 14,  cost: 0,  speed:true,   color:0xffd45c }
    ]
  };

  // 30 quests, chained progression
  const quests = [
    { id:1,  name:"Boars at the Meadow",       type:"kill",   target:"Boar", need:4,  have:0, done:false, rewardGold:18, rewardXP:90,  note:"Thin the boars near the pond." },
    { id:2,  name:"Wolf Tracks",              type:"kill",   target:"Wolf", need:3,  have:0, done:false, rewardGold:22, rewardXP:110, note:"Wolves roam the deeper shade." },
    { id:3,  name:"Torn Hide Collection",     type:"collect",item:"Torn Hide", need:4, have:0, done:false, rewardGold:26, rewardXP:120, note:"Loot hides and bring them back." },
    { id:4,  name:"Visit the Pond",           type:"visit",  spot:"pond", need:1, have:0, done:false, rewardGold:20, rewardXP:120, note:"Touch the water and return." },
    { id:5,  name:"Boar Cleanup",             type:"kill",   target:"Boar", need:6, have:0, done:false, rewardGold:34, rewardXP:160, note:"The camp wants the meadow clear." },
    { id:6,  name:"Fangs for the Fire",       type:"collect",item:"Worn Fang", need:5, have:0, done:false, rewardGold:38, rewardXP:170, note:"Fangs make good charms." },
    { id:7,  name:"Stone Shrine Check",       type:"visit",  spot:"shrine", need:1, have:0, done:false, rewardGold:28, rewardXP:160, note:"See if the shrine is intact." },
    { id:8,  name:"Wolves at Dusk",           type:"kill",   target:"Wolf", need:6, have:0, done:false, rewardGold:44, rewardXP:200, note:"They hunt when the light fades." },
    { id:9,  name:"Frayed Pelts",             type:"collect",item:"Frayed Pelt", need:6, have:0, done:false, rewardGold:48, rewardXP:210, note:"Bring pelts for warm wraps." },
    { id:10, name:"Old Oak Vigil",            type:"visit",  spot:"oak", need:1, have:0, done:false, rewardGold:35, rewardXP:200, note:"Listen to the forest near the oak." },

    { id:11, name:"Meadow Control",           type:"kill",   target:"Boar", need:8, have:0, done:false, rewardGold:60, rewardXP:250, note:"Boars keep coming back." },
    { id:12, name:"Shiny Trinkets",           type:"collect",item:"Shiny Trinket", need:3, have:0, done:false, rewardGold:66, rewardXP:260, note:"Bring anything that gleams." },
    { id:13, name:"Windy Hill Marker",        type:"visit",  spot:"hill", need:1, have:0, done:false, rewardGold:52, rewardXP:240, note:"Climb the hill, mark the path." },
    { id:14, name:"Wolf Alpha",               type:"kill",   target:"Wolf", need:8, have:0, done:false, rewardGold:78, rewardXP:300, note:"Push them back from the trails." },
    { id:15, name:"Tusk Run",                 type:"collect",item:"Boar Tusk", need:8, have:0, done:false, rewardGold:84, rewardXP:310, note:"The quartermaster wants tusks." },

    { id:16, name:"Boar Cull II",             type:"kill",   target:"Boar", need:10, have:0, done:false, rewardGold:98, rewardXP:360, note:"Finish what you started." },
    { id:17, name:"Charm of Green",           type:"collect",item:"Green Charm", need:2, have:0, done:false, rewardGold:120, rewardXP:400, note:"Rare charms drop sometimes." },
    { id:18, name:"Shrine and Return",        type:"visit",  spot:"shrine", need:1, have:0, done:false, rewardGold:90, rewardXP:340, note:"The stones feel warm tonight." },
    { id:19, name:"Wolves at the Oak",        type:"kill",   target:"Wolf", need:10, have:0, done:false, rewardGold:128, rewardXP:420, note:"They gather near the oak." },
    { id:20, name:"Pondside Offerings",       type:"visit",  spot:"pond", need:1, have:0, done:false, rewardGold:95, rewardXP:360, note:"Leave an offering, then return." },

    { id:21, name:"Meadow Lockdown",          type:"kill",   target:"Boar", need:12, have:0, done:false, rewardGold:150, rewardXP:480, note:"Hold the meadow for the camp." },
    { id:22, name:"Fangs and Pelts",          type:"collect",item:"Frayed Pelt", need:10, have:0, done:false, rewardGold:165, rewardXP:500, note:"The camp is building supplies." },
    { id:23, name:"Hill Watch",               type:"visit",  spot:"hill", need:1, have:0, done:false, rewardGold:120, rewardXP:460, note:"Scan the forest lines from above." },
    { id:24, name:"Wolves Break the Line",    type:"kill",   target:"Wolf", need:12, have:0, done:false, rewardGold:190, rewardXP:540, note:"They are bold now." },
    { id:25, name:"Trinkets for the Elder",   type:"collect",item:"Shiny Trinket", need:6, have:0, done:false, rewardGold:210, rewardXP:560, note:"The elder pays well for shine." },

    { id:26, name:"The Boar King (Rumor)",    type:"kill",   target:"Boar", need:14, have:0, done:false, rewardGold:240, rewardXP:620, note:"No king exists. Prove it." },
    { id:27, name:"The Oak Remembers",        type:"visit",  spot:"oak", need:1, have:0, done:false, rewardGold:170, rewardXP:560, note:"Stand still. Let the place speak." },
    { id:28, name:"Wolf Pack Collapse",       type:"kill",   target:"Wolf", need:14, have:0, done:false, rewardGold:280, rewardXP:700, note:"Break their confidence." },
    { id:29, name:"Green Charms II",          type:"collect",item:"Green Charm", need:4, have:0, done:false, rewardGold:340, rewardXP:820, note:"Luck favors the persistent." },
    { id:30, name:"Forest Secured",           type:"visit",  spot:"camp", need:1, have:0, done:false, rewardGold:420, rewardXP:1000, note:"Return to camp. Collect the final reward." },
  ];

  let questIndex = 0;
  function activeQuest() { return quests[clamp(questIndex, 0, quests.length - 1)]; }

  let mobs = [];
  let loot = [];
  let chatLines = [];
  let inventory = [];

  function invInit() {
    inventory = [];
    for (let i = 0; i < 16; i++) inventory.push(null);
  }
  function invAdd(item) {
    for (let i = 0; i < inventory.length; i++) {
      if (!inventory[i]) { inventory[i] = item; return true; }
    }
    return false;
  }
  function invCount() {
    let n = 0;
    for (const it of inventory) if (it) n++;
    return n;
  }
  function invCountItem(name) {
    let n = 0;
    for (const it of inventory) if (it && it.name === name) n++;
    return n;
  }
  function invRender() {
    invGrid.innerHTML = "";
    for (let i = 0; i < 16; i++) {
      const it = inventory[i];
      const div = document.createElement("div");
      div.className = "invItem";
      if (!it) div.innerHTML = `<div>Empty</div><span>Slot ${i+1}</span>`;
      else div.innerHTML = `<div>${it.name}</div><span>Value ${it.value}g</span>`;
      invGrid.appendChild(div);
    }
    document.getElementById("invSub").textContent = `Bags: ${invCount()}/16 | Space to loot nearby`;
  }

  const items = [
    { name: "Torn Hide", value: 2 },
    { name: "Boar Tusk", value: 3 },
    { name: "Worn Fang", value: 2 },
    { name: "Frayed Pelt", value: 2 },
    { name: "Shiny Trinket", value: 6 },
    { name: "Green Charm", value: 12 }
  ];

  function abilitiesSetup() {
    const a = abilitiesByClass[player.cls];
    a1n.textContent = a[0].name;
    a2n.textContent = a[1].name;
    a3n.textContent = a[2].name;
    a4n.textContent = a[3].name;

    if (player.cls === "Warrior") { player.hpMax = 120; player.mpMax = 40; player.aaSpeed = 1.9; player.range = 120; player.armor = 0.18; }
    if (player.cls === "Mage")    { player.hpMax = 95;  player.mpMax = 95; player.aaSpeed = 2.1; player.range = 260; player.armor = 0.07; }
    if (player.cls === "Rogue")   { player.hpMax = 105; player.mpMax = 55; player.aaSpeed = 1.6; player.range = 120; player.armor = 0.11; }
    player.hp = player.hpMax;
    player.mp = player.mpMax;
  }

  function levelUp() {
    player.lvl += 1;
    player.xp -= player.xpNeed;
    player.xpNeed = Math.floor(player.xpNeed * 1.28 + 40);
    player.hpMax = Math.floor(player.hpMax * 1.10 + 12);
    player.mpMax = Math.floor(player.mpMax * 1.07 + 8);
    player.hp = player.hpMax;
    player.mp = player.mpMax;
    chat(`<span style="color:rgba(92,255,176,0.92)">DING</span> You reached level ${player.lvl}.`);
    state.shake = Math.min(14, state.shake + 7);
  }
  function giveXP(n) {
    player.xp += Math.floor(n * D.xp);
    while (player.xp >= player.xpNeed) levelUp();
  }

  function questProgressFromInventory() {
    const q = activeQuest();
    if (q.type !== "collect") return;
    q.have = Math.min(q.need, invCountItem(q.item));
    if (q.have >= q.need) q.done = true;
  }

  function tryCompleteQuest() {
    const q = activeQuest();
    if (q.done) return;

    if (q.type === "collect") {
      questProgressFromInventory();
      return;
    }

    if (q.type === "visit") {
      const spot = landmarks[q.spot];
      if (!spot) return;
      if (dist2(player.x, player.y, spot.x, spot.y) < 140) {
        q.have = 1;
        q.done = true;
        chat(`<span style="color:rgba(255,212,92,0.95)">Quest</span> Objective complete.`);
        state.shake = Math.min(14, state.shake + 6);
      }
      return;
    }
  }

  function turnInQuestAtCamp() {
    const q = activeQuest();
    if (!q.done) return;
    if (dist2(player.x, player.y, landmarks.camp.x, landmarks.camp.y) > 160) return;

    state.gold += q.rewardGold;
    giveXP(q.rewardXP);

    chat(`<span style="color:rgba(255,212,92,0.95)">Quest</span> Turned in: ${q.name}.`);
    questIndex = clamp(questIndex + 1, 0, quests.length - 1);
    const nq = activeQuest();
    chat(`<span style="color:rgba(120,170,255,0.92)">New quest</span>: ${nq.name}.`);

    saveGame();
  }

  // Mob system
  function newMob(kind, lvl, x, y) {
    const base = (kind === "Boar") ? { hp: 60, dmg: 8, sp: 120 } : { hp: 80, dmg: 10, sp: 115 };
    const hp = Math.floor((base.hp + lvl * 10) * D.mobHp);
    return {
      id: "m" + Math.floor(state.rng() * 1e9),
      kind, lvl, x, y,
      vx: 0, vy: 0,
      r: kind === "Boar" ? 30 : 34,
      hp, hpMax: hp,
      dmg: Math.floor((base.dmg + lvl * 1.6) * D.mobDmg),
      sp: (base.sp + lvl * 3.0) * D.mobSp,
      aggro: false,
      leashX: x,
      leashY: y,
      leash: 520,
      hit: 0,
      slow: 0,
      root: 0,
      dot: 0,
      dotT: 0,
      armorDown: 0,
      mesh: null,
      ring: null
    };
  }

  function spawnPack() {
    mobs = [];

    const count = 14 + Math.floor(state.rng() * 10);
    for (let i = 0; i < count; i++) {
      const roll = state.rng();
      const kind = roll < 0.62 ? "Boar" : "Wolf";

      // region bias: boars near pond/meadow, wolves near oak/woods
      const anchor = (kind === "Boar") ? landmarks.pond : landmarks.oak;
      const spread = (kind === "Boar") ? 560 : 720;

      const lvl = clamp(1 + Math.floor(i/4) + Math.floor(state.rng() * 2), 1, 14);
      const x = clamp(anchor.x + (state.rng() - 0.5) * spread, 180, world.w - 180);
      const y = clamp(anchor.y + (state.rng() - 0.5) * spread, 180, world.h - 180);

      mobs.push(newMob(kind, lvl, x, y));
    }
  }

  function dropLoot(m) {
    let it = items[Math.floor(state.rng() * (items.length - 1))];
    if (state.rng() > 0.93) it = { name: "Green Charm", value: 12 };

    loot.push({
      id: "l" + Math.floor(state.rng() * 1e9),
      x: m.x,
      y: m.y,
      r: 20,
      item: it,
      t: state.rng() * Math.PI * 2,
      mesh: null
    });
  }

  function lootNearby() {
    let picked = false;
    for (let i = loot.length - 1; i >= 0; i--) {
      const l = loot[i];
      if (dist2(player.x, player.y, l.x, l.y) < 120) {
        if (invAdd(l.item)) {
          if (l.mesh) scene.remove(l.mesh);
          loot.splice(i, 1);
          chat(`Looted <span style="color:rgba(255,212,92,0.95)">${l.item.name}</span>.`);
          picked = true;
        } else {
          chat(`<span style="color:rgba(255,110,110,0.95)">Bags full</span>.`);
          break;
        }
      }
    }
    if (picked) {
      invRender();
      questProgressFromInventory();
      if (activeQuest().done) chat(`<span style="color:rgba(255,212,92,0.95)">Quest</span> Objective complete.`);
      saveGame();
    }
  }

  function sellJunk() {
    if (dist2(player.x, player.y, landmarks.camp.x, landmarks.camp.y) > 160) return;
    let gold = 0;
    for (let i = 0; i < inventory.length; i++) {
      const it = inventory[i];
      if (it) { gold += it.value; inventory[i] = null; }
    }
    if (gold > 0) {
      const g = Math.floor(gold * D.gold);
      state.gold += g;
      chat(`Sold junk for <span style="color:rgba(255,212,92,0.95)">${g}g</span>.`);
      invRender();
      saveGame();
    }
  }

  function hitMob(m, dmg, fx) {
    m.aggro = true;
    m.hit = 1.0;
    const armorMul = (m.armorDown > 0) ? 1.12 : 1.0;
    m.hp -= Math.floor(dmg * armorMul);

    if (fx.slow) m.slow = Math.max(m.slow, 2.5);
    if (fx.root) m.root = Math.max(m.root, 1.4);
    if (fx.dot)  m.dot  = Math.max(m.dot, 6.0);
    if (fx.armorDown) m.armorDown = Math.max(m.armorDown, 6.0);

    state.shake = Math.min(12, state.shake + 4);

    if (m.hp <= 0) mobDie(m);
  }

  function mobDie(m) {
    mobs = mobs.filter(z => z.id !== m.id);
    if (m.mesh) scene.remove(m.mesh);
    if (m.ring) scene.remove(m.ring);

    giveXP(34 + m.lvl * 12);

    const q = activeQuest();
    if (!q.done && q.type === "kill" && q.target === m.kind) {
      q.have = clamp(q.have + 1, 0, q.need);
      if (q.have >= q.need) {
        q.done = true;
        chat(`<span style="color:rgba(255,212,92,0.95)">Quest</span> Objective complete.`);
      }
    }

    dropLoot(m);
    if (player.targetId === m.id) player.targetId = null;

    saveGame();
  }

  let armorBuffT = 0;
  let speedBuffT = 0;

  function useAbility(slot) {
    if (!state.running || state.paused) return;
    const a = abilitiesByClass[player.cls][slot - 1];

    if (player.gcd > 0) return;
    const cdKey = "a" + slot;
    if (player[cdKey] > 0) return;

    if (a.cost && player.mp < a.cost) {
      chat(`<span style="color:rgba(255,110,110,0.95)">Not enough mana</span>.`);
      return;
    }

    if (a.blink) {
      player.mp -= a.cost;
      player[cdKey] = a.cd;
      player.gcd = 0.35;
      const ang = Math.atan2(mouseN.y, mouseN.x);
      player.x += Math.cos(ang) * 260;
      player.y += Math.sin(ang) * 260;
      player.x = clamp(player.x, 80, world.w - 80);
      player.y = clamp(player.y, 80, world.h - 80);
      chat(`Cast <span style="color:rgba(92,255,176,0.85)">${a.name}</span>.`);
      return;
    }

    if (a.stealth) {
      player[cdKey] = a.cd;
      player.gcd = 0.35;
      player.armor = Math.min(0.35, player.armor + 0.08);
      chat(`Used <span style="color:rgba(120,170,255,0.80)">${a.name}</span>.`);
      return;
    }

    if (a.speed) {
      player[cdKey] = a.cd;
      player.gcd = 0.35;
      speedBuffT = 5.0;
      chat(`Used <span style="color:rgba(255,212,92,0.92)">${a.name}</span>.`);
      return;
    }

    if (a.buff === "armor") {
      player[cdKey] = a.cd;
      player.gcd = 0.35;
      armorBuffT = 10.0;
      chat(`Cast <span style="color:rgba(92,255,176,0.90)">${a.name}</span>.`);
      return;
    }

    if (a.aoe) {
      player.mp -= a.cost;
      player[cdKey] = a.cd;
      player.gcd = 0.45;

      for (const m of mobs) {
        const d = dist2(player.x, player.y, m.x, m.y);
        if (d < 240) hitMob(m, a.dmg, { slow:false, root: !!a.root, dot:false, armorDown:false });
      }
      chat(`Cast <span style="color:rgba(255,140,230,0.75)">${a.name}</span>.`);
      return;
    }

    if (a.dash) {
      if (!player.targetId) { chat("No target."); return; }
      const m = mobs.find(z => z.id === player.targetId);
      if (!m) return;
      player[cdKey] = a.cd;
      player.gcd = 0.35;
      const dx = m.x - player.x;
      const dy = m.y - player.y;
      const d = Math.hypot(dx, dy) + 1e-6;
      player.x += (dx / d) * 260;
      player.y += (dy / d) * 260;
      hitMob(m, a.dmg, { slow:false, root:false, dot:false, armorDown:false });
      chat(`Used <span style="color:rgba(255,212,92,0.92)">${a.name}</span>.`);
      return;
    }

    if (!player.targetId) { chat("No target."); return; }
    const m = mobs.find(z => z.id === player.targetId);
    if (!m) return;

    const d = dist2(player.x, player.y, m.x, m.y);
    if (d > player.range) { chat("Out of range."); return; }

    player.mp -= (a.cost || 0);
    player[cdKey] = a.cd;
    player.gcd = 0.35;

    hitMob(m, a.dmg, { slow: !!a.slow, root: !!a.root, dot: !!a.dot, armorDown: (a.debuff === "armor") });
    chat(`Used <span style="color:rgba(245,245,250,0.92)">${a.name}</span>.`);
  }

  function autoAttack(dt) {
    if (!player.targetId) return;
    const m = mobs.find(z => z.id === player.targetId);
    if (!m) return;

    const d = dist2(player.x, player.y, m.x, m.y);
    if (d > player.range) return;

    player.aaCd -= dt;
    if (player.aaCd <= 0) {
      player.aaCd = player.aaSpeed;
      const base = (player.cls === "Mage") ? 10 : 14;
      hitMob(m, base + Math.floor(player.lvl * 1.6), { slow:false, root:false, dot:false, armorDown:false });
    }
  }

  let mobHitCd = 0;
  function mobAttack(dt, m) {
    mobHitCd -= dt;
    if (mobHitCd > 0) return;
    mobHitCd = 1.25;

    const armor = clamp(player.armor + (armorBuffT > 0 ? 0.06 : 0), 0, 0.45);
    const dmg = Math.floor(m.dmg * (1 - armor));
    player.hp -= dmg;

    state.shake = Math.min(16, state.shake + 8);
    chat(`${m.kind} hits you for <span style="color:rgba(255,110,110,0.95)">${dmg}</span>.`);

    if (player.hp <= 0) {
      player.hp = 0;
      state.running = false;
      overlay.style.display = "flex";
      overlay.innerHTML = `
        <div class="card">
          <div class="title">You died</div>
          <p class="sub">Press <span class="kbd">R</span> to respawn at the campfire.</p>
          <p class="sub" style="opacity:0.78; margin:0;">Tip: loot with Space, sell at campfire, turn in quest at campfire.</p>
        </div>
      `;
      saveGame();
    }
  }

  function mobAI(dt) {
    for (const m of mobs) {
      m.hit = Math.max(0, m.hit - 3.2 * dt);
      m.slow = Math.max(0, m.slow - dt);
      m.root = Math.max(0, m.root - dt);
      m.dot = Math.max(0, m.dot - dt);
      m.armorDown = Math.max(0, m.armorDown - dt);

      if (m.dot > 0) {
        m.dotT -= dt;
        if (m.dotT <= 0) {
          m.dotT = 1.0;
          m.hp -= 6 + Math.floor(player.lvl * 0.6);
          if (m.hp <= 0) { mobDie(m); continue; }
        }
      }

      const leashD = dist2(m.x, m.y, m.leashX, m.leashY);
      const aggroD = dist2(player.x, player.y, m.x, m.y);

      if (!m.aggro && aggroD < 280) m.aggro = true;

      let tx = m.leashX, ty = m.leashY;
      if (m.aggro) { tx = player.x; ty = player.y; }

      if (m.aggro && leashD > m.leash) {
        m.aggro = false;
        tx = m.leashX; ty = m.leashY;
      }

      const dx = tx - m.x;
      const dy = ty - m.y;
      const d = Math.hypot(dx, dy) + 1e-6;

      const slowMul = (m.slow > 0) ? 0.62 : 1.0;
      const rootMul = (m.root > 0) ? 0.0 : 1.0;

      const sp = m.sp * slowMul * rootMul;
      m.vx = lerp(m.vx, (dx / d) * sp, 0.08);
      m.vy = lerp(m.vy, (dy / d) * sp, 0.08);

      m.x += m.vx * dt;
      m.y += m.vy * dt;

      // melee
      if (m.aggro && aggroD < 92) mobAttack(dt, m);
    }
  }

  function respawn() {
    player.x = landmarks.camp.x;
    player.y = landmarks.camp.y;
    player.hp = player.hpMax;
    player.mp = player.mpMax;
    player.targetId = null;
    state.running = true;
    overlay.style.display = "none";
    chat("You feel the warmth of the campfire.");
  }

  function uiUpdate(fps=0) {
    pName.textContent = `${player.name} (${player.cls})`;
    pLvl.textContent = `Lv ${player.lvl}`;
    pHP.style.width = `${clamp(player.hp / player.hpMax, 0, 1) * 100}%`;
    pMP.style.width = `${clamp(player.mp / player.mpMax, 0, 1) * 100}%`;

    const xpT = clamp(player.xp / player.xpNeed, 0, 1) * 100;
    xpFill.style.width = `${xpT}%`;

    pillL.textContent = `Zone: ${state.zone} | Gold: ${state.gold} | Bags: ${invCount()}/16`;

    const q = activeQuest();
    if (q.type === "kill") pillM.textContent = q.done ? `Quest: ${q.name} (Complete)` : `Quest: ${q.name} (${q.have}/${q.need})`;
    if (q.type === "collect") pillM.textContent = q.done ? `Quest: ${q.name} (Complete)` : `Quest: ${q.name} (${q.have}/${q.need})`;
    if (q.type === "visit") pillM.textContent = q.done ? `Quest: ${q.name} (Complete)` : `Quest: ${q.name} (Go to ${landmarks[q.spot]?.name || q.spot})`;

    const extra = CONFIG.show_fps ? ` | FPS ${fps.toFixed(0)}` : "";
    pillR.textContent = `Ping: ${state.ping}ms${extra}`;

    const a = abilitiesByClass[player.cls];
    const cds = [
      { el: a1cd, v: player.a1, max: a[0].cd },
      { el: a2cd, v: player.a2, max: a[1].cd },
      { el: a3cd, v: player.a3, max: a[2].cd },
      { el: a4cd, v: player.a4, max: a[3].cd }
    ];
    for (const c of cds) {
      const t = clamp(c.v / c.max, 0, 1);
      c.el.style.transform = `translateY(${t * 100}%)`;
      c.el.style.display = (c.v > 0) ? "block" : "none";
    }

    if (player.targetId) {
      const m = mobs.find(z => z.id === player.targetId);
      if (m) {
        tFrame.style.display = "block";
        tName.textContent = `${m.kind}`;
        tLvl.textContent = `Lv ${m.lvl}`;
        tHP.style.width = `${clamp(m.hp / m.hpMax, 0, 1) * 100}%`;
        const d = dist2(player.x, player.y, m.x, m.y);
        tNote.textContent = d <= player.range ? "In range. Fight." : "Out of range. Move closer.";
      } else {
        tFrame.style.display = "none";
      }
    } else {
      tFrame.style.display = "none";
    }

    const q2 = activeQuest();
    if (q2.type === "kill") {
      questText.textContent = q2.done
        ? `Return to campfire to turn in. Reward ${q2.rewardGold}g and ${q2.rewardXP} XP.`
        : `${q2.note} Kill ${q2.need} ${q2.target}s. Progress ${q2.have}/${q2.need}. Turn in at camp.`;
    } else if (q2.type === "collect") {
      questText.textContent = q2.done
        ? `Return to campfire to turn in. Reward ${q2.rewardGold}g and ${q2.rewardXP} XP.`
        : `${q2.note} Collect ${q2.need}x ${q2.item}. You have ${q2.have}/${q2.need}. Turn in at camp.`;
    } else {
      questText.textContent = q2.done
        ? `Return to campfire to turn in. Reward ${q2.rewardGold}g and ${q2.rewardXP} XP.`
        : `${q2.note} Travel to ${landmarks[q2.spot]?.name || q2.spot}. Turn in at camp.`;
    }
  }

  // Minimap (2D overlay)
  function minimapDraw() {
    mctx.clearRect(0,0,mm.width,mm.height);
    mctx.save();
    mctx.fillStyle = "rgba(0,0,0,0.28)";
    mctx.fillRect(0,0,mm.width,mm.height);

    const sx = mm.width / world.w;
    const sy = mm.height / world.h;

    // landmarks
    const dots = [
      { p: landmarks.camp, c: "rgba(255,212,92,0.85)" },
      { p: landmarks.pond, c: "rgba(120,170,255,0.75)" },
      { p: landmarks.oak,  c: "rgba(92,255,176,0.70)" },
      { p: landmarks.hill, c: "rgba(255,170,110,0.70)" },
      { p: landmarks.shrine,c:"rgba(255,140,230,0.70)" },
    ];
    for (const d of dots) {
      mctx.fillStyle = d.c;
      mctx.beginPath();
      mctx.arc(d.p.x*sx, d.p.y*sy, 3.5, 0, Math.PI*2);
      mctx.fill();
    }

    // mobs
    for (const m of mobs) {
      mctx.fillStyle = m.aggro ? "rgba(255,110,110,0.80)" : "rgba(255,170,110,0.60)";
      mctx.fillRect(m.x*sx, m.y*sy, 2.2, 2.2);
    }

    // loot
    for (const l of loot) {
      mctx.fillStyle = "rgba(255,212,92,0.85)";
      mctx.fillRect(l.x*sx, l.y*sy, 2.2, 2.2);
    }

    // player
    mctx.fillStyle = "rgba(92,255,176,0.92)";
    mctx.beginPath();
    mctx.arc(player.x*sx, player.y*sy, 3.5, 0, Math.PI*2);
    mctx.fill();

    mctx.restore();
  }

  // 3D setup
  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x08110c, Q.fog);

  const camera = new THREE.PerspectiveCamera(55, W / H, 0.1, 6000);
  const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
  renderer.setSize(W, H, false);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const hemi = new THREE.HemisphereLight(0xb8ffd9, 0x08110c, 0.85);
  scene.add(hemi);

  const sun = new THREE.DirectionalLight(0xfff1cf, 0.85);
  sun.position.set(-800, 1200, -400);
  sun.castShadow = false;
  scene.add(sun);

  const fill = new THREE.DirectionalLight(0x88b9ff, 0.30);
  fill.position.set(900, 700, 600);
  scene.add(fill);

  const root = new THREE.Group();
  scene.add(root);

  const WORLD_ORIGIN = { x: world.w * 0.5, y: world.h * 0.5 };
  function to3(x, y) {
    return new THREE.Vector3(x - WORLD_ORIGIN.x, 0, y - WORLD_ORIGIN.y);
  }

  // ground
  const groundGeo = new THREE.PlaneGeometry(world.w, world.h, 60, 60);
  groundGeo.rotateX(-Math.PI/2);

  // vertex color variation for grassy feel
  const pos = groundGeo.attributes.position;
  const colors = [];
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const z = pos.getZ(i);
    const n = (Math.sin(x*0.004) + Math.cos(z*0.004) + Math.sin((x+z)*0.002)) * 0.25;
    const g = clamp(0.35 + 0.20*n, 0.18, 0.62);
    const r = clamp(0.08 + 0.05*n, 0.02, 0.18);
    const b = clamp(0.08 + 0.04*n, 0.02, 0.16);
    colors.push(r, g, b);
  }
  groundGeo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  const groundMat = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 1.0,
    metalness: 0.0
  });
  const ground = new THREE.Mesh(groundGeo, groundMat);
  ground.receiveShadow = false;
  root.add(ground);

  // pond
  const pondGeo = new THREE.CircleGeometry(240, 40);
  pondGeo.rotateX(-Math.PI/2);
  const pondMat = new THREE.MeshStandardMaterial({ color: 0x2a6aa5, roughness: 0.15, metalness: 0.15, transparent:true, opacity:0.85 });
  const pond = new THREE.Mesh(pondGeo, pondMat);
  const pondP = to3(landmarks.pond.x, landmarks.pond.y);
  pond.position.set(pondP.x, 0.05, pondP.z);
  root.add(pond);

  // hill
  const hillGeo = new THREE.ConeGeometry(260, 110, 24);
  const hillMat = new THREE.MeshStandardMaterial({ color: 0x1b3d24, roughness: 1.0, metalness: 0.0 });
  const hill = new THREE.Mesh(hillGeo, hillMat);
  const hillP = to3(landmarks.hill.x, landmarks.hill.y);
  hill.position.set(hillP.x, 55, hillP.z);
  root.add(hill);

  // shrine
  const shrine = new THREE.Group();
  const shrineP = to3(landmarks.shrine.x, landmarks.shrine.y);
  shrine.position.set(shrineP.x, 0, shrineP.z);
  const sMat = new THREE.MeshStandardMaterial({ color: 0x6a6f77, roughness: 0.95, metalness: 0.0 });
  const slab = new THREE.Mesh(new THREE.BoxGeometry(220, 26, 120), sMat);
  slab.position.y = 13;
  shrine.add(slab);
  const stone1 = new THREE.Mesh(new THREE.BoxGeometry(40, 110, 40), sMat);
  stone1.position.set(-70, 55, -20);
  shrine.add(stone1);
  const stone2 = new THREE.Mesh(new THREE.BoxGeometry(44, 140, 44), sMat);
  stone2.position.set(60, 70, 10);
  shrine.add(stone2);
  root.add(shrine);

  // old oak
  const oak = new THREE.Group();
  const oakP = to3(landmarks.oak.x, landmarks.oak.y);
  oak.position.set(oakP.x, 0, oakP.z);
  const trunkMat = new THREE.MeshStandardMaterial({ color: 0x3b2a1e, roughness: 1.0, metalness: 0.0 });
  const leafMat  = new THREE.MeshStandardMaterial({ color: 0x1b6b3a, roughness: 1.0, metalness: 0.0 });
  const trunk = new THREE.Mesh(new THREE.CylinderGeometry(36, 52, 240, 14), trunkMat);
  trunk.position.y = 120;
  oak.add(trunk);
  const crown = new THREE.Mesh(new THREE.SphereGeometry(170, 18, 14), leafMat);
  crown.position.y = 280;
  oak.add(crown);
  root.add(oak);

  // trees + grass
  const treeMat1 = new THREE.MeshStandardMaterial({ color: 0x194f2a, roughness: 1.0, metalness: 0.0 });
  const treeMat2 = new THREE.MeshStandardMaterial({ color: 0x1a3f2a, roughness: 1.0, metalness: 0.0 });
  const barkMat  = new THREE.MeshStandardMaterial({ color: 0x2a1d14, roughness: 1.0, metalness: 0.0 });

  function randInWorld() {
    return {
      x: 160 + state.rng() * (world.w - 320),
      y: 160 + state.rng() * (world.h - 320),
    };
  }

  for (let i = 0; i < Q.trees; i++) {
    const p = randInWorld();
    const dCamp = dist2(p.x, p.y, landmarks.camp.x, landmarks.camp.y);
    if (dCamp < 260) continue;

    const g = new THREE.Group();
    const t = to3(p.x, p.y);
    g.position.set(t.x, 0, t.z);

    const h = 140 + state.rng() * 170;
    const r = 14 + state.rng() * 20;

    const trunk = new THREE.Mesh(new THREE.CylinderGeometry(r*0.65, r, h, 10), barkMat);
    trunk.position.y = h * 0.5;
    g.add(trunk);

    const cone = new THREE.Mesh(new THREE.ConeGeometry(r * 4.0, h * 0.95, 10), state.rng() < 0.5 ? treeMat1 : treeMat2);
    cone.position.y = h * 0.95;
    g.add(cone);

    root.add(g);
  }

  const grassGeo = new THREE.PlaneGeometry(14, 36);
  const grassMat = new THREE.MeshStandardMaterial({ color: 0x2b8a4b, roughness: 1.0, metalness: 0.0, side: THREE.DoubleSide });
  for (let i = 0; i < Q.grass; i++) {
    const p = randInWorld();
    const dPond = dist2(p.x, p.y, landmarks.pond.x, landmarks.pond.y);
    const dCamp = dist2(p.x, p.y, landmarks.camp.x, landmarks.camp.y);
    if (dCamp < 220) continue;
    if (dPond < 240 && state.rng() < 0.7) continue;

    const blade = new THREE.Mesh(grassGeo, grassMat);
    const t = to3(p.x, p.y);
    blade.position.set(t.x, 18, t.z);
    blade.rotation.y = state.rng() * Math.PI;
    blade.rotation.x = -Math.PI/2 + (state.rng()-0.5)*0.25;
    blade.scale.setScalar(0.6 + state.rng() * 1.2);
    blade.material = grassMat.clone();
    blade.material.opacity = 0.75;
    blade.material.transparent = true;
    root.add(blade);
  }

  // models
  function makeWarrior() {
    const g = new THREE.Group();

    const armor = new THREE.MeshStandardMaterial({ color: 0x2a2f3a, roughness: 0.8, metalness: 0.35 });
    const skin  = new THREE.MeshStandardMaterial({ color: 0xffd1b3, roughness: 1.0, metalness: 0.0 });
    const cloth = new THREE.MeshStandardMaterial({ color: 0x2e6bff, roughness: 1.0, metalness: 0.0 });
    const steel = new THREE.MeshStandardMaterial({ color: 0xb9c1cf, roughness: 0.35, metalness: 0.85 });

    const torso = new THREE.Mesh(new THREE.BoxGeometry(44, 54, 26), armor);
    torso.position.y = 54;
    g.add(torso);

    const head = new THREE.Mesh(new THREE.SphereGeometry(16, 12, 10), skin);
    head.position.y = 94;
    g.add(head);

    const belt = new THREE.Mesh(new THREE.BoxGeometry(46, 10, 28), cloth);
    belt.position.y = 30;
    g.add(belt);

    const leg1 = new THREE.Mesh(new THREE.BoxGeometry(14, 30, 14), armor);
    leg1.position.set(-10, 15, 0);
    g.add(leg1);
    const leg2 = leg1.clone();
    leg2.position.set(10, 15, 0);
    g.add(leg2);

    const arm1 = new THREE.Mesh(new THREE.BoxGeometry(12, 34, 12), armor);
    arm1.position.set(-32, 58, 0);
    g.add(arm1);
    const arm2 = arm1.clone();
    arm2.position.set(32, 58, 0);
    g.add(arm2);

    const sword = new THREE.Mesh(new THREE.BoxGeometry(6, 70, 10), steel);
    sword.position.set(42, 50, 0);
    sword.rotation.z = 0.35;
    g.add(sword);

    return g;
  }

  function makeBoar() {
    const g = new THREE.Group();
    const bodyMat = new THREE.MeshStandardMaterial({ color: 0x5a3d2a, roughness: 1.0, metalness: 0.0 });
    const darkMat = new THREE.MeshStandardMaterial({ color: 0x2a1d14, roughness: 1.0, metalness: 0.0 });
    const boneMat = new THREE.MeshStandardMaterial({ color: 0xe7e2d3, roughness: 0.9, metalness: 0.0 });

    const body = new THREE.Mesh(new THREE.BoxGeometry(70, 34, 46), bodyMat);
    body.position.y = 26;
    g.add(body);

    const head = new THREE.Mesh(new THREE.BoxGeometry(34, 24, 28), bodyMat);
    head.position.set(46, 30, 0);
    g.add(head);

    const snout = new THREE.Mesh(new THREE.BoxGeometry(18, 14, 18), darkMat);
    snout.position.set(62, 24, 0);
    g.add(snout);

    const tusk1 = new THREE.Mesh(new THREE.CylinderGeometry(2.2, 2.2, 18, 8), boneMat);
    tusk1.position.set(62, 18, 10);
    tusk1.rotation.z = 1.2;
    g.add(tusk1);
    const tusk2 = tusk1.clone();
    tusk2.position.set(62, 18, -10);
    tusk2.rotation.z = -1.2;
    g.add(tusk2);

    for (const sx of [-20, 10, 34, 54]) {
      const leg = new THREE.Mesh(new THREE.BoxGeometry(10, 22, 10), darkMat);
      leg.position.set(sx, 11, 16);
      g.add(leg);
      const leg2 = leg.clone();
      leg2.position.z = -16;
      g.add(leg2);
    }

    return g;
  }

  function makeWolf() {
    const g = new THREE.Group();
    const fur = new THREE.MeshStandardMaterial({ color: 0x3a3e45, roughness: 1.0, metalness: 0.0 });
    const dark = new THREE.MeshStandardMaterial({ color: 0x1a1c20, roughness: 1.0, metalness: 0.0 });

    const body = new THREE.Mesh(new THREE.BoxGeometry(86, 30, 34), fur);
    body.position.y = 26;
    g.add(body);

    const neck = new THREE.Mesh(new THREE.BoxGeometry(24, 22, 24), fur);
    neck.position.set(48, 34, 0);
    g.add(neck);

    const head = new THREE.Mesh(new THREE.BoxGeometry(28, 20, 22), fur);
    head.position.set(64, 36, 0);
    g.add(head);

    const muzzle = new THREE.Mesh(new THREE.BoxGeometry(18, 12, 14), dark);
    muzzle.position.set(78, 32, 0);
    g.add(muzzle);

    const tail = new THREE.Mesh(new THREE.BoxGeometry(34, 10, 10), fur);
    tail.position.set(-60, 34, 0);
    tail.rotation.z = 0.25;
    g.add(tail);

    for (const sx of [-26, 4, 28, 52]) {
      const leg = new THREE.Mesh(new THREE.BoxGeometry(10, 22, 10), dark);
      leg.position.set(sx, 11, 12);
      g.add(leg);
      const leg2 = leg.clone();
      leg2.position.z = -12;
      g.add(leg2);
    }

    return g;
  }

  function makeLootMesh() {
    const mat = new THREE.MeshStandardMaterial({ color: 0xffd45c, roughness: 0.35, metalness: 0.75, emissive: 0x332200, emissiveIntensity: 0.6 });
    const geo = new THREE.TorusGeometry(10, 4.2, 10, 22);
    const m = new THREE.Mesh(geo, mat);
    m.rotation.x = Math.PI/2;
    return m;
  }

  function makeTargetRing() {
    const geo = new THREE.RingGeometry(34, 44, 40);
    const mat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.26, side: THREE.DoubleSide });
    const ring = new THREE.Mesh(geo, mat);
    ring.rotation.x = -Math.PI/2;
    ring.position.y = 0.2;
    return ring;
  }

  let playerMesh = null;

  function rebuildPlayerMesh() {
    if (playerMesh) root.remove(playerMesh);
    if (player.cls === "Warrior") playerMesh = makeWarrior();
    else if (player.cls === "Mage") playerMesh = makeWarrior(); // keep model simple; class affects stats/abilities
    else playerMesh = makeWarrior();
    root.add(playerMesh);
  }

  function attachMobMeshes() {
    for (const m of mobs) {
      const mesh = (m.kind === "Boar") ? makeBoar() : makeWolf();
      m.mesh = mesh;
      root.add(mesh);

      const ring = makeTargetRing();
      m.ring = ring;
      root.add(ring);
    }
    for (const l of loot) {
      const mesh = makeLootMesh();
      l.mesh = mesh;
      root.add(mesh);
    }
  }

  function syncMeshes(dt) {
    // camera follow, mild shake
    const camTarget = to3(player.x, player.y);
    const shake = state.shake * Q.shakeMul;
    state.shake = Math.max(0, state.shake - dt * 18);

    // set camera in isometric-ish angle
    const yaw = Math.PI * 0.25;
    const pitch = 0.9;
    const dist = 820;

    const sx = (state.rng() - 0.5) * shake * 0.8;
    const sz = (state.rng() - 0.5) * shake * 0.8;

    camera.position.set(
      camTarget.x + Math.cos(yaw) * dist + sx,
      620 + shake * 0.6,
      camTarget.z + Math.sin(yaw) * dist + sz
    );
    camera.lookAt(camTarget.x, 0, camTarget.z);

    if (playerMesh) {
      const p = to3(player.x, player.y);
      playerMesh.position.set(p.x, 0, p.z);
      const mv = Math.hypot(player.vx, player.vy);
      if (mv > 10) playerMesh.rotation.y = Math.atan2(player.vx, player.vy);
    }

    for (const m of mobs) {
      if (!m.mesh) continue;
      const p = to3(m.x, m.y);
      m.mesh.position.set(p.x, 0, p.z);

      const mv = Math.hypot(m.vx, m.vy);
      if (mv > 10) m.mesh.rotation.y = Math.atan2(m.vx, m.vy);

      if (m.ring) {
        m.ring.position.set(p.x, 0.2, p.z);
        m.ring.visible = (player.targetId === m.id);
        if (m.ring.visible) m.ring.material.opacity = 0.20 + 0.12 * (0.5 + 0.5*Math.sin(state.t*3.0));
      }
    }

    for (const l of loot) {
      if (!l.mesh) continue;
      l.t += dt * 2.2;
      const bob = Math.sin(l.t) * 8.0;
      const p = to3(l.x, l.y);
      l.mesh.position.set(p.x, 18 + bob, p.z);
      l.mesh.rotation.z += dt * 0.8;
    }
  }

  // Raycasting target selection
  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  function pickTarget(screenX, screenY) {
    ndc.x = (screenX / W) * 2 - 1;
    ndc.y = -(screenY / H) * 2 + 1;
    raycaster.setFromCamera(ndc, camera);

    const candidates = mobs.map(m => m.mesh).filter(Boolean);
    const hits = raycaster.intersectObjects(candidates, true);
    if (!hits.length) return null;

    // map hit object back to mob
    const obj = hits[0].object;
    for (const m of mobs) {
      if (!m.mesh) continue;
      if (obj === m.mesh || m.mesh.children.includes(obj) || m.mesh.getObjectById(obj.id)) return m;
    }
    return null;
  }

  function minimapAndQuestTick() {
    // quest visit detection
    tryCompleteQuest();
    minimapDraw();
  }

  // Ambient soundtrack (procedural), loops forever
  const audio = {
    ctx: null,
    master: null,
    on: !!CONFIG.music_on,
    vol: clamp((CONFIG.music_vol ?? 35) / 100, 0, 1),
    started: false,
    chirpT: 0
  };

  function audioStart() {
    if (audio.started) return;
    audio.started = true;

    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    audio.ctx = new AC();

    audio.master = audio.ctx.createGain();
    audio.master.gain.value = audio.on ? audio.vol : 0.0;
    audio.master.connect(audio.ctx.destination);

    // warm pad
    const pad = audio.ctx.createOscillator();
    pad.type = "sine";
    pad.frequency.value = 110;

    const pad2 = audio.ctx.createOscillator();
    pad2.type = "triangle";
    pad2.frequency.value = 220;

    const lfo = audio.ctx.createOscillator();
    lfo.type = "sine";
    lfo.frequency.value = 0.08;

    const lfoGain = audio.ctx.createGain();
    lfoGain.gain.value = 0.18;

    const padGain = audio.ctx.createGain();
    padGain.gain.value = 0.0;

    const filter = audio.ctx.createBiquadFilter();
    filter.type = "lowpass";
    filter.frequency.value = 900;
    filter.Q.value = 0.6;

    lfo.connect(lfoGain);
    lfoGain.connect(padGain.gain);

    pad.connect(filter);
    pad2.connect(filter);
    filter.connect(padGain);
    padGain.connect(audio.master);

    const now = audio.ctx.currentTime;
    padGain.gain.setValueAtTime(0.0, now);
    padGain.gain.linearRampToValueAtTime(0.18, now + 2.5);

    pad.start();
    pad2.start();
    lfo.start();

    audio._pad = pad;
    audio._pad2 = pad2;
    audio._filter = filter;
  }

  function audioSet(on, vol) {
    audio.on = on;
    audio.vol = clamp(vol, 0, 1);
    if (!audio.master) return;
    audio.master.gain.value = audio.on ? audio.vol : 0.0;
  }

  function audioChirp() {
    if (!audio.ctx || !audio.master) return;
    const o = audio.ctx.createOscillator();
    const g = audio.ctx.createGain();
    const f = audio.ctx.createBiquadFilter();
    o.type = "sine";
    f.type = "bandpass";
    f.frequency.value = 2200 + Math.random() * 1400;
    f.Q.value = 6;

    const now = audio.ctx.currentTime;
    o.frequency.setValueAtTime(1500 + Math.random()*600, now);
    o.frequency.exponentialRampToValueAtTime(900 + Math.random()*400, now + 0.12);

    g.gain.setValueAtTime(0.0, now);
    g.gain.linearRampToValueAtTime(0.08, now + 0.02);
    g.gain.linearRampToValueAtTime(0.0, now + 0.18);

    o.connect(f);
    f.connect(g);
    g.connect(audio.master);
    o.start(now);
    o.stop(now + 0.22);
  }

  // main tick
  function tick(dt) {
    if (!state.running || state.paused) return;

    state.t += dt;
    state.ping = 10 + Math.floor(6 * (0.5 + Math.sin(state.t * 0.3) * 0.5));

    player.gcd = Math.max(0, player.gcd - dt);
    player.a1 = Math.max(0, player.a1 - dt);
    player.a2 = Math.max(0, player.a2 - dt);
    player.a3 = Math.max(0, player.a3 - dt);
    player.a4 = Math.max(0, player.a4 - dt);

    player.mp = Math.min(player.mpMax, player.mp + 6.5 * dt);

    armorBuffT = Math.max(0, armorBuffT - dt);
    speedBuffT = Math.max(0, speedBuffT - dt);

    const accel = 2100;
    const maxVBase = 460;
    const speedMul = speedBuffT > 0 ? 1.35 : 1.0;
    const maxV = maxVBase * speedMul;
    const friction = 0.86;

    let ax = 0, ay = 0;
    if (keys.has("w") || keys.has("arrowup")) ay -= 1;
    if (keys.has("s") || keys.has("arrowdown")) ay += 1;
    if (keys.has("a") || keys.has("arrowleft")) ax -= 1;
    if (keys.has("d") || keys.has("arrowright")) ax += 1;

    player.vx += ax * accel * dt;
    player.vy += ay * accel * dt;

    player.vx = clamp(player.vx, -maxV, maxV);
    player.vy = clamp(player.vy, -maxV, maxV);

    player.x += player.vx * dt;
    player.y += player.vy * dt;

    player.vx *= Math.pow(friction, dt * 60);
    player.vy *= Math.pow(friction, dt * 60);

    player.x = clamp(player.x, 80, world.w - 80);
    player.y = clamp(player.y, 80, world.h - 80);

    autoAttack(dt);
    mobAI(dt);

    // quest turn-in
    turnInQuestAtCamp();

    // occasional bird chirps
    if (audio.on && audio.ctx) {
      audio.chirpT -= dt;
      if (audio.chirpT <= 0) {
        audio.chirpT = 3.5 + Math.random() * 5.5;
        if (Math.random() < 0.7) audioChirp();
      }
    }

    minimapAndQuestTick();
  }

  function start() {
    state.running = true;
    overlay.style.display = "none";
    canvas.focus();
    audioStart();
    audioSet(!!CONFIG.music_on, clamp((CONFIG.music_vol ?? 35)/100, 0, 1));
  }

  function resetAll() {
    state.rng = mulberry32(state.seed);
    state.gold = 0;

    player.lvl = 1;
    player.xp = 0;
    player.xpNeed = 120;
    player.targetId = null;

    abilitiesSetup();
    rebuildPlayerMesh();

    invInit();
    invRender();

    // reset quests
    questIndex = 0;
    for (const q of quests) { q.have = 0; q.done = false; }

    // clear entities from scene
    for (const m of mobs) { if (m.mesh) root.remove(m.mesh); if (m.ring) root.remove(m.ring); }
    for (const l of loot) { if (l.mesh) root.remove(l.mesh); }
    mobs = [];
    loot = [];
    chatLines = [];
    chatEl.innerHTML = "";

    spawnPack();
    attachMobMeshes();

    player.x = landmarks.camp.x;
    player.y = landmarks.camp.y;

    chat("Welcome to Greenhollow Forest.");
    chat("Click to target. Use 1-4 abilities. Loot with Space.");
    chat("Turn in quests at campfire.");
    uiUpdate();
    saveGame();
  }

  function tryLoad() {
    if (CONFIG.reset_save) clearSave();
    const s = loadSave();
    if (!s) return false;

    state.gold = s.gold ?? 0;

    player.lvl = s.lvl ?? 1;
    player.xp = s.xp ?? 0;
    player.xpNeed = s.xpNeed ?? 120;

    abilitiesSetup();
    rebuildPlayerMesh();

    invInit();
    if (Array.isArray(s.inv)) {
      for (let i = 0; i < 16; i++) inventory[i] = s.inv[i] ?? null;
    }

    questIndex = clamp(s.questIndex ?? 0, 0, quests.length - 1);

    // reset quest state, then restore active quest progress
    for (const q of quests) { q.have = 0; q.done = false; }
    if (s.questState) {
      activeQuest().have = s.questState.have ?? 0;
      activeQuest().done = !!s.questState.done;
    }
    questProgressFromInventory();

    spawnPack();
    attachMobMeshes();

    player.x = landmarks.camp.x;
    player.y = landmarks.camp.y;

    invRender();
    chat("Save loaded.");
    uiUpdate();
    return true;
  }

  // input
  window.addEventListener("keydown", (e) => {
    const k = e.key.toLowerCase();

    if (k === "p") {
      e.preventDefault();
      if (!state.running) return;
      state.paused = !state.paused;
      if (state.paused) {
        overlay.style.display = "flex";
        overlay.innerHTML = `
          <div class="card">
            <div class="title">Paused</div>
            <p class="sub">Press <span class="kbd">P</span> to resume. Press <span class="kbd">R</span> to respawn if dead.</p>
            <p class="sub" style="opacity:0.78; margin:0;">Press <span class="kbd">I</span> inventory and <span class="kbd">Q</span> quest.</p>
          </div>
        `;
      } else overlay.style.display = "none";
      return;
    }

    if (k === "r") { e.preventDefault(); respawn(); return; }

    if (k === "i") {
      e.preventDefault();
      invPanel.style.display = (invPanel.style.display === "block") ? "none" : "block";
      questPanel.style.display = "none";
      invRender();
      return;
    }

    if (k === "q") {
      e.preventDefault();
      questPanel.style.display = (questPanel.style.display === "block") ? "none" : "block";
      invPanel.style.display = "none";
      return;
    }

    if (k === " ") {
      e.preventDefault();
      if (!state.running) start();
      if (!state.running || state.paused) return;

      // campfire sells, plus quest turn-in
      if (dist2(player.x, player.y, landmarks.camp.x, landmarks.camp.y) < 160) {
        sellJunk();
        turnInQuestAtCamp();
      }
      lootNearby();
      return;
    }

    if (k === "1") { e.preventDefault(); useAbility(1); return; }
    if (k === "2") { e.preventDefault(); useAbility(2); return; }
    if (k === "3") { e.preventDefault(); useAbility(3); return; }
    if (k === "4") { e.preventDefault(); useAbility(4); return; }

    keys.add(k);
  });

  window.addEventListener("keyup", (e) => keys.delete(e.key.toLowerCase()));

  canvas.addEventListener("mousemove", (e) => {
    const p = canvasToLocal(e);
    const cx = W * 0.5, cy = H * 0.5;
    const dx = p.x - cx;
    const dy = p.y - cy;
    const mag = Math.hypot(dx, dy) + 1e-6;
    mouseN.x = dx / mag;
    mouseN.y = dy / mag;
  });

  canvas.addEventListener("mousedown", (e) => {
    canvas.focus();
    if (!state.running) start();
    const p = canvasToLocal(e);

    const m = pickTarget(p.x, p.y);
    if (m) {
      player.targetId = m.id;
      chat(`Targeting ${m.kind} (Lv ${m.lvl}).`);
    } else {
      player.targetId = null;
    }

    audioStart();
  });

  // boot and main loop
  if (!tryLoad()) resetAll();

  overlay.style.display = "flex";

  let last = performance.now();
  let fps = 60;
  let fpsAcc = 0, fpsN = 0, fpsT = 0;

  function frame(now) {
    requestAnimationFrame(frame);

    let dtMs = now - last;
    last = now;
    dtMs = Math.min(dtMs, 50);
    const dt = dtMs / 1000.0;

    fpsT += dtMs;
    fpsAcc += 1000.0 / Math.max(1, dtMs);
    fpsN += 1;
    if (fpsT > 350) { fps = fpsAcc / fpsN; fpsAcc = 0; fpsN = 0; fpsT = 0; }

    tick(dt);
    syncMeshes(dt);

    renderer.render(scene, camera);
    uiUpdate(fps);
  }

  requestAnimationFrame(frame);
})();
</script>
</body>
</html>
"""

html = html.replace("__CONFIG_JSON__", json.dumps(cfg))
components.html(html, height=910, scrolling=False)
