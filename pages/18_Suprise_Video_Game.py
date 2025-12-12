import json
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Mini Classic RPG", layout="wide")
st.title("Mini Classic RPG")

with st.sidebar:
    st.subheader("Settings")
    player_class = st.selectbox("Class", ["Warrior", "Mage", "Rogue"], index=0)
    difficulty = st.selectbox("Difficulty", ["Chill", "Normal", "Hard"], index=1)
    quality = st.selectbox("Graphics", ["Ultra", "High", "Medium", "Low"], index=1)
    show_fps = st.toggle("Show FPS", value=False)
    reduce_motion = st.toggle("Reduce motion", value=False)
    seed = st.number_input("Seed", min_value=1, max_value=999999999, value=1337, step=1)
    reset_save = st.button("Reset save")

cfg = {
    "player_class": player_class,
    "difficulty": difficulty,
    "quality": quality,
    "show_fps": show_fps,
    "reduce_motion": reduce_motion,
    "seed": int(seed),
    "reset_save": bool(reset_save),
}

st.caption("WASD move | Click to target | Space to interact/loot | 1-4 abilities | I inventory | Q quest log | R respawn | P pause")

html = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <style>
    html, body { margin:0; padding:0; background: transparent; }
    .wrap { width:100%; display:flex; justify-content:center; }
    .frame { position: relative; width: min(1400px, 98vw); }
    canvas {
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
        <div class="pill" id="pillM">Quest: Boar Cleanup (0/6)</div>
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

      <canvas id="game" width="1400" height="780" tabindex="0"></canvas>

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
          <div class="title">Mini Classic RPG</div>
          <p class="sub">
            A WoW Classic inspired mini-zone built for Streamlit. Original assets and names, familiar loop.<br/>
            Target mobs, manage cooldowns, loot, level, finish the quest, then the zone ramps.
          </p>
          <p class="sub">
            <span class="kbd">W</span><span class="kbd">A</span><span class="kbd">S</span><span class="kbd">D</span> move |
            <span class="kbd">Click</span> target |
            <span class="kbd">Space</span> interact/loot |
            <span class="kbd">1-4</span> abilities |
            <span class="kbd">I</span> inventory |
            <span class="kbd">Q</span> quest |
            <span class="kbd">P</span> pause |
            <span class="kbd">R</span> respawn
          </p>
          <p class="sub" style="opacity:0.78; margin:0;">Click the game area, then press 1 to begin pulling.</p>
        </div>
      </div>

    </div>
  </div>

<script>
(() => {
  const CONFIG = __CONFIG_JSON__;

  const canvas = document.getElementById("game");
  const ctx = canvas.getContext("2d");
  const mm = document.getElementById("minimap");
  const mctx = mm.getContext("2d");

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

  const W = canvas.width;
  const H = canvas.height;

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

  const Q = (() => {
    const rm = !!CONFIG.reduce_motion;
    const q = (CONFIG.quality || "High").toLowerCase();
    if (q === "ultra") return { bloomScale: 0.5, blur1: rm ? 7 : 10, blur2: rm ? 14 : 20, grass: 900, trees: 44, shakeMul: rm ? 0.55 : 1.0 };
    if (q === "high")  return { bloomScale: 0.6, blur1: rm ? 6 : 8,  blur2: rm ? 12 : 16, grass: 700, trees: 38, shakeMul: rm ? 0.55 : 1.0 };
    if (q === "medium")return { bloomScale: 0.7, blur1: rm ? 5 : 6,  blur2: rm ? 10 : 12, grass: 520, trees: 30, shakeMul: rm ? 0.45 : 0.85 };
    return               { bloomScale: 0.8, blur1: rm ? 4 : 5,  blur2: rm ? 8  : 10, grass: 360, trees: 22, shakeMul: rm ? 0.40 : 0.75 };
  })();

  const D = (() => {
    const d = (CONFIG.difficulty || "Normal").toLowerCase();
    if (d === "chill")  return { mobHp: 0.85, mobDmg: 0.85, mobSp: 0.92, xp: 1.10, gold: 1.05 };
    if (d === "hard")   return { mobHp: 1.18, mobDmg: 1.18, mobSp: 1.08, xp: 0.95, gold: 0.95 };
    return               { mobHp: 1.00, mobDmg: 1.00, mobSp: 1.00, xp: 1.00, gold: 1.00 };
  })();

  const scene = document.createElement("canvas");
  scene.width = W; scene.height = H;
  const sctx = scene.getContext("2d");

  const bloom = document.createElement("canvas");
  bloom.width = Math.floor(W * Q.bloomScale);
  bloom.height = Math.floor(H * Q.bloomScale);
  const bctx = bloom.getContext("2d");

  const SAVE_KEY = "mini_classic_rpg_save_v1";

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
        quest: { kills: quest.kills, done: quest.done },
        seed: state.seed
      };
      localStorage.setItem(SAVE_KEY, JSON.stringify(payload));
    } catch {}
  }
  function clearSave() {
    try { localStorage.removeItem(SAVE_KEY); } catch {}
  }

  const keys = new Set();
  let mouse = { x: W * 0.5, y: H * 0.5 };

  function canvasToLocal(e) {
    const r = canvas.getBoundingClientRect();
    const x = (e.clientX - r.left) * (W / r.width);
    const y = (e.clientY - r.top) * (H / r.height);
    return { x, y };
  }

  function chat(line) {
    chatLines.unshift(line);
    if (chatLines.length > 9) chatLines.pop();
    chatEl.innerHTML = chatLines.map(s => `<div class="line">${s}</div>`).join("");
  }

  const state = {
    seed: CONFIG.seed || 1337,
    rng: mulberry32(CONFIG.seed || 1337),
    running: false,
    paused: false,
    t: 0,
    shake: 0,
    gold: 0,
    zone: "Greenhollow",
    ping: 12
  };

  const player = {
    cls: (CONFIG.player_class || "Warrior"),
    name: "Adventurer",
    x: 2200,
    y: 2100,
    vx: 0,
    vy: 0,
    r: 18,

    lvl: 1,
    xp: 0,
    xpNeed: 120,

    hp: 100,
    hpMax: 100,

    mp: 60,
    mpMax: 60,

    ap: 10,
    sp: 10,

    armor: 0.10,

    gcd: 0,
    aaCd: 0,
    aaSpeed: 1.8,
    range: 62,

    targetId: null,

    a1: 0, a2: 0, a3: 0, a4: 0
  };

  const quest = {
    name: "Boar Cleanup",
    need: 6,
    kills: 0,
    done: false,
    rewardGold: 55,
    rewardXP: 180
  };

  const abilitiesByClass = {
    Warrior: [
      { key:"1", name:"Strike", cd: 3.5, cost: 0,  dmg: 22, type:"melee", color:"rgba(255,140,230,0.90)" },
      { key:"2", name:"Sunder", cd: 8.0, cost: 0,  dmg: 14, type:"melee", debuff:"armor", color:"rgba(120,170,255,0.90)" },
      { key:"3", name:"Shout",  cd: 20,  cost: 0,  buff:"armor", color:"rgba(92,255,176,0.90)" },
      { key:"4", name:"Charge", cd: 14,  cost: 0,  dash:true, dmg: 10, type:"melee", color:"rgba(255,212,92,0.92)" }
    ],
    Mage: [
      { key:"1", name:"Frostbolt", cd: 2.2, cost: 12, dmg: 20, type:"ranged", slow:true, color:"rgba(120,210,255,0.92)" },
      { key:"2", name:"Fireball",  cd: 3.8, cost: 16, dmg: 28, type:"ranged", dot:true, color:"rgba(255,170,110,0.92)" },
      { key:"3", name:"Nova",      cd: 16,  cost: 18, dmg: 14, aoe:true, root:true, color:"rgba(255,140,230,0.75)" },
      { key:"4", name:"Blink",     cd: 12,  cost: 10, blink:true, color:"rgba(92,255,176,0.85)" }
    ],
    Rogue: [
      { key:"1", name:"Stab",    cd: 2.6, cost: 0,  dmg: 20, type:"melee", color:"rgba(255,140,230,0.90)" },
      { key:"2", name:"Poison",  cd: 10,  cost: 0,  dmg: 10, type:"melee", dot:true, color:"rgba(92,255,176,0.88)" },
      { key:"3", name:"Vanish",  cd: 22,  cost: 0,  stealth:true, color:"rgba(120,170,255,0.80)" },
      { key:"4", name:"Dash",    cd: 14,  cost: 0,  speed:true, color:"rgba(255,212,92,0.92)" }
    ]
  };

  let mobs = [];
  let loot = [];
  let particles = [];
  let grass = [];
  let trees = [];
  let chatLines = [];

  const world = {
    w: 4200,
    h: 3600
  };

  function initDecor() {
    grass = [];
    trees = [];
    for (let i = 0; i < Q.grass; i++) {
      grass.push({
        x: state.rng() * world.w,
        y: state.rng() * world.h,
        s: 0.6 + state.rng() * 1.3,
        a: 0.06 + state.rng() * 0.10
      });
    }
    for (let i = 0; i < Q.trees; i++) {
      trees.push({
        x: 260 + state.rng() * (world.w - 520),
        y: 260 + state.rng() * (world.h - 520),
        r: 28 + state.rng() * 46,
        hue: state.rng() < 0.65 ? 1 : 2
      });
    }
  }

  function addParticles(x, y, color, n=18, spread=1.0) {
    for (let i = 0; i < n; i++) {
      const a = state.rng() * Math.PI * 2;
      const sp = (40 + state.rng() * 260) * spread;
      particles.push({
        x, y,
        vx: Math.cos(a) * sp,
        vy: Math.sin(a) * sp,
        life: 0.35 + state.rng() * 0.35,
        r: 1.2 + state.rng() * 3.2,
        c: color
      });
    }
  }

  function dist(ax, ay, bx, by) {
    const dx = ax - bx, dy = ay - by;
    return Math.hypot(dx, dy);
  }

  function clampWorld(x, y) {
    return {
      x: clamp(x, 40, world.w - 40),
      y: clamp(y, 40, world.h - 40)
    };
  }

  function newMob(kind, lvl, x, y) {
    const base = (kind === "Boar") ? { hp: 58, dmg: 8, sp: 105 } : { hp: 78, dmg: 10, sp: 95 };
    const hp = Math.floor((base.hp + lvl * 10) * D.mobHp);
    return {
      id: "m" + Math.floor(state.rng() * 1e9),
      kind,
      lvl,
      x, y,
      vx: 0, vy: 0,
      r: kind === "Boar" ? 18 : 20,
      hp, hpMax: hp,
      dmg: Math.floor((base.dmg + lvl * 1.6) * D.mobDmg),
      sp: (base.sp + lvl * 3.0) * D.mobSp,
      aggro: false,
      leashX: x,
      leashY: y,
      leash: 360,
      hit: 0,
      slow: 0,
      root: 0,
      dot: 0,
      dotT: 0,
      armorDown: 0,
      target: false
    };
  }

  function spawnPack() {
    const packs = 10 + Math.floor(state.rng() * 10);
    mobs = [];
    for (let i = 0; i < packs; i++) {
      const kind = (state.rng() < 0.82) ? "Boar" : "Wolf";
      const lvl = clamp(1 + Math.floor(i/4) + Math.floor(state.rng() * 2), 1, 8);
      const x = 400 + state.rng() * (world.w - 800);
      const y = 400 + state.rng() * (world.h - 800);
      mobs.push(newMob(kind, lvl, x, y));
    }
  }

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
  function invRender() {
    invGrid.innerHTML = "";
    for (let i = 0; i < 16; i++) {
      const it = inventory[i];
      const div = document.createElement("div");
      div.className = "invItem";
      if (!it) {
        div.innerHTML = `<div>Empty</div><span>Slot ${i+1}</span>`;
      } else {
        div.innerHTML = `<div>${it.name}</div><span>Value ${it.value}g</span>`;
      }
      invGrid.appendChild(div);
    }
    document.getElementById("invSub").textContent = `Bags: ${invCount()}/16 | Space to loot nearby`;
  }

  function abilitiesSetup() {
    const a = abilitiesByClass[player.cls];
    a1n.textContent = a[0].name;
    a2n.textContent = a[1].name;
    a3n.textContent = a[2].name;
    a4n.textContent = a[3].name;

    if (player.cls === "Warrior") { player.hpMax = 115; player.mpMax = 40; player.aaSpeed = 1.9; player.range = 62; player.armor = 0.16; }
    if (player.cls === "Mage")    { player.hpMax = 90;  player.mpMax = 90; player.aaSpeed = 2.1; player.range = 210; player.armor = 0.06; }
    if (player.cls === "Rogue")   { player.hpMax = 100; player.mpMax = 55; player.aaSpeed = 1.6; player.range = 62; player.armor = 0.10; }
    player.hp = player.hpMax;
    player.mp = player.mpMax;
  }

  function uiUpdate(fps=0) {
    pName.textContent = `${player.name} (${player.cls})`;
    pLvl.textContent = `Lv ${player.lvl}`;
    pHP.style.width = `${clamp(player.hp / player.hpMax, 0, 1) * 100}%`;
    pMP.style.width = `${clamp(player.mp / player.mpMax, 0, 1) * 100}%`;

    const xpT = clamp(player.xp / player.xpNeed, 0, 1) * 100;
    xpFill.style.width = `${xpT}%`;

    pillL.textContent = `Zone: ${state.zone} | Gold: ${state.gold} | Bags: ${invCount()}/16`;
    pillM.textContent = quest.done ? `Quest: ${quest.name} (Complete) | Return for reward` : `Quest: ${quest.name} (${quest.kills}/${quest.need})`;
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
        const d = dist(player.x, player.y, m.x, m.y);
        tNote.textContent = d <= player.range ? "In range. Fight." : "Out of range. Move closer.";
      } else {
        tFrame.style.display = "none";
      }
    } else {
      tFrame.style.display = "none";
    }

    questText.textContent = quest.done
      ? `Return to the campfire (center of zone) to claim ${quest.rewardGold}g and ${quest.rewardXP} XP.`
      : `Kill ${quest.need} boars in Greenhollow. You have ${quest.kills}/${quest.need}.`;
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
    addParticles(player.x, player.y, "rgba(92,255,176,0.92)", 70, 1.4);
    state.shake = Math.min(18, state.shake + 10);
  }

  function giveXP(n) {
    player.xp += Math.floor(n * D.xp);
    while (player.xp >= player.xpNeed) levelUp();
  }

  function awardQuest() {
    if (quest.done && dist(player.x, player.y, world.w * 0.5, world.h * 0.5) < 120) {
      quest.done = false;
      quest.kills = 0;
      state.gold += quest.rewardGold;
      giveXP(quest.rewardXP);
      chat(`<span style="color:rgba(255,212,92,0.95)">Quest</span> Reward claimed. The zone stirs...`);
      quest.rewardGold = Math.floor(quest.rewardGold * 1.18 + 12);
      quest.rewardXP = Math.floor(quest.rewardXP * 1.15 + 18);
      spawnPack();
      saveGame();
    }
  }

  function dropLoot(m) {
    const roll = state.rng();
    const items = [
      { name: "Torn Hide", value: 2 },
      { name: "Boar Tusk", value: 3 },
      { name: "Worn Fang", value: 2 },
      { name: "Frayed Pelt", value: 2 },
      { name: "Shiny Trinket", value: 6 }
    ];

    let it = items[Math.floor(state.rng() * items.length)];
    if (roll > 0.92) it = { name: "Green Charm", value: 12 };

    loot.push({
      x: m.x,
      y: m.y,
      r: 14,
      item: it,
      t: state.rng() * Math.PI * 2
    });
  }

  function lootNearby() {
    let picked = false;
    for (let i = loot.length - 1; i >= 0; i--) {
      const l = loot[i];
      if (dist(player.x, player.y, l.x, l.y) < 70) {
        if (invAdd(l.item)) {
          loot.splice(i, 1);
          chat(`Looted <span style="color:rgba(255,212,92,0.95)">${l.item.name}</span>.`);
          addParticles(l.x, l.y, "rgba(255,212,92,0.90)", 26, 1.1);
          picked = true;
        } else {
          chat(`<span style="color:rgba(255,110,110,0.95)">Bags full</span>.`);
          break;
        }
      }
    }
    if (picked) { invRender(); saveGame(); }
  }

  function sellJunk() {
    let gold = 0;
    for (let i = 0; i < inventory.length; i++) {
      const it = inventory[i];
      if (it) { gold += it.value; inventory[i] = null; }
    }
    if (gold > 0) {
      state.gold += Math.floor(gold * D.gold);
      chat(`Sold junk for <span style="color:rgba(255,212,92,0.95)">${Math.floor(gold * D.gold)}g</span>.`);
      invRender();
      saveGame();
    }
  }

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
      const ang = Math.atan2(mouse.y - (H*0.5), mouse.x - (W*0.5));
      player.x += Math.cos(ang) * 240;
      player.y += Math.sin(ang) * 240;
      const p = clampWorld(player.x, player.y);
      player.x = p.x; player.y = p.y;
      addParticles(player.x, player.y, a.color, 40, 1.2);
      state.shake = Math.min(16, state.shake + 8);
      chat(`Cast <span style="color:${a.color}">${a.name}</span>.`);
      return;
    }

    if (a.stealth) {
      player[cdKey] = a.cd;
      player.gcd = 0.35;
      player.armor = Math.min(0.35, player.armor + 0.08);
      addParticles(player.x, player.y, a.color, 36, 1.1);
      chat(`Used <span style="color:${a.color}">${a.name}</span>.`);
      return;
    }

    if (a.speed) {
      player[cdKey] = a.cd;
      player.gcd = 0.35;
      speedBuffT = 5.0;
      addParticles(player.x, player.y, a.color, 36, 1.1);
      chat(`Used <span style="color:${a.color}">${a.name}</span>.`);
      return;
    }

    if (a.buff === "armor") {
      player[cdKey] = a.cd;
      player.gcd = 0.35;
      armorBuffT = 10.0;
      addParticles(player.x, player.y, a.color, 42, 1.15);
      chat(`Cast <span style="color:${a.color}">${a.name}</span>.`);
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
      player.x += (dx / d) * 200;
      player.y += (dy / d) * 200;
      addParticles(player.x, player.y, a.color, 40, 1.25);
      hitMob(m, a.dmg, { slow:false, root:false, dot:false, armorDown:false }, a.color);
      chat(`Used <span style="color:${a.color}">${a.name}</span>.`);
      return;
    }

    if (a.aoe) {
      player.mp -= a.cost;
      player[cdKey] = a.cd;
      player.gcd = 0.45;
      addParticles(player.x, player.y, a.color, 60, 1.4);

      for (const m of mobs) {
        const d = dist(player.x, player.y, m.x, m.y);
        if (d < 170) {
          hitMob(m, a.dmg, { slow:false, root: !!a.root, dot:false, armorDown:false }, a.color);
        }
      }
      chat(`Cast <span style="color:${a.color}">${a.name}</span>.`);
      return;
    }

    if (!player.targetId) { chat("No target."); return; }
    const m = mobs.find(z => z.id === player.targetId);
    if (!m) return;

    const d = dist(player.x, player.y, m.x, m.y);
    if (d > player.range) {
      chat("Out of range.");
      return;
    }

    player.mp -= (a.cost || 0);
    player[cdKey] = a.cd;
    player.gcd = 0.35;

    hitMob(m, a.dmg, { slow: !!a.slow, root: !!a.root, dot: !!a.dot, armorDown: (a.debuff === "armor") }, a.color);
    chat(`Used <span style="color:${a.color}">${a.name}</span>.`);
  }

  function hitMob(m, dmg, fx, color) {
    m.aggro = true;
    m.hit = 1.0;
    const armorMul = (m.armorDown > 0) ? 1.12 : 1.0;
    m.hp -= Math.floor(dmg * armorMul);

    addParticles(m.x, m.y, color, 20, 1.0);
    state.shake = Math.min(16, state.shake + 6);

    if (fx.slow) m.slow = Math.max(m.slow, 2.5);
    if (fx.root) m.root = Math.max(m.root, 1.4);
    if (fx.dot)  m.dot  = Math.max(m.dot, 6.0);
    if (fx.armorDown) m.armorDown = Math.max(m.armorDown, 6.0);

    if (m.hp <= 0) mobDie(m);
  }

  function mobDie(m) {
    mobs = mobs.filter(z => z.id !== m.id);
    addParticles(m.x, m.y, "rgba(255,110,110,0.80)", 52, 1.35);

    giveXP(34 + m.lvl * 12);

    if (m.kind === "Boar" && !quest.done) {
      quest.kills += 1;
      if (quest.kills >= quest.need) {
        quest.done = true;
        chat(`<span style="color:rgba(255,212,92,0.95)">Quest</span> Objective complete.`);
        addParticles(player.x, player.y, "rgba(255,212,92,0.90)", 70, 1.4);
      }
    }

    dropLoot(m);
    if (player.targetId === m.id) player.targetId = null;

    saveGame();
  }

  function autoAttack(dt) {
    if (!player.targetId) return;
    const m = mobs.find(z => z.id === player.targetId);
    if (!m) return;

    const d = dist(player.x, player.y, m.x, m.y);
    if (d > player.range) return;

    player.aaCd -= dt;
    if (player.aaCd <= 0) {
      player.aaCd = player.aaSpeed;
      const base = (player.cls === "Mage") ? 10 : 14;
      hitMob(m, base + Math.floor(player.lvl * 1.6), { slow:false, root:false, dot:false, armorDown:false }, "rgba(255,255,255,0.70)");
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
          addParticles(m.x, m.y, "rgba(92,255,176,0.70)", 10, 0.9);
          if (m.hp <= 0) { mobDie(m); continue; }
        }
      }

      const leashD = dist(m.x, m.y, m.leashX, m.leashY);
      const aggroD = dist(player.x, player.y, m.x, m.y);

      if (!m.aggro && aggroD < 220) m.aggro = true;

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

      if (m.aggro && aggroD < 52) {
        mobAttack(dt, m);
      }
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

    addParticles(player.x, player.y, "rgba(255,110,110,0.72)", 26, 1.1);
    state.shake = Math.min(20, state.shake + 10);
    chat(`${m.kind} hits you for <span style="color:rgba(255,110,110,0.95)">${dmg}</span>.`);

    if (player.hp <= 0) {
      player.hp = 0;
      state.running = false;
      overlay.style.display = "flex";
      overlay.innerHTML = `
        <div class="card">
          <div class="title">You died</div>
          <p class="sub">Press <span class="kbd">R</span> to respawn at the campfire.</p>
          <p class="sub" style="opacity:0.78; margin:0;">Tip: loot with Space, sell at campfire with Space, then turn in quest.</p>
        </div>
      `;
      saveGame();
    }
  }

  function respawn() {
    player.x = world.w * 0.5;
    player.y = world.h * 0.5;
    player.hp = player.hpMax;
    player.mp = player.mpMax;
    player.targetId = null;
    state.running = true;
    overlay.style.display = "none";
    chat("You feel the warmth of the campfire.");
  }

  function drawWorld(dt) {
    sctx.clearRect(0,0,W,H);

    const camX = clamp(player.x - W * 0.5, 0, world.w - W);
    const camY = clamp(player.y - H * 0.5, 0, world.h - H);

    // background
    const g = sctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, "rgba(10,14,12,1)");
    g.addColorStop(1, "rgba(6,9,8,1)");
    sctx.fillStyle = g;
    sctx.fillRect(0,0,W,H);

    // soft zone glows
    const rg1 = sctx.createRadialGradient(W*0.30, H*0.22, 50, W*0.30, H*0.22, 820);
    rg1.addColorStop(0, "rgba(92,255,176,0.10)");
    rg1.addColorStop(1, "rgba(0,0,0,0)");
    sctx.fillStyle = rg1; sctx.fillRect(0,0,W,H);

    const rg2 = sctx.createRadialGradient(W*0.78, H*0.20, 50, W*0.78, H*0.20, 820);
    rg2.addColorStop(0, "rgba(120,170,255,0.10)");
    rg2.addColorStop(1, "rgba(0,0,0,0)");
    sctx.fillStyle = rg2; sctx.fillRect(0,0,W,H);

    // ground tiles
    const tile = 70;
    sctx.save();
    sctx.globalAlpha = 0.10;
    sctx.strokeStyle = "rgba(255,255,255,0.10)";
    for (let x = -((camX % tile)); x < W + tile; x += tile) {
      sctx.beginPath(); sctx.moveTo(x, 0); sctx.lineTo(x, H); sctx.stroke();
    }
    for (let y = -((camY % tile)); y < H + tile; y += tile) {
      sctx.beginPath(); sctx.moveTo(0, y); sctx.lineTo(W, y); sctx.stroke();
    }
    sctx.restore();

    // grass
    sctx.save();
    for (const gr of grass) {
      const x = gr.x - camX;
      const y = gr.y - camY;
      if (x < -30 || y < -30 || x > W + 30 || y > H + 30) continue;
      sctx.globalAlpha = gr.a;
      sctx.fillStyle = "rgba(92,255,176,0.55)";
      sctx.fillRect(x, y, 2.0 * gr.s, 10.0 * gr.s);
    }
    sctx.restore();

    // trees
    for (const tr of trees) {
      const x = tr.x - camX;
      const y = tr.y - camY;
      if (x < -160 || y < -160 || x > W + 160 || y > H + 160) continue;

      const col = tr.hue === 1 ? "rgba(92,255,176,0.14)" : "rgba(120,170,255,0.12)";
      sctx.save();
      sctx.shadowColor = col;
      sctx.shadowBlur = 20;
      sctx.beginPath();
      sctx.arc(x, y, tr.r, 0, Math.PI*2);
      sctx.fillStyle = "rgba(0,0,0,0.22)";
      sctx.fill();
      sctx.restore();

      sctx.save();
      sctx.shadowColor = col;
      sctx.shadowBlur = 24;
      sctx.beginPath();
      sctx.arc(x, y, tr.r * 0.82, 0, Math.PI*2);
      sctx.fillStyle = col;
      sctx.fill();
      sctx.restore();
    }

    // campfire at center
    const cx = world.w * 0.5 - camX;
    const cy = world.h * 0.5 - camY;
    sctx.save();
    sctx.shadowColor = "rgba(255,212,92,0.30)";
    sctx.shadowBlur = 26;
    sctx.beginPath();
    sctx.arc(cx, cy, 26, 0, Math.PI*2);
    sctx.fillStyle = "rgba(255,212,92,0.18)";
    sctx.fill();
    sctx.restore();

    sctx.save();
    sctx.globalAlpha = 0.65;
    sctx.fillStyle = "rgba(255,212,92,0.30)";
    sctx.fillRect(cx - 2, cy - 20, 4, 10);
    sctx.fillRect(cx - 8, cy - 8, 16, 6);
    sctx.restore();

    // loot
    for (const l of loot) {
      l.t += dt * 2.2;
      const bob = Math.sin(l.t) * 4.5;
      const x = l.x - camX;
      const y = l.y - camY + bob;
      if (x < -50 || y < -50 || x > W + 50 || y > H + 50) continue;

      sctx.save();
      sctx.shadowColor = "rgba(255,212,92,0.70)";
      sctx.shadowBlur = 18;
      sctx.beginPath();
      sctx.arc(x, y, l.r, 0, Math.PI*2);
      sctx.fillStyle = "rgba(255,212,92,0.78)";
      sctx.fill();
      sctx.strokeStyle = "rgba(255,255,255,0.14)";
      sctx.lineWidth = 2;
      sctx.stroke();
      sctx.restore();
    }

    // mobs
    for (const m of mobs) {
      const x = m.x - camX;
      const y = m.y - camY;
      if (x < -80 || y < -80 || x > W + 80 || y > H + 80) continue;

      const hostile = m.aggro ? "rgba(255,110,110,0.90)" : "rgba(255,170,110,0.72)";
      sctx.save();
      sctx.shadowColor = hostile;
      sctx.shadowBlur = 18 + m.hit * 12;
      sctx.beginPath();
      sctx.arc(x, y, m.r, 0, Math.PI*2);
      const eg = sctx.createRadialGradient(x - 6, y - 8, 2, x, y, m.r * 2.2);
      eg.addColorStop(0, "rgba(255,255,255,0.70)");
      eg.addColorStop(0.35, hostile);
      eg.addColorStop(1, "rgba(0,0,0,0)");
      sctx.fillStyle = eg;
      sctx.fill();
      sctx.restore();

      // hp ring
      const hpT = clamp(m.hp / m.hpMax, 0, 1);
      sctx.save();
      sctx.beginPath();
      sctx.arc(x, y, m.r + 9, -Math.PI/2, -Math.PI/2 + Math.PI*2*hpT);
      sctx.strokeStyle = `rgba(255,255,255,${0.10 + 0.20*hpT})`;
      sctx.lineWidth = 2;
      sctx.stroke();
      sctx.restore();

      // target marker
      if (player.targetId === m.id) {
        sctx.save();
        sctx.globalAlpha = 0.65;
        sctx.strokeStyle = "rgba(255,255,255,0.25)";
        sctx.lineWidth = 2;
        sctx.beginPath();
        sctx.arc(x, y, m.r + 16, 0, Math.PI*2);
        sctx.stroke();
        sctx.restore();
      }
    }

    // player
    const px = player.x - camX;
    const py = player.y - camY;

    sctx.save();
    sctx.shadowColor = "rgba(92,255,176,0.70)";
    sctx.shadowBlur = 24;
    sctx.beginPath();
    sctx.arc(px, py, player.r + 3, 0, Math.PI*2);
    const pg = sctx.createRadialGradient(px - 8, py - 10, 2, px, py, player.r * 2.6);
    pg.addColorStop(0, "rgba(255,255,255,0.92)");
    pg.addColorStop(0.35, "rgba(92,255,176,0.86)");
    pg.addColorStop(1, "rgba(0,0,0,0)");
    sctx.fillStyle = pg;
    sctx.fill();
    sctx.restore();

    // particles
    for (const p of particles) {
      const x = p.x - camX;
      const y = p.y - camY;
      if (x < -80 || y < -80 || x > W + 80 || y > H + 80) continue;
      sctx.save();
      sctx.globalAlpha = clamp(p.life / 0.6, 0, 1);
      sctx.shadowColor = p.c;
      sctx.shadowBlur = 14;
      sctx.beginPath();
      sctx.arc(x, y, p.r, 0, Math.PI*2);
      sctx.fillStyle = p.c;
      sctx.fill();
      sctx.restore();
    }

    // interaction prompt
    const nearCamp = dist(player.x, player.y, world.w * 0.5, world.h * 0.5) < 120;
    const nearLoot = loot.some(l => dist(player.x, player.y, l.x, l.y) < 70);

    sctx.save();
    sctx.globalAlpha = 0.88;
    sctx.fillStyle = "rgba(0,0,0,0.35)";
    sctx.fillRect(W*0.5 - 240, H - 118, 480, 44);
    sctx.strokeStyle = "rgba(255,255,255,0.12)";
    sctx.strokeRect(W*0.5 - 240, H - 118, 480, 44);
    sctx.fillStyle = "rgba(255,255,255,0.86)";
    sctx.font = "700 13px system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial";
    sctx.textAlign = "center";
    sctx.textBaseline = "middle";

    let msg = "Click a mob to target. Use 1-4 abilities.";
    if (nearLoot) msg = "Press Space to loot nearby items.";
    if (nearCamp) msg = "At campfire: Space sells junk. Turn in quest by standing here.";
    sctx.fillText(msg, W*0.5, H - 96);
    sctx.restore();

    // camera reticle
    sctx.save();
    sctx.globalAlpha = 0.55;
    sctx.strokeStyle = "rgba(255,255,255,0.18)";
    sctx.lineWidth = 1.5;
    sctx.beginPath();
    sctx.arc(W*0.5 + (mouse.x - W*0.5)*0.0, H*0.5 + (mouse.y - H*0.5)*0.0, 10, 0, Math.PI*2);
    sctx.stroke();
    sctx.restore();
  }

  function compositeBloom() {
    bctx.clearRect(0,0,bloom.width,bloom.height);
    bctx.save();
    bctx.scale(bloom.width / W, bloom.height / H);
    bctx.drawImage(scene, 0, 0);
    bctx.restore();

    bctx.save();
    bctx.globalCompositeOperation = "source-in";
    bctx.filter = `blur(${Q.blur1}px)`;
    bctx.globalAlpha = 0.85;
    bctx.drawImage(bloom, 0, 0);
    bctx.restore();

    bctx.save();
    bctx.filter = `blur(${Q.blur2}px)`;
    bctx.globalAlpha = 0.40;
    bctx.drawImage(bloom, 0, 0);
    bctx.restore();

    ctx.clearRect(0,0,W,H);
    ctx.drawImage(scene, 0, 0);

    ctx.save();
    ctx.globalCompositeOperation = "screen";
    ctx.globalAlpha = 0.75;
    ctx.drawImage(bloom, 0, 0, W, H);
    ctx.restore();
  }

  function minimapDraw() {
    mctx.clearRect(0,0,mm.width,mm.height);
    mctx.save();
    mctx.fillStyle = "rgba(0,0,0,0.28)";
    mctx.fillRect(0,0,mm.width,mm.height);

    const sx = mm.width / world.w;
    const sy = mm.height / world.h;

    // camp
    mctx.fillStyle = "rgba(255,212,92,0.85)";
    mctx.beginPath();
    mctx.arc(world.w*0.5*sx, world.h*0.5*sy, 4, 0, Math.PI*2);
    mctx.fill();

    // mobs
    for (const m of mobs) {
      mctx.fillStyle = m.aggro ? "rgba(255,110,110,0.80)" : "rgba(255,170,110,0.65)";
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

  function tick(dt) {
    if (!state.running || state.paused) return;

    state.t += dt;
    state.ping = 10 + Math.floor(6 * (0.5 + Math.sin(state.t * 0.3) * 0.5));

    // timers
    player.gcd = Math.max(0, player.gcd - dt);
    player.a1 = Math.max(0, player.a1 - dt);
    player.a2 = Math.max(0, player.a2 - dt);
    player.a3 = Math.max(0, player.a3 - dt);
    player.a4 = Math.max(0, player.a4 - dt);

    // regen
    player.mp = Math.min(player.mpMax, player.mp + 6.5 * dt);

    // buffs
    armorBuffT = Math.max(0, armorBuffT - dt);
    speedBuffT = Math.max(0, speedBuffT - dt);

    // movement
    const accel = 1900;
    const maxVBase = 410;
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

    const p = clampWorld(player.x, player.y);
    player.x = p.x; player.y = p.y;

    // combat
    autoAttack(dt);
    mobAI(dt);

    // particles
    for (let i = particles.length - 1; i >= 0; i--) {
      const p = particles[i];
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.vx *= Math.pow(0.97, dt * 60);
      p.vy *= Math.pow(0.97, dt * 60);
      p.life -= dt;
      if (p.life <= 0) particles.splice(i, 1);
    }

    // campfire interactions
    awardQuest();
  }

  let armorBuffT = 0;
  let speedBuffT = 0;

  function start() {
    state.running = true;
    overlay.style.display = "none";
    canvas.focus();
  }

  function resetAll() {
    state.rng = mulberry32(state.seed);
    state.gold = 0;
    player.lvl = 1;
    player.xp = 0;
    player.xpNeed = 120;
    player.targetId = null;

    abilitiesSetup();
    invInit();
    invRender();

    quest.kills = 0;
    quest.done = false;
    quest.rewardGold = 55;
    quest.rewardXP = 180;

    loot = [];
    particles = [];
    chatLines = [];
    chatEl.innerHTML = "";

    initDecor();
    spawnPack();

    player.x = world.w * 0.5;
    player.y = world.h * 0.5;

    chat("Welcome to Greenhollow.");
    chat("Pull boars. Loot with Space. Sell at campfire.");
    saveGame();
    uiUpdate();
  }

  function tryLoad() {
    if (CONFIG.reset_save) clearSave();
    const s = loadSave();
    if (!s) return false;

    state.gold = s.gold ?? 0;
    player.lvl = s.lvl ?? 1;
    player.xp = s.xp ?? 0;
    player.xpNeed = s.xpNeed ?? 120;

    invInit();
    if (Array.isArray(s.inv)) {
      for (let i = 0; i < 16; i++) inventory[i] = s.inv[i] ?? null;
    }

    quest.kills = s.quest?.kills ?? 0;
    quest.done = s.quest?.done ?? false;

    abilitiesSetup();
    initDecor();
    spawnPack();

    player.x = world.w * 0.5;
    player.y = world.h * 0.5;

    invRender();
    chat("Save loaded.");
    uiUpdate();
    return true;
  }

  // Input
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
            <p class="sub" style="opacity:0.78; margin:0;">Press <span class="kbd">I</span> for inventory and <span class="kbd">Q</span> for quest log.</p>
          </div>
        `;
      } else {
        overlay.style.display = "none";
      }
      return;
    }

    if (k === "r") {
      e.preventDefault();
      respawn();
      return;
    }

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
      // campfire sells
      const nearCamp = dist(player.x, player.y, world.w * 0.5, world.h * 0.5) < 120;
      if (nearCamp) sellJunk();
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
    mouse.x = p.x;
    mouse.y = p.y;
  });

  canvas.addEventListener("mousedown", (e) => {
    canvas.focus();
    if (!state.running) start();
    const p = canvasToLocal(e);
    mouse.x = p.x;
    mouse.y = p.y;

    // clicking targets nearest mob under cursor in screen space
    const camX = clamp(player.x - W * 0.5, 0, world.w - W);
    const camY = clamp(player.y - H * 0.5, 0, world.h - H);

    let best = null;
    let bestD = 999999;

    for (const m of mobs) {
      const sx = m.x - camX;
      const sy = m.y - camY;
      const d = dist(sx, sy, mouse.x, mouse.y);
      if (d < m.r + 18 && d < bestD) { bestD = d; best = m; }
    }

    if (best) {
      player.targetId = best.id;
      chat(`Targeting ${best.kind} (Lv ${best.lvl}).`);
    } else {
      player.targetId = null;
    }
  });

  // Main loop
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
    drawWorld(dt);
    compositeBloom();
    minimapDraw();
    uiUpdate(fps);
  }

  // boot
  if (!tryLoad()) resetAll();

  overlay.style.display = "flex";
  requestAnimationFrame(frame);
})();
</script>
</body>
</html>
"""

html = html.replace("__CONFIG_JSON__", json.dumps(cfg))
components.html(html, height=910, scrolling=False)
