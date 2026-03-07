import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Neon Frontier", page_icon="🎮", layout="wide")

# =========================================================
# Styling
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(83, 52, 131, 0.35), transparent 28%),
            radial-gradient(circle at top right, rgba(0, 200, 255, 0.20), transparent 22%),
            linear-gradient(135deg, #0a0f1e 0%, #0f172a 55%, #111827 100%);
        color: #e5e7eb;
    }
    .glass {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 16px 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        margin-bottom: 12px;
    }
    .title-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.16), rgba(59, 130, 246, 0.16), rgba(168, 85, 247, 0.16));
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 24px;
        padding: 22px 24px;
        box-shadow: 0 18px 40px rgba(0,0,0,0.28);
        margin-bottom: 14px;
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 10px 14px;
        min-height: 102px;
    }
    .small-label {
        font-size: 0.82rem;
        color: #9ca3af;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .big-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f9fafb;
        line-height: 1.2;
        margin-top: 4px;
    }
    .flavor {
        color: #cbd5e1;
        font-size: 0.96rem;
        line-height: 1.55;
    }
    .quest-box {
        background: rgba(99,102,241,0.12);
        border-left: 4px solid #818cf8;
        padding: 12px 14px;
        border-radius: 12px;
        margin: 8px 0;
    }
    .danger-box {
        background: rgba(239,68,68,0.12);
        border-left: 4px solid #f87171;
        padding: 12px 14px;
        border-radius: 12px;
        margin: 8px 0;
    }
    .success-box {
        background: rgba(34,197,94,0.12);
        border-left: 4px solid #4ade80;
        padding: 12px 14px;
        border-radius: 12px;
        margin: 8px 0;
    }
    .map-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        background: rgba(255,255,255,0.06);
        border-radius: 12px;
        padding-left: 16px;
        padding-right: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Game data
# =========================================================
ZONES = {
    "Neon District": {
        "danger": 1,
        "description": "The starter zone. Low-level gangs, neon markets, and a skyline soaked in rain.",
        "loot": ["Scrap", "Medkit", "Data Shard"],
        "vendors": ["Arms Dealer", "Street Doc", "Fixer"],
    },
    "Iron Docks": {
        "danger": 2,
        "description": "Cargo cranes, black market imports, and corporate smugglers working after midnight.",
        "loot": ["Titanium", "Fuel Cell", "Encrypted Key"],
        "vendors": ["Smuggler", "Mechanic", "Dock Broker"],
    },
    "Ash Barrens": {
        "danger": 3,
        "description": "A wasteland beyond the city grid where rogue drones patrol the dunes.",
        "loot": ["Alloy Plate", "Rare Crystal", "Drone Core"],
        "vendors": ["Nomad Trader", "Salvage Hunter"],
    },
    "Oracle Heights": {
        "danger": 4,
        "description": "The richest district in the world-state. Corporate citadels, elite security, hidden vaults.",
        "loot": ["Quantum Chip", "Vault Pass", "Legendary Cache"],
        "vendors": ["Broker", "Black AI", "Luxury Dealer"],
    },
}

ENEMIES = {
    1: [
        {"name": "Street Punk", "hp": 28, "atk": 6, "credits": 25, "xp": 18},
        {"name": "Drone Scout", "hp": 24, "atk": 7, "credits": 30, "xp": 20},
    ],
    2: [
        {"name": "Dock Enforcer", "hp": 40, "atk": 9, "credits": 48, "xp": 30},
        {"name": "Cargo Mech", "hp": 46, "atk": 10, "credits": 54, "xp": 34},
    ],
    3: [
        {"name": "Rogue Drone", "hp": 55, "atk": 12, "credits": 70, "xp": 46},
        {"name": "Sand Reaver", "hp": 62, "atk": 13, "credits": 84, "xp": 52},
    ],
    4: [
        {"name": "Oracle Guard", "hp": 82, "atk": 16, "credits": 130, "xp": 80},
        {"name": "Sentinel AI", "hp": 95, "atk": 18, "credits": 155, "xp": 92},
    ],
}

ITEM_SHOP = {
    "Medkit": {"type": "consumable", "price": 35, "heal": 35},
    "Nano Pack": {"type": "consumable", "price": 90, "heal": 90},
    "Iron Blade": {"type": "weapon", "price": 120, "power": 5},
    "Plasma Rifle": {"type": "weapon", "price": 260, "power": 10},
    "Titan Armor": {"type": "armor", "price": 220, "armor": 4},
    "Aegis Shell": {"type": "armor", "price": 420, "armor": 8},
    "Boost Chip": {"type": "augment", "price": 180, "crit": 0.08},
}

QUESTS = [
    {
        "name": "First Blood",
        "description": "Defeat 3 enemies anywhere in Neon District.",
        "goal": 3,
        "type": "kills",
        "zone": "Neon District",
        "reward_credits": 120,
        "reward_xp": 80,
    },
    {
        "name": "Dockside Runner",
        "description": "Win 2 races in Iron Docks.",
        "goal": 2,
        "type": "races",
        "zone": "Iron Docks",
        "reward_credits": 180,
        "reward_xp": 120,
    },
    {
        "name": "Data Heist",
        "description": "Complete 2 hacks in Oracle Heights.",
        "goal": 2,
        "type": "hacks",
        "zone": "Oracle Heights",
        "reward_credits": 300,
        "reward_xp": 180,
    },
]

LORE_EVENTS = [
    "A blackout rolls across the skyline and every tower blinks blood-red for three seconds.",
    "A pirate radio broadcast claims the city governor is a simulated construct.",
    "Street rumors say a vault train crossed the Ash Barrens at dawn under military escort.",
    "Rain begins falling inside the transit tunnels though the weather grid says clear skies.",
    "A trader offers you a map fragment, then vanishes into a crowd that parts too quickly.",
]

# =========================================================
# Helpers
# =========================================================
def level_threshold(level: int) -> int:
    return 100 + (level - 1) * 90


def ensure_state() -> None:
    if "game" in st.session_state:
        return
    st.session_state.game = {
        "name": "Operator",
        "class": "Rogue",
        "zone": "Neon District",
        "level": 1,
        "xp": 0,
        "credits": 180,
        "hp": 100,
        "max_hp": 100,
        "weapon_bonus": 0,
        "armor": 0,
        "crit": 0.06,
        "inventory": {"Medkit": 2, "Scrap": 1},
        "kills": 0,
        "races": 0,
        "hacks": 0,
        "world_clock": 1,
        "notoriety": 0,
        "story_log": ["You arrive in Neon District with a fake ID, a low-grade sidearm, and a debt you do not remember taking on."],
        "market_history": [],
        "active_enemy": None,
        "leaderboard": [],
        "quests": [{**q, "progress": 0, "completed": False} for q in QUESTS],
        "properties": [],
        "music_on": True,
    }
    seed_market()


def seed_market() -> None:
    game = st.session_state.game
    if game["market_history"]:
        return
    items = ["Scrap", "Fuel Cell", "Rare Crystal", "Quantum Chip", "Medkit"]
    base_prices = {"Scrap": 18, "Fuel Cell": 60, "Rare Crystal": 115, "Quantum Chip": 240, "Medkit": 35}
    for tick in range(1, 26):
        for item in items:
            drift = math.sin(tick / 3) * 0.04
            vol = random.uniform(-0.16, 0.16)
            px = max(5, round(base_prices[item] * (1 + drift + vol), 2))
            game["market_history"].append({"day": tick, "item": item, "price": px})


def get_market_snapshot() -> pd.DataFrame:
    df = pd.DataFrame(st.session_state.game["market_history"])
    latest_day = df["day"].max()
    return df[df["day"] == latest_day].copy()


def evolve_market() -> None:
    game = st.session_state.game
    df = pd.DataFrame(game["market_history"])
    latest_day = int(df["day"].max())
    latest = df[df["day"] == latest_day]
    next_rows = []
    for _, row in latest.iterrows():
        macro = random.choice([0.94, 0.98, 1.00, 1.03, 1.07])
        shock = random.uniform(-0.08, 0.08)
        px = max(5, round(row["price"] * macro * (1 + shock), 2))
        next_rows.append({"day": latest_day + 1, "item": row["item"], "price": px})
    game["market_history"].extend(next_rows)
    game["world_clock"] += 1


def current_stats() -> Dict[str, float]:
    game = st.session_state.game
    attack = 10 + game["level"] * 2 + game["weapon_bonus"]
    defense = game["armor"]
    crit = game["crit"]
    return {"attack": attack, "defense": defense, "crit": crit}


def log_event(text: str) -> None:
    st.session_state.game["story_log"].insert(0, text)
    st.session_state.game["story_log"] = st.session_state.game["story_log"][:18]


def maybe_level_up() -> None:
    game = st.session_state.game
    while game["xp"] >= level_threshold(game["level"]):
        game["xp"] -= level_threshold(game["level"])
        game["level"] += 1
        hp_gain = 18
        game["max_hp"] += hp_gain
        game["hp"] = game["max_hp"]
        game["credits"] += 50
        log_event(f"Level up. You are now level {game['level']}. Max HP increased by {hp_gain} and your systems recalibrated.")


def spawn_enemy() -> Dict:
    zone = st.session_state.game["zone"]
    danger = ZONES[zone]["danger"]
    template = random.choice(ENEMIES[danger]).copy()
    lvl = st.session_state.game["level"]
    template["hp"] += lvl * 3
    template["atk"] += max(0, lvl - 1)
    return template


def update_quests(action_type: str, zone: str) -> None:
    game = st.session_state.game
    for q in game["quests"]:
        if q["completed"]:
            continue
        if q["type"] == action_type and q["zone"] == zone:
            q["progress"] += 1
            if q["progress"] >= q["goal"]:
                q["completed"] = True
                game["credits"] += q["reward_credits"]
                game["xp"] += q["reward_xp"]
                log_event(f"Quest completed: {q['name']}. Rewards collected: {q['reward_credits']} credits and {q['reward_xp']} XP.")
                maybe_level_up()


def inventory_add(item: str, qty: int = 1) -> None:
    inv = st.session_state.game["inventory"]
    inv[item] = inv.get(item, 0) + qty


def inventory_remove(item: str, qty: int = 1) -> bool:
    inv = st.session_state.game["inventory"]
    if inv.get(item, 0) < qty:
        return False
    inv[item] -= qty
    if inv[item] <= 0:
        del inv[item]
    return True


def rest_at_safehouse() -> None:
    game = st.session_state.game
    fee = 25 + game["level"] * 5
    if game["credits"] < fee:
        st.warning(f"You need {fee} credits to recover at the safehouse.")
        return
    game["credits"] -= fee
    game["hp"] = game["max_hp"]
    evolve_market()
    log_event(f"You spend {fee} credits at a safehouse. Wounds close, ammo resets, and the city moves one day forward.")


def buy_item(item_name: str) -> None:
    game = st.session_state.game
    data = ITEM_SHOP[item_name]
    if game["credits"] < data["price"]:
        st.warning("Not enough credits.")
        return
    game["credits"] -= data["price"]
    if data["type"] == "consumable":
        inventory_add(item_name, 1)
    elif data["type"] == "weapon":
        game["weapon_bonus"] = max(game["weapon_bonus"], data["power"])
    elif data["type"] == "armor":
        game["armor"] = max(game["armor"], data["armor"])
    elif data["type"] == "augment":
        game["crit"] = max(game["crit"], 0.06 + data["crit"])
    log_event(f"Purchased {item_name} for {data['price']} credits.")


def use_medkit(name: str) -> None:
    if not inventory_remove(name, 1):
        st.warning(f"No {name} available.")
        return
    heal_amt = ITEM_SHOP[name]["heal"]
    game = st.session_state.game
    before = game["hp"]
    game["hp"] = min(game["max_hp"], game["hp"] + heal_amt)
    healed = game["hp"] - before
    log_event(f"Used {name}. Restored {healed} HP.")


def do_explore() -> None:
    game = st.session_state.game
    zone = game["zone"]
    pool = ZONES[zone]["loot"]
    roll = random.random()
    if roll < 0.52:
        loot = random.choice(pool)
        inventory_add(loot, 1)
        found = random.randint(8, 28) + ZONES[zone]["danger"] * 5
        game["credits"] += found
        log_event(f"You sweep {zone} and come back with {loot} plus {found} loose credits.")
    elif roll < 0.82:
        dmg = random.randint(4, 14) + ZONES[zone]["danger"] * 2
        game["hp"] = max(1, game["hp"] - dmg)
        log_event(f"An ambush clips you during exploration. You lose {dmg} HP but escape into the alleys.")
    else:
        game["active_enemy"] = spawn_enemy()
        log_event(f"A hostile target emerges in {zone}: {game['active_enemy']['name']}.")
    evolve_market()


def do_combat_turn(action: str) -> None:
    game = st.session_state.game
    enemy = game["active_enemy"]
    if not enemy:
        st.warning("No enemy active.")
        return

    stats = current_stats()
    narrative = []

    if action == "attack":
        crit_hit = random.random() < stats["crit"]
        player_dmg = random.randint(int(stats["attack"] * 0.7), int(stats["attack"] * 1.25))
        if crit_hit:
            player_dmg = int(player_dmg * 1.8)
        enemy["hp"] -= player_dmg
        narrative.append(f"You hit {enemy['name']} for {player_dmg} damage" + (" with a critical strike." if crit_hit else "."))
    elif action == "ability":
        overload = random.randint(int(stats["attack"] * 1.1), int(stats["attack"] * 1.8))
        self_cost = random.randint(4, 10)
        enemy["hp"] -= overload
        game["hp"] = max(1, game["hp"] - self_cost)
        narrative.append(f"You fire an overload burst for {overload} damage and suffer {self_cost} recoil damage.")
    elif action == "flee":
        if random.random() < 0.6:
            log_event(f"You vanish into the grid and escape {enemy['name']}.")
            game["active_enemy"] = None
            evolve_market()
            return
        narrative.append("Escape attempt fails. The enemy closes distance.")

    if enemy["hp"] > 0:
        enemy_dmg = max(1, enemy["atk"] - stats["defense"] + random.randint(-2, 3))
        game["hp"] = max(0, game["hp"] - enemy_dmg)
        narrative.append(f"{enemy['name']} hits back for {enemy_dmg} damage.")

    if enemy["hp"] <= 0:
        game["credits"] += enemy["credits"]
        game["xp"] += enemy["xp"]
        game["kills"] += 1
        zone = game["zone"]
        update_quests("kills", zone)
        maybe_level_up()
        inventory_add(random.choice(ZONES[zone]["loot"]), 1)
        log_event(" ".join(narrative) + f" Target eliminated. You loot {enemy['credits']} credits and gain {enemy['xp']} XP.")
        game["active_enemy"] = None
        evolve_market()
        return

    if game["hp"] <= 0:
        game["hp"] = max(18, int(game["max_hp"] * 0.35))
        loss = min(game["credits"], 90)
        game["credits"] -= loss
        game["notoriety"] = max(0, game["notoriety"] - 1)
        log_event(" ".join(narrative) + f" You black out and wake up at a clinic. {loss} credits are gone.")
        game["active_enemy"] = None
        evolve_market()
        return

    log_event(" ".join(narrative))


def do_trade(item: str, qty: int, direction: str) -> None:
    game = st.session_state.game
    market = get_market_snapshot().set_index("item")
    if item not in market.index:
        st.warning("Item unavailable.")
        return
    px = float(market.loc[item, "price"])
    total = round(px * qty, 2)
    if direction == "buy":
        if game["credits"] < total:
            st.warning("Insufficient credits.")
            return
        game["credits"] -= total
        inventory_add(item, qty)
        log_event(f"Bought {qty}x {item} at {px} for {total} credits.")
    else:
        if not inventory_remove(item, qty):
            st.warning("Not enough inventory.")
            return
        game["credits"] += total
        log_event(f"Sold {qty}x {item} at {px} for {total} credits.")
    evolve_market()


def acquire_property() -> None:
    game = st.session_state.game
    zone = game["zone"]
    cost = ZONES[zone]["danger"] * 350
    if zone in game["properties"]:
        st.info("You already control an outpost here.")
        return
    if game["credits"] < cost:
        st.warning(f"You need {cost} credits.")
        return
    game["credits"] -= cost
    game["properties"].append(zone)
    game["max_hp"] += 10
    game["hp"] = game["max_hp"]
    log_event(f"You establish an outpost in {zone} for {cost} credits. Income routes and local influence expand.")


def collect_property_income() -> None:
    game = st.session_state.game
    if not game["properties"]:
        st.info("No controlled outposts yet.")
        return
    income = sum(ZONES[z]["danger"] * 40 for z in game["properties"])
    game["credits"] += income
    game["notoriety"] += len(game["properties"])
    evolve_market()
    log_event(f"Your network pays out {income} credits. The city also notices. Notoriety rises.")


def save_score() -> None:
    game = st.session_state.game
    score = game["level"] * 100 + game["credits"] + game["kills"] * 25 + len(game["properties"]) * 150
    entry = {
        "name": game["name"],
        "class": game["class"],
        "score": score,
        "level": game["level"],
        "credits": game["credits"],
        "kills": game["kills"],
    }
    board = game["leaderboard"]
    board.append(entry)
    board.sort(key=lambda x: x["score"], reverse=True)
    game["leaderboard"] = board[:10]
    log_event(f"Score saved to local leaderboard at {score}.")


def start_new_run(name: str, klass: str) -> None:
    st.session_state.pop("game", None)
    ensure_state()
    st.session_state.game["name"] = name
    st.session_state.game["class"] = klass
    if klass == "Mercenary":
        st.session_state.game["weapon_bonus"] = 2
        st.session_state.game["credits"] += 40
    elif klass == "Rogue":
        st.session_state.game["crit"] = 0.12
    elif klass == "Techno Mage":
        st.session_state.game["max_hp"] = 92
        st.session_state.game["hp"] = 92
        st.session_state.game["weapon_bonus"] = 4
    log_event(f"A new run begins. {name}, the {klass}, enters Neon District.")


def render_audio() -> None:
    if not st.session_state.game.get("music_on", True):
        return
    html = """
    <script>
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    function tone(freq, start, dur, gain) {
        const o = ctx.createOscillator();
        const g = ctx.createGain();
        o.connect(g); g.connect(ctx.destination);
        o.type = 'sine'; o.frequency.value = freq;
        g.gain.value = gain;
        o.start(ctx.currentTime + start);
        o.stop(ctx.currentTime + start + dur);
    }
    const now = 0.05;
    [220,330,440,392,330].forEach((f, i) => tone(f, now + i*0.12, 0.10, 0.015));
    </script>
    """
    components.html(html, height=0)


def render_city_map() -> None:
    current_zone = st.session_state.game["zone"]
    labels = list(ZONES.keys())
    x = [0, 1, 2, 3]
    y = [0, 1.2, 0.2, 1.4]
    danger = [ZONES[k]["danger"] for k in labels]
    df = pd.DataFrame({"Zone": labels, "x": x, "y": y, "Danger": danger})
    chart = (
        alt.Chart(df)
        .mark_circle(size=850)
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            color=alt.Color("Danger:Q", scale=alt.Scale(scheme="plasma"), legend=None),
            tooltip=["Zone", "Danger"],
            stroke=alt.condition(alt.datum.Zone == current_zone, alt.value("#ffffff"), alt.value("#111827")),
            strokeWidth=alt.condition(alt.datum.Zone == current_zone, alt.value(3), alt.value(1)),
        )
        .properties(height=280)
    )
    text = alt.Chart(df).mark_text(fontSize=14, dy=0, color="white").encode(x="x:Q", y="y:Q", text="Zone")
    st.altair_chart(chart + text, use_container_width=True)


def render_market_chart() -> None:
    df = pd.DataFrame(st.session_state.game["market_history"])
    chart = (
        alt.Chart(df)
        .mark_line(point=False)
        .encode(
            x=alt.X("day:Q", title="World Day"),
            y=alt.Y("price:Q", title="Price"),
            color=alt.Color("item:N", legend=alt.Legend(orient="bottom")),
            tooltip=["day", "item", "price"],
        )
        .properties(height=280)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def render_hacking_minigame() -> None:
    st.markdown("<div class='glass'><h4>Neural Breach</h4><div class='flavor'>Beat the timer and match the code. This uses a self-contained HTML canvas mini-game inside Streamlit.</div></div>", unsafe_allow_html=True)
    html = """
    <html>
    <body style='margin:0;background:#0b1020;color:#e5e7eb;font-family:Inter,system-ui;'>
      <div style='padding:12px 14px;'>
        <div style='font-size:14px;margin-bottom:8px;'>Type the displayed 4-digit code before the timer expires, then press Submit.</div>
        <canvas id='hack' width='720' height='180' style='width:100%;background:linear-gradient(135deg,#111827,#0f172a);border:1px solid rgba(255,255,255,0.12);border-radius:16px;'></canvas>
        <div style='margin-top:12px;display:flex;gap:10px;align-items:center;'>
          <input id='guess' maxlength='4' style='padding:10px 12px;border-radius:10px;border:none;width:120px;font-size:18px;background:#1f2937;color:white;' />
          <button onclick='submitGuess()' style='padding:10px 14px;border-radius:10px;border:none;background:#2563eb;color:white;font-weight:700;'>Submit</button>
          <button onclick='newRound()' style='padding:10px 14px;border-radius:10px;border:none;background:#7c3aed;color:white;font-weight:700;'>Reset</button>
          <span id='status' style='font-size:14px;color:#cbd5e1;'></span>
        </div>
      </div>
      <script>
        const canvas = document.getElementById('hack');
        const ctx = canvas.getContext('2d');
        let secret = '';
        let start = Date.now();
        let won = false;
        function randCode(){ return Math.floor(1000 + Math.random()*9000).toString(); }
        function newRound(){
          secret = randCode();
          start = Date.now();
          won = false;
          document.getElementById('guess').value = '';
          document.getElementById('status').textContent = 'New breach initialized.';
        }
        function draw(){
          const elapsed = (Date.now() - start)/1000;
          const left = Math.max(0, 10 - elapsed);
          ctx.clearRect(0,0,canvas.width,canvas.height);
          for(let i=0;i<70;i++){
            ctx.fillStyle = `rgba(34,211,238,${Math.random()*0.35})`;
            ctx.fillRect(Math.random()*canvas.width, Math.random()*canvas.height, 2, 8+Math.random()*18);
          }
          ctx.fillStyle = '#e5e7eb';
          ctx.font = 'bold 28px Inter';
          ctx.fillText('ACCESS CODE: ' + secret, 24, 48);
          ctx.font = '18px Inter';
          ctx.fillText('Time left: ' + left.toFixed(1) + 's', 24, 84);
          ctx.fillText('Signal stability: ' + Math.max(0, Math.round(left*10)) + '%', 24, 114);
          ctx.fillStyle = left > 3 ? '#22c55e' : '#ef4444';
          ctx.fillRect(24, 136, left/10 * 640, 16);
          if(left <= 0 && !won){
            document.getElementById('status').textContent = 'Breach failed. Reset and try again.';
          } else {
            requestAnimationFrame(draw);
          }
        }
        function submitGuess(){
          const g = document.getElementById('guess').value;
          const elapsed = (Date.now() - start)/1000;
          if(elapsed > 10){
            document.getElementById('status').textContent = 'Too late.';
            return;
          }
          if(g === secret){
            won = true;
            document.getElementById('status').textContent = 'Success. You cracked the node.';
          } else {
            document.getElementById('status').textContent = 'Mismatch. Try faster.';
          }
        }
        newRound();
        draw();
      </script>
    </body>
    </html>
    """
    components.html(html, height=290)
    if st.button("Resolve hack in the RPG layer", key="hack_resolve"):
        success = random.random() < 0.68 + min(0.18, st.session_state.game["level"] * 0.02)
        if success:
            reward = 80 + ZONES[st.session_state.game["zone"]]["danger"] * 50
            st.session_state.game["credits"] += reward
            st.session_state.game["xp"] += 45
            st.session_state.game["hacks"] += 1
            update_quests("hacks", st.session_state.game["zone"])
            maybe_level_up()
            log_event(f"Hack successful. You siphon {reward} credits and extract sensitive intel.")
        else:
            loss = random.randint(8, 22)
            st.session_state.game["hp"] = max(1, st.session_state.game["hp"] - loss)
            log_event(f"Counter-intrusion detected. Neural burn costs you {loss} HP.")
        evolve_market()


def render_racing_minigame() -> None:
    st.markdown("<div class='glass'><h4>Arc Sprint</h4><div class='flavor'>A lightweight lane racer embedded with HTML and JavaScript. It is not GTA, but inside Streamlit this is about as far as you can push the shell while keeping it runnable in one file.</div></div>", unsafe_allow_html=True)
    html = """
    <html>
    <body style='margin:0;background:#0b1020;color:#e5e7eb;font-family:Inter,system-ui;'>
      <div style='padding:8px 12px;'>
        <div style='font-size:14px;margin-bottom:6px;'>Use A and D to dodge. Survive 20 seconds.</div>
        <canvas id='race' width='720' height='320' style='width:100%;background:#111827;border:1px solid rgba(255,255,255,0.12);border-radius:16px;'></canvas>
      </div>
      <script>
        const c = document.getElementById('race');
        const ctx = c.getContext('2d');
        let player = {lane:1, y:260};
        let obs = [];
        let t0 = Date.now();
        let over = false;
        document.addEventListener('keydown', (e)=>{
          if(e.key === 'a' || e.key === 'ArrowLeft') player.lane = Math.max(0, player.lane-1);
          if(e.key === 'd' || e.key === 'ArrowRight') player.lane = Math.min(2, player.lane+1);
        });
        function spawn(){
          obs.push({lane:Math.floor(Math.random()*3), y:-20, speed:4 + Math.random()*4});
        }
        function drawLane(){
          ctx.fillStyle='#0f172a'; ctx.fillRect(0,0,c.width,c.height);
          [240,480].forEach(x=>{ ctx.strokeStyle='rgba(255,255,255,0.18)'; ctx.setLineDash([12,18]); ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,c.height); ctx.stroke();});
          ctx.setLineDash([]);
        }
        function rectForLane(lane,y){ return {x: lane*240 + 85, y, w:70, h:42}; }
        function hit(a,b){ return a.x < b.x+b.w && a.x+a.w > b.x && a.y < b.y+b.h && a.y+a.h > b.y; }
        function loop(){
          if(over) return;
          drawLane();
          const elapsed = (Date.now()-t0)/1000;
          if(Math.random() < 0.05) spawn();
          const p = rectForLane(player.lane, player.y);
          ctx.fillStyle='#22c55e'; ctx.fillRect(p.x,p.y,p.w,p.h);
          ctx.fillStyle='#e5e7eb'; ctx.font='18px Inter'; ctx.fillText('Time: ' + elapsed.toFixed(1), 16, 26);
          for(let i=obs.length-1;i>=0;i--){
            obs[i].y += obs[i].speed;
            const o = rectForLane(obs[i].lane, obs[i].y);
            ctx.fillStyle='#ef4444'; ctx.fillRect(o.x,o.y,o.w,o.h);
            if(hit(p,o)) { over = true; ctx.fillStyle='white'; ctx.font='bold 32px Inter'; ctx.fillText('CRASH', 300, 160); return; }
            if(obs[i].y > 360) obs.splice(i,1);
          }
          if(elapsed >= 20){ over = true; ctx.fillStyle='white'; ctx.font='bold 28px Inter'; ctx.fillText('WIN', 324, 160); return; }
          requestAnimationFrame(loop);
        }
        loop();
      </script>
    </body>
    </html>
    """
    components.html(html, height=360)
    if st.button("Resolve race in the RPG layer", key="race_resolve"):
        success = random.random() < 0.62 + min(0.15, st.session_state.game["level"] * 0.02)
        if success:
            reward = 95 + ZONES[st.session_state.game["zone"]]["danger"] * 60
            st.session_state.game["credits"] += reward
            st.session_state.game["xp"] += 55
            st.session_state.game["races"] += 1
            update_quests("races", st.session_state.game["zone"])
            maybe_level_up()
            log_event(f"You win the street race and bank {reward} credits.")
        else:
            loss = random.randint(10, 25)
            st.session_state.game["hp"] = max(1, st.session_state.game["hp"] - loss)
            log_event(f"The run goes wrong. You lose {loss} HP in the aftermath.")
        evolve_market()


# =========================================================
# App
# =========================================================
ensure_state()
render_audio()
game = st.session_state.game

with st.sidebar:
    st.markdown("## New Run")
    new_name = st.text_input("Operator Name", value=game["name"])
    new_class = st.selectbox("Class", ["Rogue", "Mercenary", "Techno Mage"], index=["Rogue", "Mercenary", "Techno Mage"].index(game["class"]))
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start Over", use_container_width=True):
            start_new_run(new_name, new_class)
            st.rerun()
    with c2:
        if st.button("Save Score", use_container_width=True):
            save_score()

    st.markdown("---")
    game["music_on"] = st.toggle("Synth Stinger", value=game.get("music_on", True))
    game["zone"] = st.selectbox("Travel", list(ZONES.keys()), index=list(ZONES.keys()).index(game["zone"]))
    if st.button("Rest at Safehouse", use_container_width=True):
        rest_at_safehouse()
        st.rerun()
    if st.button("Collect Outpost Income", use_container_width=True):
        collect_property_income()
        st.rerun()
    if st.button("Acquire Local Outpost", use_container_width=True):
        acquire_property()
        st.rerun()

st.markdown(
    f"""
    <div class='title-card'>
        <div style='display:flex;justify-content:space-between;gap:18px;align-items:flex-start;flex-wrap:wrap;'>
            <div>
                <div style='font-size:2.3rem;font-weight:800;line-height:1;'>Neon Frontier</div>
                <div style='margin-top:8px;color:#cbd5e1;font-size:1.02rem;'>A Streamlit-native cyberpunk sandbox with RPG progression, trading, procedural combat, property control, charts, quests, HTML canvas mini-games, local leaderboard logic, custom CSS, session state persistence, and a deliberately overbuilt interface.</div>
            </div>
            <div class='glass' style='min-width:280px;'>
                <div class='small-label'>Live world flavor</div>
                <div class='flavor' style='margin-top:8px;'>{random.choice(LORE_EVENTS)}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

stats = current_stats()
c1, c2, c3, c4, c5, c6 = st.columns(6)
for col, label, value, sub in [
    (c1, "Operator", game["name"], f"{game['class']} | Level {game['level']}"),
    (c2, "HP", f"{game['hp']} / {game['max_hp']}", f"Armor {game['armor']}"),
    (c3, "Credits", f"{game['credits']}", f"World Day {game['world_clock']}"),
    (c4, "Attack", f"{stats['attack']}", f"Crit {round(stats['crit']*100, 1)}%"),
    (c5, "Zone", game["zone"], f"Danger {ZONES[game['zone']]['danger']}"),
    (c6, "Notoriety", f"{game['notoriety']}", f"Kills {game['kills']}"),
]:
    with col:
        st.markdown(f"<div class='metric-card'><div class='small-label'>{label}</div><div class='big-number'>{value}</div><div class='flavor'>{sub}</div></div>", unsafe_allow_html=True)

left, right = st.columns([1.25, 1])
with left:
    st.markdown("<div class='glass'><h4>City Grid</h4><div class='flavor'>Each district scales enemy difficulty, loot quality, and property economics. This is the macro map of your run.</div></div>", unsafe_allow_html=True)
    render_city_map()
with right:
    st.markdown(f"<div class='glass'><h4>{game['zone']}</h4><div class='flavor'>{ZONES[game['zone']]['description']}</div><div class='flavor' style='margin-top:10px;'>Vendors: {', '.join(ZONES[game['zone']]['vendors'])}</div><div class='flavor'>Potential loot: {', '.join(ZONES[game['zone']]['loot'])}</div></div>", unsafe_allow_html=True)
    q_html = "".join([
        f"<div class='quest-box'><b>{q['name']}</b><br>{q['description']}<br>Progress: {q['progress']} / {q['goal']} {'✅' if q['completed'] else ''}</div>"
        for q in game['quests']
    ])
    st.markdown(f"<div class='glass'><h4>Quest Board</h4>{q_html}</div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Adventure", "Combat", "Market", "Shop & Inventory", "Mini-Games", "Logs & Leaderboard"])

with tab1:
    a1, a2, a3 = st.columns([1, 1, 1.1])
    with a1:
        st.markdown("<div class='glass'><h4>Explore</h4><div class='flavor'>Sweep the zone for loot, credits, danger, or a combat encounter.</div></div>", unsafe_allow_html=True)
        if st.button("Explore the Zone", use_container_width=True):
            do_explore()
            st.rerun()
        if st.button("Generate Random World Event", use_container_width=True):
            log_event(random.choice(LORE_EVENTS))
            evolve_market()
            st.rerun()
    with a2:
        st.markdown("<div class='glass'><h4>Run Snapshot</h4></div>", unsafe_allow_html=True)
        st.progress(min(1.0, game["xp"] / level_threshold(game["level"])))
        st.caption(f"XP to next level: {game['xp']} / {level_threshold(game['level'])}")
        st.metric("Properties Controlled", len(game["properties"]))
        st.metric("Races Won", game["races"])
        st.metric("Successful Hacks", game["hacks"])
    with a3:
        inv_df = pd.DataFrame([
            {"Item": k, "Qty": v} for k, v in sorted(game["inventory"].items())
        ]) if game["inventory"] else pd.DataFrame({"Item": [], "Qty": []})
        st.markdown("<div class='glass'><h4>Inventory Snapshot</h4></div>", unsafe_allow_html=True)
        st.dataframe(inv_df, use_container_width=True, hide_index=True)

with tab2:
    st.markdown("<div class='glass'><h4>Combat Arena</h4><div class='flavor'>Turn-based combat tied into your persistent run. Abilities, risk, escape attempts, loot, and leveling all feed back into the world state.</div></div>", unsafe_allow_html=True)
    enemy = game["active_enemy"]
    if enemy:
        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("Enemy", enemy["name"])
        ec2.metric("Enemy HP", max(0, enemy["hp"]))
        ec3.metric("Enemy Attack", enemy["atk"])
        x1, x2, x3 = st.columns(3)
        with x1:
            if st.button("Attack", use_container_width=True):
                do_combat_turn("attack")
                st.rerun()
        with x2:
            if st.button("Special Ability", use_container_width=True):
                do_combat_turn("ability")
                st.rerun()
        with x3:
            if st.button("Flee", use_container_width=True):
                do_combat_turn("flee")
                st.rerun()
    else:
        st.info("No active enemy. Exploring can trigger an encounter.")

with tab3:
    st.markdown("<div class='glass'><h4>Dynamic Commodity Market</h4><div class='flavor'>A toy economy with evolving daily prices, item inventory, and a tradable spread between safe and dangerous zones. This gives the run a second loop beyond combat.</div></div>", unsafe_allow_html=True)
    render_market_chart()
    snap = get_market_snapshot().reset_index(drop=True)
    t1, t2, t3 = st.columns([1.2, 1, 1])
    with t1:
        st.dataframe(snap, use_container_width=True, hide_index=True)
    with t2:
        item = st.selectbox("Market Item", list(snap["item"]))
        qty = st.slider("Quantity", 1, 10, 1)
        if st.button("Buy", use_container_width=True):
            do_trade(item, qty, "buy")
            st.rerun()
        if st.button("Sell", use_container_width=True):
            do_trade(item, qty, "sell")
            st.rerun()
    with t3:
        market_df = pd.DataFrame(game["market_history"])
        latest = market_df.groupby("item").tail(1).set_index("item")
        first = market_df.groupby("item").head(1).set_index("item")
        perf = ((latest["price"] / first["price"] - 1) * 100).round(1).reset_index()
        perf.columns = ["Item", "Return %"]
        st.dataframe(perf, use_container_width=True, hide_index=True)

with tab4:
    st.markdown("<div class='glass'><h4>Shop & Loadout</h4><div class='flavor'>Permanent upgrades, healing, and inventory management.</div></div>", unsafe_allow_html=True)
    s1, s2 = st.columns([1.15, 1])
    with s1:
        shop_df = pd.DataFrame([
            {"Item": k, **v} for k, v in ITEM_SHOP.items()
        ])
        st.dataframe(shop_df, use_container_width=True, hide_index=True)
        purchase_item = st.selectbox("Shop Item", list(ITEM_SHOP.keys()))
        if st.button("Purchase Selected Item", use_container_width=True):
            buy_item(purchase_item)
            st.rerun()
    with s2:
        st.markdown("<div class='glass'><h4>Consumables</h4></div>", unsafe_allow_html=True)
        med_cols = st.columns(2)
        with med_cols[0]:
            if st.button("Use Medkit", use_container_width=True):
                use_medkit("Medkit")
                st.rerun()
        with med_cols[1]:
            if st.button("Use Nano Pack", use_container_width=True):
                use_medkit("Nano Pack")
                st.rerun()
        st.markdown("<div class='glass'><h4>Inventory</h4></div>", unsafe_allow_html=True)
        inv_df = pd.DataFrame([
            {"Item": k, "Qty": v} for k, v in sorted(game["inventory"].items())
        ]) if game["inventory"] else pd.DataFrame({"Item": [], "Qty": []})
        st.dataframe(inv_df, use_container_width=True, hide_index=True)

with tab5:
    mg1, mg2 = st.columns(2)
    with mg1:
        render_hacking_minigame()
    with mg2:
        render_racing_minigame()

with tab6:
    l1, l2 = st.columns([1.15, 0.85])
    with l1:
        st.markdown("<div class='glass'><h4>Story Log</h4></div>", unsafe_allow_html=True)
        for line in game["story_log"]:
            st.markdown(f"<div class='quest-box'>{line}</div>", unsafe_allow_html=True)
    with l2:
        st.markdown("<div class='glass'><h4>Leaderboard</h4></div>", unsafe_allow_html=True)
        board = pd.DataFrame(game["leaderboard"]) if game["leaderboard"] else pd.DataFrame(columns=["name", "class", "score", "level", "credits", "kills"])
        st.dataframe(board, use_container_width=True, hide_index=True)

st.markdown(
    "<div class='glass'><div class='flavor'>What you asked for at the conceptual level, meaning a full World of Warcraft or Grand Theft Auto equivalent, is far beyond what Streamlit is meant to ship as a single self-contained app. What this does instead is push the platform in the right direction: custom styling, persistent run state, embedded JavaScript mini-games, an economy, quests, combat, progression, charts, and a game shell that actually runs inside Streamlit without external services.</div></div>",
    unsafe_allow_html=True,
)
