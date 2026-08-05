from pathlib import Path

TARGET = Path("pages/1_ADFM_Public_Equities_Baskets.py")
NEEDLE = "                                   'Private Robotics Access Vehicles': ['BOT'],\n"

text = TARGET.read_text(encoding="utf-8")
count = text.count(NEEDLE)
if count != 1:
    raise RuntimeError(f"Expected exactly one Private Robotics Access Vehicles basket, found {count}.")

TARGET.write_text(text.replace(NEEDLE, "", 1), encoding="utf-8")
print("Removed Private Robotics Access Vehicles basket.")
