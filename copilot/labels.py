"""Rule-based labels on normalised text.

The dataset has no part / fault-type columns, so the search filters use these
derived labels. Rules are ordered: the first match wins, so more specific parts
(ROCKER COVER) come before generic ones (OIL).
"""
import re

COMPONENTS = [
    ("rocker cover", r"ROCKER"),
    ("pushrod", r"PUSHROD"),
    ("intake / induction", r"INTAKE|INDUCTION"),
    ("exhaust", r"EXHAUST|MUFFLER|STACK"),
    ("ignition", r"PLUG|MAGNETO|IGNITION|HARNESS|\bLEADS?\b"),
    ("baffle / cooling", r"BAFFL|SHROUD|COOLING|SEAL"),
    ("oil system", r"OIL|DIPSTICK|SUMP|FILLER|BREATHER"),
    ("fuel system", r"FUEL|CARBURETOR|MIXTURE|PRIMER|INJECTOR|SERVO|SNIFFLER"),
    ("propeller / governor", r"PROPELLER|GOVERNOR|SPINNER"),
    ("starter / electrical", r"STARTER|ALTERNATOR|GENERATOR|WIRE|WIRING|ELECTRIC|BATTERY|RING GEAR|VOLT"),
    ("cylinder / valves", r"COMPRESSION|PISTON|VALVE|CYLINDER HEAD|TAPPET|LIFTER|CYLINDER$|CYLINDER [A-Z]*CRACK"),
    ("cowling", r"COWL"),
    ("engine mount / case", r"MOUNT|CRANK|\bCASE\b|SUMP"),
    ("engine operation", r"ROUGH|VIBRAT|OVERSPEED|IDLE|RPM|POWER|QUIT|DIED|SURG|STOP|START|TEMPERATURE|PRESSURE"),
]

FAULT_TYPES = [
    ("low compression", r"COMPRESSION"),
    ("leak", r"LEAK|DRIP|SEEP"),
    ("crack", r"CRACK"),
    ("missing / loose hardware", r"MISSING|LOOSE|SHEARED|BACKED OUT|BACKING OUT|FELL"),
    ("broken / damaged / worn", r"BROKE|DAMAGE|WORN|TORN|CHAF|BENT|FRAY|RUB|DETERIORAT|DENT"),
    ("running problem", r"ROUGH|VIBRAT|IDLE|QUIT|DIED|SURG|RPM|POWER|OVERSPEED|START|STALL|KNOCK|BACKFIR"),
    ("temperature / pressure", r"TEMPERATURE|HOT|OVERHEAT|PRESSURE"),
]

# Action types for the fix leaderboard. Shared by the LLM labeller (as its
# allowed list) and by the rule fallback, so both produce the same labels.
ACTION_TYPES = [
    ("Stop-drilled crack", r"STOP DRILL"),
    ("Replaced", r"REPLACED|REPLACE|INSTALLED NEW|INSTALLED (A )?SERVICEABLE|CHANGED|NEW [A-Z]+ INSTALLED|SWAPPED"),
    ("Repaired / patched", r"REPAIR|PATCH|RIVET|FABRICAT|WELD|DOUBLER|RTV|SILICONE|SEALANT|SEALED"),
    ("Tightened / resecured", r"TIGHTEN|TORQUE|SECURED|SAFETIED|SAFETY WIRE|REATTACH|ATTACHED|STAKED"),
    ("Installed missing hardware", r"INSTALLED .*(SCREW|BOLT|NUT|RIVET|CLAMP|TIE|WASHER|HARDWARE)"),
    ("Adjusted", r"ADJUST|\bSET\b|TIMED|TIMING|REPOSITION"),
    ("Cleaned", r"CLEAN|WASH|FLUSH"),
    ("Inspected - no fault found", r"COULD NOT DUPLICATE|UNABLE TO DUPLICATE|NO DEFECT|NO LEAK|CHECK(ED)? GOOD|"
                                   r"OPERATIONS (CHECK )?GOOD|WITHIN LIMITS|NORMAL|NO DISCREPANC|\bOK\b"),
    ("Inspected / troubleshot", r"INSPECT|CHECK|TROUBLESHOOT|RAN|RUN|REVIEWED|FOUND"),
    ("Removed", r"REMOVED"),
]
ACTION_TYPE_NAMES = [name for name, _ in ACTION_TYPES] + ["Other"]


def _first_match(text: str, rules, default: str) -> str:
    for name, pat in rules:
        if re.search(pat, text):
            return name
    return default


def component(normalized_problem: str) -> str:
    return _first_match(normalized_problem, COMPONENTS, "other")


def fault_type(normalized_problem: str) -> str:
    return _first_match(normalized_problem, FAULT_TYPES, "other")


def action_type(normalized_action: str) -> str:
    return _first_match(normalized_action, ACTION_TYPES, "Other")


COMPONENT_NAMES = [name for name, _ in COMPONENTS] + ["other"]
FAULT_TYPE_NAMES = [name for name, _ in FAULT_TYPES] + ["other"]
