# app.py
import re
from datetime import datetime
import streamlit as st

st.set_page_config(page_title="Lawyer Chat")

DISCLAIMER = (
    "General information only, not legal advice. I am not your lawyer. "
    "For anything time-sensitive or high-stakes, speak to a licensed attorney in the relevant jurisdiction."
)

# Simple jurisdiction sniffing (optional, purely for tone and placeholders)
JURISDICTION_HINTS = {
    "new york": "New York",
    "ny": "New York",
    "delaware": "Delaware",
    "california": "California",
    "ca": "California",
    "texas": "Texas",
    "uk": "England and Wales",
    "england": "England and Wales",
    "london": "England and Wales",
    "turkey": "Turkey",
    "dubai": "UAE (Dubai)",
    "uae": "UAE",
}

def infer_jurisdiction(text: str) -> str:
    t = " " + text.lower().strip() + " "
    for k, v in JURISDICTION_HINTS.items():
        # " ny " guards against matching random substrings
        if f" {k} " in t:
            return v
    return "unspecified"

def extract_dates(text: str) -> str:
    # Lightweight: find ISO-like dates
    matches = re.findall(r"\b(20\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b", text)
    if not matches:
        return "none mentioned"
    # Render the first one nicely
    y, m, d = matches[0]
    try:
        dt = datetime(int(y), int(m), int(d))
        return dt.strftime("%B %d, %Y")
    except Exception:
        return f"{y}-{m}-{d}"

# Rules: (pattern, intent_key)
RULES = [
    (r"\b(sue|lawsuit|litigation|complaint|summons|served|court|judge)\b", "litigation"),
    (r"\b(contract|agreement|msa|sow|term sheet|purchase agreement|lease)\b", "contract"),
    (r"\b(liability|indemnif|hold harmless|damages|cap|limitation of liability)\b", "liability"),
    (r"\b(employment|terminate|fired|severance|noncompete|nonsolicit|confidentiality)\b", "employment"),
    (r"\b(ip|intellectual property|copyright|trademark|patent|assignment|license)\b", "ip"),
    (r"\b(privacy|gdpr|ccpa|data breach|personal data|pii)\b", "privacy"),
    (r"\b(llc|corporation|incorporat|bylaws|operating agreement|cap table)\b", "corporate"),
    (r"\b(real estate|tenant|landlord|rent|evict|deposit|habitability)\b", "real_estate"),
    (r"\b(defamation|libel|slander)\b", "defamation"),
    (r"\b(arbitration|mediate|settlement|demand letter)\b", "dispute_resolution"),
]

TEMPLATES = {
    "litigation": (
        "Under {jurisdiction} practice, the first question is posture: are we pre-suit, threatened, or already filed. "
        "From there I look for (a) the specific causes of action being asserted, (b) what evidence actually exists today, "
        "(c) deadlines like limitation periods and service/response timing, and (d) venue and any arbitration or forum clause.\n\n"
        "Based on what you wrote: \"{user_message}\".\n\n"
        "If you want a usable next step, answer this in one pass: who are the parties, what happened, when it happened (date: {date_hint}), "
        "what documents exist, and what outcome you want (money, injunction, walk-away, reputational containment)."
    ),
    "contract": (
        "Contract review starts with leverage and failure modes. I read for business deal terms first, then the clauses that decide who eats the loss.\n\n"
        "On your point: \"{user_message}\".\n\n"
        "In {jurisdiction}, I would focus on scope, payment triggers, term and termination, assignment, confidentiality, IP ownership, "
        "indemnities, limitation of liability, dispute resolution, and governing law. If you paste the clause or the section headings, "
        "I can map the risk and propose cleaner language."
    ),
    "liability": (
        "Liability analysis is allocation plus caps. The key is what claims can be brought, what damages are recoverable, and what the contract forces one side to cover.\n\n"
        "From what you wrote: \"{user_message}\".\n\n"
        "In {jurisdiction}, I would check: negligence standard, indemnity scope (third-party vs first-party), exclusions (gross negligence, willful misconduct), "
        "damage categories waived (consequential, lost profits), cap level and carve-outs, and insurance requirements. If you tell me the relationship "
        "(vendor, customer, employer, landlord), I will answer in that frame."
    ),
    "employment": (
        "Employment issues turn on jurisdiction and the paper trail. The same facts read very differently depending on contract terms, policy documents, and timing.\n\n"
        "You said: \"{user_message}\".\n\n"
        "In {jurisdiction}, I would identify whether this is at-will, whether there is an offer letter or employment agreement, what policies were acknowledged, "
        "and whether any protected-activity or retaliation angle is being alleged. If you want drafting help, paste the relevant clauses and I will rewrite them tighter."
    ),
    "ip": (
        "IP questions are usually ownership, scope, and chain of title. If ownership is unclear, everything downstream becomes a litigation risk.\n\n"
        "You wrote: \"{user_message}\".\n\n"
        "In {jurisdiction}, I would ask: who created it, under what agreement, using whose resources, and whether there is a written assignment. "
        "If this is a license, the key is field of use, sublicensing, exclusivity, term, and termination effects."
    ),
    "privacy": (
        "Privacy and data issues are facts first, law second. The obligations depend on what data you collected, where the users are, and what your notices promised.\n\n"
        "You said: \"{user_message}\".\n\n"
        "In {jurisdiction}, I would triage: data categories (PII, financial, health), where it is stored, who has access, incident timeline, "
        "breach notification triggers, and vendor contracts. If you describe your product and user geography, I can outline the compliance posture."
    ),
    "corporate": (
        "Corporate structure work is about control, economics, and future dispute prevention.\n\n"
        "On: \"{user_message}\".\n\n"
        "In {jurisdiction}, the core documents are operating agreement/bylaws, equity issuance approvals, IP assignment into the entity, "
        "and clear decision rights. If you tell me whether you care more about control, fundraising flexibility, or tax simplicity, "
        "I will steer the structure accordingly."
    ),
    "real_estate": (
        "Real estate disputes are document-driven. The lease and local rules decide most outcomes before anyone argues.\n\n"
        "You wrote: \"{user_message}\".\n\n"
        "In {jurisdiction}, I would pull the lease section on notices, default, cure periods, maintenance obligations, and fees. "
        "If you paste the relevant lease language and the timeline (date: {date_hint}), I can frame your leverage and the cleanest next move."
    ),
    "defamation": (
        "Defamation analysis is publication, falsity, and damages, plus the practical question of whether litigation makes the story bigger.\n\n"
        "You said: \"{user_message}\".\n\n"
        "In {jurisdiction}, I would ask what exactly was said, to whom, whether it is provably false vs opinion, and what evidence exists. "
        "If you want, I can draft a demand letter that preserves optionality without overreaching."
    ),
    "dispute_resolution": (
        "Dispute resolution is strategy and forum. The fastest path is usually the one the contract forces on you.\n\n"
        "From your message: \"{user_message}\".\n\n"
        "In {jurisdiction}, I would locate any arbitration clause, venue clause, attorney-fee provision, and notice requirements. "
        "If you want a demand letter, tell me the counterparty, the breach, what you want, and your outside deadline."
    ),
    "fallback": (
        "I can respond, but I need a tighter fact pattern. Tell me the jurisdiction (right now it is {jurisdiction}), the parties, what happened, "
        "what documents exist, and what outcome you want.\n\n"
        "Your message: \"{user_message}\".\n\n"
        "If you paste the exact clause or the exact allegation, I will answer in that language and keep it practical."
    ),
}

def classify_intent(text: str) -> str:
    t = text.lower()
    for pat, key in RULES:
        if re.search(pat, t, flags=re.IGNORECASE):
            return key
    return "fallback"

def generate_response(user_message: str) -> str:
    jurisdiction = infer_jurisdiction(user_message)
    date_hint = extract_dates(user_message)
    intent = classify_intent(user_message)

    template = TEMPLATES.get(intent, TEMPLATES["fallback"])
    return (
        template.format(
            user_message=user_message.strip(),
            jurisdiction=jurisdiction,
            date_hint=date_hint,
        )
        + "\n\n"
        + DISCLAIMER
    )

st.title("Lawyer Chat (canned replies)")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Tell me what you are dealing with and include jurisdiction if you have it.\n\n" + DISCLAIMER}
    ]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

prompt = st.chat_input("Type your message...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    reply = generate_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)
