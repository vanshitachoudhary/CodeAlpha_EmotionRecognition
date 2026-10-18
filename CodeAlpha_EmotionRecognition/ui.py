"""Pure HTML/CSS components (no streamlit import -> unit-testable and previewable in a browser)."""
import html
import re

from data import COLOR, EMOJI

e = html.escape
VIOLET = "#7c3aed"


def _c(s: str) -> str:
    """Ek line mein collapse: markdown indented/blank lines ko code-block banne se rokta hai."""
    return re.sub(r"\s*\n\s*", "", s).strip()


CSS = _c("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea {font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;}
#MainMenu, footer {visibility:hidden;}
.block-container{padding-top:1.3rem;max-width:1180px}
.stTabs [data-baseweb="tab-list"]{gap:.35rem;border-bottom:1px solid #e2e8f0}
.stTabs [data-baseweb="tab"]{border-radius:10px 10px 0 0;padding:.55rem 1rem;font-weight:600}
.stButton>button,.stDownloadButton>button{border-radius:10px;font-weight:600}
.hero{position:relative;overflow:hidden;background:linear-gradient(135deg,#2e1065 0%,#6d28d9 55%,#c026d3 100%);
padding:2rem 2.2rem;border-radius:20px;color:#fff;margin-bottom:1.1rem;box-shadow:0 10px 30px rgba(109,40,217,.28)}
.hero:before{content:"";position:absolute;right:-60px;top:-80px;width:300px;height:300px;border-radius:50%;background:rgba(255,255,255,.08)}
.hero:after{content:"";position:absolute;right:140px;bottom:-110px;width:220px;height:220px;border-radius:50%;background:rgba(255,255,255,.06)}
.hero h1{margin:0;font-size:2.15rem;font-weight:800;letter-spacing:-.02em;color:#fff;position:relative}
.hero p{margin:.45rem 0 .9rem;opacity:.93;font-size:1.02rem;max-width:780px;position:relative}
.chip{display:inline-block;padding:.28rem .8rem;margin:0 .4rem .3rem 0;border-radius:999px;background:rgba(255,255,255,.18);font-size:.8rem;font-weight:600;position:relative}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:.9rem;margin-bottom:1.1rem}
.kpi{background:#fff;border:1px solid #e2e8f0;border-radius:16px;padding:1rem 1.2rem;box-shadow:0 1px 3px rgba(15,23,42,.05)}
.kpi .l{font-size:.72rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:#64748b}
.kpi .v{font-size:1.75rem;font-weight:800;color:#0f172a;line-height:1.25}
.kpi .s{font-size:.76rem;color:#64748b}
.panel{background:#fff;border:1px solid #e2e8f0;border-radius:18px;padding:1.2rem 1.4rem;box-shadow:0 1px 3px rgba(15,23,42,.05);color:#0f172a;margin-bottom:.8rem}
.panel h4{margin:0 0 .75rem;font-size:1rem;font-weight:700;color:#0f172a}
.panel .sub{font-size:.8rem;color:#64748b;margin:-.45rem 0 .8rem}
.note{font-size:.78rem;color:#64748b}
.pill{display:inline-block;padding:.34rem .9rem;border-radius:999px;font-weight:700;font-size:.85rem}
.pill.ok{background:#ede9fe;color:#5b21b6}.pill.warn{background:#fef3c7;color:#92400e}
.emo{text-align:center;padding:.4rem 0 .2rem}
.emo .big{font-size:4.2rem;line-height:1.1}
.emo .nm{font-size:1.8rem;font-weight:800;text-transform:capitalize;letter-spacing:-.01em}
.emo .cf{font-size:.9rem;color:#64748b;margin:.2rem 0 .7rem}
.brow{display:grid;grid-template-columns:118px 1fr 52px;align-items:center;gap:.6rem;margin:.32rem 0;font-size:.85rem;color:#334155}
.btrack{height:14px;background:#f1f5f9;border-radius:999px;overflow:hidden}
.bfill{height:100%;border-radius:999px}
.brow .vl{font-weight:700;text-align:right;font-variant-numeric:tabular-nums}
.tl{display:flex;gap:3px;margin:.2rem 0}
.tlc{flex:1;min-width:0;border-radius:8px;padding:.45rem .1rem;text-align:center;color:#fff;font-size:.72rem;font-weight:600}
.tlc b{display:block;font-size:1.15rem;line-height:1.2}
.step{display:flex;gap:.9rem;align-items:flex-start;margin:.7rem 0}
.step .n{flex:0 0 34px;height:34px;border-radius:50%;background:#ede9fe;color:#5b21b6;font-weight:800;display:flex;align-items:center;justify-content:center}
.step h5{margin:0;font-size:.98rem;color:#0f172a}.step p{margin:.15rem 0 0;font-size:.86rem;color:#475569}
.tip{background:#f5f3ff;border-left:4px solid #7c3aed;padding:.7rem 1rem;border-radius:10px;margin-top:.8rem;font-size:.9rem;color:#4c1d95}
.foot{text-align:center;font-size:.76rem;color:#94a3b8;margin-top:1.6rem}
@media (max-width:760px){.kpis{grid-template-columns:repeat(2,1fr)}.hero h1{font-size:1.6rem}.brow{grid-template-columns:96px 1fr 46px}}
</style>""")


def hero(title, subtitle, chips):
    cs = "".join(f'<span class="chip">{e(str(c))}</span>' for c in chips)
    return _c(f'<div class="hero"><h1>🎙️ {e(title)}</h1><p>{e(subtitle)}</p><div>{cs}</div></div>')


def kpi_row(items):
    cells = "".join(f'<div class="kpi"><div class="l">{e(l)}</div><div class="v">{e(v)}</div><div class="s">{e(s)}</div></div>' for l, v, s in items)
    return _c(f'<div class="kpis">{cells}</div>')


def panel(title, inner, sub=""):
    sb = f'<div class="sub">{e(sub)}</div>' if sub else ""
    return _c(f'<div class="panel"><h4>{e(title)}</h4>{sb}{inner}</div>')


def pill(kind, text):
    return _c(f'<span class="pill {kind}">{e(text)}</span>')


def emotion_card(emotion, prob, runner_up=None, low=False):
    ru = f" · runner-up: {e(runner_up[0])} {runner_up[1]:.0%}" if runner_up else ""
    note = pill("warn", "⚠️ Low confidence, treat with caution") if low else pill("ok", "Model's top prediction")
    return _c(f'<div class="emo"><div class="big">{EMOJI.get(emotion, "🎙️")}</div>'
              f'<div class="nm" style="color:{COLOR.get(emotion, "#334155")}">{e(emotion)}</div>'
              f'<div class="cf">{prob:.0%} confidence{ru}</div>{note}</div>')


def prob_bars(probs, classes):
    rows = []
    for p, c in sorted(zip(probs, classes), reverse=True):
        rows.append(f'<div class="brow"><span>{EMOJI.get(c, "")} {e(c)}</span><div class="btrack">'
                    f'<div class="bfill" style="width:{p * 100:.1f}%;background:{COLOR.get(c, VIOLET)}"></div></div>'
                    f'<span class="vl">{p:.0%}</span></div>')
    return _c("".join(rows))


def timeline_strip(times, labels):
    """Har window ka predicted emotion ek coloured cell mein (time ke order mein)."""
    cells = "".join(f'<div class="tlc" style="background:{COLOR.get(l, VIOLET)}"><b>{EMOJI.get(l, "")}</b>{t:.1f}s</div>'
                    for t, l in zip(times, labels))
    return _c(f'<div class="tl">{cells}</div>')


def steps(items):
    return _c("".join(f'<div class="step"><div class="n">{i}</div><div><h5>{e(t)}</h5><p>{e(d)}</p></div></div>'
                      for i, (t, d) in enumerate(items, 1)))


def footer():
    return '<div class="foot">Educational demo · not for hiring, surveillance or any decision about people · CodeAlpha ML Internship</div>'
