"""Human-readable output: terminal text, markdown (for PR comments), and one
self-contained HTML page with run history."""
from __future__ import annotations

import html
import json
from pathlib import Path


def pct(x) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def money(x) -> str:
    return "n/a" if x is None else (f"${x:.4f}" if x < 1 else f"${x:.2f}")


def run_text(run: dict, verbose: bool = False) -> str:
    m = run["metrics"]
    lo, hi = m["pass_all_ci95"]
    lines = []
    for cid, c in run["cases"].items():
        s = c["summary"]
        if s["status"] == "unavailable":
            mark, info = "?", "unavailable"
        else:
            mark = "✓" if s["passed_all"] else ("~" if s["flaky"] else "✗")
            info = f"{s['mean']:.0%}" + ("  flaky" if s["flaky"] else "")
        lines.append(f"  {mark} {cid}: {info}")
        if (verbose or not s.get("passed_all")) and s["status"] != "unavailable":
            last = next((r for r in reversed(c["runs"]) if r["status"] != "unavailable"), {})
            if last.get("error"):
                lines.append(f"      error: {last['error']}")
            for ck in last.get("checks", []):
                if ck["score"] < 1 or verbose:
                    flag = " [critical]" if ck["critical"] else ""
                    lines.append(f"      {'·' if ck['score'] >= 1 else '✗'} {ck['type']}{flag}: {ck['detail']}")
    head = (f"\n{run['suite']['n']} cases × {run['config']['repeat']} repeat(s)"
            f"  ·  mean {pct(m['mean'])}  ·  pass^k {m['pass_all_k']}/{m['n_scored']} = {pct(m['pass_all'])}"
            f" (95% CI {lo:.0%}–{hi:.0%})")
    tail = [head]
    if m["flaky"]:
        tail.append(f"flaky: {', '.join(m['flaky'])}")
    if m["unavailable_runs"]:
        tail.append(f"⚠ {m['unavailable_runs']} run(s) UNAVAILABLE: excluded from scores, run marked incomplete")
    con = m.get("consistency")
    if con:
        def pc(x):
            return "n/a" if x is None else f"{x:.0%}"
        tail.append(f"same steps every repeat: {pc(con['path_consistent'])} of {con['n']} cases"
                    f" (pass^k {pc(con['pass_all_when_same_path'])} when steps match, "
                    f"{pc(con['pass_all_when_path_varies'])} when they vary)"
                    + (f" · median token spread {con['median_token_spread']:.0%}" if con['median_token_spread'] is not None else ""))
    tail.append(f"latency p50 {m['latency_ms']['p50']:.0f}ms p95 {m['latency_ms']['p95']:.0f}ms"
                f"  ·  tokens {m['tokens']['input']}/{m['tokens']['output']}  ·  cost {money(m['cost_usd'])}")
    return "\n".join(lines + tail)


def compare_text(cmp: dict) -> str:
    d = cmp["delta"]
    out = [f"{cmp['base']}  →  {cmp['cand']}",
           f"verdict: {cmp['verdict']}",
           f"mean {_signed(d['mean'])}  ·  pass^k {_signed(d['pass_all'])}  ·  "
           f"p95 latency {d['latency_p95_ms']:+.0f}ms  ·  cost {_signed(d['cost_usd'], money=True)}"]
    for key, label in (("regressions", "REGRESSED"), ("fixes", "fixed"), ("newly_flaky", "newly flaky"),
                       ("steadied", "no longer flaky"), ("degraded", "score down, same verdict"),
                       ("improved", "score up, same verdict"), ("added", "new cases"),
                       ("removed", "removed cases"), ("changed_checks", "checks changed (not compared)"),
                       ("unscored", "unavailable in one run")):
        if cmp[key]:
            out.append(f"  {label}: {', '.join(cmp[key])}")
    return "\n".join(out)


def _signed(x, money=False) -> str:
    if x is None:
        return "n/a"
    return (f"{'+' if x >= 0 else '-'}${abs(x):.4f}") if money else f"{x * 100:+.1f} pts"


def compare_markdown(cmp: dict, run: dict, ok: bool | None = None, reasons: list | None = None) -> str:
    m = run["metrics"]
    lines = [f"### goldloop {'✅ pass' if ok else '❌ fail' if ok is not None else ''}",
             f"**{cmp['verdict']}**", "",
             "| | mean | pass^k | flaky | p95 | cost |", "|---|---|---|---|---|---|",
             f"| this run | {pct(m['mean'])} | {pct(m['pass_all'])} | {len(m['flaky'])} | "
             f"{m['latency_ms']['p95']:.0f}ms | {money(m['cost_usd'])} |",
             f"| Δ vs baseline | {_signed(cmp['delta']['mean'])} | {_signed(cmp['delta']['pass_all'])} | | "
             f"{cmp['delta']['latency_p95_ms']:+.0f}ms | {_signed(cmp['delta']['cost_usd'], money=True)} |", ""]
    for key, label in (("regressions", "Regressed"), ("fixes", "Fixed"), ("newly_flaky", "Newly flaky")):
        if cmp[key]:
            lines.append(f"- **{label}:** " + ", ".join(f"`{c}`" for c in cmp[key]))
    for r in reasons or []:
        lines.append(f"- ⛔ {r}")
    return "\n".join(lines)


def html_report(runs: list[dict], out: Path) -> Path:
    """One page: the latest run in detail, and the trend across every run."""
    latest = runs[-1]
    hist = [{"id": r["id"], "label": r.get("label", ""), "sha": (r.get("git") or {}).get("sha"),
             "mean": r["metrics"]["mean"], "pass_all": r["metrics"]["pass_all"],
             "flaky": len(r["metrics"]["flaky"]), "cost": r["metrics"]["cost_usd"],
             "n": r["suite"]["n"], "suite": r["suite"]["fingerprint"]} for r in runs]
    rows = []
    for cid, c in latest["cases"].items():
        s = c["summary"]
        cls = "unk" if s["status"] == "unavailable" else "ok" if s["passed_all"] else "flaky" if s["flaky"] else "bad"
        last = next((r for r in reversed(c["runs"]) if r["status"] != "unavailable"), {})
        fails = [f"{ck['type']}: {ck['detail']}" for ck in last.get("checks", []) if ck["score"] < 1]
        if last.get("error"):
            fails.insert(0, "error: " + last["error"])
        scores = " ".join("·" if r["status"] == "unavailable" else f"{r['score']:.0%}" for r in c["runs"])
        rows.append(f"<tr class='{cls}'><td><code>{html.escape(cid)}</code></td>"
                    f"<td>{html.escape(', '.join(c.get('tags') or []))}</td>"
                    f"<td>{html.escape(c.get('source') or '')}</td><td class='num'>{scores}</td>"
                    f"<td>{'<br>'.join(html.escape(f) for f in fails[:4])}</td></tr>")
    m = latest["metrics"]
    lo, hi = m["pass_all_ci95"]
    tiles = [("mean", pct(m["mean"])), ("pass^k", f"{pct(m['pass_all'])}<small>95% CI {lo:.0%}–{hi:.0%}</small>"),
             ("flaky", str(len(m["flaky"]))), ("p95 latency", f"{m['latency_ms']['p95']:.0f}ms"),
             ("cost", money(m["cost_usd"])), ("complete", "yes" if m["complete"] else "NO")]
    page = _TEMPLATE.replace("{{TITLE}}", "goldloop report").replace(
        "{{RUN}}", html.escape(f"{latest['id']} · {latest['suite']['n']} cases × {latest['config']['repeat']} · "
                               f"git {(latest.get('git') or {}).get('sha')}")).replace(
        "{{TILES}}", "".join(f"<div class='tile'><b>{k}</b><span>{v}</span></div>" for k, v in tiles)).replace(
        "{{ROWS}}", "\n".join(rows)).replace("{{HIST}}", json.dumps(hist))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return out


_TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{{TITLE}}</title><style>
:root{--bg:#fbfaf7;--fg:#1d1d1b;--mute:#6b6a65;--line:#e4e2dc;--ok:#1f7a4d;--bad:#b3261e;--warn:#a8650b;--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--fg:#ecebe6;--mute:#9a9891;--line:#2c2b28;--ok:#5cc28d;--bad:#ef7b72;--warn:#e3a54a;--card:#1c1c1a}}
body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px}
main{max-width:1100px;margin:auto}h1{font-size:20px;margin:0}.sub{color:var(--mute);margin:4px 0 20px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:24px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.tile b{display:block;font-size:12px;color:var(--mute);font-weight:500}.tile span{font-size:20px;font-variant-numeric:tabular-nums}
.tile small{display:block;font-size:11px;color:var(--mute)}
h2{font-size:15px;margin:24px 0 8px}svg{width:100%;height:160px;background:var(--card);border:1px solid var(--line);border-radius:8px}
.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;background:var(--card)}
td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top;font-size:13px}
th{color:var(--mute);font-weight:500}.num{font-variant-numeric:tabular-nums;white-space:nowrap}
tr.ok td:first-child{border-left:3px solid var(--ok)}tr.bad td:first-child{border-left:3px solid var(--bad)}
tr.flaky td:first-child{border-left:3px solid var(--warn)}tr.unk td:first-child{border-left:3px solid var(--mute)}
.legend{color:var(--mute);font-size:12px}.legend i{display:inline-block;width:10px;height:3px;margin:0 4px 2px 10px}
</style></head><body><main><h1>goldloop report</h1><div class="sub">{{RUN}}</div>
<div class="tiles">{{TILES}}</div>
<h2>History</h2><div class="legend"><i style="background:var(--ok)"></i>mean<i style="background:var(--warn)"></i>pass^k
 · dashed line = the suite changed (new or edited cases), so points either side aren't the same exam</div>
<svg id="h" viewBox="0 0 1000 160" preserveAspectRatio="none"></svg>
<h2>Cases (latest run)</h2><div class="wrap"><table><tr><th>case</th><th>tags</th><th>source</th><th>scores per repeat</th><th>what failed</th></tr>
{{ROWS}}</table></div></main><script>
const H={{HIST}},s=document.getElementById('h'),W=1000,T=10,B=150;
const x=i=>H.length<2?W/2:20+i*(W-40)/(H.length-1),y=v=>B-(v??0)*(B-T);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n);
let g='';H.forEach((r,i)=>{if(i&&r.suite!==H[i-1].suite)g+=`<line x1="${x(i)-6}" x2="${x(i)-6}" y1="${T}" y2="${B}" stroke="${css('--mute')}" stroke-dasharray="3 3"/>`});
for(const[k,c]of[['mean','--ok'],['pass_all','--warn']]){g+=`<polyline fill="none" stroke-width="2" stroke="${css(c)}" points="${H.map((r,i)=>x(i)+','+y(r[k])).join(' ')}"/>`;
H.forEach((r,i)=>g+=`<circle cx="${x(i)}" cy="${y(r[k])}" r="3.5" fill="${css(c)}"><title>${r.id} ${r.label||''} git ${r.sha||'?'}\\n${k} ${((r[k]??0)*100).toFixed(1)}%  ·  ${r.n} cases</title></circle>`)}
s.innerHTML=g;</script></body></html>"""
