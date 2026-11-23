"""Stand-alone, offline view of a calibrated fluorescence/flux scan.

Usage: write_flux_plots(outdir, payload). The caller supplies fractions (not
percent), flux in atoms/s, and normalized signals evaluated on phase_rad.
Plotting never resamples, clips, or fits the data.
"""

import json
import math
from pathlib import Path
import numpy as np


def _plain(v):
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_plain(x) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    return v


def _prepare(payload):
    p = _plain(payload)
    phases = p["phase_rad"]
    if len(phases) < 3 or any(v is None for v in phases) or len(set(phases)) < 3:
        raise ValueError("Three or more finite, distinct phase samples are needed")
    if not p["recipes"]:
        raise ValueError("No recipes supplied for the flux scan")
    for r in p["recipes"]:
        if len(r["ideal_signal"]) != len(phases) or not r["rows"]:
            raise ValueError("Missing phase or flux scan for %s" % r["name"])
        for row in r["rows"]:
            if row["flux_per_s"] is None or row["flux_per_s"] <= 0:
                raise ValueError("Input flux must be positive and finite")
            for k in ("example_signal", "example_fit"):
                if row.get(k) is None:
                    row[k] = [None] * len(phases)
                if len(row[k]) != len(phases):
                    raise ValueError("Phase grid does not match %s" % k)
        r["rows"].sort(key=lambda row: row["flux_per_s"])
        if len(set(row["flux_per_s"] for row in r["rows"])) != len(r["rows"]):
            raise ValueError("Duplicate flux setting")
    p.setdefault("metadata", {})
    return p


def _static(path, p):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter, LogLocator, NullFormatter
    import warnings

    phase = np.asarray(p["phase_rad"])
    n = len(p["recipes"])
    colors = dict(ideal="#00776c", obs="#aa5524", fit="#ba5828", band="#f4d1af")
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    ):
        fig, axes = plt.subplots(n, 2, figsize=(10.6, 3.3 * n + 0.35), squeeze=False)
        for j, r in enumerate(p["recipes"]):
            rr = r["rows"]
            row = min(rr, key=lambda v: abs(math.log10(v["flux_per_s"]) - 5))
            a, b = axes[j]
            label = r.get("name_en")
            if not label:
                name = r["name"].replace("∶", ":").replace("：", ":")
                label = (
                    "3:1 pump split"
                    if "3:1" in name
                    else "1:1 pump split" if "1:1" in name else "Preparation %d" % (j + 1)
                )
            a.plot(phase, r["ideal_signal"], color=colors["ideal"], lw=2, label="Expected")
            a.plot(
                phase,
                np.array(row["example_fit"], float),
                color=colors["fit"],
                ls="--",
                label="Single-scan fit",
            )
            a.scatter(
                phase,
                np.array(row["example_signal"], float),
                s=14,
                facecolors="white",
                edgecolors=colors["obs"],
                linewidths=0.9,
                label="Single scan",
            )
            a.set(
                xlabel="Third-pulse phase (rad)",
                ylabel="Normalized fluorescence (P2)",
                title="%s  |  %.2g incident atoms/s" % (label, row["flux_per_s"]),
            )
            a.yaxis.set_major_locator(MaxNLocator(5))
            a.legend(fontsize=8, loc="best", frameon=False)
            x = np.array([v["flux_per_s"] for v in rr])

            def col(k):
                return np.array([np.nan if v.get(k) is None else v[k] for v in rr])

            lo, hi = col("lo95"), col("hi95")
            b.fill_between(
                x, lo, hi, color=colors["band"], alpha=0.82, label="Central 95% of valid fits"
            )
            b.plot(x, col("median_contrast"), "o-", color=colors["fit"], ms=4, label="Median fit")
            b.plot(
                x,
                col("ideal_contrast"),
                ls="--",
                color=colors["ideal"],
                lw=1.8,
                label="Expected contrast",
            )
            b.set(
                xscale="log",
                xlabel="Incident atom flux (atoms/s)",
                ylabel="Fringe contrast",
                title="%s  |  flux scan" % label,
            )
            if len(x) > 1:
                b.set_xlim(x[0] * 0.85, x[-1] * 1.18)
            b.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            b.xaxis.set_major_locator(LogLocator(base=10, numticks=8))
            b.xaxis.set_minor_formatter(NullFormatter())
            b.legend(fontsize=8, loc="best", frameon=False)
            for ax in (a, b):
                ax.grid(alpha=0.18)
                ax.set_axisbelow(True)
        fig.tight_layout(pad=1.6, h_pad=2, w_pad=2.5)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"Glyph .* missing from .*font")
            fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)


def write_flux_plots(outdir, payload):
    """Return paths to the self-contained HTML explorer and static PNG."""
    p = _prepare(payload)
    out = Path(outdir)
    out.mkdir(exist_ok=True, parents=True)
    html = out / "flux_visibility.html"
    png = out / "flux_visibility.png"
    safe = json.dumps(p, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace(
        "<", "\\u003c"
    )
    html.write_text(_HTML.replace("__PAYLOAD__", safe), encoding="utf-8")
    _static(png, p)
    return html, png


_HTML = r"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>原子通量与三脉冲干涉条纹</title><style>
:root{--ink:#173535;--tone:#486462;--border:#d4dfdb;--teal:#00796e;--tan:#af5228;--band:#f4ceac}
*{box-sizing:border-box}body{margin:0;background:#f7f9f6;color:var(--ink);font:15px/1.55 system-ui,"Microsoft YaHei",sans-serif}
main{max-width:1240px;margin:auto;padding:32px 24px 48px}h1,h2,h3,p{margin:0}small,.muted{color:var(--tone)}
.kicker{font-size:12px;font-weight:700;letter-spacing:.1em;color:var(--teal);margin-bottom:4px}h1{font-size:clamp(23px,4vw,34px);letter-spacing:-.035em}h2{font-size:19px}h3{font-size:16px}
.intro{max-width:950px;color:var(--tone);margin-top:7px}.controls,.card,.extra{background:white;border:1px solid var(--border);border-radius:10px}
.controls{display:grid;grid-template-columns:minmax(180px,1fr) minmax(280px,2fr);gap:22px 40px;margin:20px 0 12px;padding:15px 20px}
label{font-weight:650}select,input,button{font:inherit}select,button{border:1px solid #91a6a2;background:white;color:var(--ink);border-radius:5px;padding:6px 9px}button{cursor:pointer}
select:focus-visible,input:focus-visible,button:focus-visible,summary:focus-visible{outline:3px solid #1b79a6;outline-offset:2px}#case{display:block;margin-top:4px;max-width:100%}
.range{display:flex;align-items:center;gap:12px}.range input{flex:1;min-width:85px;accent-color:var(--teal)}.range select{max-width:140px}
.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:var(--border);border:1px solid var(--border);border-radius:8px;overflow:hidden}
.metrics div{background:#eef5f1;padding:12px}.metrics small,.metrics strong{display:block}.metrics strong{font-size:clamp(17px,2.2vw,25px);font-variant-numeric:tabular-nums}
.charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:14px 0}.card,.extra{padding:16px}
.legend{font-size:12px;margin:5px 0;color:var(--tone)}.leg{display:inline-block;width:18px;height:3px;background:var(--teal);vertical-align:middle;margin:0 4px 0 7px}.leg:first-child{margin-left:0}
.leg.fit{background:transparent;border-top:2px dashed var(--tan)}.leg.dot{height:7px;width:7px;border:1.4px solid var(--tan);background:white;border-radius:50%}
.leg.median{background:var(--tan)}.leg.band{height:9px;background:var(--band)}
.chart{width:100%;height:auto;display:block}.chart text{fill:#425953;font:12px system-ui,"Microsoft YaHei",sans-serif}
.chart text.title{font-size:13px}.axis{stroke:#889b95;stroke-width:1}.gridline{stroke:#e2e8e4;stroke-width:1}
.note{margin:5px 0 0;color:var(--tone);font-size:13px}.extra{margin:14px 0}.extra p{margin:.35em 0}.tablewrap{overflow-x:auto}
table{min-width:430px;width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:6px 10px;text-align:right;border-bottom:1px solid #e6ebe7}th{background:#f4f7f3}tr.sel td{background:#e6f3ec}summary{cursor:pointer;font-weight:650}
.buttons{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin:10px 0}#copyarea{width:100%;height:90px}
footer{color:var(--tone);font-size:12px;margin-top:14px}.apcontrols{display:flex;align-items:center;gap:9px;flex-wrap:wrap}
@media(max-width:780px){main{padding:18px 12px 28px}.charts,.controls{grid-template-columns:1fr}.controls{gap:9px}.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style>
<main>
<header><p class="kicker">⁸⁷Rb · 三脉冲 Raman 干涉仪</p><h1>原子通量与干涉条纹对比度</h1>
<p class="intro">扫描第三脉冲相位，读取归一化荧光信号。主图只改变每秒入射的原子数，保持速度分布、制备光、Raman 光和探测孔径不变。可调节通量观察单次测量的散点与拟合区间。</p></header>
<section class="controls"><div><label for="case">态制备方案</label><select id="case"></select></div>
<div><label for="level">入射原子通量 Φ（原子/s）</label><div class="range"><input type="range" id="level" min="0" step="1">
<select id="levelSelect" aria-label="直接选择入射通量"></select></div></div></section>
<section class="metrics" aria-live="polite"><div><small>选中入射通量（原子/s）</small><strong id="inputFlux"></strong></div>
<div><small>理想荧光对比度</small><strong id="ideal"></strong></div><div><small>单次示例拟合对比度</small><strong id="exC"></strong></div>
<div><small>重复扫相拟合中位数</small><strong id="median"></strong></div></section>
<div class="charts"><section class="card"><h2>一次三脉冲相位扫描</h2><p class="legend"><i class="leg"></i>理想均值 <i class="leg fit"></i>单次拟合 <i class="leg dot"></i>单次观测</p><svg id="fringe" viewBox="0 0 530 290" class="chart" role="img" aria-label="第三脉冲相位与双通道归一化荧光"></svg>
<p class="note" id="sampleNote"></p></section>
<section class="card"><h2>通量扫描：对比度及其波动</h2><p class="legend"><i class="leg"></i>理想 <i class="leg median"></i>拟合中位数 <i class="leg band"></i>有效拟合的中间 95%</p>
<svg id="visibility" viewBox="0 0 530 290" class="chart" role="img" aria-label="以对数坐标显示入射通量与对比度"></svg><p class="note" id="scatterNote"></p></section></div>
<section class="extra" id="apPanel"><h2>另一种改变通量的方式：缩小探测孔径</h2><p class="muted">固定源通量，改变探测端接收的原子。若被挡住的原子和留下的原子对条纹的贡献不同，理想对比度也会改变。这个比较不包含密度引起的碰撞退相干。</p>
<div class="apcontrols"><label for="ap">孔径半径</label><select id="ap"></select><span id="apText"></span></div>
<div class="tablewrap"><table><thead><tr><th>半径 (mm)</th><th>接收通量 (原子/s)</th><th>有效目标态通量 (原子/s)</th><th>理想对比度</th><th>接收源编号数</th></tr></thead><tbody id="apBody"></tbody></table></div>
<p class="note">小孔径可能只接收到少量模拟源原子。可用最后一列判断有限样本对结果的影响；未接收到原子时，对比度无定义。</p></section>
<section class="extra"><details><summary>查看通量扫描数值与模拟设置</summary><p id="setup"></p>
<div class="buttons"><button id="copy" type="button">复制当前方案数据</button><button id="download" type="button">下载 CSV</button><span aria-live="polite" id="status"></span></div>
<textarea readonly id="copyarea" hidden aria-label="待复制数据"></textarea>
<div class="tablewrap"><table><thead id="dataHeading"></thead><tbody id="dataBody"></tbody></table></div></details></section>
<footer>阴影是独立重复一次完整扫相时的拟合分布，区别于拟合均值的置信区间。有限计数时，扣背景的归一化读数及拟合对比度可能超出物理范围；图中保留这些数值，无法拟合者不记作零。</footer>
</main>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
'use strict';
const D=JSON.parse(document.getElementById('payload').textContent), $=id=>document.getElementById(id);
const finite=x=>typeof x==='number'&&Number.isFinite(x), dash='—', sci=x=>finite(x)?x.toExponential(2).replace('e+','e'):dash,
 pct=x=>finite(x)?(100*x).toFixed(2)+'%':dash, num=x=>finite(x)?String(Number(x.toPrecision(4))):dash;
const recipes=D.recipes, phase=D.phase_rad, nums=(arr)=>arr.filter(finite), vmin=a=>Math.min(...a),vmax=a=>Math.max(...a);
function option(sel,label,value){const o=document.createElement('option');o.value=value;o.textContent=label;sel.append(o)}
function el(svg,name,attrs={},label=null){const node=document.createElementNS('http://www.w3.org/2000/svg',name);for(const k in attrs)node.setAttribute(k,attrs[k]);if(label!==null)node.textContent=label;svg.append(node);return node}
const C={x0:68,x1:510,y0:239,y1:16};function scale(domain,range){return x=>range[0]+(x-domain[0])*(range[1]-range[0])/(domain[1]-domain[0])}
function span(ys,includeZero){const vals=nums(ys);if(!vals.length)return[0,1];let lo=vmin(vals),hi=vmax(vals);if(includeZero)lo=Math.min(lo,0);
 let pad=(hi-lo)*.10||.05;return [includeZero&&lo===0?0:lo-pad,hi+pad]}
function ticks(lo,hi,n){const raw=(hi-lo)/n,p=Math.pow(10,Math.floor(Math.log10(raw)));const step=p*([1,2,2.5,5,10].find(x=>x>=raw/p)||10);
 let xs=[];for(let i=Math.ceil(lo/step-1e-10);i<=Math.floor(hi/step+1e-10)&&xs.length<12;i++)xs.push(i*step);return xs}
const tick=x=>Math.abs(x)>=1e4||x!==0&&Math.abs(x)<.001?x.toExponential(0):String(+x.toPrecision(4));
function axes(svg,x,y,xt,yt,xlab,ylab){svg.replaceChildren();for(const [v,lab] of yt){el(svg,'line',{x1:C.x0,x2:C.x1,y1:y(v),y2:y(v),class:'gridline'});
 el(svg,'text',{x:C.x0-8,y:y(v)+4,'text-anchor':'end'},lab)}for(const [v,lab] of xt){el(svg,'line',{x1:x(v),x2:x(v),y1:C.y0,y2:C.y0+4,class:'axis'});el(svg,'text',{x:x(v),y:C.y0+20,'text-anchor':'middle'},lab)}
 el(svg,'path',{d:'M'+C.x0+' '+C.y1+' V'+C.y0+' H'+C.x1,fill:'none',class:'axis'});
 el(svg,'text',{x:(C.x0+C.x1)/2,y:283,'text-anchor':'middle',class:'title'},xlab);el(svg,'text',{transform:'translate(16 128) rotate(-90)','text-anchor':'middle',class:'title'},ylab)}
function line(svg,xx,yy,x,y,color,dashed=false){let path='',active=false;xx.forEach((v,i)=>{if(finite(v)&&finite(yy[i])){path+=(active?'L':'M')+x(v)+' '+y(yy[i]);active=true}else active=false});
 el(svg,'path',{d:path,stroke:color,'stroke-width':2.3,fill:'none','stroke-dasharray':dashed?'6 4':'none'})}
function dots(svg,xx,yy,x,y,color,r=3){xx.forEach((v,i)=>{if(finite(v)&&finite(yy[i])){const a=el(svg,'circle',{cx:x(v),cy:y(yy[i]),r:r,stroke:color,'stroke-width':1.4,fill:'white'});el(a,'title',{},'x = '+num(v)+'；y = '+num(yy[i]))}})}
function plotFringe(r,row){const svg=$('fringe'),xd=[vmin(phase),vmax(phase)],yd=span([...r.ideal_signal,...row.example_fit,...row.example_signal],false);
 const x=scale(xd,[C.x0,C.x1]),y=scale(yd,[C.y0,C.y1]);const k0=Math.ceil(xd[0]/(Math.PI/2)-1e-9),k1=Math.floor(xd[1]/(Math.PI/2)+1e-9);
 const pix=n=>n===0?'0':n%2===0?((n===2?'':n===-2?'−':n/2)+'π'):((n===1?'':n===-1?'−':n)+'π/2');
 const xt=k1-k0<=6&&k1>k0?Array.from({length:k1-k0+1},(_,i)=>[(i+k0)*Math.PI/2,pix(i+k0)]):ticks(...xd,5).map(v=>[v,tick(v)]);
 axes(svg,x,y,xt,ticks(...yd,5).map(v=>[v,tick(v)]),'第三脉冲附加相位（rad）','归一化荧光 P₂');
 line(svg,phase,r.ideal_signal,x,y,'#00796e');line(svg,phase,row.example_fit,x,y,'#af5228',true);dots(svg,phase,row.example_signal,x,y,'#af5228')}
function plotVisibility(r,row){const svg=$('visibility'),rs=r.rows,xx=rs.map(d=>d.flux_per_s),gg=k=>rs.map(d=>finite(d[k])?d[k]*100:null),
 med=gg('median_contrast'),lo=gg('lo95'),hi=gg('hi95'),ideal=gg('ideal_contrast'),yd=span([...lo,...hi,...med,...ideal],true);
 const xlo=Math.log10(xx[0]),xhi=Math.log10(xx[xx.length-1]);const xlog=scale([xlo-.06,xhi+.06],[C.x0,C.x1]),x=v=>xlog(Math.log10(v)),y=scale(yd,[C.y0,C.y1]);
 let xt=[];const stride=Math.max(1,Math.ceil((xhi-xlo)/6));for(let a=Math.ceil(xlo-1e-9);a<=Math.floor(xhi+1e-9);a+=stride)xt.push([10**a,'1e'+a]);
 if(xt.length<2)xt=xx.filter((_,i)=>i===0||i===xx.length-1).map(v=>[v,sci(v)]);axes(svg,x,y,xt,ticks(...yd,5).map(v=>[v,tick(v)]),'入射原子通量（原子/s，对数轴）','条纹对比度（%）');
 let block=[];function fill(){if(block.length){let poly=block.map(i=>[x(xx[i]),y(hi[i])]).concat(block.slice().reverse().map(i=>[x(xx[i]),y(lo[i])]));
 el(svg,'polygon',{points:poly.map(t=>t.join(',')).join(' '),fill:'#f4ceac','fill-opacity':.8});block=[]}}
 xx.forEach((v,i)=>{if(finite(hi[i])&&finite(lo[i]))block.push(i);else fill()});fill();
 el(svg,'line',{x1:x(row.flux_per_s),x2:x(row.flux_per_s),y1:C.y1,y2:C.y0,stroke:'#6f8781','stroke-width':1,'stroke-dasharray':'2 4'});
 line(svg,xx,med,x,y,'#af5228');dots(svg,xx,med,x,y,'#af5228',2.8);line(svg,xx,ideal,x,y,'#00796e',true);
 if(finite(row.median_contrast))el(svg,'circle',{cx:x(row.flux_per_s),cy:y(100*row.median_contrast),r:5,fill:'none',stroke:'#713209','stroke-width':1.8})}
const head=['入射通量(/s)','探测接收通量(/s)','有效目标态通量(/s)','理想对比度','拟合中位数','2.5%分位','97.5%分位','拟合成功率','单次对比度'];
const cols=['flux_per_s','detected_flux_per_s','effective_target_flux_per_s','ideal_contrast','median_contrast','lo95','hi95','fit_success_rate','example_contrast'];
function tr(parent,cells,tag='td',chosen=false){let row=document.createElement('tr');if(chosen)row.className='sel';cells.forEach(c=>{let cell=document.createElement(tag);cell.textContent=c;row.append(cell)});parent.append(row)}
function paintAperture(r){const a=r.apertures||[];$('apPanel').hidden=!a.length;if(!a.length)return;
 let i=Number($('ap').value)||0;const v=a[i];$('apText').textContent='入射 '+sci(v.incident_flux_per_s)+' 原子/s → 接收 '+sci(v.detected_flux_per_s)+' 原子/s，理想对比度 '+pct(v.ideal_contrast)+'。';
 $('apBody').replaceChildren();a.forEach((u,k)=>tr($('apBody'),[num(u.radius_mm),sci(u.detected_flux_per_s),sci(u.effective_target_flux_per_s),pct(u.ideal_contrast),
 Array.isArray(u.accepted_source_ids)?u.accepted_source_ids.length:num(u.accepted_source_ids)],'td',i===k))}
function render(){let r=recipes[Number($('case').value)],j=Number($('level').value),row=r.rows[j];$('levelSelect').value=j;$('level').setAttribute('aria-valuetext',sci(row.flux_per_s)+' 原子每秒');
 $('inputFlux').textContent=sci(row.flux_per_s);$('ideal').textContent=pct(row.ideal_contrast);$('exC').textContent=pct(row.example_contrast);$('median').textContent=pct(row.median_contrast);
 $('sampleNote').textContent='探测接收 '+sci(row.detected_flux_per_s)+' 原子/s；有效目标态 '+sci(row.effective_target_flux_per_s)+' 原子/s。每个相位点的期望接收原子数：'+num(row.expected_atoms_per_gate)+'。';
 $('scatterNote').textContent='拟合范围 '+pct(row.lo95)+' 至 '+pct(row.hi95)+'，拟合成功率 '+pct(row.fit_success_rate)+(finite(row.unphysical_fit_fraction)?'；有效拟合中超出物理范围的比例 '+pct(row.unphysical_fit_fraction):'')+'。';
 plotFringe(r,row);plotVisibility(r,row);paintAperture(r);$('dataBody').replaceChildren();r.rows.forEach((u,k)=>tr($('dataBody'),cols.map((c,i)=>i<3?sci(u[c]):pct(u[c])),'td',k===j));
 $('copyarea').hidden=true;$('status').textContent=''}
function loadRecipe(prevFlux){const r=recipes[Number($('case').value)],rr=r.rows;let i=0;rr.forEach((v,j)=>{if(Math.abs(Math.log(v.flux_per_s/prevFlux))<Math.abs(Math.log(rr[i].flux_per_s/prevFlux)))i=j});
 $('level').max=rr.length-1;$('level').value=i;$('levelSelect').replaceChildren();rr.forEach((v,j)=>option($('levelSelect'),sci(v.flux_per_s),j));$('ap').replaceChildren();(r.apertures||[]).forEach((v,j)=>option($('ap'),num(v.radius_mm)+' mm',j));
 if((r.apertures||[]).length)$('ap').value=r.apertures.length-1;render()}
function textRows(sep){const r=recipes[Number($('case').value)];const cell=v=>v===null||v===undefined?'':String(v);
 return [cols.join(sep),...r.rows.map(u=>cols.map(k=>cell(u[k])).join(sep))].join('\r\n')}
$('copy').onclick=async()=>{const t=textRows('\t');try{await navigator.clipboard.writeText(t);$('status').textContent='已复制；对比度以 0–1 比例表示。'}catch(e){$('copyarea').hidden=false;$('copyarea').value=t;$('copyarea').select();let ok=false;try{ok=document.execCommand('copy')}catch(_){}
 $('status').textContent=ok?'已复制。':'已选中数值，可按 Ctrl+C 复制。';if(ok)$('copyarea').hidden=true}};
$('download').onclick=()=>{const url=URL.createObjectURL(new Blob(['\ufeff'+textRows(',')],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download='flux_visibility_recipe_'+(1+Number($('case').value))+'.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
recipes.forEach((r,i)=>option($('case'),r.name,i));let primary=recipes.findIndex(r=>/3\s*[:∶：]\s*1/.test(r.name));$('case').value=Math.max(primary,0);
const m=D.metadata;$('setup').textContent='每相位积分 '+num(m.gate_ms)+' ms；每条扫描 '+num(m.phase_points||phase.length)+' 个相位点；每档 '+num(m.repeats)+' 次完整扫描。MOT 入射样本 '+num(m.source_samples)+' 个，每个源编号 '+num(m.recoil_replicas)+' 条反冲轨迹。每通道每门背景 '+num(m.background_counts)+' 计数，电子噪声标准差 '+num(m.electronic_noise_std_counts)+' 计数。'+(m.source_note?'制备数据：'+m.source_note+'。':'');
tr($('dataHeading'),head,'th');$('level').oninput=render;$('levelSelect').onchange=()=>{$('level').value=$('levelSelect').value;render()};$('case').onchange=()=>loadRecipe(Number($('levelSelect').selectedOptions[0].textContent));
$('ap').onchange=()=>paintAperture(recipes[Number($('case').value)]);loadRecipe(1e5);
</script></html>"""
