import pandas as pd, numpy as np, matplotlib.pyplot as plt
import matplotlib as mpl, statsmodels.api as sm
mpl.rcParams.update({"font.size":11,"axes.splines.top":False,"axes.spines.right":False} if False else {"font.size":11})
for s in ["top","right"]: mpl.rcParams[f"axes.spines.{s}"]=False
df=pd.read_csv("data/dataset_with_index.csv")
KZ="Kazakhstan"; hl=lambda c:"#c0392b" if c==KZ else "#4c72b0"
DPI=300; OUT="figures"

# F1: DAR ranking
d=df.sort_values("DAR")
plt.figure(figsize=(7,4.6))
plt.barh(d["country"],d["DAR"],color=[hl(c) for c in d["country"]])
plt.xlabel("Data-for-AI Readiness (DAR) index, 0–100"); plt.title("Figure 1. DAR index across economies (2024)")
for y,(c,v) in enumerate(zip(d["country"],d["DAR"])): plt.text(v+0.5,y,f"{v:.0f}",va="center",fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_dar_ranking.png",dpi=DPI); plt.close()

# F2: conversion quadrant — DAR (x) vs research intensity R&D%GDP (y)
plt.figure(figsize=(6.6,5.4))
x=df["DAR"]; y=df["rnd_pct_gdp"]
plt.scatter(x,y,s=60,color=[hl(c) for c in df["country"]],zorder=3)
plt.axvline(x.median(),ls="--",c="grey",lw=.8); plt.axhline(y.median(),ls="--",c="grey",lw=.8)
for c,xi,yi in zip(df["country"],x,y):
    plt.annotate(c,(xi,yi),fontsize=8,xytext=(4,3),textcoords="offset points",
                 color="#c0392b" if c==KZ else "black",fontweight="bold" if c==KZ else "normal")
plt.xlabel("Data readiness (DAR)"); plt.ylabel("Research intensity (R&D, % of GDP)")
plt.title("Figure 2. Data readiness vs research intensity\n(KZ: data-ready, research-thin quadrant)")
plt.text(0.98,0.02,"high-data / low-research",transform=plt.gca().transAxes,ha="right",fontsize=8,color="grey")
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_conversion_quadrant.png",dpi=DPI); plt.close()

# F3: under-conversion — actual vs predicted AI output, KZ residual
d=df.copy(); d["l_ai"]=np.log(d["ai_works"]); d["l_pop"]=np.log(d["population"]); d["l_gdp"]=np.log(d["gdppc_ppp"])
X=sm.add_constant(d[["l_pop","l_gdp","DAR"]]); m=sm.OLS(d["l_ai"],X).fit(); d["pred"]=m.fittedvalues
plt.figure(figsize=(6.4,5.4))
lo,hi=d["pred"].min()-.5,d["pred"].max()+.5
plt.plot([lo,hi],[lo,hi],c="grey",ls="--",lw=.9,label="perfect conversion")
plt.scatter(d["pred"],d["l_ai"],s=60,color=[hl(c) for c in d["country"]],zorder=3)
for c,xi,yi in zip(d["country"],d["pred"],d["l_ai"]):
    plt.annotate(c,(xi,yi),fontsize=8,xytext=(4,-2),textcoords="offset points",
                 color="#c0392b" if c==KZ else "black",fontweight="bold" if c==KZ else "normal")
kz=d[d.country==KZ].iloc[0]
plt.annotate("under-conversion\n(resid = %.2f)"%m.resid[d.index[d.country==KZ][0]],
             (kz["pred"],kz["l_ai"]),xytext=(kz["pred"]+0.3,kz["l_ai"]-1.4),fontsize=8,color="#c0392b",
             arrowprops=dict(arrowstyle="->",color="#c0392b"))
plt.xlabel("Predicted log(AI works)  [size+income+DAR]"); plt.ylabel("Actual log(AI works)")
plt.title("Figure 3. Under-conversion of inputs into AI output"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_underconversion.png",dpi=DPI); plt.close()

# F4: KZ input profile vs comparator means (percentile ranks)
metrics={"DAR (data)":"DAR","Openness P2":"spi_p2_open","Infra P5":"spi_p5_infra",
         "R&D %GDP":"rnd_pct_gdp","Researchers/mil":"researchers_permil","AI works/mil":"ai_per_mil"}
kzrank={k:(df[v].rank(pct=True)[df.country==KZ].iloc[0]*100) for k,v in metrics.items()}
plt.figure(figsize=(7,4.4))
ks=list(kzrank); vs=[kzrank[k] for k in ks]
plt.bar(ks,vs,color=["#4c72b0" if v>=50 else "#c0392b" for v in vs])
plt.axhline(50,ls="--",c="grey",lw=.8); plt.ylabel("Kazakhstan percentile within sample")
plt.title("Figure 4. Kazakhstan: data-ready, conversion-thin"); plt.xticks(rotation=25,ha="right")
for i,v in enumerate(vs): plt.text(i,v+2,f"{v:.0f}",ha="center",fontsize=9)
plt.ylim(0,105); plt.tight_layout(); plt.savefig(f"{OUT}/fig4_kz_profile.png",dpi=DPI); plt.close()

print("figures saved:")
import os
for f in sorted(os.listdir(OUT)): print("  ",f, os.path.getsize(f"{OUT}/{f}"),"bytes")
