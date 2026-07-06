import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm

df = pd.read_csv("data/dataset.csv")

# --- DAR index (0-100 pillars already) ---
df["DAR"]      = df[["spi_p2_open","spi_p4_source","spi_p5_infra"]].mean(axis=1)   # primary: openness+capacity+infra
df["DAR_all5"] = df[["spi_p1_use","spi_p2_open","spi_p3_prod","spi_p4_source","spi_p5_infra"]].mean(axis=1)
# PCA robustness on the 4 supply pillars
X = StandardScaler().fit_transform(df[["spi_p2_open","spi_p3_prod","spi_p4_source","spi_p5_infra"]])
pc = PCA(n_components=1).fit_transform(X)[:,0]
# orient PCA so higher=better
if np.corrcoef(pc, df["DAR"])[0,1] < 0: pc = -pc
df["DAR_pca"] = (pc - pc.min())/(pc.max()-pc.min())*100

# --- AI output intensities ---
df["ai_per_mil"]   = df["ai_works"]/df["population"]*1e6
df["rsch_total"]   = df["researchers_permil"]*df["population"]/1e6
df["ai_per_1k_rsch"]= df["ai_works"]/(df["rsch_total"]/1000)
df["ai_share"]     = df["ai_works"]/df["total_works"]*100

pd.set_option("display.width",160); pd.set_option("display.max_columns",30)
print("===== DAR ranking =====")
print(df[["country","DAR","DAR_pca","spi_p2_open","spi_p4_source","spi_p5_infra","rnd_pct_gdp","ai_per_mil","ai_share"]]
      .sort_values("DAR",ascending=False).round(1).to_string(index=False))

def corr(a,b):
    m=df[[a,b]].dropna()
    from scipy.stats import pearsonr,spearmanr
    return pearsonr(m[a],m[b])[0], spearmanr(m[a],m[b])[0]
print("\n===== Correlations (Pearson, Spearman) =====")
for y in ["ai_per_mil","ai_share","ai_per_1k_rsch"]:
    df["_ly"]=np.log(df[y].replace(0,np.nan))
    p,s=corr("DAR","_ly"); print(f"DAR vs log({y}):   r={p:.2f}  rho={s:.2f}")
p,s=corr("DAR","rnd_pct_gdp"); print(f"DAR vs R&D%GDP:     r={p:.2f}  rho={s:.2f}")
p,s=corr("rnd_pct_gdp","_ly"); print(f"R&D vs log(ai/1krsch): computed above set")

# --- demonstrative OLS (n=11, analytical not inferential) ---
d=df.copy()
d["l_ai"]=np.log(d["ai_works"]); d["l_pop"]=np.log(d["population"]); d["l_gdp"]=np.log(d["gdppc_ppp"])
X=sm.add_constant(d[["l_pop","l_gdp","DAR"]]); m=sm.OLS(d["l_ai"],X).fit()
print("\n===== OLS log(AI_works) ~ log(pop)+log(GDPpc)+DAR  (n=11, DESCRIPTIVE) =====")
print(m.params.round(4).to_string()); print("R2=%.3f"%m.rsquared)
d["resid"]=m.resid
print("\nResiduals (below 0 = under-converts inputs into AI output):")
print(d[["country","resid"]].sort_values("resid").round(2).to_string(index=False))

# conversion model: add R&D and researchers
d2=d.dropna(subset=["researchers_permil"]).copy()
d2["l_rsch"]=np.log(d2["rsch_total"])
X2=sm.add_constant(d2[["l_rsch","DAR"]]); m2=sm.OLS(d2["l_ai"],X2).fit()
print("\n===== OLS log(AI_works) ~ log(researchers)+DAR  (n=%d) ====="%len(d2))
print(m2.params.round(4).to_string()); print("R2=%.3f"%m2.rsquared)
df.to_csv("data/dataset_with_index.csv",index=False)
print("\nsaved dataset_with_index.csv")
