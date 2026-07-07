# Dark-themed figures for the Chinchilla PDF, computed from the paper's fit.
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK="#0A0D13"; PANEL="#141A28"; LINE="#262E43"; TXT="#E8ECF5"; MUT="#8B95AC"
AMBER="#F5A524"; COOL="#5B8DEF"; GOOD="#5BE3A6"; VIO="#B98CFF"; BAD="#EF5B7B"
plt.rcParams.update({
    "figure.facecolor": INK, "axes.facecolor": INK, "savefig.facecolor": INK,
    "axes.edgecolor": LINE, "axes.labelcolor": MUT, "text.color": TXT,
    "xtick.color": MUT, "ytick.color": MUT, "grid.color": "#1C2333",
    "font.size": 11, "axes.linewidth": 1.1, "figure.dpi": 200})

# --- fitted constants (Eq. 10) ---
E,A,B,al,be = 1.69, 406.4, 410.7, 0.34, 0.28
L   = lambda N,D: E + A/N**al + B/D**be
G   = ((al*A)/(be*B))**(1/(al+be))
a,b = be/(al+be), al/(al+be)
N3  = lambda C: G*(C/6)**a          # Approach-3 frontier
D3  = lambda C: (1/G)*(C/6)**b
N12 = lambda C: np.sqrt(C/120)      # Approach-1/2 (a=b=0.5, 20 tok/param)
isoL= lambda N,C: L(N, C/(6*N))
Nk  = lambda C,C0=5.76e23: N3(C0)*(C/C0)**0.73   # Kaplan rule, anchored
Lk  = lambda C: L(Nk(C), C/(6*Nk(C)))

def style(ax):
    ax.grid(True,which="both",lw=0.7,alpha=0.5)
    for s in ax.spines.values(): s.set_color(LINE)
    ax.tick_params(labelsize=9)
def save(fig,name):
    fig.tight_layout(); fig.savefig(name,bbox_inches="tight",pad_inches=0.12)
    plt.close(fig); print("wrote",name)

# 1) parametric law: loss vs D at several N
fig,ax=plt.subplots(figsize=(6.6,3.5))
D=np.logspace(9,14,200)
for Nv,col in [(1e9,"#5C6680"),(1e10,COOL),(7e10,AMBER),(1e12,GOOD)]:
    ax.semilogx(D,L(Nv,D),color=col,lw=2.4,label=f"N={Nv:.0e}")
ax.axhline(E,color=BAD,lw=1.2,ls=":")
ax.text(1.5e9,E+0.03,"E = 1.69 — entropy of text",color=BAD,fontsize=8.5)
ax.set_xlabel("D — training tokens"); ax.set_ylabel("L(N,D)  (nats/token)")
ax.set_ylim(1.6,3.2); ax.legend(facecolor=PANEL,edgecolor=LINE,labelcolor=TXT,fontsize=8.5)
style(ax); save(fig,"fig_law.png")

# 2) IsoFLOP valleys
fig,ax=plt.subplots(figsize=(6.6,3.5))
N=np.logspace(7,13,300)
for C,col in [(1e19,"#5C6680"),(1e20,COOL),(1e21,AMBER),(1e22,GOOD),(1e23,VIO)]:
    ax.semilogx(N,isoL(N,C),color=col,lw=2.2,label=f"C={C:.0e}")
    No=N3(C); ax.plot([No],[isoL(No,C)],"o",ms=6,color=col)
ax.set_xlabel("N — parameters"); ax.set_ylabel("final loss (nats/token)")
ax.set_ylim(1.7,4); ax.set_title("each fixed budget has an optimal model — the valley",color=MUT,fontsize=9)
ax.legend(facecolor=PANEL,edgecolor=LINE,labelcolor=TXT,fontsize=8)
style(ax); save(fig,"fig_isoflop.png")

# 3) frontier + landmark models
fig,ax=plt.subplots(figsize=(6.6,3.6))
C=np.logspace(19,26,200)
ax.loglog(C,N3(C),color=AMBER,lw=2.6,label=r"$N_{opt}$ (Approach 3)")
ax.loglog(C,N12(C),color=AMBER,lw=1.4,ls="--",label=r"$N_{opt}$ (A1/2: 20 tok/param)")
ax.loglog(C,Nk(C),color=BAD,lw=1.6,ls=":",label=r"Kaplan rule $N\propto C^{0.73}$")
models={"Chinchilla":(70e9,1.4e12,GOOD),"Gopher":(280e9,300e9,BAD),
        "GPT-3":(175e9,300e9,BAD),"MT-NLG":(530e9,270e9,BAD)}
for name,(Nv,Dv,col) in models.items():
    ax.plot([6*Nv*Dv],[Nv],"o",ms=7,color=col)
    ax.annotate(name,(6*Nv*Dv,Nv),textcoords="offset points",xytext=(6,4),color=col,fontsize=8)
ax.set_xlabel("C — training compute (FLOPs)"); ax.set_ylabel("parameters")
ax.legend(facecolor=PANEL,edgecolor=LINE,labelcolor=TXT,fontsize=8,loc="upper left")
style(ax); save(fig,"fig_frontier.png")

# 4) the allocation penalty: L_opt vs Kaplan-rule loss
fig,ax=plt.subplots(figsize=(6.6,3.3))
C=np.logspace(20,27,200)
ax.semilogx(C,L(N3(C),D3(C)),color=GOOD,lw=2.8,label="compute-optimal L(C)")
ax.semilogx(C,Lk(C),color=BAD,lw=2.2,ls="--",label="Kaplan-rule allocation")
ax.axvline(5.76e23,color=AMBER,lw=1.2,ls=":")
ax.text(7e23,2.7,"Gopher budget",color=AMBER,fontsize=8.5)
ax.set_xlabel("C — training compute (FLOPs)"); ax.set_ylabel("achievable loss (nats/token)")
ax.legend(facecolor=PANEL,edgecolor=LINE,labelcolor=TXT,fontsize=8.5)
style(ax); save(fig,"fig_penalty.png")

print("all figures done")
