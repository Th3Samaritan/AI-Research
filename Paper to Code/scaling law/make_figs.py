# Generate dark-themed figures for the Scaling Laws PDF, matching the site palette.
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

INK="#0A0D13"; PANEL="#141A28"; LINE="#262E43"; TXT="#E8ECF5"; MUT="#8B95AC"
AMBER="#F5A524"; COOL="#5B8DEF"; GOOD="#5BE3A6"; VIO="#B98CFF"; BAD="#EF5B7B"

plt.rcParams.update({
    "figure.facecolor": INK, "axes.facecolor": INK, "savefig.facecolor": INK,
    "axes.edgecolor": LINE, "axes.labelcolor": MUT, "text.color": TXT,
    "xtick.color": MUT, "ytick.color": MUT, "grid.color": "#1C2333",
    "font.size": 11, "axes.linewidth": 1.1, "figure.dpi": 200,
})

# ---- constants (Table 5/6) ----
aN,Nc=0.076,8.8e13; aD,Dc=0.095,5.4e13; aCm,Ccm=0.050,3.1e8
aB,Bs=0.21,2.1e8; aS,Sc=0.76,2.1e3
Ne,pN=1.3e9,0.73; De,pD=2.0e10,0.27; Se,pS=5.4e3,0.03

L_N=lambda N:(Nc/N)**aN
L_D=lambda D:(Dc/D)**aD
L_Cm=lambda C:(Ccm/C)**aCm
Bcrit=lambda L:Bs/L**(1/aB)
L_ND=lambda N,D:((Nc/N)**(aN/aD)+Dc/D)**aD
L_NS=lambda N,S:(Nc/N)**aN+(Sc/S)**aS
N_opt=lambda C:Ne*C**pN
D_opt=lambda C:De*C**pD
S_opt=lambda C:Se*C**pS
D_epoch=lambda C:4e10*C**0.26

def style(ax):
    ax.grid(True,which="both",lw=0.7,alpha=0.5)
    for s in ax.spines.values(): s.set_color(LINE)
    ax.tick_params(labelsize=9)

def save(fig,name):
    fig.tight_layout()
    fig.savefig(name,bbox_inches="tight",pad_inches=0.12)
    plt.close(fig); print("wrote",name)

# 1) three basic laws
fig,ax=plt.subplots(figsize=(6.6,3.5))
N=np.logspace(3,13,200); ax.loglog(N,L_N(N),color=AMBER,lw=2.6,label=r"$L(N)=(N_c/N)^{\alpha_N}$")
D=np.logspace(5,13,200); ax.loglog(D,L_D(D),color=COOL,lw=2.6,label=r"$L(D)=(D_c/D)^{\alpha_D}$")
C=np.logspace(-4,6,200); ax.loglog(C,L_Cm(C),color=GOOD,lw=2.6,label=r"$L(C_{min})=(C_c/C_{min})^{\alpha}$")
ax.set_xlabel("scale factor  (N params · D tokens · $C_{min}$ PF-days)"); ax.set_ylabel("test loss  (nats/token)")
ax.set_ylim(1,8); ax.legend(facecolor=PANEL,edgecolor=LINE,labelcolor=TXT,fontsize=8.5,loc="upper right")
style(ax); save(fig,"fig_basics.png")

# 2) Bcrit(L)
fig,ax=plt.subplots(figsize=(6.6,3.3))
L=np.linspace(0.8,7,200); ax.loglog(L,Bcrit(L),color=VIO,lw=2.8)
ax.set_xlabel("loss L  (nats/token — smaller = better trained)"); ax.set_ylabel(r"$B_{crit}$  (tokens)")
ax.annotate("doubles per ~13% loss drop",xy=(2.5,Bcrit(2.5)),xytext=(3.2,3e5),color=MUT,fontsize=8.5,
    arrowprops=dict(arrowstyle="->",color=MUT))
style(ax); save(fig,"fig_batch.png")

# 3) L(N,D) overfitting family
fig,ax=plt.subplots(figsize=(6.6,3.5))
D=np.logspace(6,13,200)
for Nv,col in [(1e6,"#5C6680"),(1e7,COOL),(1e8,AMBER),(1e9,GOOD)]:
    ax.semilogx(D,L_ND(Nv,D),color=col,lw=2.4,label=f"N={Nv:.0e}")
    ax.axhline(L_N(Nv),color=col,lw=1,ls=":",alpha=0.7)
ax.set_xlabel("dataset size D  (tokens)"); ax.set_ylabel("L(N,D)  (nats/token)")
ax.set_ylim(2,6); ax.legend(facecolor=PANEL,edgecolor=LINE,labelcolor=TXT,fontsize=8.5)
ax.set_title("solid = finite-data loss · dotted = capacity floor L(N)",color=MUT,fontsize=9)
style(ax); save(fig,"fig_nd.png")

# 4) learning curves
fig,ax=plt.subplots(figsize=(6.6,3.3))
S=np.logspace(2,6,200)
for Nv,col in [(1e6,"#5C6680"),(1e7,COOL),(1e8,AMBER),(1e9,GOOD)]:
    ax.loglog(S,L_NS(Nv,S),color=col,lw=2.4,label=f"N={Nv:.0e}")
ax.set_xlabel(r"$S_{min}$  (optimization steps)"); ax.set_ylabel("L(N,S)  (nats/token)")
ax.set_ylim(2,7); ax.legend(facecolor=PANEL,edgecolor=LINE,labelcolor=TXT,fontsize=8.5)
style(ax); save(fig,"fig_train.png")

# 5) optimal allocation
fig,ax=plt.subplots(figsize=(6.6,3.5))
C=np.logspace(-4,6,200)
ax.loglog(C,N_opt(C),color=AMBER,lw=2.6,label=r"$N_{opt}\propto C^{0.73}$")
ax.loglog(C,D_opt(C),color=COOL,lw=2.6,label=r"$D_{opt}\propto C^{0.27}$")
ax.loglog(C,S_opt(C),color=GOOD,lw=2.6,label=r"$S_{min}\propto C^{0.03}$")
ax.set_xlabel(r"compute budget $C_{min}$  (PF-days)"); ax.set_ylabel("optimal quantity")
ax.legend(facecolor=PANEL,edgecolor=LINE,labelcolor=TXT,fontsize=8.5,loc="upper left")
style(ax); save(fig,"fig_alloc.png")

# 6) breakdown crossing
fig,ax=plt.subplots(figsize=(6.6,3.3))
C=np.logspace(-2,7,200)
ax.loglog(C,L_Cm(C),color=AMBER,lw=2.8,label=r"$L(C_{min})$")
ax.loglog(C,L_D(D_epoch(C)),color=COOL,lw=2.8,ls="--",label=r"$L(D(C_{min}))$ data-limited")
ax.axvline(1e4,color=BAD,lw=1.2,ls=":"); ax.scatter([1e4],[L_Cm(1e4)],color=BAD,zorder=5,s=40)
ax.annotate("breakdown\n~$10^4$ PF-days\nL*≈1.7",xy=(1e4,L_Cm(1e4)),xytext=(30,2.4),color=BAD,fontsize=8.5,
    arrowprops=dict(arrowstyle="->",color=BAD))
ax.set_xlabel(r"compute budget $C_{min}$  (PF-days)"); ax.set_ylabel("L  (nats/token)")
ax.set_ylim(1,6); ax.legend(facecolor=PANEL,edgecolor=LINE,labelcolor=TXT,fontsize=8.5,loc="upper right")
style(ax); save(fig,"fig_breakdown.png")

print("all figures done")
