import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D #Viktig å ha 3D, engang i tiden var det skikkelig kult. Synd jeg ikke fikk sett Minecraft movie i 3D
from matplotlib import cm

# Fysiske konstanter
gasskonstant = 8.314  # J/mol·K
optimal_temp = 93 + 273.15  # K (93°C)
maks_løselighet = 0.3  #Søkte på google, hva er max løselighet på vann, fikk ish 0.3 ergo ish 30%
partikkel_tetthet = 500  # kg/m³

# Kaffe-komponenter og deres egenskaper
sammensetning = {
    'Syrer': {'maks_løselighet': 0.08, 'aktiveringsenergi': 38000, 'farge': 'red'},
    'Sukkerarter': {'maks_løselighet': 0.12, 'aktiveringsenergi': 45000, 'farge': 'green'},
    'Bitterstoffer': {'maks_løselighet': 0.1, 'aktiveringsenergi': 52000, 'farge': 'blue'}
} #Ett rask google søk, og dette var det jeg fant ut var de mest avgjørdene komponente som bidrar til smak.

def viskositet(T):
    
    return 0.001 * np.exp(2000 * (1/T - 1/373.15))

def ekstraksjonsrate(konsentrasjon, kverningsgrad, temp, trykk, dose=18e-3, 
                     maks_løs=0.3, Ea=45000):
    
    k = 5e-3 * np.exp(-Ea / (gasskonstant * temp))
    
    høyde_bed = 0.03  # meter, altså da høyden på "Portafilteret(nerdeemoji)"
    permeabilitet = (kverningsgrad ** 2) / 150  # Kozeny-Carman-relasjon
    mu = viskositet(temp)
    
    gjennomstrømning = (np.pi * 0.03**2 * permeabilitet * trykk) / (mu * høyde_bed)
    
    overflate = 3 * dose / (partikkel_tetthet * (kverningsgrad / 2))
    
    return k * overflate * (maks_løs - konsentrasjon) * gjennomstrømning

def ekstraksjons_ode(y, t, d, T, P, dose=18e-3, maks_løs=0.3, Ea=45000):
   
    C, S = y
    rate = ekstraksjonsrate(C, d, T, P, dose, maks_løs, Ea)
    if rate * t > S and S > 0:
        rate = S / t
    
    return [rate, -rate]

def ekstraksjonsmodell(kverningsgrad, temp=optimal_temp, trykk=9e5, dose=18e-3, 
                       kontakttid=25, maks_løs=0.3, Ea=45000):

    t = np.linspace(0.01, kontakttid, 100)
    starttilstand = [0, maks_løs]
    
    løsning = odeint(ekstraksjons_ode, starttilstand, t, args=(kverningsgrad, temp, trykk, dose, maks_løs, Ea))
    
    return t, løsning[:, 0], løsning[:, 1]

def lag_espresso_analyse():
    fig = plt.figure(figsize=(15, 15))
    
    # Plot 1: Effekt av kverningsgrad på ekstraksjon
    ax1 = fig.add_subplot(221)
    kverningsgrader = [150e-6, 250e-6, 350e-6, 450e-6]
    for d in kverningsgrader:
        t, C, _ = ekstraksjonsmodell(d)
        ax1.plot(t, C, label=f"{d*1e6:.0f} μm")
    ax1.set_xlabel('Tid (s)')
    ax1.set_ylabel('Konsentrasjon')
    ax1.set_title('Effekt av kverningsgrad på ekstraksjon')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Optimal kverningsgrad og følsomhet
    ax2 = fig.add_subplot(222)
    d_range = np.linspace(100e-6, 500e-6, 40)
    utbytter = [ekstraksjonsmodell(d)[1][-1] / maks_løselighet * 100 for d in d_range]
    følsomhet = np.gradient(utbytter, d_range * 1e6)
    
    ax2.plot(d_range*1e6, utbytter, 'b-', label='Utbytte (%)')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(d_range*1e6, følsomhet, 'r--', label='Følsomhet')
    
    ax2.set_xlabel('Kverningsgrad (μm)')
    ax2.set_ylabel('Ekstraksjonsutbytte (%)', color='b')
    ax2_twin.set_ylabel('Følsomhet', color='r')
    ax2.set_title('Optimal kverningsgrad')
    ax2.axvspan(180, 380, alpha=0.2, color='green', label='Optimalt område')
    
    linjer1, etiketter1 = ax2.get_legend_handles_labels()
    linjer2, etiketter2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(linjer1 + linjer2, etiketter1 + etiketter2)
    ax2.grid(True)

    # Plot 3: Ekstraksjon av ulike smaksstoffer
    ax3 = fig.add_subplot(223)
    for navn, egenskaper in sammensetning.items():
        t, C, _ = ekstraksjonsmodell(
            250e-6, maks_løs=egenskaper['maks_løselighet'], Ea=egenskaper['aktiveringsenergi']
        )
        ax3.plot(t, C, color=egenskaper['farge'], label=navn)
    
    t = np.linspace(0.01, 25, 100)
    total = np.zeros_like(t)
    for egenskaper in sammensetning.values():
        _, C, _ = ekstraksjonsmodell(
            250e-6, maks_løs=egenskaper['maks_løselighet'], Ea=egenskaper['aktiveringsenergi']
        )
        total += C

    ax3.plot(t, total, 'k--', label='Total')
    ax3.set_xlabel('Tid (s)')
    ax3.set_ylabel('Konsentrasjon')
    ax3.set_title('Ekstraksjon av smaksprofiler')
    ax3.legend()
    ax3.grid(True)

    # Plot 4: 3D – Temperatur og kverningsgrad vs utbytte
    ax4 = fig.add_subplot(224, projection='3d')
    d_vals = np.linspace(150e-6, 400e-6, 10)
    T_vals = np.linspace(88, 96, 10) + 273.15
    D, T = np.meshgrid(d_vals, T_vals)
    Z = np.zeros_like(D)

    for i in range(len(T_vals)):
        for j in range(len(d_vals)):
            _, C, _ = ekstraksjonsmodell(D[i, j], temp=T[i, j])
            Z[i, j] = C[-1] / maks_løselighet * 100

    flate = ax4.plot_surface(D*1e6, T - 273.15, Z, cmap=cm.viridis, linewidth=0, antialiased=True)
    ax4.set_xlabel('Kverningsgrad (μm)')
    ax4.set_ylabel('Temperatur (°C)')
    ax4.set_zlabel('Utbytte (%)')
    ax4.set_title('Sammenheng mellom temperatur, kverningsgrad og utbytte')
    
    fargebar = fig.colorbar(flate, ax=ax4, shrink=0.5, aspect=5)
    fargebar.set_label('Ekstraksjonsutbytte (%)')

    plt.tight_layout()
    plt.show()
    return fig


lag_espresso_analyse()
