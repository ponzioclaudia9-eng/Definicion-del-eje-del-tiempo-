import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from matplotlib.gridspec import GridSpec
import os

# Configurar matplotlib para mostrar ventanas
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 11
plt.ion()  # Modo interactivo activado

# ==================== DEFINICIONES INICIALES ====================
fs = 1000  # Frecuencia de muestreo
t = np.linspace(0, 1, fs)

print("="*70)
print("TRANSFORMADA DE FOURIER - DEFINICIONES FUNDAMENTALES")
print("="*70)
print(f"\nfs (Frecuencia de Muestreo): {fs} Hz")
print(f"Duración total: 1 segundo")
print(f"Número de muestras: {len(t)}")

# ==================== FUNCIONES ====================
def crear_senal_sinusoidal(frecuencia, amplitud=1.0):
    senal = amplitud * np.sin(2 * np.pi * frecuencia * t)
    return senal

def crear_senal_compuesta(componentes):
    senal = np.zeros_like(t)
    for frecuencia, amplitud in componentes:
        senal += amplitud * np.sin(2 * np.pi * frecuencia * t)
    return senal

def calcular_transformada_fourier(senal):
    espectro = fft(senal)
    magnitudes = np.abs(espectro) / len(senal)
    frecuencias = fftfreq(len(senal), 1/fs)
    fase = np.angle(espectro)
    return magnitudes, frecuencias, fase

def encontrar_frecuencias_dominantes(senal, n_picos=5):
    magnitudes, frecuencias, _ = calcular_transformada_fourier(senal)
    idx_positivas = frecuencias >= 0
    magnitudes_pos = magnitudes[idx_positivas]
    frecuencias_pos = frecuencias[idx_positivas]
    indices_maximos = np.argsort(magnitudes_pos)[-n_picos:][::-1]
    frecuencias_dominantes = []
    for idx in indices_maximos:
        if magnitudes_pos[idx] > 0.01:
            frecuencias_dominantes.append({
                'frecuencia': frecuencias_pos[idx],
                'magnitud': magnitudes_pos[idx]
            })
    return frecuencias_dominantes

def visualizar_senal_completa(senal, titulo, frecuencias_esperadas=None, nombre_archivo=None):
    """
    Crea visualización completa de la señal y la muestra en pantalla
    """
    mag, freq, fase = calcular_transformada_fourier(senal)
    
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)
    
    color_tiempo = '#1f77b4'
    color_freq = '#ff7f0e'
    color_fase = '#2ca02c'
    color_db = '#d62728'
    
    # 1. Señal en el tiempo (completa)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(t, senal, linewidth=2, color=color_tiempo, label='Señal')
    ax1.set_xlabel('Tiempo (s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Amplitud', fontsize=12, fontweight='bold')
    ax1.set_title(f'{titulo} - Dominio del Tiempo (Señal Completa)', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=11)
    
    # 2. Zoom en primeros 300ms
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(t[:300], senal[:300], linewidth=2.5, color=color_tiempo, marker='o', markersize=2, alpha=0.8)
    ax2.set_xlabel('Tiempo (s)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Amplitud', fontsize=11, fontweight='bold')
    ax2.set_title('Zoom (0-0.3s)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 3. Espectro de Magnitud (lineal)
    ax3 = fig.add_subplot(gs[1, 0])
    idx_pos = freq >= 0
    ax3.plot(freq[idx_pos], mag[idx_pos], linewidth=2.5, color=color_freq, label='Magnitud')
    ax3.fill_between(freq[idx_pos], mag[idx_pos], alpha=0.4, color=color_freq)
    if frecuencias_esperadas:
        for f_esp in frecuencias_esperadas:
            ax3.axvline(x=f_esp, color='red', linestyle='--', alpha=0.8, linewidth=2.5, label=f'{f_esp} Hz')
    ax3.set_xlabel('Frecuencia (Hz)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Magnitud', fontsize=11, fontweight='bold')
    ax3.set_title('Espectro de Magnitud (Escala Lineal)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(0, 150)
    ax3.legend(fontsize=10)
    
    # 4. Espectro en dB
    ax4 = fig.add_subplot(gs[1, 1])
    magnitudes_db = 20 * np.log10(mag + 1e-10)
    ax4.plot(freq[idx_pos], magnitudes_db[idx_pos], linewidth=2.5, color=color_db)
    ax4.fill_between(freq[idx_pos], magnitudes_db[idx_pos], alpha=0.4, color=color_db)
    if frecuencias_esperadas:
        for f_esp in frecuencias_esperadas:
            ax4.axvline(x=f_esp, color='red', linestyle='--', alpha=0.8, linewidth=2.5)
    ax4.set_xlabel('Frecuencia (Hz)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Magnitud (dB)', fontsize=11, fontweight='bold')
    ax4.set_title('Espectro en Escala Logarítmica (dB)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_xlim(0, 150)
    
    # 5. Espectro de Fase
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(freq[idx_pos][:len(fase)//2], fase[idx_pos][:len(fase)//2], 
             linewidth=2, color=color_fase, marker='o', markersize=2, alpha=0.7)
    ax5.set_xlabel('Frecuencia (Hz)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Fase (radianes)', fontsize=11, fontweight='bold')
    ax5.set_title('Espectro de Fase', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.set_xlim(0, 150)
    
    # 6. Gráfico de barras - Frecuencias dominantes
    ax6 = fig.add_subplot(gs[2, :2])
    freq_dom = encontrar_frecuencias_dominantes(senal, 10)
    if freq_dom:
        freq_vals = [f['frecuencia'] for f in freq_dom]
        mag_vals = [f['magnitud'] for f in freq_dom]
        colors_bar = plt.cm.viridis(np.linspace(0, 1, len(freq_vals)))
        bars = ax6.bar(range(len(freq_vals)), mag_vals, color=colors_bar, edgecolor='black', linewidth=2)
        ax6.set_xticks(range(len(freq_vals)))
        ax6.set_xticklabels([f'{f:.1f}\nHz' for f in freq_vals], rotation=0, ha='center', fontsize=10, fontweight='bold')
        ax6.set_ylabel('Magnitud', fontsize=11, fontweight='bold')
        ax6.set_title('🎯 Frecuencias Dominantes (Top 10)', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Valores en las barras
        for bar, mag in zip(bars, mag_vals):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mag:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 7. Información estadística
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    energia_total = np.sum(mag)
    potencia = np.mean(senal**2)
    amplitud_max = np.max(np.abs(senal))
    
    info_text = f"""
📊 INFORMACIÓN DE LA SEÑAL
{'='*40}

⚙️  Parámetros de Muestreo:
  • fs: {fs} Hz
  • Duración: {len(t)/fs:.3f} s
  • Muestras: {len(t)}
  • Resolución: {1/fs*1000:.2f} ms

📈 Estadísticas:
  • Amplitud Máxima: {amplitud_max:.4f}
  • Energía Total: {energia_total:.4f}
  • Potencia Promedio: {potencia:.4f}
  • Nyquist: {fs/2:.0f} Hz

🎵 Componentes Principales:
"""
    
    for i, freq_d in enumerate(freq_dom[:5], 1):
        info_text += f"\n  {i}. {freq_d['frecuencia']:.1f} Hz"
        info_text += f" ({freq_d['magnitud']:.4f})"
    
    ax7.text(0.05, 0.95, info_text, transform=ax7.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=1))
    
    plt.suptitle(titulo, fontsize=18, fontweight='bold', y=0.995)
    
    # Guardar imagen
    if nombre_archivo:
        plt.savefig(nombre_archivo, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✅ Imagen guardada: {nombre_archivo}")
    
    # Mostrar la imagen en pantalla
    plt.show(block=False)
    plt.pause(0.5)  # Pausa para asegurar que se dibuje
    
    return fig


# ==================== EJEMPLO 1: SINUSOIDE SIMPLE ====================
print("\n" + "="*70)
print("EJEMPLO 1: Señal Sinusoidal Simple (10 Hz)")
print("="*70)

senal1 = crear_senal_sinusoidal(frecuencia=10, amplitud=1.0)
print(f"\nFrecuencias dominantes:")
for i, freq_dom in enumerate(encontrar_frecuencias_dominantes(senal1, 3), 1):
    print(f"  {i}. {freq_dom['frecuencia']:.2f} Hz (Magnitud: {freq_dom['magnitud']:.4f})")

visualizar_senal_completa(senal1, "🔵 EJEMPLO 1: Señal Sinusoidal 10 Hz",
                         frecuencias_esperadas=[10],
                         nombre_archivo='ejemplo1_senal_simple.png')

input("Presiona ENTER para ver el siguiente ejemplo...")


# ==================== EJEMPLO 2: SEÑAL COMPUESTA ====================
print("\n" + "="*70)
print("EJEMPLO 2: Señal Compuesta (5 Hz + 15 Hz + 25 Hz)")
print("="*70)

componentes = [(5, 1.0), (15, 0.7), (25, 0.5)]
senal2 = crear_senal_compuesta(componentes)

print(f"\nFrecuencias dominantes:")
for i, freq_dom in enumerate(encontrar_frecuencias_dominantes(senal2, 5), 1):
    print(f"  {i}. {freq_dom['frecuencia']:.2f} Hz (Magnitud: {freq_dom['magnitud']:.4f})")

visualizar_senal_completa(senal2, "🟠 EJEMPLO 2: Señal Compuesta (5 + 15 + 25 Hz)",
                         frecuencias_esperadas=[5, 15, 25],
                         nombre_archivo='ejemplo2_senal_compuesta.png')

input("Presiona ENTER para ver el siguiente ejemplo...")


# ==================== EJEMPLO 3: SEÑAL CON RUIDO ====================
print("\n" + "="*70)
print("EJEMPLO 3: Señal con Ruido (30 Hz + ruido gaussiano)")
print("="*70)

np.random.seed(42)
senal_limpia = crear_senal_sinusoidal(frecuencia=30, amplitud=1.0)
ruido = 0.5 * np.random.randn(len(t))
senal3 = senal_limpia + ruido

print(f"\nFrecuencias dominantes (con ruido):")
for i, freq_dom in enumerate(encontrar_frecuencias_dominantes(senal3, 3), 1):
    print(f"  {i}. {freq_dom['frecuencia']:.2f} Hz (Magnitud: {freq_dom['magnitud']:.4f})")

visualizar_senal_completa(senal3, "🔴 EJEMPLO 3: Señal con Ruido (30 Hz)",
                         frecuencias_esperadas=[30],
                         nombre_archivo='ejemplo3_senal_ruido.png')

input("Presiona ENTER para ver el siguiente ejemplo...")


# ==================== EJEMPLO 4: CHIRP ====================
print("\n" + "="*70)
print("EJEMPLO 4: Señal Chirp (Frecuencia Variable 5-50 Hz)")
print("="*70)

frecuencia_inicial = 5
frecuencia_final = 50
fase_chirp = 2 * np.pi * (frecuencia_inicial * t + 
                          (frecuencia_final - frecuencia_inicial) * t**2 / (2 * 1))
senal4 = np.sin(fase_chirp)

visualizar_senal_completa(senal4, "🟣 EJEMPLO 4: Señal Chirp (5-50 Hz)",
                         nombre_archivo='ejemplo4_chirp.png')

print("\n" + "="*70)
print("✅ TODOS LOS EJEMPLOS COMPLETADOS")
print("="*70)
print("""
📁 ARCHIVOS GUARDADOS:
   • ejemplo1_senal_simple.png
   • ejemplo2_senal_compuesta.png
   • ejemplo3_senal_ruido.png
   • ejemplo4_chirp.png

🎯 CONCEPTOS PRINCIPALES:
   1. Dominio del Tiempo: Cómo varía la amplitud
   2. Transformada de Fourier: Convierte a frecuencias
   3. Espectro de Magnitud: Qué tan fuerte es cada frecuencia
   4. Espectro de Fase: La posición de cada componente
   5. Escala en dB: Visualización logarítmica

💡 APLICACIONES REALES:
   • Análisis de música y audio
   • Procesamiento de señales sísmicas
   • Compresión MP3 y JPEG
   • Filtrado de ruido en comunicaciones
   • Análisis médico (EEG, ECG)
""")

# Mantener ventanas abiertas
plt.show()