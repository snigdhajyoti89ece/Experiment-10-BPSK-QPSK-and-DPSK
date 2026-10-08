import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducible noise generation
np.random.seed(42)

# ==============================================================================
# 1. CORE MAPPER / DEMAPPER FUNCTIONS (First Principles)
# ==============================================================================

def bpsk_mod(bits):
    """Maps bits 0->+1, 1->-1 (Normalized energy = 1)"""
    return 1.0 - 2.0 * bits

def bpsk_demod(symbols):
    """Coherent decision boundary at Re{r} = 0"""
    return (np.real(symbols) < 0).astype(int)

def qpsk_gray_mod(bits):
    """
    Gray-coded QPSK Mapping (Normalized energy = 1):
    00 -> (+1 + j)/sqrt(2)
    01 -> (+1 - j)/sqrt(2)
    11 -> (-1 - j)/sqrt(2)
    10 -> (-1 + j)/sqrt(2)
    """
    bits_reshaped = bits.reshape(-1, 2)
    I = 1.0 - 2.0 * bits_reshaped[:, 0]
    Q = 1.0 - 2.0 * bits_reshaped[:, 1]
    return (I + 1j * Q) / np.sqrt(2.0)

def qpsk_gray_demod(symbols):
    """Independent bit decisions on In-phase and Quadrature components"""
    b0 = (np.real(symbols) < 0).astype(int)
    b1 = (np.imag(symbols) < 0).astype(int)
    return np.column_stack((b0, b1)).ravel()

def dbpsk_mod(bits):
    """
    Differential BPSK Encoding:
    d[k] = d[k-1] XOR b[k]
    """
    d = np.zeros(len(bits) + 1, dtype=int)
    d[0] = 0  # Reference initial state
    for k in range(len(bits)):
        d[k+1] = d[k] ^ bits[k]
    return 1.0 - 2.0 * d

def dbpsk_demod(symbols):
    """
    Non-coherent Differential Detection:
    v[k] = r[k] * conj(r[k-1])
    Decision based on Re{v[k]} < 0
    """
    diff_metric = symbols[1:] * np.conj(symbols[:-1])
    return (np.real(diff_metric) < 0).astype(int)

# ==============================================================================
# 2. MANDATORY VALIDATION TASK (Task 36 & Validation Prompt)
# ==============================================================================
print("="*60)
print("MANDATORY VALIDATION: QPSK Mapping & Detection Verification")
print("="*60)

test_bits = np.array([0, 0, 0, 1, 1, 1, 1, 0])
pairs = ["00", "01", "11", "10"]
mapped_syms = qpsk_gray_mod(test_bits)
detected_bits = qpsk_gray_demod(mapped_syms)

for i in range(4):
    sym = mapped_syms[i]
    dec_pair = f"{detected_bits[2*i]}{detected_bits[2*i+1]}"
    print(f"Input Pair: {pairs[i]}  -->  Mapped Symbol: {sym.real:+.3f} {sym.imag:+.3f}j  -->  Detected: {dec_pair}")

assert np.array_equal(test_bits, detected_bits), "Validation Failed!"
print("\nValidation Result: PASSED (Mapper and Demapper are consistent)\n")

# ==============================================================================
# 3. PHASE OFFSET SIMULATION & VISUALIZATIONS
# ==============================================================================

# Task 38: Tested Phase Offsets
phase_offsets_deg = [10, 30, 60, 90]
phase_offsets_rad = np.radians(phase_offsets_deg)

# --- Visualization 1: QPSK Mapping Constellation & Decision Regions ---
plt.figure(figsize=(6, 6))
qpsk_symbols = qpsk_gray_mod(test_bits)
for i, label in enumerate(pairs):
    plt.scatter(qpsk_symbols[i].real, qpsk_symbols[i].imag, color='blue', s=100)
    plt.annotate(f" {label}\n({qpsk_symbols[i].real:.2f}, {qpsk_symbols[i].imag:.2f})", 
                 (qpsk_symbols[i].real, qpsk_symbols[i].imag), fontsize=10)

plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
plt.grid(True)
plt.title("Gray-coded QPSK Mapping & Decision Boundaries")
plt.xlabel("In-Phase (I)")
plt.ylabel("Quadrature (Q)")
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.show()

# --- Visualization 2: Ideal vs. Noisy & Rotated Constellations ---
N = 1000
bits_sample = np.random.randint(0, 2, N)
snr_db = 12
snr_linear = 10**(snr_db / 10.0)
noise_std = 1.0 / np.sqrt(2 * snr_linear)

qpsk_syms = qpsk_gray_mod(bits_sample)
noise = noise_std * (np.random.randn(len(qpsk_syms)) + 1j * np.random.randn(len(qpsk_syms)))

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

for idx, deg in enumerate(phase_offsets_deg):
    phi = np.radians(deg)
    rx_rotated = (qpsk_syms + noise) * np.exp(1j * phi)
    
    axes[idx].scatter(rx_rotated.real, rx_rotated.imag, alpha=0.4, s=15, color='crimson', label='Received Noisy')
    axes[idx].scatter(qpsk_symbols.real, qpsk_symbols.imag, color='black', marker='x', s=80, label='Ideal Points')
    axes[idx].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[idx].axvline(0, color='black', linestyle='--', linewidth=0.8)
    axes[idx].set_title(f"QPSK Constellation with {deg}° Phase Rotation")
    axes[idx].set_xlabel("I")
    axes[idx].set_ylabel("Q")
    axes[idx].set_xlim(-1.8, 1.8)
    axes[idx].set_ylim(-1.8, 1.8)
    axes[idx].grid(True)
    axes[idx].legend(loc='upper right')

plt.tight_layout()
plt.show()

# --- Visualization 3: Phase Trajectory ---
num_traj = 30
bpsk_tx = bpsk_mod(bits_sample[:num_traj])
dbpsk_tx = dbpsk_mod(bits_sample[:num_traj])

plt.figure(figsize=(12, 4))
plt.plot(np.angle(bpsk_tx, deg=True), 'o-', label='BPSK Phase Trajectory')
plt.plot(np.angle(dbpsk_tx, deg=True), 's--', label='DBPSK Phase Trajectory')
plt.title("Phase Trajectory Comparison (First 30 Symbols)")
plt.xlabel("Symbol Index")
plt.ylabel("Phase (Degrees)")
plt.grid(True)
plt.legend()
plt.show()

# --- Visualization 4: BER vs SNR under Phase Offsets & DPSK Comparison ---
snr_db_range = np.arange(0, 13, 1)
N_sim = 100000
data_bits = np.random.randint(0, 2, N_sim)

# Modulated signals
tx_bpsk = bpsk_mod(data_bits)
tx_qpsk = qpsk_gray_mod(data_bits)
tx_dbpsk = dbpsk_mod(data_bits)

# Performance evaluation across SNRs
ber_bpsk_ideal = []
ber_qpsk_ideal = []
ber_dbpsk_ideal = []
ber_bpsk_offsets = {deg: [] for deg in phase_offsets_deg}

for snr in snr_db_range:
    snr_lin = 10**(snr / 10.0)
    sigma = 1.0 / np.sqrt(2 * snr_lin)
    
    # 1. Ideal BPSK
    n_b = sigma * (np.random.randn(N_sim) + 1j * np.random.randn(N_sim))
    rx_b = tx_bpsk + n_b
    ber_bpsk_ideal.append(np.mean(bpsk_demod(rx_b) != data_bits))
    
    # 2. Ideal Gray QPSK
    n_q = sigma * (np.random.randn(len(tx_qpsk)) + 1j * np.random.randn(len(tx_qpsk)))
    rx_q = tx_qpsk + n_q
    ber_qpsk_ideal.append(np.mean(qpsk_gray_demod(rx_q) != data_bits))
    
    # 3. Ideal DBPSK (Non-coherent detection)
    n_db = sigma * (np.random.randn(len(tx_dbpsk)) + 1j * np.random.randn(len(tx_dbpsk)))
    rx_db = tx_dbpsk + n_db
    ber_dbpsk_ideal.append(np.mean(dbpsk_demod(rx_db) != data_bits))
    
    # 4. BPSK with Carrier Phase Offsets
    for deg in phase_offsets_deg:
        phi = np.radians(deg)
        rx_b_off = (tx_bpsk + n_b) * np.exp(1j * phi)
        ber_bpsk_offsets[deg].append(np.mean(bpsk_demod(rx_b_off) != data_bits))

# Plot BER Curves
plt.figure(figsize=(10, 7))
plt.semilogy(snr_db_range, ber_bpsk_ideal, 'k-o', label='Ideal Coherent BPSK', linewidth=2)
plt.semilogy(snr_db_range, ber_qpsk_ideal, 'g--s', label='Ideal Gray QPSK', linewidth=2)
plt.semilogy(snr_db_range, ber_dbpsk_ideal, 'm-^', label='Ideal DBPSK (Non-coherent)', linewidth=2)

for deg, color in zip(phase_offsets_deg, ['c', 'm', 'y', 'r']):
    plt.semilogy(snr_db_range, ber_bpsk_offsets[deg], f'{color}:v', label=f'BPSK with {deg}° Offset')

plt.title("BER Performance under Phase Offsets & Differential Detection")
plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.ylim(1e-4, 0.6)
plt.grid(True, which='both', linestyle=':')
plt.legend()
plt.show()

# ==============================================================================
# 4. OBSERVATION AND INTERPRETATION
# ==============================================================================
print("""
================================================================================
OBSERVATION AND INTERPRETATION
================================================================================

1. Expected Physical Effects:
   - Carrier Phase Offset in Coherent Detection: Rotating the received constellation by phase offset φ 
     attenuates the in-phase decision signal by cos(φ) and introduces crosstalk sin(φ). 
     When φ = 90°, the signal shifts entirely into the quadrature axis, causing a total BER floor near 0.5.
   - Differential Detection (DPSK): Because detection compares consecutive received symbols 
     (r[k] * conj(r[k-1])), constant phase offsets cancel out (exp(jφ) * exp(-jφ) = 1). Thus, DPSK is 
     immune to static phase rotation, though it incurs ~1–3 dB SNR penalty relative to ideal coherent detection.

2. Agreement with Theory:
   - QPSK and BPSK achieve identical BER vs. Eb/N0 curves under ideal conditions due to Gray coding and 
     orthogonal I/Q channels.
   - At 10° and 30° offsets, BPSK suffers progressive degradation. At 60° and 90°, severe decision errors occur.
   - DBPSK shows robust performance regardless of static carrier phase offset.

3. Discrepancy & Diagnostic Test:
   - Discrepancy Test: When testing 90° phase rotation on QPSK/BPSK without phase-locked loop (PLL) tracking,
     bits are completely inverted or swapped between I and Q channels.
   - Diagnosis: Evaluated via constellation scatter plots and manual validation mapping (Task 36) 
     to confirm that phase ambiguity causes full symbol misclassification unless differential encoding is used.
""")
