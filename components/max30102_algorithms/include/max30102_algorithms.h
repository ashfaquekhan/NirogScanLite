/**
 * @file max30102_algorithms.h
 * @brief Combined Heart Rate and SpO2 Algorithms for MAX30102 Sensor
 * 
 * This header provides a unified interface for both heart rate detection using the 
 * Peripheral Beat Amplitude (PBA) algorithm and SpO2 calculation algorithms optimized
 * for the MAX30102 pulse oximetry sensor.
 * 
 * KEY CONCEPTS:
 * 
 * 1. PHOTOPLETHYSMOGRAPHY (PPG):
 *    - Uses light absorption changes in blood vessels to detect heartbeats
 *    - RED light (660nm): Highly absorbed by oxygenated hemoglobin
 *    - IR light (880nm): Absorbed by both oxy and deoxy hemoglobin
 *    - The ratio of these absorptions determines SpO2
 * 
 * 2. PBA ALGORITHM:
 *    - Peripheral Beat Amplitude algorithm detects beat-to-beat variations
 *    - Uses AC component analysis and zero-crossing detection
 *    - Requires signal conditioning: DC removal, low-pass filtering
 * 
 * 3. SPO2 CALCULATION:
 *    - Based on Beer-Lambert law for light absorption
 *    - Ratio R = (AC_red/DC_red) / (AC_ir/DC_ir)
 *    - SpO2 = f(R) using empirical calibration curves
 * 
 * @author Y3X Innovatech
 * @date 2025
 * 
 * @note This is for educational/development purposes only, not for medical diagnosis
 */

#ifndef MAX30102_ALGORITHMS_H
#define MAX30102_ALGORITHMS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

/* ============================================================================
 * ALGORITHM CONFIGURATION CONSTANTS
 * ============================================================================ */

/** @brief Sampling frequency for algorithm calculations (Hz) */
#define MAX30102_SAMPLING_FREQ          25

/** @brief Buffer size for SpO2 calculations (4 seconds of data) */
#define MAX30102_BUFFER_SIZE            (MAX30102_SAMPLING_FREQ * 4)

/** @brief Moving average filter size */
#define MAX30102_MA4_SIZE               4

/** @brief Maximum number of peaks to detect */
#define MAX30102_MAX_PEAKS              15

/** @brief FIR filter coefficient count */
#define MAX30102_FIR_COEFFS             12

/** @brief Circular buffer size for FIR filter */
#define MAX30102_FIR_BUFFER_SIZE        32

/* ============================================================================
 * DATA STRUCTURES
 * ============================================================================ */

/**
 * @brief Heart rate detection algorithm state
 * 
 * This structure maintains the internal state of the PBA algorithm
 * including signal processing variables and edge detection states.
 */
typedef struct {
    // AC signal analysis
    int16_t ir_ac_signal_current;       ///< Current AC component
    int16_t ir_ac_signal_previous;      ///< Previous AC component for edge detection
    int16_t ir_ac_signal_min;           ///< Minimum value in current cycle
    int16_t ir_ac_signal_max;           ///< Maximum value in current cycle
    int16_t ir_ac_max;                  ///< Global AC maximum
    int16_t ir_ac_min;                  ///< Global AC minimum
    
    // DC estimation
    int16_t ir_average_estimated;       ///< Estimated DC component
    int32_t ir_avg_reg;                 ///< DC estimation accumulator
    
    // Edge detection states
    int16_t positive_edge;              ///< Rising edge detection flag
    int16_t negative_edge;              ///< Falling edge detection flag
    
    // FIR filter state
    int16_t fir_buffer[MAX30102_FIR_BUFFER_SIZE];  ///< Circular buffer for FIR
    uint8_t fir_offset;                 ///< Current position in FIR buffer
    
    // Beat detection parameters
    uint32_t last_beat_time;            ///< Timestamp of last detected beat
    bool initialized;                   ///< Algorithm initialization flag
} max30102_hr_state_t;

/**
 * @brief SpO2 calculation results
 * 
 * Contains the calculated SpO2 value and validity flags
 */
typedef struct {
    int32_t spo2_value;                 ///< Calculated SpO2 percentage
    int8_t spo2_valid;                  ///< 1 if SpO2 calculation is valid
    int32_t heart_rate;                 ///< Calculated heart rate (BPM)
    int8_t hr_valid;                    ///< 1 if heart rate calculation is valid
    int32_t ratio_average;              ///< R-ratio used for SpO2 calculation
} max30102_spo2_result_t;

/* ============================================================================
 * FUNCTION PROTOTYPES
 * ============================================================================ */

/**
 * @brief Initialize the heart rate detection algorithm
 * 
 * This function initializes the PBA algorithm state machine and prepares
 * all internal variables for heart rate detection.
 * 
 * @param[out] hr_state Pointer to heart rate algorithm state structure
 * 
 * IMPLEMENTATION DETAILS:
 * - Clears all signal processing buffers
 * - Initializes FIR filter coefficients
 * - Resets edge detection states
 * - Sets up DC estimation parameters
 */
void max30102_hr_init(max30102_hr_state_t *hr_state);

/**
 * @brief Process a single IR sample for heart rate detection
 * 
 * This is the core PBA algorithm implementation that processes each incoming
 * IR sample and detects heart beats using sophisticated signal processing.
 * 
 * ALGORITHM FLOW:
 * 1. DC Estimation: Remove low-frequency baseline drift
 * 2. AC Extraction: Get the pulsatile component
 * 3. Low-pass Filtering: Remove high-frequency noise
 * 4. Zero-crossing Detection: Find rising/falling edges
 * 5. Peak Analysis: Validate beat authenticity
 * 
 * @param[in,out] hr_state Heart rate algorithm state
 * @param[in] ir_sample Raw IR sensor reading from MAX30102
 * @return true if a valid heartbeat was detected, false otherwise
 * 
 * SIGNAL PROCESSING THEORY:
 * - The IR signal contains both DC (tissue absorption) and AC (blood volume changes)
 * - Heartbeats appear as periodic changes in the AC component
 * - Zero-crossing detection finds the transition points
 * - Amplitude thresholding validates genuine beats vs. noise
 */
bool max30102_check_for_beat(max30102_hr_state_t *hr_state, int32_t ir_sample);

/**
 * @brief Calculate heart rate and SpO2 from buffer data
 * 
 * This function implements the complete SpO2 calculation algorithm using
 * buffered red and IR data. It performs peak detection, ratio calculation,
 * and empirical SpO2 estimation.
 * 
 * ALGORITHM OVERVIEW:
 * 1. Peak Detection: Find valleys in inverted IR signal
 * 2. Heart Rate: Calculate from peak intervals
 * 3. AC/DC Extraction: Get pulsatile vs. baseline components
 * 4. Ratio Calculation: R = (AC_red/DC_red) / (AC_ir/DC_ir)
 * 5. SpO2 Lookup: Convert ratio to SpO2 using calibration table
 * 
 * @param[in] ir_buffer Array of IR sensor readings
 * @param[in] buffer_length Number of samples in buffer
 * @param[in] red_buffer Array of RED sensor readings
 * @param[out] result Structure containing calculated values
 * 
 * MATHEMATICAL FOUNDATION:
 * SpO2 calculation is based on the Beer-Lambert law:
 * A = ε × c × l
 * Where: A=absorbance, ε=extinction coefficient, c=concentration, l=path length
 * 
 * The ratio of ratios eliminates path length dependency:
 * R = (ΔA_red/A_red) / (ΔA_ir/A_ir)
 * SpO2 = f(R) determined empirically
 */
void max30102_calculate_spo2(uint32_t *ir_buffer, int32_t buffer_length,
                           uint32_t *red_buffer, max30102_spo2_result_t *result);

/**
 * @brief Get the current heart rate estimate
 * 
 * Returns the most recent heart rate calculation based on beat-to-beat intervals.
 * Uses averaging to provide stable readings.
 * 
 * @param[in] hr_state Heart rate algorithm state
 * @return Heart rate in beats per minute, or -1 if invalid
 */
int32_t max30102_get_heart_rate(const max30102_hr_state_t *hr_state);

/**
 * @brief Reset the algorithm state
 * 
 * Clears all internal buffers and resets the algorithm to initial state.
 * Useful when starting a new measurement session.
 * 
 * @param[out] hr_state Heart rate algorithm state to reset
 */
void max30102_reset_algorithm(max30102_hr_state_t *hr_state);

/* ============================================================================
 * UTILITY FUNCTIONS (Internal)
 * ============================================================================ */

/**
 * @brief DC estimation using exponential moving average
 * 
 * This function implements a first-order IIR filter to estimate the DC
 * component of the PPG signal. The DC component represents the baseline
 * tissue absorption and must be removed to isolate the AC pulsatile signal.
 * 
 * FILTER EQUATION:
 * y[n] = α × x[n] + (1-α) × y[n-1]
 * Where α determines the filter time constant
 * 
 * @param[in,out] accumulator Pointer to filter state accumulator
 * @param[in] sample Current input sample
 * @return Estimated DC component
 */
int16_t max30102_average_dc_estimator(int32_t *accumulator, uint16_t sample);

/**
 * @brief Low-pass FIR filter implementation
 * 
 * Applies a 12-tap finite impulse response filter to remove high-frequency
 * noise while preserving the heart rate signal content. The filter is
 * designed with a cutoff frequency appropriate for heart rate detection.
 * 
 * FILTER CHARACTERISTICS:
 * - Type: Linear phase FIR
 * - Taps: 12 coefficients
 * - Cutoff: ~5 Hz (appropriate for HR signals)
 * - Implementation: Circular buffer for efficiency
 * 
 * @param[in,out] hr_state Heart rate state containing filter buffer
 * @param[in] input Raw input sample
 * @return Filtered output sample
 */
int16_t max30102_low_pass_fir_filter(max30102_hr_state_t *hr_state, int16_t input);

#ifdef __cplusplus
}
#endif

#endif /* MAX30102_ALGORITHMS_H */