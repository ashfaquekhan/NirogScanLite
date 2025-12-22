#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "driver/i2c.h"
#include "driver/gpio.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_heap_caps.h"
#include "esp_check.h"
#include <inttypes.h>
#include <string.h>

#define ENABLE_LOGS 1

static const char *TAG = "NirogScan";

#define I2C_MASTER_SCL_IO_0           9
#define I2C_MASTER_SDA_IO_0           8
#define I2C_MASTER_FREQ_HZ_0          400000
#define I2C_MASTER_NUM_0              I2C_NUM_0

#define I2C_MASTER_SCL_IO_1           11
#define I2C_MASTER_SDA_IO_1           10
#define I2C_MASTER_FREQ_HZ_1          400000
#define I2C_MASTER_NUM_1              I2C_NUM_1

#define MAX30101_I2C_ADDR             0x57
#define MAX17048_I2C_ADDR             0x36

#define ECG_ADC_CHANNEL               ADC_CHANNEL_1
#define ECG_ADC_ATTEN                 ADC_ATTEN_DB_12
#define ECG_GPIO_NUM                  GPIO_NUM_2

#define LO_PLUS_PIN                   GPIO_NUM_16
#define LO_MINUS_PIN                  GPIO_NUM_17

#define MAX30101_REG_FIFO_WR_PTR      0x04
#define MAX30101_REG_FIFO_OVF_COUNTER 0x05
#define MAX30101_REG_FIFO_RD_PTR      0x06
#define MAX30101_REG_FIFO_DATA        0x07
#define MAX30101_REG_FIFO_CONFIG      0x08
#define MAX30101_REG_MODE_CONFIG      0x09
#define MAX30101_REG_SPO2_CONFIG      0x0A
#define MAX30101_REG_LED1_PA          0x0C
#define MAX30101_REG_LED2_PA          0x0D
#define MAX30101_REG_LED3_PA          0x0E
#define MAX30101_REG_PART_ID          0xFF

#define MAX17048_REG_VCELL            0x02
#define MAX17048_REG_SOC              0x04

#define PPG_SAMPLE_RATE_HZ            64
#define ECG_SAMPLE_RATE_HZ            50
#define PPG_TIMER_PERIOD_US           (1000000 / PPG_SAMPLE_RATE_HZ)
#define ECG_TIMER_PERIOD_US           (1000000 / ECG_SAMPLE_RATE_HZ)
#define ALGORITHM_BUFFER_SIZE         256

#define PPG_READY_BIT                 BIT0
#define ECG_READY_BIT                 BIT1

#if ENABLE_LOGS
    #define LOG_I(tag, format, ...) ESP_LOGI(tag, format, ##__VA_ARGS__)
    #define LOG_E(tag, format, ...) ESP_LOGE(tag, format, ##__VA_ARGS__)
#else
    #define LOG_I(tag, format, ...)
    #define LOG_E(tag, format, ...)
#endif

typedef struct {
    int16_t ir_ac_signal_current;
    int16_t ir_ac_signal_previous;
    int16_t ir_ac_signal_min;
    int16_t ir_ac_signal_max;
    int16_t ir_ac_max;
    int16_t ir_ac_min;
    int16_t ir_average_estimated;
    int32_t ir_avg_reg;
    int16_t positive_edge;
    int16_t negative_edge;
    uint32_t last_beat_time;
    bool initialized;
} hr_state_t;

typedef struct {
    int32_t spo2_value;
    int8_t spo2_valid;
    int32_t heart_rate;
    int8_t hr_valid;
    int32_t ratio_average;
} spo2_result_t;

typedef struct {
    uint32_t red;
    uint32_t ir;
    int ecg;
    float battery_voltage;
    float battery_percent;
    int lo_plus;
    int lo_minus;
    int32_t heart_rate;
    int32_t spo2;
    int8_t hr_valid;
    int8_t spo2_valid;
} sensor_data_t;

static adc_oneshot_unit_handle_t adc1_handle;
static sensor_data_t sensor_data = {0};
static EventGroupHandle_t sync_event_group;
static esp_timer_handle_t ppg_timer;
static esp_timer_handle_t ecg_timer;
static volatile uint32_t battery_counter = 0;

static uint32_t red_buffer[ALGORITHM_BUFFER_SIZE];
static uint32_t ir_buffer[ALGORITHM_BUFFER_SIZE];
static int buffer_index = 0;
static hr_state_t hr_state = {0};

static const uint8_t spo2_lookup_table[184] = {
    95, 95, 95, 96, 96, 96, 97, 97, 97, 97, 97, 98, 98, 98, 98, 98, 
    99, 99, 99, 99, 99, 99, 99, 99, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 99, 99, 99, 99,
    99, 99, 99, 99, 98, 98, 98, 98, 98, 98, 97, 97, 97, 97, 96, 96,
    96, 96, 95, 95, 95, 94, 94, 94, 93, 93, 93, 92, 92, 92, 91, 91,
    90, 90, 89, 89, 89, 88, 88, 87, 87, 86, 86, 85, 85, 84, 84, 83,
    82, 82, 81, 81, 80, 80, 79, 78, 78, 77, 76, 76, 75, 74, 74, 73,
    72, 72, 71, 70, 69, 69, 68, 67, 66, 66, 65, 64, 63, 62, 62, 61,
    60, 59, 58, 57, 56, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46,
    45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 31, 30, 29,
    28, 27, 26, 25, 23, 22, 21, 20, 19, 17, 16, 15, 14, 12, 11, 10,
    9, 7, 6, 5, 3, 2, 1
};

static esp_err_t i2c_master_init_0(void) {
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO_0,
        .scl_io_num = I2C_MASTER_SCL_IO_0,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_MASTER_FREQ_HZ_0,
    };
    ESP_RETURN_ON_ERROR(i2c_param_config(I2C_MASTER_NUM_0, &conf), TAG, "I2C0 config failed");
    return i2c_driver_install(I2C_MASTER_NUM_0, conf.mode, 0, 0, 0);
}

static esp_err_t i2c_master_init_1(void) {
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO_1,
        .scl_io_num = I2C_MASTER_SCL_IO_1,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_MASTER_FREQ_HZ_1,
    };
    ESP_RETURN_ON_ERROR(i2c_param_config(I2C_MASTER_NUM_1, &conf), TAG, "I2C1 config failed");
    return i2c_driver_install(I2C_MASTER_NUM_1, conf.mode, 0, 0, 0);
}

static esp_err_t max30101_write_reg(uint8_t reg, uint8_t data) {
    uint8_t write_buf[2] = {reg, data};
    return i2c_master_write_to_device(I2C_MASTER_NUM_0, MAX30101_I2C_ADDR, write_buf, 2, pdMS_TO_TICKS(100));
}

static esp_err_t max30101_read_reg(uint8_t reg, uint8_t *data) {
    return i2c_master_write_read_device(I2C_MASTER_NUM_0, MAX30101_I2C_ADDR, &reg, 1, data, 1, pdMS_TO_TICKS(100));
}

static esp_err_t max30101_read_fifo(uint8_t *buffer, size_t len) {
    uint8_t reg = MAX30101_REG_FIFO_DATA;
    return i2c_master_write_read_device(I2C_MASTER_NUM_0, MAX30101_I2C_ADDR, &reg, 1, buffer, len, pdMS_TO_TICKS(100));
}

static esp_err_t max30101_get_fifo_status(uint8_t *available_samples) {
    uint8_t wr_ptr, rd_ptr;
    esp_err_t ret;
    
    ret = max30101_read_reg(MAX30101_REG_FIFO_WR_PTR, &wr_ptr);
    if (ret != ESP_OK) return ret;
    
    ret = max30101_read_reg(MAX30101_REG_FIFO_RD_PTR, &rd_ptr);
    if (ret != ESP_OK) return ret;
    
    *available_samples = (wr_ptr - rd_ptr) & 0x1F;
    return ESP_OK;
}

static bool check_for_beat(int32_t ir_sample) {
    bool beat_detected = false;
    
    int32_t ir_avg_reg = hr_state.ir_avg_reg;
    
    ir_avg_reg += ((((long) ir_sample << 15) - ir_avg_reg) >> 4);
    hr_state.ir_avg_reg = ir_avg_reg;
    hr_state.ir_average_estimated = (int16_t) (ir_avg_reg >> 15);
    
    int16_t ir_ac = (int16_t) ((long) ir_sample - (long) hr_state.ir_average_estimated);
    hr_state.ir_ac_signal_previous = hr_state.ir_ac_signal_current;
    hr_state.ir_ac_signal_current = ir_ac;
    
    hr_state.ir_ac_max = ir_ac;
    hr_state.ir_ac_min = ir_ac;
    
    if (hr_state.ir_ac_signal_previous * hr_state.ir_ac_signal_current < 0) {
        if (hr_state.ir_ac_signal_current > 0 && !hr_state.positive_edge) {
            hr_state.positive_edge = 1;
            hr_state.negative_edge = 0;
            hr_state.ir_ac_max = hr_state.ir_ac_signal_current;
        }
        
        if (hr_state.ir_ac_signal_current < 0 && hr_state.positive_edge) {
            hr_state.positive_edge = 0;
            hr_state.negative_edge = 1;
            hr_state.ir_ac_min = hr_state.ir_ac_signal_current;
            
            int16_t threshold = (hr_state.ir_ac_max - hr_state.ir_ac_min) >> 2;
            
            if (hr_state.ir_ac_max > 20 && hr_state.ir_ac_min < -20 && 
                (hr_state.ir_ac_max - hr_state.ir_ac_min) > 20 && 
                hr_state.ir_ac_max > threshold && (-hr_state.ir_ac_min) > threshold) {
                    
                uint32_t current_time = esp_timer_get_time() / 1000;
                if (current_time - hr_state.last_beat_time > 300) {
                    beat_detected = true;
                    hr_state.last_beat_time = current_time;
                }
            }
        }
    }
    
    return beat_detected;
}

static void calculate_spo2(uint32_t *ir_buf, uint32_t *red_buf, int32_t buf_len, spo2_result_t *result) {
    result->spo2_valid = 0;
    result->hr_valid = 0;
    result->spo2_value = -999;
    result->heart_rate = -999;
    
    if (buf_len < 100) return;
    
    uint32_t ir_mean = 0, red_mean = 0;
    for (int i = 0; i < buf_len; i++) {
        ir_mean += ir_buf[i];
        red_mean += red_buf[i];
    }
    ir_mean /= buf_len;
    red_mean /= buf_len;
    
    int32_t ir_ac_sum = 0, red_ac_sum = 0;
    for (int i = 0; i < buf_len; i++) {
        int32_t ir_ac = (int32_t)ir_buf[i] - (int32_t)ir_mean;
        int32_t red_ac = (int32_t)red_buf[i] - (int32_t)red_mean;
        
        if (ir_ac < 0) ir_ac = -ir_ac;
        if (red_ac < 0) red_ac = -red_ac;
        
        ir_ac_sum += ir_ac;
        red_ac_sum += red_ac;
    }
    
    if (ir_ac_sum == 0 || red_mean == 0 || ir_mean == 0) return;
    
    int32_t ratio_x100 = (red_ac_sum * ir_mean * 100) / (red_mean * ir_ac_sum);
    
    if (ratio_x100 >= 50 && ratio_x100 < 184) {
        result->spo2_value = spo2_lookup_table[ratio_x100 - 50];
        result->spo2_valid = 1;
        result->ratio_average = ratio_x100;
    }
}

static esp_err_t max30101_init(void) {
    uint8_t part_id;
    ESP_RETURN_ON_ERROR(max30101_read_reg(MAX30101_REG_PART_ID, &part_id), TAG, "Part ID read failed");
    LOG_I(TAG, "MAX30101 Part ID: 0x%02X", part_id);

    ESP_RETURN_ON_ERROR(max30101_write_reg(MAX30101_REG_MODE_CONFIG, 0x40), TAG, "Reset failed");
    vTaskDelay(pdMS_TO_TICKS(100));

    ESP_RETURN_ON_ERROR(max30101_write_reg(MAX30101_REG_FIFO_WR_PTR, 0x00), TAG, "WR PTR failed");
    ESP_RETURN_ON_ERROR(max30101_write_reg(MAX30101_REG_FIFO_OVF_COUNTER, 0x00), TAG, "OVF failed");
    ESP_RETURN_ON_ERROR(max30101_write_reg(MAX30101_REG_FIFO_RD_PTR, 0x00), TAG, "RD PTR failed");
    ESP_RETURN_ON_ERROR(max30101_write_reg(MAX30101_REG_FIFO_CONFIG, 0x4F), TAG, "FIFO config failed");
    ESP_RETURN_ON_ERROR(max30101_write_reg(MAX30101_REG_MODE_CONFIG, 0x03), TAG, "Mode config failed");
    ESP_RETURN_ON_ERROR(max30101_write_reg(MAX30101_REG_SPO2_CONFIG, 0x24), TAG, "SpO2 config failed");
    ESP_RETURN_ON_ERROR(max30101_write_reg(MAX30101_REG_LED1_PA, 0x24), TAG, "LED1 failed");
    ESP_RETURN_ON_ERROR(max30101_write_reg(MAX30101_REG_LED2_PA, 0x24), TAG, "LED2 failed");

    return ESP_OK;
}

static esp_err_t max17048_read_reg(uint8_t reg, uint16_t *data) {
    uint8_t buffer[2];
    esp_err_t ret = i2c_master_write_read_device(I2C_MASTER_NUM_1, MAX17048_I2C_ADDR, &reg, 1, buffer, 2, pdMS_TO_TICKS(100));
    if (ret == ESP_OK) {
        *data = (buffer[0] << 8) | buffer[1];
    }
    return ret;
}

static esp_err_t adc_init(void) {
    adc_oneshot_unit_init_cfg_t init_config = {
        .unit_id = ADC_UNIT_1,
    };
    ESP_RETURN_ON_ERROR(adc_oneshot_new_unit(&init_config, &adc1_handle), TAG, "ADC init failed");

    adc_oneshot_chan_cfg_t config = {
        .bitwidth = ADC_BITWIDTH_DEFAULT,
        .atten = ECG_ADC_ATTEN,
    };
    return adc_oneshot_config_channel(adc1_handle, ECG_ADC_CHANNEL, &config);
}

static void gpio_init_all(void) {
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << LO_PLUS_PIN) | (1ULL << LO_MINUS_PIN),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&io_conf);
}

static void IRAM_ATTR ppg_timer_callback(void* arg) {
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    xEventGroupSetBitsFromISR(sync_event_group, PPG_READY_BIT, &xHigherPriorityTaskWoken);
    if (xHigherPriorityTaskWoken) {
        portYIELD_FROM_ISR();
    }
}

static void IRAM_ATTR ecg_timer_callback(void* arg) {
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    xEventGroupSetBitsFromISR(sync_event_group, ECG_READY_BIT, &xHigherPriorityTaskWoken);
    if (xHigherPriorityTaskWoken) {
        portYIELD_FROM_ISR();
    }
}

static void ppg_task(void *pvParameters) {
    uint8_t fifo_data[6];
    uint8_t available_samples;
    uint32_t red, ir;
    static uint32_t sample_count = 0;

    while (1) {
        xEventGroupWaitBits(sync_event_group, PPG_READY_BIT, pdTRUE, pdFALSE, portMAX_DELAY);

        if (max30101_get_fifo_status(&available_samples) == ESP_OK && available_samples > 0) {
            for (uint8_t i = 0; i < available_samples; i++) {
                if (max30101_read_fifo(fifo_data, 6) == ESP_OK) {
                    red = ((uint32_t)fifo_data[0] << 16) | ((uint32_t)fifo_data[1] << 8) | fifo_data[2];
                    red &= 0x3FFFF;

                    ir = ((uint32_t)fifo_data[3] << 16) | ((uint32_t)fifo_data[4] << 8) | fifo_data[5];
                    ir &= 0x3FFFF;

                    if (red > 1000 && ir > 1000) {
                        sensor_data.red = red;
                        sensor_data.ir = ir;
                        
                        red_buffer[buffer_index] = red;
                        ir_buffer[buffer_index] = ir;
                        buffer_index = (buffer_index + 1) % ALGORITHM_BUFFER_SIZE;
                        
                        check_for_beat((int32_t)ir);
                        
                        sample_count++;
                        if (sample_count % 64 == 0 && sample_count > 192) {
                            spo2_result_t result;
                            calculate_spo2(ir_buffer, red_buffer, ALGORITHM_BUFFER_SIZE, &result);
                            
                            if (result.spo2_valid) {
                                sensor_data.spo2 = result.spo2_value;
                                sensor_data.spo2_valid = 1;
                            } else {
                                sensor_data.spo2_valid = 0;
                            }
                        }
                    } else {
                        sensor_data.spo2_valid = 0;
                    }
                }
            }
        }
    }
}

static void ecg_task(void *pvParameters) {
    int adc_raw;
    uint16_t vcell, soc;
    static uint32_t ecg_sample_count = 0;

    while (1) {
        xEventGroupWaitBits(sync_event_group, ECG_READY_BIT, pdTRUE, pdFALSE, portMAX_DELAY);

        sensor_data.lo_plus = gpio_get_level(LO_PLUS_PIN);
        sensor_data.lo_minus = gpio_get_level(LO_MINUS_PIN);
        
        if (adc_oneshot_read(adc1_handle, ECG_ADC_CHANNEL, &adc_raw) == ESP_OK) {
            sensor_data.ecg = adc_raw;
        }

        ecg_sample_count++;
        
        battery_counter++;
        if (battery_counter >= 256) {
            battery_counter = 0;
            if (max17048_read_reg(MAX17048_REG_VCELL, &vcell) == ESP_OK &&
                max17048_read_reg(MAX17048_REG_SOC, &soc) == ESP_OK) {
                sensor_data.battery_voltage = (vcell >> 4) * 1.25 / 1000.0;
                sensor_data.battery_percent = (soc >> 8) + (soc & 0xFF) / 256.0;
            }
        }

        if (ecg_sample_count % 2 == 0) {
            if (sensor_data.spo2_valid) {
                printf(">ecg:%d,ir:%lu,red:%lu,spo2:%ld,%.2f,%.1f,%d,%d\n",
                    sensor_data.ecg,
                    (unsigned long)sensor_data.ir,
                    (unsigned long)sensor_data.red,
                    sensor_data.spo2,
                    sensor_data.battery_voltage,
                    sensor_data.battery_percent,
                    sensor_data.lo_minus,
                    sensor_data.lo_plus);
            } else {
                printf(">ecg:%d,ir:%lu,red:%lu,%.2f,%.1f,%d,%d\n",
                    sensor_data.ecg,
                    (unsigned long)sensor_data.ir,
                    (unsigned long)sensor_data.red,
                    sensor_data.battery_voltage,
                    sensor_data.battery_percent,
                    sensor_data.lo_minus,
                    sensor_data.lo_plus);
            }
        }
    }
}

void app_main(void) {
    LOG_I(TAG, "NirogScan Lite Starting...");
    LOG_I(TAG, "PPG Sample Rate: %d Hz, ECG Sample Rate: %d Hz", PPG_SAMPLE_RATE_HZ, ECG_SAMPLE_RATE_HZ);

    sync_event_group = xEventGroupCreate();
    if (sync_event_group == NULL) {
        LOG_E(TAG, "Failed to create event group");
        return;
    }

    memset(&hr_state, 0, sizeof(hr_state_t));

    gpio_init_all();
    vTaskDelay(pdMS_TO_TICKS(100));

    if (i2c_master_init_0() != ESP_OK) {
        LOG_E(TAG, "I2C0 init failed");
        return;
    }

    if (i2c_master_init_1() != ESP_OK) {
        LOG_E(TAG, "I2C1 init failed");
        return;
    }

    if (adc_init() != ESP_OK) {
        LOG_E(TAG, "ADC init failed");
        return;
    }

    if (max30101_init() != ESP_OK) {
        LOG_E(TAG, "MAX30101 init failed");
        return;
    }

    LOG_I(TAG, "All sensors initialized with dual sample rates");

    printf("ecg,ir,red,spo2(optional),battery_v,battery_percent,lo_minus,lo_plus\n");

    xTaskCreatePinnedToCore(ppg_task, "ppg_task", 8192, NULL, 10, NULL, 0);
    xTaskCreatePinnedToCore(ecg_task, "ecg_task", 4096, NULL, 9, NULL, 1);

    const esp_timer_create_args_t ppg_timer_args = {
        .callback = &ppg_timer_callback,
        .name = "ppg_timer"
    };

    const esp_timer_create_args_t ecg_timer_args = {
        .callback = &ecg_timer_callback,
        .name = "ecg_timer"
    };

    ESP_ERROR_CHECK(esp_timer_create(&ppg_timer_args, &ppg_timer));
    ESP_ERROR_CHECK(esp_timer_create(&ecg_timer_args, &ecg_timer));
    
    ESP_ERROR_CHECK(esp_timer_start_periodic(ppg_timer, PPG_TIMER_PERIOD_US));
    ESP_ERROR_CHECK(esp_timer_start_periodic(ecg_timer, ECG_TIMER_PERIOD_US));

    LOG_I(TAG, "Started: PPG @ %d Hz, ECG @ %d Hz with HR/SpO2 algorithms", 
          PPG_SAMPLE_RATE_HZ, ECG_SAMPLE_RATE_HZ);
}