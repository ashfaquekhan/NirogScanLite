#include <stdio.h>
#include <string.h>
#include <math.h>
#include "esp_log.h"
#include "esp_vfs_fat.h"
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tinyusb.h"
#include "tinyusb_default_config.h"
#include "tinyusb_msc.h"

static const char *TAG = "PDF_DEMO";
#define BASE_PATH "/data"

void app_main(void)
{
    ESP_LOGI(TAG, "=== Advanced PDF Demo ===");
    ESP_LOGI(TAG, "Initializing FatFS storage");

    static wl_handle_t wl_handle = WL_INVALID_HANDLE;
    
    const esp_partition_t *data_partition = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_DATA_FAT, NULL);
    
    if (data_partition == NULL) {
        ESP_LOGE(TAG, "Failed to find FAT partition");
        return;
    }

    esp_err_t err = wl_mount(data_partition, &wl_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to mount wear leveling");
        return;
    }

    tinyusb_msc_storage_config_t storage_cfg = {
        .mount_point = TINYUSB_MSC_STORAGE_MOUNT_APP,
        .medium.wl_handle = wl_handle,
        .fat_fs = {
            .base_path = BASE_PATH,
            .config.max_files = 5,
        },
    };

    tinyusb_msc_storage_handle_t storage_handle = NULL;
    err = tinyusb_msc_new_storage_spiflash(&storage_cfg, &storage_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create MSC storage");
        return;
    }

    vTaskDelay(pdMS_TO_TICKS(100));

    ESP_LOGI(TAG, "Creating advanced PDF with graphs...");
    FILE *f = fopen(BASE_PATH "/report.pdf", "wb");
    if (!f) {
        ESP_LOGE(TAG, "Failed to create PDF file");
        return;
    }

    static char content[8192];
    int len = 0;

    len += snprintf(content + len, sizeof(content) - len,
        "BT /F2 20 Tf 180 750 Td (NirogScan Health Report) Tj ET\n");
    len += snprintf(content + len, sizeof(content) - len,
        "BT /F1 10 Tf 50 725 Td (Patient: Demo User  |  Date: 2025-12-29  |  ID: NS-001) Tj ET\n");
    len += snprintf(content + len, sizeof(content) - len,
        "0.7 0.7 0.7 RG 0.5 w 50 715 m 560 715 l S\n");

    len += snprintf(content + len, sizeof(content) - len,
        "BT /F2 12 Tf 50 690 Td (ECG Waveform) Tj ET\n");

    len += snprintf(content + len, sizeof(content) - len,
        "0.95 0.98 0.95 rg 50 540 510 140 re f\n");

    len += snprintf(content + len, sizeof(content) - len,
        "0.8 0.9 0.8 RG 0.3 w\n");
    for (int i = 0; i <= 10; i++) {
        int x = 50 + i * 51;
        len += snprintf(content + len, sizeof(content) - len, "%d 540 m %d 680 l S\n", x, x);
    }
    for (int i = 0; i <= 7; i++) {
        int y = 540 + i * 20;
        len += snprintf(content + len, sizeof(content) - len, "50 %d m 560 %d l S\n", y, y);
    }

    len += snprintf(content + len, sizeof(content) - len,
        "0 0.6 0 RG 1.5 w\n");

    for (int i = 0; i < 200; i++) {
        float t = i * 0.03f;
        float ecg = 0;
        float phase = fmodf(t, 0.8f);

        if (phase < 0.06f) {
            ecg = 0.08f * sinf(phase * 52.36f);
        } else if (phase < 0.10f) {
            ecg = -0.12f;
        } else if (phase < 0.14f) {
            float q = (phase - 0.10f) / 0.04f;
            ecg = -0.12f + 1.2f * sinf(q * 3.14159f);
        } else if (phase < 0.18f) {
            ecg = -0.15f;
        } else if (phase < 0.35f) {
            float s = (phase - 0.18f) / 0.17f;
            ecg = -0.15f + 0.15f * s;
        } else if (phase < 0.50f) {
            float tw = (phase - 0.35f) / 0.15f;
            ecg = 0.22f * sinf(tw * 3.14159f);
        }

        int x = 50 + (i * 510 / 200);
        int y = 610 + (int)(ecg * 55);

        if (i == 0) {
            len += snprintf(content + len, sizeof(content) - len, "%d %d m\n", x, y);
        } else {
            len += snprintf(content + len, sizeof(content) - len, "%d %d l\n", x, y);
        }
    }
    len += snprintf(content + len, sizeof(content) - len, "S\n");

    len += snprintf(content + len, sizeof(content) - len,
        "BT /F1 8 Tf 55 525 Td (25mm/s | 10mm/mV | HR: 75 BPM) Tj ET\n");

    len += snprintf(content + len, sizeof(content) - len,
        "BT /F2 12 Tf 50 500 Td (PPG Waveform) Tj ET\n");

    len += snprintf(content + len, sizeof(content) - len,
        "0.95 0.95 1.0 rg 50 400 250 90 re f\n");

    len += snprintf(content + len, sizeof(content) - len,
        "0.8 0.2 0.2 RG 1.2 w\n");
    for (int i = 0; i < 80; i++) {
        float t = i * 0.08f;
        float ppg = 0.35f * sinf(t * 1.2f) + 0.12f * sinf(t * 2.4f + 0.5f);
        int x = 50 + (i * 250 / 80);
        int y = 445 + (int)(ppg * 40);
        if (i == 0) {
            len += snprintf(content + len, sizeof(content) - len, "%d %d m\n", x, y);
        } else {
            len += snprintf(content + len, sizeof(content) - len, "%d %d l\n", x, y);
        }
    }
    len += snprintf(content + len, sizeof(content) - len, "S\n");

    len += snprintf(content + len, sizeof(content) - len,
        "0.5 0.2 0.5 RG 1.0 w\n");
    for (int i = 0; i < 80; i++) {
        float t = i * 0.08f;
        float ppg = 0.40f * sinf(t * 1.2f) + 0.15f * sinf(t * 2.4f + 0.5f);
        int x = 50 + (i * 250 / 80);
        int y = 445 + (int)(ppg * 40);
        if (i == 0) {
            len += snprintf(content + len, sizeof(content) - len, "%d %d m\n", x, y);
        } else {
            len += snprintf(content + len, sizeof(content) - len, "%d %d l\n", x, y);
        }
    }
    len += snprintf(content + len, sizeof(content) - len, "S\n");

    len += snprintf(content + len, sizeof(content) - len,
        "BT /F1 8 Tf 55 388 Td (Red 660nm | IR 940nm | SpO2: 98%%) Tj ET\n");

    len += snprintf(content + len, sizeof(content) - len,
        "BT /F2 12 Tf 320 500 Td (Heart Rate Trend) Tj ET\n");

    len += snprintf(content + len, sizeof(content) - len,
        "0.95 0.95 0.95 rg 320 400 240 90 re f\n");
    len += snprintf(content + len, sizeof(content) - len,
        "0.5 0.5 0.5 RG 0.5 w 320 400 240 90 re S\n");

    int hr[] = {72, 74, 78, 82, 85, 80, 77, 74, 73, 75, 76, 74};
    for (int i = 0; i < 12; i++) {
        int bh = (hr[i] - 65) * 3;
        int bx = 325 + i * 19;

        float g = (float)(hr[i] - 70) / 20.0f;
        if (g < 0) g = 0;
        if (g > 1) g = 1;

        len += snprintf(content + len, sizeof(content) - len,
            "%.2f %.2f 0.3 rg %d 405 15 %d re f\n",
            0.3f + g * 0.5f, 0.6f - g * 0.3f, bx, bh);
    }

    len += snprintf(content + len, sizeof(content) - len,
        "BT /F1 7 Tf 330 392 Td (0h) Tj 400 392 Td (6h) Tj 470 392 Td (12h) Tj ET\n");

    len += snprintf(content + len, sizeof(content) - len,
        "BT /F2 12 Tf 50 360 Td (Vital Signs) Tj ET\n");

    len += snprintf(content + len, sizeof(content) - len,
        "0.9 0.95 1.0 rg 50 280 150 70 re f\n");
    len += snprintf(content + len, sizeof(content) - len,
        "0.3 0.5 0.8 RG 2 w 50 280 150 70 re S\n");
    len += snprintf(content + len, sizeof(content) - len,
        "BT /F2 16 Tf 90 320 Td (75) Tj /F1 10 Tf ( BPM) Tj ET\n");
    len += snprintf(content + len, sizeof(content) - len,
        "BT /F1 9 Tf 85 300 Td (Heart Rate) Tj ET\n");

    len += snprintf(content + len, sizeof(content) - len,
        "0.9 1.0 0.9 rg 220 280 150 70 re f\n");
    len += snprintf(content + len, sizeof(content) - len,
        "0.2 0.7 0.3 RG 2 w 220 280 150 70 re S\n");
    len += snprintf(content + len, sizeof(content) - len,
        "BT /F2 16 Tf 265 320 Td (98) Tj /F1 10 Tf ( %%) Tj ET\n");
    len += snprintf(content + len, sizeof(content) - len,
        "BT /F1 9 Tf 270 300 Td (SpO2) Tj ET\n");

    len += snprintf(content + len, sizeof(content) - len,
        "1.0 0.95 0.9 rg 390 280 150 70 re f\n");
    len += snprintf(content + len, sizeof(content) - len,
        "0.8 0.5 0.2 RG 2 w 390 280 150 70 re S\n");
    len += snprintf(content + len, sizeof(content) - len,
        "BT /F2 16 Tf 430 320 Td (36.5) Tj /F1 10 Tf ( C) Tj ET\n");
    len += snprintf(content + len, sizeof(content) - len,
        "BT /F1 9 Tf 430 300 Td (Temperature) Tj ET\n");

    len += snprintf(content + len, sizeof(content) - len,
        "BT /F2 12 Tf 50 250 Td (Health Metrics) Tj ET\n");

    float metrics[] = {0.85f, 0.70f, 0.92f, 0.60f};
    const char *names[] = {"HRV", "Stress", "Recovery", "Activity"};
    for (int i = 0; i < 4; i++) {
        int by = 220 - i * 22;
        len += snprintf(content + len, sizeof(content) - len,
            "0.85 0.85 0.85 rg 100 %d 200 15 re f\n", by);
        len += snprintf(content + len, sizeof(content) - len,
            "0.3 0.6 0.3 rg 100 %d %d 15 re f\n", by, (int)(200 * metrics[i]));
        len += snprintf(content + len, sizeof(content) - len,
            "BT /F1 9 Tf 55 %d Td (%s) Tj ET\n", by + 3, names[i]);
        len += snprintf(content + len, sizeof(content) - len,
            "BT /F1 8 Tf 310 %d Td (%.0f%%) Tj ET\n", by + 3, metrics[i] * 100);
    }

    len += snprintf(content + len, sizeof(content) - len,
        "BT /F2 10 Tf 400 250 Td (Device: NirogScan Pro) Tj ET\n");
    len += snprintf(content + len, sizeof(content) - len,
        "BT /F1 9 Tf 400 235 Td (Firmware: v2.9.0) Tj ET\n");
    len += snprintf(content + len, sizeof(content) - len,
        "BT /F1 9 Tf 400 220 Td (Battery: 85%%) Tj ET\n");

    len += snprintf(content + len, sizeof(content) - len,
        "1 0 0 rg\n");
    len += snprintf(content + len, sizeof(content) - len,
        "500 180 m 510 195 l 520 180 l 520 165 510 155 500 165 l 500 180 l f\n");
    len += snprintf(content + len, sizeof(content) - len,
        "480 180 m 490 195 l 500 180 l 500 165 490 155 480 165 l 480 180 l f\n");

    len += snprintf(content + len, sizeof(content) - len,
        "0.6 0.6 0.6 RG 0.5 w 50 100 m 560 100 l S\n");
    len += snprintf(content + len, sizeof(content) - len,
        "BT /F1 8 Tf 50 85 Td (This report is for informational purposes. Consult a doctor for medical advice.) Tj ET\n");
    len += snprintf(content + len, sizeof(content) - len,
        "BT /F1 8 Tf 50 70 Td (Generated by NirogScan | Report ID: RPT-2025-001) Tj ET\n");

    fprintf(f, "%%PDF-1.4\n");
    fprintf(f, "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n");
    fprintf(f, "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n");
    fprintf(f, "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
               "/Contents 4 0 R /Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> >> endobj\n");
    fprintf(f, "4 0 obj << /Length %d >> stream\n%sendstream endobj\n", len, content);
    fprintf(f, "5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n");
    fprintf(f, "6 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >> endobj\n");
    fprintf(f, "xref\n0 7\n");
    fprintf(f, "0000000000 65535 f \n");
    fprintf(f, "0000000009 00000 n \n");
    fprintf(f, "0000000058 00000 n \n");
    fprintf(f, "0000000115 00000 n \n");
    fprintf(f, "0000000250 00000 n \n");
    fprintf(f, "0000%06d 00000 n \n", 300 + len);
    fprintf(f, "0000%06d 00000 n \n", 380 + len);
    fprintf(f, "trailer << /Size 7 /Root 1 0 R >>\n");
    fprintf(f, "startxref\n%d\n%%%%EOF\n", 450 + len);

    fclose(f);
    ESP_LOGI(TAG, "PDF created: %d bytes of content", len);

    vTaskDelay(pdMS_TO_TICKS(100));

    ESP_LOGI(TAG, "Starting USB MSC");
    const tinyusb_config_t tusb_cfg = TINYUSB_DEFAULT_CONFIG();
    err = tinyusb_driver_install(&tusb_cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to install TinyUSB");
        return;
    }

    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "SUCCESS! Connect USB to get report.pdf");
    ESP_LOGI(TAG, "========================================");
}