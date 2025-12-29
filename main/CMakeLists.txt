idf_component_register(
    SRCS "main.c"
    INCLUDE_DIRS "."
    REQUIRES 
        fatfs 
        vfs
        wear_levelling 
        tinyusb
        driver
        esp_partition
        spiffs
)
